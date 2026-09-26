"""The reply engine flow: segments, tools, refusals, and the record."""

import base64
import json
from datetime import datetime, timezone
from types import SimpleNamespace

import discord

from AgentEliza.history import History
from AgentEliza.llm_chat import ChatEngine, ChatError, IncomingMessage
from AgentEliza.providers.base import analyze_image_impl
from tests.AgentEliza.fakes import (
    FakeApi, FakeBot, FakeCompactor, FakeConfig, FakeHarnessTools, FakeMCP,
    FakeMemory, FakePreset, FakeScopeStats,
)


def build_engine(api: FakeApi, *, harness_tools=None, stats=None, config=None, compactor=None, bot=None, mcp=None) -> ChatEngine:
    memory = FakeMemory()
    return ChatEngine(
        bot or FakeBot()
        , config or FakeConfig({"api_key": "test-key"})
        , History(memory)
        , memory
        , mcp or FakeMCP()
        , harness_tools or FakeHarnessTools()
        , stats or FakeScopeStats()
        , compactor or FakeCompactor()
        , api
    )


async def drain(engine: ChatEngine, content: str) -> list[str]:
    return [
        segment
        async for segment in engine.generate_reply(IncomingMessage(
            channel_id=100, content=content, user_id=7, user_name="Madrang", bot_name="TestBot"
        ))
    ]


def call(name: str, call_id: str = "1") -> dict:
    return {"choices": [{"message": {
        "content": ""
        , "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": "{}"}}]
    }}]}


def close(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


async def test_a_plain_reply_yields_the_closing_text() -> None:
    api = FakeApi([close("Hello there.")])
    stats = FakeScopeStats()
    engine = build_engine(api, stats=stats)
    assert await drain(engine, "hi") == ["Hello there."]
    request = api.requests[0]
    assert request["model"] == "test-model"
    assert request["messages"][0]["role"] == "system"
    assert "TestBot" in request["messages"][0]["content"]
    # The user turn carries the speaker tag.
    assert request["messages"][-1]["content"].endswith("Madrang <@7>: hi")
    # The usage of the reply landed in the stats.
    assert stats.records and stats.records[0]["prompt_tokens"] == 0


async def test_a_note_yields_collapsed_before_the_tools_run() -> None:
    first = {"choices": [{"message": {
        "content": "Let me check.\n\n\n\n"
        , "tool_calls": [{"id": "1", "function": {"name": "echo", "arguments": "{}"}}]
    }}]}
    api = FakeApi([first, close("Done.")])
    tools = FakeHarnessTools(["echo"])
    engine = build_engine(api, harness_tools=tools)
    assert await drain(engine, "hi") == ["Let me check.", "Done."]
    assert tools.calls == [("echo", {})]
    # The turn and the whole exchange stay in the session.
    session = engine.history.sessions[7]
    assert session.messages[-1]["content"] == "Done."


async def test_the_no_reply_tag_yields_nothing() -> None:
    assert await drain(build_engine(FakeApi([close("[no-reply]")])), "hi") == []


async def test_an_empty_close_after_a_note_stays_silent() -> None:
    noted = {"choices": [{"message": {
        "content": "Let me check."
        , "tool_calls": [{"id": "1", "function": {"name": "echo", "arguments": "{}"}}]
    }}]}
    api = FakeApi([noted, close("")])
    engine = build_engine(api, harness_tools=FakeHarnessTools(["echo"]))
    # The note went out: the notice must not replace it.
    assert await drain(engine, "hi") == ["Let me check."]


async def test_an_empty_close_with_no_note_yields_the_notice() -> None:
    assert await drain(build_engine(FakeApi([close("")])), "hi") == ["(empty answer)"]


async def test_a_failing_tool_becomes_the_result_text() -> None:
    api = FakeApi([call("boom"), close("It failed.")])
    tools = FakeHarnessTools(["boom"], results={"boom": RuntimeError("kaput")})
    engine = build_engine(api, harness_tools=tools)
    assert await drain(engine, "hi") == ["It failed."]
    tool_messages = [
        message for message in api.requests[-1]["messages"]
        if message.get("role") == "tool"
    ]
    assert tool_messages[0]["content"] == "Error: the tool boom failed: RuntimeError: kaput"


async def test_a_full_media_window_refuses_the_generation() -> None:
    ran: list = []

    async def handler(arguments, engine):
        ran.append(arguments)
        return "The image has been sent."

    preset = FakePreset(native=[{
        "name": "generate_image"
        , "description": "Generate an image."
        , "parameters": {"type": "object", "properties": {}}
        , "handler": handler
        , "media": "images"
    }])
    api = FakeApi([call("generate_image"), close("Done.")], preset=preset)
    stats = FakeScopeStats(refusal="Error: the hourly media window is full. It resets later.")
    engine = build_engine(api, stats=stats)
    assert await drain(engine, "hi") == ["Done."]
    # The full window stops the call before the provider: the handler never
    # runs, nothing counts, and the refusal becomes the tool result.
    assert ran == []
    assert stats.media_counts == []
    tool_messages = [
        message for message in api.requests[-1]["messages"]
        if message.get("role") == "tool"
    ]
    assert tool_messages[0]["content"] == "Error: the hourly media window is full. It resets later."


async def test_a_successful_generation_counts_and_consumes() -> None:
    async def handler(arguments, engine):
        return "The image has been sent."

    preset = FakePreset(native=[{
        "name": "generate_image"
        , "description": "Generate an image."
        , "parameters": {"type": "object", "properties": {}}
        , "handler": handler
        , "media": "images"
    }])
    api = FakeApi([call("generate_image"), close("Done.")], preset=preset)
    stats = FakeScopeStats()
    engine = build_engine(api, stats=stats)
    assert await drain(engine, "hi") == ["Done."]
    # The generation counted into the scope of the message.
    assert stats.media_counts[0].guild_id is None
    assert stats.media_counts[0].channel_id == 100
    assert stats.media_counts[0].user_id == 7
    assert stats.records[-1]["images"] == 1
    assert stats.records[-1]["tool_calls"] == 1


async def test_a_fresh_session_backfills_the_channel_history() -> None:
    earlier = SimpleNamespace(
        id=11
      , author=SimpleNamespace(id=7, bot=False, display_name="Madrang")
      , type=discord.MessageType.default
      , content="earlier"
      , attachments=[]
      , reference=None
      , guild=None
      , mentions=[]
      , created_at=datetime(2026, 9, 8, 6, 0, tzinfo=timezone.utc)
    )

    class Channel:
        def history(self, limit=100, oldest_first=False):
            async def walk():
                yield earlier

            return walk()

    class BackfillBot(FakeBot):
        user = SimpleNamespace(id=5, name="TestBot")

        def get_channel(self, channel_id):
            return Channel()

    api = FakeApi([close("Hello there.")])
    engine = build_engine(api, bot=BackfillBot())
    assert await drain(engine, "hi") == ["Hello there."]
    # The turn of the earlier message rode the fresh context.
    assert any(
        isinstance(message["content"], str) and "earlier" in message["content"]
        for message in api.requests[0]["messages"]
    )


async def test_a_blocked_usage_yields_the_notice_and_calls_nothing() -> None:
    api = FakeApi([], blocked="The provider usage is at 95% (Balance), over the 90% threshold.")
    assert await drain(build_engine(api), "hi") == [
        "The provider usage is at 95% (Balance), over the 90% threshold."
    ]
    assert api.requests == []


async def test_a_missing_api_key_yields_the_notice() -> None:
    engine = build_engine(FakeApi([]), config=FakeConfig())
    assert await drain(engine, "hi") == [
        "The API key is not set. An admin can set it with the `eliza setkey` command."
    ]


def call_with(name: str, arguments: dict, call_id: str = "1") -> dict:
    return {"choices": [{"message": {
        "content": ""
        , "tool_calls": [{"id": call_id, "function": {"name": name, "arguments": json.dumps(arguments)}}]
    }}]}


def switch_tool(record: list):
    """A native tool that moves the conversation model and records the answer."""

    async def handler(arguments, engine):
        record.append(await engine.set_conversation_model(arguments["model"]))
        return "switched"

    return {
        "name": "switch_model"
        , "description": "Switch the model."
        , "parameters": {
            "type": "object"
            , "properties": {"model": {"type": "string", "description": "The target model."}}
            , "required": ["model"]
        }
        , "handler": handler
    }


def show_tool(record: list):
    """A native tool that offers an image to the conversation and records the answer."""

    async def handler(arguments, engine):
        record.append((engine.vision_chat, await engine.show_image("photo.png", b"pngbytes", "image/png")))
        return "read"

    return {
        "name": "read_image"
        , "description": "Read an image."
        , "parameters": {"type": "object", "properties": {}}
        , "handler": handler
    }


async def test_a_switch_onto_a_smaller_window_condenses_and_stores() -> None:
    answers: list = []
    preset = FakePreset(
        native=[switch_tool(answers)]
        , context_lengths={"test-model": 1_000_000, "small": 128_000}
    )
    compactor = FakeCompactor(usage={"prompt_tokens": 100, "completion_tokens": 40, "cost": 0.1})
    stats = FakeScopeStats()
    api = FakeApi(
        [close("Hello there."), call_with("switch_model", {"model": "small"}), close("Done.")]
        , preset=preset
    )
    engine = build_engine(api, stats=stats, compactor=compactor)
    assert await drain(engine, "hi") == ["Hello there."]
    assert await drain(engine, "switch") == ["Done."]
    # The move onto the smaller window condenses first, keeping no verbatim
    # turns, and the override stores.
    assert answers == [None]
    assert compactor.calls == [(7, 0)]
    assert engine.history.sessions[7].model_override == "small"
    # The condense usage merged into the interaction stats of the reply.
    assert stats.records[-1]["prompt_tokens"] == 100
    assert stats.records[-1]["cost"] == 0.1
    # The request that runs the tool keeps its model.
    assert api.requests[-1]["model"] == "test-model"


async def test_a_failed_condense_refuses_the_switch() -> None:
    answers: list = []
    preset = FakePreset(
        native=[switch_tool(answers)]
        , context_lengths={"test-model": 1_000_000, "small": 128_000}
    )
    api = FakeApi(
        [close("Hello there."), call_with("switch_model", {"model": "small"}), close("Done.")]
        , preset=preset
    )
    engine = build_engine(api, compactor=FakeCompactor())
    await drain(engine, "hi")
    await drain(engine, "switch")
    assert len(answers) == 1
    assert answers[0].startswith("Error: the condense before the move to small failed")
    assert engine.history.sessions[7].model_override is None


async def test_the_default_restore_onto_a_smaller_window_condenses() -> None:
    answers: list = []

    async def handler(arguments, engine):
        await engine.set_conversation_model("big")
        answers.append(await engine.set_conversation_model(None))
        return "switched"

    preset = FakePreset(
        native=[{
            "name": "switch_model"
            , "description": "Switch the model."
            , "parameters": {"type": "object", "properties": {}}
            , "handler": handler
        }]
        , context_lengths={"test-model": 128_000, "big": 1_000_000}
    )
    compactor = FakeCompactor(usage={"prompt_tokens": 10})
    api = FakeApi([close("Hello there."), call("switch_model"), close("Done.")], preset=preset)
    engine = build_engine(api, compactor=compactor)
    await drain(engine, "hi")
    await drain(engine, "switch")
    # The growth stores without a condense; the restore onto the smaller
    # configured window condenses.
    assert answers == [None]
    assert compactor.calls == [(7, 0)]
    assert engine.history.sessions[7].model_override is None


async def test_the_overload_fallback_rides_the_condensed_context() -> None:
    def condense(session):
        session.apply_compaction("The summary.", len(session.messages) - 1)

    preset = FakePreset(
        context_lengths={"test-model": 1_000_000, "cheap": 128_000}
        , fallback="cheap"
    )
    compactor = FakeCompactor(usage={"prompt_tokens": 5}, apply=condense)
    api = FakeApi([
        close("Hello there.")
        , ChatError("rate_limit", "Too many requests", {"error": "the model is overloaded"})
        , close("Rescued.")
    ], preset=preset)
    engine = build_engine(api, compactor=compactor)
    assert await drain(engine, "hi") == ["Hello there."]
    segments = await drain(engine, "again")
    assert "The preset cheap answers this conversation for now." in segments[0]
    assert segments[-1] == "Rescued."
    # The move onto the smaller window condensed the session, and the retry
    # rides the condensed context with this reply's turn after it.
    assert compactor.calls == [(7, 0)]
    session = engine.history.sessions[7]
    assert session.model_override == "cheap"
    retry = api.requests[-1]
    assert retry["model"] == "cheap"
    assert [message["role"] for message in retry["messages"]] == ["system", "user"]


async def test_the_overload_fallback_keeps_the_exchange_after_the_condense() -> None:
    def condense(session):
        session.apply_compaction("The summary.", len(session.messages) - 1)

    async def echo(arguments, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model):
        return "ok"

    preset = FakePreset(
        native=[{
            "name": "echo"
            , "description": "Echo."
            , "parameters": {"type": "object", "properties": {}}
            , "handler": echo
        }]
        , context_lengths={"test-model": 1_000_000, "cheap": 128_000}
        , fallback="cheap"
    )
    compactor = FakeCompactor(usage={"prompt_tokens": 5}, apply=condense)
    api = FakeApi([
        close("Hello there.")
        , ChatError("rate_limit", "Too many requests", {"error": "the model is overloaded"})
        , call_with("echo", {})
        , close("Done.")
    ], preset=preset)
    engine = build_engine(api, compactor=compactor)
    assert await drain(engine, "hi") == ["Hello there."]
    assert (await drain(engine, "again"))[-1] == "Done."
    # The round after the retry appends to the same condensed list: its
    # exchange reaches the provider.
    last = api.requests[-1]
    assert [message["role"] for message in last["messages"]] == ["system", "user", "assistant", "tool"]
    assert last["messages"][1]["content"].endswith("Madrang <@7>: again")


async def test_a_second_overload_move_in_one_reply_keeps_the_exchange() -> None:
    class Ladder(FakePreset):
        """preset_fallback walks a ladder: every model names its successor."""

        def preset_fallback(self, model):
            return {"test-model": "cheap", "cheap": "cheaper2"}.get(model)

    def condense(session):
        session.apply_compaction("The summary.", len(session.messages) - 1)

    async def echo(arguments, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model):
        return "ok"

    preset = Ladder(
        native=[{
            "name": "echo"
            , "description": "Echo."
            , "parameters": {"type": "object", "properties": {}}
            , "handler": echo
        }]
        , context_lengths={"test-model": 1_000_000, "cheap": 128_000, "cheaper2": 64_000}
    )
    compactor = FakeCompactor(usage={"prompt_tokens": 5}, apply=condense)
    api = FakeApi([
        close("Hello there.")
        , ChatError("rate_limit", "Too many requests", {"error": "the model is overloaded"})
        , call_with("echo", {})
        , ChatError("rate_limit", "Too many requests", {"error": "the model is overloaded"})
        , close("Done.")
    ], preset=preset)
    engine = build_engine(api, compactor=compactor)
    assert await drain(engine, "hi") == ["Hello there."]
    segments = await drain(engine, "again")
    assert "The preset cheaper2 answers this conversation for now." in segments[1]
    assert segments[-1] == "Done."
    session = engine.history.sessions[7]
    assert session.model_override == "cheaper2"
    # The first move condensed; the second found an emptied session and
    # stored without another condense.
    assert compactor.calls == [(7, 0)]
    # The second retry rides the whole exchange of this reply: the user
    # turn, the call, and its result stay in order and linked.
    retry = api.requests[-1]
    assert retry["model"] == "cheaper2"
    assert [message["role"] for message in retry["messages"]] == ["system", "user", "assistant", "tool"]
    assert retry["messages"][1]["content"].endswith("Madrang <@7>: again")
    assert retry["messages"][3]["tool_call_id"] == retry["messages"][2]["tool_calls"][0]["id"]


async def test_a_vision_conversation_takes_the_image_a_tool_reads() -> None:
    record: list = []
    preset = FakePreset(native=[show_tool(record)])
    preset.vision_models = {"test-model"}
    api = FakeApi([close("Hello there."), call_with("read_image", {}), close("Done.")], preset=preset)
    engine = build_engine(api)
    assert await drain(engine, "hi") == ["Hello there."]
    assert await drain(engine, "look") == ["Done."]
    # The conversation model sees images, so the offer joined.
    assert record == [(True, True)]
    # The note rides the round after the tool result, as image parts.
    last = api.requests[-1]
    assert [message["role"] for message in last["messages"]] == [
        "system", "user", "assistant", "user", "assistant", "tool", "user"
    ]
    note = last["messages"][-1]["content"]
    assert [part["type"] for part in note] == ["text", "image_url"]
    assert note[1]["image_url"]["url"] == f"data:image/png;base64,{base64.b64encode(b'pngbytes').decode('ascii')}"


async def test_a_plain_conversation_refuses_the_image_offer() -> None:
    record: list = []
    preset = FakePreset(native=[show_tool(record)])
    api = FakeApi([close("Hello there."), call_with("read_image", {}), close("Done.")], preset=preset)
    engine = build_engine(api)
    assert await drain(engine, "hi") == ["Hello there."]
    assert await drain(engine, "look") == ["Done."]
    # The conversation model reads text only: the offer refused, no note.
    assert record == [(False, False)]
    last = api.requests[-1]
    assert [message["role"] for message in last["messages"]] == [
        "system", "user", "assistant", "user", "assistant", "tool"
    ]


async def test_analyze_image_hands_a_vision_conversation_the_image() -> None:
    calls: list = []
    fetches: list = []
    shown: list = []

    async def call_api(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "a description"}}]}

    async def fetch_url(url):
        fetches.append(url)
        return (b"img", "image/png")

    async def show_image(name, data, mime):
        shown.append((name, data, mime))
        return True

    answer = await analyze_image_impl(
        {"url": "https://example.com/pic.jpg", "question": "what is it"}
        , call_api, fetch_url, model="vision-model", vision_chat=True, show_image=show_image
    )
    assert answer.startswith("The image joined this conversation")
    # Any host downloads for the offer, and no second model answers between.
    assert fetches == ["https://example.com/pic.jpg"]
    assert shown == [("pic.jpg", b"img", "image/png")]
    assert calls == []


async def test_analyze_image_of_a_plain_conversation_answers_through_the_vision_model() -> None:
    calls: list = []
    fetches: list = []
    shown: list = []

    async def call_api(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "a description"}}]}

    async def fetch_url(url):
        fetches.append(url)
        return (b"img", "image/png")

    async def show_image(name, data, mime):
        shown.append((name, data, mime))
        return True

    answer = await analyze_image_impl(
        {"url": "https://cdn.discordapp.com/attachments/1/2/photo.png"}
        , call_api, fetch_url, model="vision-model", show_image=show_image
    )
    assert answer == "a description"
    # The Discord download feeds the vision call as a data URI, and the
    # offer never ran: the conversation reads text only.
    assert fetches == ["https://cdn.discordapp.com/attachments/1/2/photo.png"]
    assert shown == []
    part = calls[0]["messages"][0]["content"][0]
    assert part["image_url"]["url"].startswith("data:image/png;base64,")


async def test_analyze_image_of_a_plain_conversation_passes_a_foreign_url_remote() -> None:
    calls: list = []
    fetches: list = []

    async def call_api(payload):
        calls.append(payload)
        return {"choices": [{"message": {"content": "a description"}}]}

    async def fetch_url(url):
        fetches.append(url)
        return (b"img", "image/png")

    answer = await analyze_image_impl(
        {"url": "https://example.com/pic.jpg"}
        , call_api, fetch_url, model="vision-model"
    )
    assert answer == "a description"
    # A foreign URL rides the vision call as-is: no download of a page the
    # conversation cannot see.
    assert fetches == []
    part = calls[0]["messages"][0]["content"][0]
    assert part["image_url"]["url"] == "https://example.com/pic.jpg"


class DecidingApi(FakeApi):
    """The cog stand-in with the decision router of the provider."""

    def __init__(self, *args, decision=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.decision = decision
        self.decision_calls: list[str] = []

    async def decide_session_model(self, state_text, nsfw_allowed=False):
        self.decision_calls.append(state_text)
        return self.decision


async def test_a_fresh_session_opens_on_the_decision_pick() -> None:
    api = DecidingApi([close("Hello.")], decision=("DeepSeek Pro", "coding"))
    engine = build_engine(api)
    assert await drain(engine, "fix this python bug") == ["Hello."]
    # The state carries the opening message, the pick rides the session
    # override, and the reported choice seeds the presence label.
    assert api.decision_calls == ["Madrang: fix this python bug"]
    assert api.requests[0]["model"] == "DeepSeek Pro"
    assert engine.history.sessions[7].model_override == "DeepSeek Pro"
    assert engine.history.sessions[7].activity == "coding"
    # The next message keeps the pick without a second decision call.
    api.answers.append(close("Done."))
    assert await drain(engine, "thanks") == ["Done."]
    assert len(api.decision_calls) == 1
    assert api.requests[1]["model"] == "DeepSeek Pro"


async def test_a_failed_decision_keeps_the_configured_model() -> None:
    api = DecidingApi([close("Hello.")], decision=None)
    engine = build_engine(api)
    assert await drain(engine, "hi") == ["Hello."]
    assert api.decision_calls == ["Madrang: hi"]
    assert api.requests[0]["model"] == "test-model"
    assert engine.history.sessions[7].model_override is None


async def test_a_compacted_session_reroutes_on_the_next_message() -> None:
    api = DecidingApi([close("Hello.")], decision=("DeepSeek Pro", None))
    engine = build_engine(api)
    session = await engine.history.get(7, "user")
    session.start_context("system")
    session.append("user", "an old turn")
    session.summary = "The user fixed a python bug."
    session.reroute = True
    assert await drain(engine, "and another thing") == ["Hello."]
    # The state carries the fresh summary beside the new message.
    assert api.decision_calls == [
        "Summary of the conversation so far:\nThe user fixed a python bug.\n"
        "New message: Madrang: and another thing"
    ]
    # The pick rides the override, no condense ran for the move.
    assert api.requests[0]["model"] == "DeepSeek Pro"
    assert session.model_override == "DeepSeek Pro"
    assert session.reroute is False
    # The next message keeps the pick without a second decision call.
    api.answers.append(close("Done."))
    assert await drain(engine, "thanks") == ["Done."]
    assert len(api.decision_calls) == 1


async def test_a_fresh_session_opens_from_the_compaction_exchange() -> None:
    api = FakeApi([close("Hello.")])
    engine = build_engine(api)
    engine.memory.summaries[("user", 7)] = "The user fixed a python bug."
    assert await drain(engine, "hi") == ["Hello."]
    messages = api.requests[0]["messages"]
    # The stored summary rides after the system message as the compaction
    # exchange: the resume note, then the summary as the agent answer.
    assert messages[1]["role"] == "user"
    assert messages[1]["content"].startswith("[harness] The conversation resumes")
    assert messages[1]["content"].endswith("[/harness]")
    assert "condense" not in messages[1]["content"]
    assert messages[2] == {"role": "assistant", "content": "The user fixed a python bug."}
    assert messages[-1]["content"].endswith("Madrang <@7>: hi")


class CeilingApi(FakeApi):
    """The cog stand-in with media ceilings for the render descriptors."""

    def __init__(self, *args, ceilings=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ceiling_answers = list(ceilings or [])
        self.ceiling_reads = 0

    async def media_ceilings(self):
        self.ceiling_reads += 1
        if len(self.ceiling_answers) > 1:
            return self.ceiling_answers.pop(0)
        return self.ceiling_answers[0] if self.ceiling_answers else {"image": None, "edit": None}


async def test_the_render_descriptors_freeze_for_the_context() -> None:
    native = [{
        "name": "generate_image", "description": "d", "parameters": {"type": "object", "properties": {}}
      , "media": "images", "handler": None
    }]
    preset = FakePreset(native=native)
    tight = {"image": 0.01, "edit": 0.02}
    open_ = {"image": None, "edit": None}
    api = CeilingApi([close("Hello."), close("Done.")], preset=preset, ceilings=[tight, open_])
    engine = build_engine(api)
    assert await drain(engine, "hi") == ["Hello."]
    assert preset.native_ceilings[0] is tight
    assert await drain(engine, "more") == ["Done."]
    # The second reply reuses the frozen ceilings: the descriptors and the
    # prompt cache hold, and the cog read the ceilings once.
    assert preset.native_ceilings[1] is tight
    assert api.ceiling_reads == 1


async def test_a_reported_render_price_joins_the_recorded_cost() -> None:
    async def handler(arguments, engine):
        await engine.add_media_cost(0.29)
        return "posted"

    native = [{
        "name": "generate_image", "description": "d", "parameters": {"type": "object", "properties": {}}
      , "media": "images", "handler": handler
    }]
    api = FakeApi([call("generate_image"), close("Done.")], preset=FakePreset(native=native))
    stats = FakeScopeStats()
    engine = build_engine(api, stats=stats)
    assert await drain(engine, "hi") == ["Done."]
    # The render price rode the usage of the reply into the stats, beside
    # the generation count.
    assert stats.records[0]["images"] == 1
    assert stats.records[0]["cost"] == 0.29


async def test_a_reported_activity_lands_on_the_session() -> None:
    async def handler(arguments, engine):
        await engine.set_activity("debugging a script")
        return "ok"

    native = [{
        "name": "set_activity", "description": "d", "parameters": {"type": "object", "properties": {}}
      , "handler": handler
    }]
    api = FakeApi([call("set_activity"), close("Done.")], preset=FakePreset(native=native))
    engine = build_engine(api)
    assert await drain(engine, "hi") == ["Done."]
    # The activity label rides the session for the Discord presence.
    assert engine.history.sessions[7].activity == "debugging a script"


async def test_a_session_without_the_router_skips_the_decision_call() -> None:
    # The plain stand-in carries no decide_session_model: a provider without
    # the capability never sees a call.
    api = FakeApi([close("Hello.")])
    engine = build_engine(api)
    assert await drain(engine, "hi") == ["Hello."]
    assert api.requests[0]["model"] == "test-model"
    assert engine.history.sessions[7].model_override is None


async def test_a_filtered_preset_carries_only_the_fixed_tools() -> None:
    async def noop(arguments, engine):
        return "ok"

    native = [
        {"name": "generate_image", "description": "d", "parameters": {}, "handler": noop}
      , {"name": "web_search", "description": "d", "parameters": {}, "handler": noop}
    ]
    preset = FakePreset(native, tool_filter=frozenset({"generate_image", "propose_choices"}))
    api = FakeApi([close("Hi.")], preset=preset)
    engine = build_engine(api, harness_tools=FakeHarnessTools(["propose_choices", "read_history"]))
    # The conversation sits on the filtered preset: a session override names it.
    session = await engine.history.get(7, "user")
    session.model_override = "Qwen"
    assert await drain(engine, "hi") == ["Hi."]
    names = {tool["function"]["name"] for tool in api.requests[0]["tools"]}
    # The harness defaults and the provider tools outside the filter stay off.
    assert names == {"generate_image", "propose_choices"}


async def test_a_preset_that_bars_mcp_keeps_the_other_tools() -> None:
    async def noop(arguments, engine):
        return "ok"

    native = [
        {"name": "generate_image", "description": "d", "parameters": {}, "handler": noop}
    ]
    preset = FakePreset(native, mcp=False)
    api = FakeApi([close("Hi.")], preset=preset)
    mcp_tool = {"type": "function", "function": {"name": "user_server_tool", "description": "d", "parameters": {}}}
    engine = build_engine(
        api
        , harness_tools=FakeHarnessTools(["propose_choices", "read_history"])
        , mcp=FakeMCP([mcp_tool])
    )
    # The conversation sits on the barring preset: a session override names it.
    session = await engine.history.get(7, "user")
    session.model_override = "Gemini"
    assert await drain(engine, "hi") == ["Hi."]
    names = {tool["function"]["name"] for tool in api.requests[0]["tools"]}
    # The MCP tool of the user server stays out, the harness defaults and the provider tool stay.
    assert "user_server_tool" not in names
    assert names == {"propose_choices", "read_history", "generate_image"}
