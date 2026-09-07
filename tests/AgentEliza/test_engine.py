"""The reply engine flow: segments, tools, refusals, and the record."""

import json

from AgentEliza.history import History
from AgentEliza.llm_chat import ChatEngine, ChatError
from tests.AgentEliza.fakes import (
    FakeApi, FakeBot, FakeCompactor, FakeConfig, FakeHarnessTools, FakeMCP,
    FakeMemory, FakePreset, FakeScopeStats,
)


def build_engine(api: FakeApi, *, harness_tools=None, stats=None, config=None, compactor=None) -> ChatEngine:
    memory = FakeMemory()
    return ChatEngine(
        FakeBot()
        , config or FakeConfig({"api_key": "test-key"})
        , History(memory)
        , memory
        , FakeMCP()
        , harness_tools or FakeHarnessTools()
        , stats or FakeScopeStats()
        , compactor or FakeCompactor()
        , api
    )


async def drain(engine: ChatEngine, content: str) -> list[str]:
    return [
        segment
        async for segment in engine.generate_reply(
            100, content, user_id=7, user_name="Madrang", bot_name="TestBot"
        )
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

    async def handler(arguments, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model):
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
    async def handler(arguments, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model):
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
    assert stats.media_counts == [(None, 100, 7)]
    assert stats.records[-1]["images"] == 1
    assert stats.records[-1]["tool_calls"] == 1


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

    async def handler(arguments, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model):
        record.append(await set_conversation_model(arguments["model"]))
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

    async def handler(arguments, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model):
        await set_conversation_model("big")
        answers.append(await set_conversation_model(None))
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
