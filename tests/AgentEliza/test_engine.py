"""The reply engine flow: segments, tools, refusals, and the record."""

from AgentEliza.history import History
from AgentEliza.llm_chat import ChatEngine
from tests.AgentEliza.fakes import (
    FakeApi, FakeBot, FakeCompactor, FakeConfig, FakeHarnessTools, FakeMCP,
    FakeMemory, FakePreset, FakeScopeStats,
)


def build_engine(api: FakeApi, *, harness_tools=None, stats=None, config=None) -> ChatEngine:
    memory = FakeMemory()
    return ChatEngine(
        FakeBot()
        , config or FakeConfig({"api_key": "test-key"})
        , History(memory)
        , memory
        , FakeMCP()
        , harness_tools or FakeHarnessTools()
        , stats or FakeScopeStats()
        , FakeCompactor()
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
