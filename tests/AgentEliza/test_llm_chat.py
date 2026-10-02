"""The engine pure functions and the native tool availability filter."""

import pytest

from AgentEliza.history import History
from AgentEliza.llm_chat import ChatEngine, IncomingMessage, _drop_overcap_images, collapse_blank_lines
from tests.AgentEliza.fakes import FakeApi, FakeBot, FakeCompactor, FakeConfig, FakeHarnessTools, FakeMCP, FakeMemory, FakePreset, FakeScopeStats


def _image_part(n: int) -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{'a' * n}"}}


def test_the_image_trim_drops_the_oldest_parts_for_good() -> None:
    # Five images ride two notes and a message, the model takes four: the
    # oldest image parts drop for good — the session records lose them —
    # and the text parts stay.
    early = {
        "role": "user"
      , "content": [{"type": "text", "text": "early note"}, _image_part(1), _image_part(2)],
    }
    late = {
        "role": "user"
      , "content": [{"type": "text", "text": "late note"}, _image_part(3), _image_part(4)],
    }
    message = {
        "role": "user"
      , "content": [{"type": "text", "text": "the turn"}, _image_part(5)],
    }
    _drop_overcap_images([early, late, message], 4)
    # The oldest note lost one image, the rest stay whole.
    assert early["content"] == [{"type": "text", "text": "early note"}, _image_part(2)]
    assert late["content"] == [{"type": "text", "text": "late note"}, _image_part(3), _image_part(4)]
    assert message["content"] == [{"type": "text", "text": "the turn"}, _image_part(5)]
    # Under the cap nothing changes.
    _drop_overcap_images([early, late, message], 4)
    assert len(early["content"]) == 2


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # A run of blank lines collapses to one blank line.
        ("Let me check.\n\n\n\n\nDone checking.", "Let me check.\n\nDone checking."),
        ("a\n\n\nb\n\n\nc", "a\n\nb\n\nc"),
        # A single blank line stays.
        ("a\n\nb", "a\n\nb"),
        ("a\nb", "a\nb"),
        # Whitespace-only lines count as blank.
        ("a\n \n \n \nb", "a\n\nb"),
        # Blank runs at both ends drop.
        ("\n\n\nLeading and trailing\n\n\n", "Leading and trailing"),
        ("   \n\n  spaced note  \n\n  ", "spaced note"),
        # A padded no-reply tag still matches the tag.
        ("[no-reply]\n\n\n", "[no-reply]"),
        ("line1\n\n\n\t\n \nline2", "line1\n\nline2"),
    ],
)
def test_collapse_blank_lines(raw: str, expected: str) -> None:
    assert collapse_blank_lines(raw) == expected


class GatedApi(FakeApi):
    """The cog stand-in with the media windows of the song tool."""

    def __init__(self, *args, gate=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = gate

    async def media_limits(self):
        # None names the disable band of the credit ratio: the media tools
        # stay off. A dict names the hourly windows.
        return None if self.gate else {"user": 8, "channel": 32}


async def _tool_names(api, *, guild_id=None, is_owner=False) -> set:
    memory = FakeMemory()
    engine = ChatEngine(
        FakeBot(), FakeConfig({"api_key": "test-key"}), History(memory), memory
        , FakeMCP(), FakeHarnessTools(), FakeScopeStats(), FakeCompactor(), api
    )
    segments = [
        segment async for segment in engine.generate_reply(IncomingMessage(
            channel_id=100, content="hi", guild_id=guild_id, user_id=7, is_owner=is_owner
        ))
    ]
    assert segments == ["Hi."]
    return {tool["function"]["name"] for tool in api.requests[0].get("tools") or []}


async def test_the_song_tool_joins_only_the_direct_message_of_the_owner() -> None:
    native = [{
        "name": "generate_song"
        , "description": "Generate one song."
        , "parameters": {"type": "object", "properties": {}}
        , "media": "music"
        , "dm_owner_only": True
        , "guild_credit_gate": True
        , "handler": None
    }]
    close = {"choices": [{"message": {"content": "Hi."}}]}
    # Another direct message user never sees the tool.
    api = FakeApi([close], preset=FakePreset(native=native))
    assert "generate_song" not in await _tool_names(api, guild_id=None, is_owner=False)
    # The owner sees it in a direct message.
    api = FakeApi([close], preset=FakePreset(native=native))
    assert "generate_song" in await _tool_names(api, guild_id=None, is_owner=True)
    # A guild sees it while the cog carries no media policy (the plain stand-in) or the windows hold.
    api = FakeApi([close], preset=FakePreset(native=native))
    assert "generate_song" in await _tool_names(api, guild_id=5, is_owner=False)
    api = GatedApi([close], gate=False, preset=FakePreset(native=native))
    assert "generate_song" in await _tool_names(api, guild_id=5, is_owner=False)
    # The disable band of the credit ratio closes the guild list.
    api = GatedApi([close], gate=True, preset=FakePreset(native=native))
    assert "generate_song" not in await _tool_names(api, guild_id=5, is_owner=False)
