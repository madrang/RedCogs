"""The speech transcription: the audio gate, the embed, and the history views."""

from datetime import datetime, timezone
from types import SimpleNamespace

import discord

from AgentEliza.history import History
from AgentEliza.llm_chat import ChatEngine, IncomingMessage
from AgentEliza.providers.kimi_code import KimiCodeProvider
from AgentEliza.providers.venice import VeniceApiProvider
from AgentEliza.tools.base import SPEECH_EMBED_TITLE, is_audio_attachment, speech_embed, speech_embed_line
from AgentEliza.tools.history import HistoryTools
from tests.AgentEliza.fakes import (
    FakeApi, FakeBot, FakeCompactor, FakeConfig, FakeHarnessTools, FakeMCP,
    FakeMemory, FakeScopeStats,
)

SPEAKER = SimpleNamespace(id=7, display_name="Madrang", display_avatar=SimpleNamespace(url="https://avatar"))


def close(text: str) -> dict:
    return {"choices": [{"message": {"content": text}}]}


def build_engine(bot) -> ChatEngine:
    memory = FakeMemory()
    return ChatEngine(
        bot
        , FakeConfig({"api_key": "test-key"})
        , History(memory)
        , memory
        , FakeMCP()
        , FakeHarnessTools()
        , FakeScopeStats()
        , FakeCompactor()
        , FakeApi([close("Hi.")])
    )


def history_message(message_id: int, author_id: int, *, content: str = "", embeds=(), attachments=(), reply_to: int | None = None, minutes: int = 0):
    """One channel history message in the shape the history views read."""
    return SimpleNamespace(
        id=message_id
      , author=SimpleNamespace(id=author_id, bot=False, display_name="Madrang" if author_id == 7 else "TestBot")
      , type=discord.MessageType.reply if reply_to is not None else discord.MessageType.default
      , content=content
      , attachments=list(attachments)
      , embeds=list(embeds)
      , reference=SimpleNamespace(message_id=reply_to, resolved=None) if reply_to is not None else None
      , guild=None
      , mentions=[]
      , created_at=datetime(2026, 9, 9, 6, minutes, tzinfo=timezone.utc)
    )


def audio_attachment(name: str = "voice-message.ogg", kind: str = "audio/ogg"):
    return SimpleNamespace(filename=name, content_type=kind, url=f"https://cdn.discordapp.com/{name}")


def test_the_speech_embed_reads_back_as_the_speaker_line() -> None:
    embed = speech_embed(SPEAKER, datetime(2026, 9, 9, 6, 4, tzinfo=timezone.utc), "hello there", 3.2, "elevenlabs/scribe-v2")
    assert embed.title == SPEECH_EMBED_TITLE
    assert speech_embed_line(SimpleNamespace(embeds=[embed])) == "Madrang <@7>: hello there"


def test_a_foreign_or_broken_embed_answers_no_line() -> None:
    assert speech_embed_line(SimpleNamespace(embeds=[])) is None
    assert speech_embed_line(SimpleNamespace(embeds=[discord.Embed(title="Now playing")])) is None
    broken = discord.Embed(title=SPEECH_EMBED_TITLE, description="hello there")
    broken.set_author(name="Madrang")
    # No User field: the speaker id is missing.
    assert speech_embed_line(SimpleNamespace(embeds=[broken])) is None


def test_a_long_speech_truncates_inside_the_embed_description() -> None:
    embed = speech_embed(SPEAKER, None, "a" * 5000, 0, "elevenlabs/scribe-v2")
    assert len(embed.description) < 4100
    assert "[truncated: " in embed.description


def test_the_audio_detection_reads_the_type_and_the_name() -> None:
    def attachment(kind, name):
        return SimpleNamespace(content_type=kind, filename=name)

    assert is_audio_attachment(attachment("audio/ogg", "voice-message.ogg"))
    assert is_audio_attachment(attachment(None, "song.mp3"))
    assert is_audio_attachment(attachment("application/octet-stream", "song.mp3"))
    assert not is_audio_attachment(attachment("video/mp4", "clip.mp4"))
    assert not is_audio_attachment(attachment("application/pdf", "doc.pdf"))
    assert not is_audio_attachment(attachment(None, "notes.txt"))


def test_the_venice_speech_request_carries_the_file_and_the_model() -> None:
    provider = VeniceApiProvider()
    assert provider.speech_model == "elevenlabs/scribe-v2"
    path, form = provider.speech_request(b"payload", "voice.ogg", "audio/ogg")
    assert path == "/audio/transcriptions"
    assert [entry[0].get("name") for entry in form._fields] == ["file", "model"]
    assert [entry[2] for entry in form._fields] == [b"payload", "elevenlabs/scribe-v2"]


def test_only_the_venice_provider_transcribes() -> None:
    assert KimiCodeProvider().speech_model is None
    assert KimiCodeProvider().speech_request(b"payload", "voice.ogg", "audio/ogg") is None


async def test_a_speech_embed_turn_rides_the_backfill_and_the_audio_drops() -> None:
    embed = speech_embed(SPEAKER, datetime(2026, 9, 9, 6, 4, tzinfo=timezone.utc), "hello there", 3.0, "elevenlabs/scribe-v2")
    audio = history_message(12, 7, attachments=[audio_attachment()], minutes=4)
    posted = history_message(13, 5, embeds=[embed], reply_to=12, minutes=5)
    earlier = history_message(11, 7, content="earlier", minutes=0)

    class Channel:
        def history(self, limit=100, oldest_first=False):
            async def walk():
                # Newest first, the order the scan reads.
                yield posted
                yield audio
                yield earlier

            return walk()

    class BackfillBot(FakeBot):
        user = SimpleNamespace(id=5, name="TestBot")

        def get_channel(self, channel_id):
            return Channel()

    api = FakeApi([close("Hi.")])
    memory = FakeMemory()
    engine = ChatEngine(
        BackfillBot(), FakeConfig({"api_key": "test-key"}), History(memory), memory,
        FakeMCP(), FakeHarnessTools(), FakeScopeStats(), FakeCompactor(), api,
    )
    assert [segment async for segment in engine.generate_reply(IncomingMessage(
        channel_id=100, content="hi", user_id=7, user_name="Madrang", bot_name="TestBot"
    ))] == ["Hi."]
    messages = api.requests[0]["messages"]
    speech = [m for m in messages if isinstance(m.get("content"), str) and "hello there" in m["content"]]
    # The embed reads as the turn of the speaker it names, and the audio
    # message it answered stays out.
    assert speech and speech[0]["role"] == "user"
    assert "2026-09-09T06:05Z Madrang <@7>: hello there" == speech[0]["content"]
    assert not any("voice-message.ogg" in str(m.get("content")) for m in messages)


async def test_a_notice_reply_and_its_message_stay_out_of_the_backfill() -> None:
    answered = history_message(21, 7, content="answer me", minutes=1)
    notice = history_message(22, 5, content="⚠️ The agent is under maintenance.", reply_to=21, minutes=2)
    earlier = history_message(11, 7, content="earlier", minutes=0)

    class Channel:
        def history(self, limit=100, oldest_first=False):
            async def walk():
                yield notice
                yield answered
                yield earlier

            return walk()

    class BackfillBot(FakeBot):
        user = SimpleNamespace(id=5, name="TestBot")

        def get_channel(self, channel_id):
            return Channel()

    engine = build_engine(BackfillBot())
    assert [segment async for segment in engine.generate_reply(IncomingMessage(
        channel_id=100, content="hi", user_id=7, user_name="Madrang", bot_name="TestBot"
    ))] == ["Hi."]
    texts = [str(m.get("content")) for m in engine.api.requests[0]["messages"]]
    assert any("earlier" in text for text in texts)
    assert not any("answer me" in text for text in texts)
    assert not any("maintenance" in text for text in texts)


async def test_read_history_renders_the_speech_line_and_drops_the_audio() -> None:
    embed = speech_embed(SPEAKER, datetime(2026, 9, 9, 6, 4, tzinfo=timezone.utc), "hello there", 3.0, "elevenlabs/scribe-v2")
    audio = history_message(12, 7, attachments=[audio_attachment()], minutes=4)
    posted = history_message(13, 5, embeds=[embed], reply_to=12, minutes=5)
    earlier = history_message(11, 7, content="earlier", minutes=0)

    class Channel:
        guild = None

        def history(self, **window):
            async def walk():
                yield posted
                yield audio
                yield earlier

            return walk()

    tools = HistoryTools()

    async def channel_getter(channel_id):
        return Channel()

    tools.channel_getter = channel_getter
    tools.bot_id_getter = lambda: 5
    result = await tools._tool_read_history({}, guild_id=None, channel_id=100, user_id=7)
    assert "2026-09-09T06:05Z Madrang <@7>: hello there" in result
    assert "2026-09-09T06:00Z Madrang <@7>: earlier" in result
    assert "voice-message.ogg" not in result
