"""The session registry, the per-session lock, and the session label."""

import asyncio
import time
from types import SimpleNamespace

from AgentEliza.history import SUMMARY_NOTE, History, Session, session_label, session_log_label


class _NoMemory:
    """The memory stand-in of the registry: every summary reads empty."""

    async def read_summary(self, scope: str, session_id: int) -> str:
        return ""


def test_plan_compaction_at_keep_zero_unloads_every_turn() -> None:
    session = Session("user")
    session.start_context("system")
    session.append("user", "one")
    session.append_message({"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "echo", "arguments": "{}"}}]})
    session.append_message({"role": "tool", "tool_call_id": "1", "content": "result"})
    # No turn stays verbatim: the block covers every turn after the system
    # message, a trailing tool exchange included.
    assert session.plan_compaction(0) == session.messages[1:]
    # A session with no turns holds nothing to unload.
    empty = Session("user")
    empty.start_context("system")
    assert empty.plan_compaction(0) == []


def test_inject_summary_places_the_exchange_after_the_system_message() -> None:
    session = Session("user")
    session.start_context("system")
    session.append("user", "one")
    session.summary = "The user fixed a python bug."
    session.inject_summary()
    # The exchange opens with the resume note, never the condense request,
    # and the summary rides as the agent answer.
    note, answer = session.messages[1], session.messages[2]
    assert note == {"role": "user", "content": SUMMARY_NOTE}
    assert "condense" not in note["content"]
    assert answer == {"role": "assistant", "content": "The user fixed a python bug."}
    assert session.messages[3]["content"] == "one"
    assert session.size == sum(Session._size_of(message) for message in session.messages)


async def test_parallel_sessions_do_not_serialize() -> None:
    history = History(_NoMemory())
    sessions = [await history.get(channel, "guild") for channel in (1, 2, 3)]
    assert len({id(session) for session in sessions}) == 3

    entered: list[float] = []
    exited: list[float] = []

    async def hold(session) -> None:
        async with session.acquire():
            entered.append(time.monotonic())
            await asyncio.sleep(0.05)
            exited.append(time.monotonic())

    await asyncio.gather(*(hold(session) for session in sessions))

    # The three held sections overlapped: no session waited on another.
    assert max(entered) < min(exited)


async def test_one_session_locks_itself() -> None:
    history = History(_NoMemory())
    session = await history.get(1, "guild")
    acquired = asyncio.Event()
    through: list[bool] = []

    async def first() -> None:
        async with session.acquire():
            acquired.set()
            await asyncio.sleep(0.05)

    async def second() -> None:
        async with session.acquire():
            through.append(True)

    holder = asyncio.create_task(first())
    await acquired.wait()
    assert session.lock.locked()

    waiter = asyncio.create_task(second())
    await asyncio.sleep(0.02)
    assert through == []

    await holder
    await waiter
    assert through == [True]


class _NamedChannel:
    """The channel stand-in of the label: a name and a guild."""

    def __init__(self, name: str, guild_name: str):
        self.name = name
        self.guild = SimpleNamespace(name=guild_name)


def test_session_label_names_a_guild_channel() -> None:
    channel = _NamedChannel("general", "Test Server")
    assert session_label("channel", 100, channel=channel) == "#general — Test Server"
    # An unresolved channel falls back to the id.
    assert session_label("channel", 100) == "channel 100"


def test_session_label_names_a_direct_message_user() -> None:
    assert session_label("user", 7, user_name="Madrang") == "DM — Madrang"
    # A wake without a user name falls back to the id.
    assert session_label("user", 7) == "DM — user 7"


async def test_session_log_label_falls_back_to_the_id() -> None:
    # Without the cog wiring, the id stands in.
    assert await session_log_label(None, 7) == "7"

    async def label_of(session_id):
        return "#general — Test Server"

    assert await session_log_label(label_of, 7) == "#general — Test Server"

    async def empty(session_id):
        return ""

    async def broken(session_id):
        raise RuntimeError("boom")

    # An empty answer and a broken getter never break the log line.
    assert await session_log_label(empty, 7) == "7"
    assert await session_log_label(broken, 7) == "7"
