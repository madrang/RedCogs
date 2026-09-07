"""The session registry and the per-session lock."""

import asyncio
import time

from AgentEliza.history import History, Session


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
