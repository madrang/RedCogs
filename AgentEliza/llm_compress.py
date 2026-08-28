"""Session compaction: summarize the older turns while the provider cache is warm."""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone

from discord.ext import tasks

from .history import (
    COMPACTION_KEEP_TURNS,
    COMPACT_REQUEST,
    DEFAULT_CACHE_TTL,
    SUMMARY_MAX_CHARS,
    Session,
)
from .llm_chat import ChatError
from .tools import MESSAGE_TIME_FORMAT
from .workspace import WORKSPACE_SWEEP_INTERVAL

log = logging.getLogger("red.agenteliza")

# Backoff of a failed compaction in the sweeper: the retry waits
# COMPACTION_RETRY_BASE times the failure count, capped at
# COMPACTION_RETRY_MAX. At COMPACTION_MAX_FAILURES consecutive failures the
# session unloads: one last compaction salvages the summary, then the
# session drops either way. The reply path is not gated: a user message
# always gets its compaction attempt.
COMPACTION_RETRY_BASE = 300
COMPACTION_RETRY_MAX = 3600
COMPACTION_MAX_FAILURES = 5
# The sweeper pre-compacts a session this close to the size budget (one or
# two messages from the reply-path trigger): the pass runs in the background
# instead of making the user wait, and leaves the same state as an idle
# compaction.
SWEEP_SIZE_AT = 0.9

# A chained summary holds one entry per compaction, oldest first. A header
# line in the user-message format (UTC time, agent name, mention id) starts
# each entry, stamped when the summary was added. The header is the
# delimiter of the chain: the roll drops the text up to a header.
_SUMMARY_HEADER_RE = re.compile(r"(?m)^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}Z [^\n]+ <@[^>\n]+>:)$")


def _summary_entries(text: str) -> list:
    """Split a chained summary into its entries. A text without headers is one entry."""
    parts = _SUMMARY_HEADER_RE.split(text)
    entries = []
    if parts[0].strip():
        entries.append(parts[0].strip())
    for header, body in zip(parts[1::2], parts[2::2]):
        entries.append(f"{header}\n{body.strip()}")
    return entries


class Compressor:
    """Summarizes the sessions before the provider cache goes cold, and on unload.

    The api parameter is the provider surface: chat_request, message_of,
    model_name, current_preset, context_length, usage_exhausted. The cog
    plays this role.
    """

    def __init__(self, config, history, memory, api):
        self.config = config
        self.history = history
        self.memory = memory
        self.api = api
        # cog_load sweeps the workspace at load: the first scan here comes
        # one interval later.
        self._last_workspace_sweep = time.monotonic()

    async def compact(self, session_id: int, session: Session, api_key: str, preset, keep: int = COMPACTION_KEEP_TURNS) -> dict | None:
        """Summarize the turns of a session and persist the summary.

        Runs while the provider cache is still warm, with the session cache
        key. The request holds the whole session: keep only sets how many
        recent turns stay loaded after the summary. The summary covers the
        kept turns too, so they can leave later without another compaction.
        The caller holds the session lock, so the session cannot change
        during the call. Returns the token usage of the call, or None on
        failure: the session then stays as it is and the next message retries.
        A content filter rejection is permanent: the session is unloaded and
        the error raised again for the caller.
        """
        if len(session.messages) <= 1:
            return None
        # The block to unload after the summary. It can be empty: a short
        # session is all tail, and the summary still updates.
        old = session.plan_compaction(keep)
        # The session goes to the API as it is: the same system message and
        # the untouched turns, so the compaction reads the context the agent
        # worked with. Only the harness request joins, as a user message.
        # An earlier summary is already in the turns as the compaction exchange.
        payload = {
            "model": await self.api.model_name(),
            "messages": [*session.messages, {"role": "user", "content": COMPACT_REQUEST}],
            "stream": False,
        }
        if preset is not None:
            payload.update(preset.extra_payload(session_id))
        try:
            data = await self.api.chat_request(api_key, payload)
        except ChatError as e:
            if e.kind != "content_filter":
                session.error = f"{type(e).__name__}: {e}"
                return None
            # The turns trip the provider filter: no summary is possible,
            # and a retry sends the same rejected content. Unload the
            # session. A fresh one starts on the next message. The caller
            # gets the error to handle.
            log.info("The content filter rejected a compaction: session %s is unloaded.", session_id)
            self.history.sessions.pop(session_id, None)
            raise
        message = self.api.message_of(data)
        if message is None:
            session.error = "The provider returned an unexpected answer."
            return None
        summary = message.get("content") or ""
        bot_user = self.api.bot.user
        name = bot_user.name if bot_user else "Eliza"
        mention = f"<@{bot_user.id}>" if bot_user else f"<@{name}>"
        stamp = f"{datetime.now(timezone.utc):{MESSAGE_TIME_FORMAT}}"
        header = f"{stamp} {name} {mention}:"
        entries = _summary_entries(session.summary) if session.summary else []
        entries.append(f"{header}\n{summary}")
        # The chain rolls: the oldest complete summaries leave first.
        while len(entries) > 1 and len("\n\n".join(entries)) > SUMMARY_MAX_CHARS:
            entries.pop(0)
        summary = "\n\n".join(entries)
        if len(summary) > SUMMARY_MAX_CHARS and summary.startswith(header):
            # A single big note: the header is optional. The content keeps the space.
            summary = summary[len(header) + 1:]
        # An over-long note loses its start, never its end.
        summary = summary[-SUMMARY_MAX_CHARS:]
        session.apply_compaction(summary, len(old))
        # The summary joins the context as the compaction exchange: the harness
        # request and the agent answer. It is not repeated in the system message.
        session.inject_summary()
        await self.memory.store_summary(session.scope, session_id, summary)
        session.error = None
        session.compaction_failures = 0
        session.compaction_retry_at = 0.0
        return data.get("usage") or {}

    @tasks.loop(seconds=60)
    async def sweep(self) -> None:
        """Compact the idle sessions to update their summaries, before the cache goes cold.

        Compaction triggered by the next message would run after the expiry
        and pay the full prompt price. The idle compaction keeps the tail: a
        user who comes back before the eviction finds a backfill-shaped
        state. The compaction itself resets the idle clock, and a session
        that reaches the expiry again with an already summarized context
        leaves the RAM: the summary persists in the Memory store, and a
        fresh session backfills the recent turns on the next message. An
        expired session without turns leaves too: it holds nothing to lose.
        The size trigger of the reply path fires when a new message pushes
        the session over the budget; the sweeper pre-compacts just below the
        budget (SWEEP_SIZE_AT), so the pass runs in the background instead
        of making the user wait. Both compactions leave the same state. A
        failed compaction retries on a per-session backoff. At
        COMPACTION_MAX_FAILURES consecutive failures the session unloads:
        one last compaction updates the summary, then the session drops
        either way.
        """
        api_key = await self.config.api_key()
        if not api_key:
            return
        if await self.api.usage_exhausted():
            return
        preset = await self.api.current_preset()
        cache_ttl = (preset.cache_ttl if preset is not None else None) or DEFAULT_CACHE_TTL
        context_length = await self.api.context_length(preset)
        for session_id, session in list(self.history.sessions.items()):
            if session.last_compaction >= session.last_active:
                continue
            if time.monotonic() < session.compaction_retry_at:
                continue
            # Idle trigger, and a size pre-trigger just below the reply-path
            # budget: the compaction runs in the background instead of on the
            # next message of the user.
            if not self.history.needs_compaction(session, cache_ttl, context_length, fill=SWEEP_SIZE_AT):
                continue
            if len(session.messages) <= 1:
                # Nothing to summarize: the session holds no turns.
                continue
            # A failed compaction marks the session. The loop task survives.
            mark = session.last_compaction
            try:
                async with session.acquire():
                    compacted = await self.compact(session_id, session, api_key, preset)
            except Exception as e:
                compacted = None
                session.error = f"{type(e).__name__}: {e}"
            if compacted is None and session.last_compaction == mark:
                if self.history.sessions.get(session_id) is not session:
                    # A content filter unloaded the session already.
                    continue
                session.compaction_failures += 1
                if session.compaction_failures >= COMPACTION_MAX_FAILURES:
                    # The session never compacts: unload it. One last
                    # compaction updates the summary (the request already
                    # covers every turn), then the session drops either way.
                    # The summary, old or new, persists, and a fresh session
                    # backfills the recent turns on the next message. A reply
                    # waiting on the lock recovers on the live session.
                    failures = session.compaction_failures
                    try:
                        async with session.acquire():
                            await self.compact(session_id, session, api_key, preset)
                    except Exception as e:
                        session.error = f"{type(e).__name__}: {e}"
                    if self.history.sessions.get(session_id) is session:
                        log.warning(
                            "The compaction of session %s failed %d times: the session is unloaded.",
                            session_id, failures,
                        )
                        del self.history.sessions[session_id]
                    continue
                session.compaction_retry_at = time.monotonic() + min(
                    COMPACTION_RETRY_BASE * session.compaction_failures, COMPACTION_RETRY_MAX
                )
        for session_id, session in list(self.history.sessions.items()):
            if session.lock.locked():
                continue
            if session.idle() < cache_ttl:
                continue
            if session.last_compaction >= session.last_active or len(session.messages) <= 1:
                # Expired, and compacted or holding no turns: nothing left
                # that the Memory store and the backfill do not hold.
                del self.history.sessions[session_id]
        # The workspace janitor rides the sweep but scans the filesystem
        # about once a day: files untouched past the age cap die here, one by
        # one, and the tools refresh the time of a file on use.
        workspace = getattr(self.api, "workspace", None)
        if workspace is not None and time.monotonic() - self._last_workspace_sweep >= WORKSPACE_SWEEP_INTERVAL:
            self._last_workspace_sweep = time.monotonic()
            await asyncio.to_thread(workspace.sweep)

    async def compact_all(self) -> None:
        """Compact every session with turns. A reboot loses the RAM history, not the summaries.

        The full-context request summarizes every turn: the RAM state dies
        with the unload, and the backfill restores the tail on the next load.
        """
        api_key = await self.config.api_key()
        if not api_key:
            return
        if await self.api.usage_exhausted():
            return
        preset = await self.api.current_preset()
        for session_id, session in list(self.history.sessions.items()):
            if len(session.messages) <= 1 or session.last_compaction >= session.last_active:
                continue
            try:
                async with session.acquire():
                    await self.compact(session_id, session, api_key, preset)
            except Exception as e:
                session.error = f"{type(e).__name__}: {e}"
