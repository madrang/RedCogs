"""Session compaction: summarize the older turns while the provider cache is warm."""

import logging
import re
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

log = logging.getLogger("red.agenteliza")

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

    async def compact(self, session_id: int, session: Session, api_key: str, preset, keep: int = COMPACTION_KEEP_TURNS) -> dict | None:
        """Summarize the turns of a session and persist the summary.

        Runs while the provider cache is still warm, with the session cache
        key. The caller holds the session lock, so the session cannot change
        during the call. Returns the token usage of the call, or None on
        failure: the session then stays as it is and the next message retries.
        A content filter rejection is permanent: the session is unloaded and
        the error raised again for the caller.
        """
        old = session.plan_compaction(keep)
        if not old:
            return None
        # The session goes to the API as it is: the same system message and
        # the untouched turns, so the compaction reads the context the agent
        # worked with. Only the harness request joins, as a user message.
        # An earlier summary is already in the turns as the compaction exchange.
        payload = {
            "model": await self.api.model_name(),
            "messages": [session.messages[0], *old, {"role": "user", "content": COMPACT_REQUEST}],
            "stream": False,
        }
        if preset is not None:
            payload.update(preset.extra_payload(session_id))
        try:
            data = await self.api.chat_request(api_key, payload)
        except ChatError as e:
            if e.kind != "content_filter":
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
        return data.get("usage") or {}

    @tasks.loop(seconds=60)
    async def sweep(self) -> None:
        """Compact the sessions half-way to cache expiry, before the cache goes cold.

        Compaction triggered by the next message would run after the expiry
        and pay the full prompt price. A failed sweep retries on the next
        loop, a successful one waits for new activity. A fully expired
        session (idle past the cache lifetime) that was compacted leaves the
        RAM: its summary persists in the Memory store, and a fresh session
        backfills the recent turns on the next message. A session that was
        never compacted stays: its verbatim turns would be lost.
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
            if not self.history.needs_compaction(session, cache_ttl, context_length):
                continue
            # A failed compaction marks the session. The loop task survives.
            try:
                async with session.lock:
                    await self.compact(session_id, session, api_key, preset)
            except Exception as e:
                session.error = f"{type(e).__name__}: {e}"
        for session_id, session in list(self.history.sessions.items()):
            if session.lock.locked():
                continue
            if session.last_compaction >= session.last_active and session.idle() >= cache_ttl:
                # Expired and compacted: nothing left that the Memory store
                # does not hold.
                del self.history.sessions[session_id]

    async def compact_all(self) -> None:
        """Compact every session with turns. A reboot loses the RAM history, not the summaries.

        An unload summarizes every turn, not only the older block: no kept
        tail survives the unload anyway.
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
                async with session.lock:
                    await self.compact(session_id, session, api_key, preset, keep=0)
            except Exception as e:
                session.error = f"{type(e).__name__}: {e}"
