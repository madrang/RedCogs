import asyncio
import contextlib
import json
import logging
import re
import time
from datetime import datetime, timezone

import aiohttp
import discord
from discord.ext import tasks
from redbot.core import commands, Config
from redbot.core.utils.chat_formatting import pagify
from redbot.core.utils.mod import is_admin_or_superior

from .history import (
    BACKFILL_MESSAGES,
    COMPACTION_KEEP_TURNS,
    COMPACT_REQUEST,
    DEFAULT_CACHE_TTL,
    SUMMARY_MAX_CHARS,
    History,
    Session,
)
from .mcp_manager import MCPManager
from .memory import MEMORY_MAX_CHARS, Memory
from .providers import DEFAULT_PROVIDER, PROVIDERS, provider_for, provider_named
from .stats import ScopeStats
from .tools import HarnessTools, MESSAGE_TIME_FORMAT

log = logging.getLogger("red.agenteliza")

SYSTEM_PROMPT = (
    "You are {name}, an AI agent on a Discord chat. Write short and clear answers.\n"
    "\n"
    "The harness marks the context with simple delimiters:\n"
    "- A user message starts with the UTC time, the sender name, the mention id, and a colon, for example '2026-08-12T14:30Z Madrang <@491487179927978014>: hello'.\n"
    "  The id lets you target that user with the memory tools or answer with a mention.\n"
    "- A memory note starts with '[memory NAME]' and ends with '[/memory]'. It shows the stored memory of that user.\n"
    "  The harness adds it before the first message of a user in this context.\n"
    "- A harness request starts with '[harness]' and ends with '[/harness]'. It arrives as a user message.\n"
    "  When the request asks to condense the conversation, answer with a summary of the conversation so far.\n"
    "\n"
    "Memory rules:\n"
    "- You have three memory scopes: server, channel, and user.\n"
    "- Use memory_read to read a scope. Use memory_write to replace the full text of a scope.\n"
    "  Use memory_append to add one new fact at the end of a scope.\n"
    "- Read a scope before you write it. A write replaces the full text. Merge the old content that you want to keep.\n"
    f"- A memory text can hold {MEMORY_MAX_CHARS} characters. The harness truncates a longer write.\n"
    "- The harness keeps the summaries of the channels and the users. The server summary changes only when you update it.\n"
    "\n"
    "The summary and the memory:\n"
    "- The summary condenses the older part of the conversation. You write it when the harness asks.\n"
    "  It keeps the thread of the talk.\n"
    "- The memory holds the facts that you choose to keep. You write it at any time with the memory tools.\n"
    "  It survives across conversations.\n"
    "- Put a durable fact in the memory. Let the summary keep the flow of the talk.\n"
    "- Do not copy memory content into a summary. The context already shows the memory notes.\n"
    "  A copy wastes context space.\n"
    "- This context shows a memory you write at once. The system message shows it again when the context restarts.\n"
    "\n"
    "Conversation rules:\n"
    "- An empty message is a poke. The user wants your attention and said nothing.\n"
    "- To stay silent, answer with only '[no-reply]'. The harness then sends nothing.\n"
    "- When a user mentions a past event that you do not know, use the history_read tool to find the exchange.\n"
    "- After 10 tool calls in one answer, finish the answer in text without tools.\n"
    "\n"
    "Discord renders your answers. You can use markdown: **bold**, *italics*, `code`, code blocks, quotes, and lists.\n"
    "You can also use the Discord forms: ||spoiler||, -# subtext, [masked links](https://url), and <t:UNIX:R> timestamps.\n"
    "A mention pings its target: <@USER_ID> for a user, <@&ROLE_ID> for a role, @here for the online members."
)
MCP_TOOL_ROUNDS = 16
# The system prompt tells the agent a limit of 10 tool calls. The gap gives
# slack when the agent miscounts its own calls.
# Cap of parallel conversation sessions (channels and direct messages
# together). A new session over the cap is refused: every live session is a
# provider cache entry and a summarization load.
MAX_SESSIONS = 3
USER_AGENT = "RedBot Chat Cog"
USAGE_CACHE_SECONDS = 300
# The reply sent when the model returns no content at all.
EMPTY_REPLY = "(empty answer)"
# The tag that lets the agent refuse to reply. The harness sends nothing.
NO_REPLY_TAG = "[no-reply]"
DEFAULT_GUILD_RULES = (
    "Be respectful. Follow the Discord rules. "
    "Do not generate illegal, harmful, or explicit content."
)
DEFAULT_DM_RULES = "Be respectful. Do not generate illegal, harmful, or explicit content."
# Cap of the rules text an admin can set. It joins the system message of every context.
RULES_MAX_CHARS = 4000
# MCP server names become tool names `<name>__<tool>`: the API accepts only these characters.
MCP_SERVER_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")
# Context backfill of a fresh session: recent channel messages from Discord.
# The message count is BACKFILL_MESSAGES from history.py: the compaction
# keeps the same count verbatim. The size cap is in characters, the context
# counts tokens: 128K characters is about 32K tokens at 4 per token.
BACKFILL_MAX_CHARS = 131_072
# Cap of the raw messages scanned to find the qualifying ones. The scan
# pages deeper into the history until it has BACKFILL_MESSAGES messages of
# the conversation with the agent, or this cap.
BACKFILL_SCAN_MAX = 199
# The timeout of a user whose exchange trips the provider content filter.
FILTER_TIMEOUT = 1800


class ChatError(Exception):
    """A classified API error. `kind` names the category, `raw` keeps the provider body or the original exception."""

    def __init__(self, kind: str, text: str, raw=None):
        super().__init__(text)
        self.kind = kind
        self.raw = raw


class Eliza(commands.Cog):
    """AgentEliza Cog - Harness that connects AI agents to Discord through an OpenAI-compatible chat API."""

    __author__ = "Madrang"
    __version__ = "0.0.1"

    def __init__(self, bot):
        self.bot = bot
        self.session: aiohttp.ClientSession | None = None
        # When closed, the agent ignores every message until the cog is reloaded.
        self._closed = False
        # Users timed out after a content filter rejection: user id -> monotonic expiry.
        self._filter_timeouts = {}
        # Usage endpoint cache: (monotonic timestamp, rows) or None.
        self._usage_cache = None
        # Init config. The identifier must stay unique and stable.
        self.config = Config.get_conf(self, identifier="agenteliza", force_registration=True)
        self.config.register_global(
            api_key=None
            , base_url=None
            , model_name=None
            , mcp_servers={}
            , usage_threshold=90
            , limit_user=20
            , limit_channel=100
            , limit_server=500
            , dm_rules=DEFAULT_DM_RULES
        )
        self.config.register_guild(rules=DEFAULT_GUILD_RULES)
        # Live MCP state lives in the manager, in memory only.
        self.mcp = MCPManager(self.config)
        # Long-term memory lives in Config. The harness tools let the agent control it.
        self.memory = Memory(self.config)
        self.harness_tools = HarnessTools(
            self.memory, self._get_session, bot.get_guild, bot.get_channel, lambda: bot.user.id if bot.user else None
        )
        self._harness_tool_names = {tool["function"]["name"] for tool in self.harness_tools.tools()}
        # Usage stats and rate windows per scope, in Config.
        self.scope_stats = ScopeStats(self.config)
        # Conversation sessions: one per guild, one per user in DMs. Verbatim turns in memory only.
        self.history = History(self.memory)

    async def cog_load(self) -> None:
        self.session = self._new_session()
        self.mcp.start()
        self._sweep_sessions.start()

    async def cog_unload(self) -> None:
        self._sweep_sessions.cancel()
        try:
            # Persist every session before the RAM state dies.
            await self._compact_all()
        finally:
            await self.mcp.close()
            if self.session:
                await self.session.close()

    #
    # Red methods
    #

    def format_help_for_context(self, ctx: commands.Context) -> str:
        """Show version in help."""
        pre_processed = super().format_help_for_context(ctx)
        return f"{pre_processed}\n\nCog Version: {self.__version__}"

    async def red_get_data_for_user(self, *, user_id: int):
        """All data stored about a user: memory, summary, stats, and rate window of the user scope."""
        return await self.config.user_from_id(user_id).all()

    async def red_delete_data_for_user(self, *, requester: str, user_id: int) -> None:
        """Delete the user scope: memory, summary, stats, rate window. Also drops the live DM session."""
        await self.config.user_from_id(user_id).clear()
        self.history.sessions.pop(user_id, None)

    #
    # Chat API
    #

    def _new_session(self) -> aiohttp.ClientSession:
        return aiohttp.ClientSession(headers={"User-Agent": f"{USER_AGENT} v{self.__version__}"})

    def _get_session(self) -> aiohttp.ClientSession:
        """The shared HTTP session, recreated when closed."""
        if self.session is None or self.session.closed:
            self.session = self._new_session()
        return self.session

    async def _compact(self, session_id: int, session: Session, api_key: str, preset, keep: int = COMPACTION_KEEP_TURNS) -> dict | None:
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
            "model": await self._model_name(),
            "messages": [session.messages[0], *old, {"role": "user", "content": COMPACT_REQUEST}],
            "stream": False,
        }
        if preset is not None:
            payload.update(preset.extra_payload(session_id))
        try:
            data = await self._chat_request(api_key, payload)
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
        message = self._message_of(data)
        if message is None:
            return None
        summary = message.get("content") or ""
        summary = summary[:SUMMARY_MAX_CHARS]
        session.apply_compaction(summary, len(old))
        # The summary joins the context as the compaction exchange: the harness
        # request and the agent answer. It is not repeated in the system message.
        session.inject_summary()
        await self.memory.store_summary(session.scope, session_id, summary)
        session.error = None
        return data.get("usage") or {}

    @tasks.loop(seconds=60)
    async def _sweep_sessions(self) -> None:
        """Compact the sessions half-way to cache expiry, before the cache goes cold.

        Compaction triggered by the next message would run after the expiry
        and pay the full prompt price. A failed sweep retries on the next
        loop, a successful one waits for new activity.
        """
        api_key = await self.config.api_key()
        if not api_key:
            return
        if await self._usage_exhausted():
            return
        preset = await self._current_preset()
        cache_ttl = (preset.cache_ttl if preset is not None else None) or DEFAULT_CACHE_TTL
        context_length = await self._context_length(preset)
        for session_id, session in list(self.history.sessions.items()):
            if session.last_compaction >= session.last_active:
                continue
            if not self.history.needs_compaction(session, cache_ttl, context_length):
                continue
            # A failed compaction marks the session. The loop task survives.
            try:
                async with session.lock:
                    await self._compact(session_id, session, api_key, preset)
            except Exception as e:
                session.error = f"{type(e).__name__}: {e}"

    async def _compact_all(self) -> None:
        """Compact every session with turns. A reboot loses the RAM history, not the summaries.

        An unload summarizes every turn, not only the older block: no kept
        tail survives the unload anyway.
        """
        api_key = await self.config.api_key()
        if not api_key:
            return
        if await self._usage_exhausted():
            return
        preset = await self._current_preset()
        for session_id, session in list(self.history.sessions.items()):
            if len(session.messages) <= 1 or session.last_compaction >= session.last_active:
                continue
            try:
                async with session.lock:
                    await self._compact(session_id, session, api_key, preset, keep=0)
            except Exception as e:
                session.error = f"{type(e).__name__}: {e}"

    def _place_block(self, guild_id, channel_id) -> str:
        """The location lines of the system message: server and channel, names and descriptions."""
        lines = []
        if guild_id is not None:
            guild = self.bot.get_guild(guild_id)
            if guild is not None:
                line = f"Server: {guild.name}"
                if guild.description:
                    line += f" — {' '.join(guild.description.split())}"
                lines.append(line)
        channel = self.bot.get_channel(channel_id)
        if isinstance(channel, discord.abc.GuildChannel):
            line = f"Channel: #{channel.name}"
            topic = getattr(channel, "topic", None)
            if topic:
                line += f" — {' '.join(topic.split())}"
            lines.append(line)
        return "\n".join(lines)

    def _system_text(self, bot_name: str, memory_entries: list, rules_block: str = "", place_block: str = "") -> str:
        """The system message of a context: prompt, place, rules, memory blocks."""
        text = SYSTEM_PROMPT.format(name=bot_name)
        if place_block:
            text += f"\n\n{place_block}"
        if rules_block:
            text += f"\n\n{rules_block}"
        for label, memory in memory_entries:
            text += f"\n\n{label} memory:\n{memory}"
        return text

    async def _rate_limits(self) -> dict:
        """The configured interaction limits per scope, 0 for unlimited."""
        return {
            "user": await self.config.limit_user()
            , "channel": await self.config.limit_channel()
            , "guild": await self.config.limit_server()
        }

    async def _base_url(self) -> str:
        return (await self.config.base_url()) or DEFAULT_PROVIDER.base_url

    async def _model_name(self) -> str:
        return (await self.config.model_name()) or DEFAULT_PROVIDER.models[0]

    async def _current_preset(self):
        """The provider matching the configured base URL, or None for a custom provider."""
        return provider_for(await self._base_url())

    async def _context_length(self, preset) -> int | None:
        """The context size of the configured model, None when unknown."""
        if preset is None:
            return None
        return preset.context_length(await self._model_name())

    async def _fetch_usage(self):
        """Query the active provider usage endpoint. Return (rows, error_message)."""
        provider = provider_for(await self._base_url())
        if provider is None:
            return None, "The current provider has no known usage endpoint."
        api_key = await self.config.api_key()
        if not api_key:
            return None, "The API key is not set. Use the `eliza setkey` command first."
        if self.session is None or self.session.closed:
            self.session = self._new_session()
        return await provider.fetch_usage(self.session, api_key)

    async def _usage_rows(self):
        """The provider usage rows, cached. None when the check fails: never block on a failure."""
        now = time.monotonic()
        if self._usage_cache and now - self._usage_cache[0] < USAGE_CACHE_SECONDS:
            return self._usage_cache[1]
        rows, error = await self._fetch_usage()
        if error:
            return None
        self._usage_cache = (now, rows)
        return rows

    async def _usage_exhausted(self) -> bool:
        """True only on a hard exhaustion report.

        The threshold stops new conversations before the plan is spent.
        Maintenance calls (sweep, unload compaction) keep working past the
        threshold, so sessions still compact cleanly. They stop here.
        """
        rows = await self._usage_rows()
        if not rows:
            return False
        return any(row.get("exhausted") for row in rows)

    async def _usage_blocked(self):
        """Return a notice when the provider usage is over the threshold, else None."""
        threshold = await self.config.usage_threshold()
        if not threshold:
            return None
        rows = await self._usage_rows()
        if not rows:
            return None
        for row in rows:
            if row.get("exhausted"):
                return "The provider balance is exhausted. An admin can check with the `eliza usage` command."
            percent = row.get("percent")
            if percent is not None and percent >= threshold:
                return (
                    f"The provider usage is at {percent:.0f}% ({row['name']}), over the {threshold}% threshold. "
                    f"I will not use more of the allowance. An admin can check with the `eliza usage` command."
                )
        return None

    async def _chat_request(self, api_key: str, payload: dict) -> dict:
        """One POST to the chat API. Return the answer data. Raise ChatError on any failure.

        A stalled connect (flaky DNS, dead route) raises a ClientError or a
        TimeoutError. Retry those like the Discord calls: 4 attempts, 4/8/16 s
        backoff. An HTTP error answer is a real reply of the API: not retried.
        The error body follows the OpenAI-style envelope `error: {code,
        message}` with an optional `contentFilter` array. The presence of
        `contentFilter` marks a provider content filter rejection, whatever
        the status code.
        """
        if self.session is None or self.session.closed:
            self.session = self._new_session()
        base_url = await self._base_url()
        delay = 4
        for attempt in range(4):
            try:
                async with self.session.post(
                    f"{base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}"},
                    json=payload,
                    # total caps the whole request: a long generation needs
                    # minutes. sock_connect fails a stalled connect fast so
                    # the retry probes again sooner.
                    timeout=aiohttp.ClientTimeout(total=300, sock_connect=15),
                ) as response:
                    try:
                        data = await response.json(content_type=None)
                    except Exception:
                        data = None
                    if isinstance(data, dict) and data.get("contentFilter"):
                        log.info(
                            "The API content filter rejected the request (HTTP %s): %s",
                            response.status, (data.get("error") or {}).get("message"),
                        )
                        raise ChatError(
                            "content_filter",
                            "The provider content filter rejected this exchange.",
                            raw=data,
                        )
                    if response.status == 400:
                        log.warning("The API rejected the request as invalid (400): %s", data)
                        message = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
                        detail = message or data
                        raise ChatError("bad_request", f"The API rejected the request (400): {detail}", raw=data)
                    if response.status == 401:
                        log.warning("The API rejected the API key (401).")
                        raise ChatError(
                            "auth",
                            "The API rejected the API key (401). An admin can check it with the `eliza status` command.",
                            raw=data,
                        )
                    if response.status == 429:
                        log.warning("The API rate limit is reached (429): %s", data)
                        raise ChatError("rate_limit", "The API rate limit is reached. Try again later.", raw=data)
                    if response.status != 200 or not data:
                        log.warning("The API request failed (HTTP %s): %s", response.status, data)
                        message = (data.get("error") or {}).get("message") if isinstance(data, dict) else None
                        detail = message or data
                        raise ChatError("http", f"The API returned an error (HTTP {response.status}): {detail}", raw=data)
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt == 3:
                    log.warning("The API request failed after %d attempts: %s: %s", attempt + 1, type(e).__name__, e)
                    raise ChatError("connection", f"The connection to the API failed: {type(e).__name__}: {e}", raw=e) from e
                log.info("The API request failed, attempt %d: %s: %s", attempt + 1, type(e).__name__, e)
                await asyncio.sleep(delay)
                delay *= 2

    @staticmethod
    def _message_of(data: dict) -> dict | None:
        """The message of the first choice, or None on a non-standard answer."""
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message")

    @staticmethod
    def _normalize_reply(message: dict | None) -> str | None:
        """The reply text of a message. None when the agent refuses with the no-reply tag."""
        content = (message or {}).get("content") or ""
        if not content.strip():
            return EMPTY_REPLY
        if content.strip() == NO_REPLY_TAG:
            return None
        return content

    async def _record_turn(self, session: Session, additions: list, exchange: list, message: dict, *, guild_id, channel_id, user_id, usage: dict) -> None:
        """Store a completed turn in the session and record its token usage.

        The exchange holds the tool rounds: the assistant calls and the tool
        results, as the API sent them. They stay in the context until a
        compaction summarizes them. Every assistant turn keeps its reasoning
        field for the same reason.
        """
        for addition in additions:
            session.append(addition["role"], addition["content"])
        for part in exchange:
            session.append_message(part)
        final = {"role": "assistant", "content": message.get("content") or ""}
        for key in ("reasoning_content", "reasoning"):
            if message.get(key):
                final[key] = message[key]
        session.append_message(final)
        if user_id is not None:
            # The turn landed in the context: the memory note does not repeat.
            session.seen_users.add(user_id)
        await self.scope_stats.record(guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage)

    async def generate_reply(self, channel_id: int, content: str, *, guild_id: int | None = None, user_id: int | None = None, bot_name: str | None = None, user_name: str | None = None, is_owner: bool = False, message_id: int | None = None) -> str | None:
        """Send one user message to the chat API and return the reply text.

        The conversation session is the channel, or the user in DMs: the
        history and the provider cache key follow the session, so a channel
        runs one agent context. Each
        stored user turn carries the speaker name as `name: content`.
        The first turn of a user in a context gets their user memory
        injected before it, once per context. Returns None when the agent
        refuses to reply with the no-reply tag. One chat request runs per
        session at a time: replies and compactions queue on the session lock.
        """
        api_key = await self.config.api_key()
        if not api_key:
            return "The API key is not set. An admin can set it with the `eliza setkey` command."
        blocked = await self._usage_blocked()
        if blocked:
            return blocked
        session_id = channel_id if guild_id is not None else user_id
        if session_id not in self.history.sessions:
            preset = await self._current_preset()
            cache_ttl = (preset.cache_ttl if preset is not None else None) or DEFAULT_CACHE_TTL
            active = sum(1 for other in self.history.sessions.values() if other.idle() < cache_ttl)
            if active >= MAX_SESSIONS:
                return "The agent is already busy in other conversations. Try again later."
        session = await self.history.get(session_id, "channel" if guild_id is not None else "user")
        async with session.lock:
            return await self._generate_locked(
                session, session_id, channel_id, content, api_key=api_key,
                guild_id=guild_id, user_id=user_id, bot_name=bot_name, user_name=user_name, is_owner=is_owner,
                message_id=message_id,
            )

    def _participates(self, message: discord.Message, bot_id: int) -> bool:
        """True when the message speaks to the bot: a direct mention, or a reply to the bot.

        A guild message of a user that does not address the bot stays out of
        the context: the user is not part of the conversation with the agent.
        In a direct message every message speaks to the bot.
        """
        if message.guild is None:
            return True
        if self.bot.user is not None and self.bot.user in message.mentions:
            return True
        if message.type != discord.MessageType.reply:
            return False
        resolved = message.reference.resolved if message.reference else None
        return isinstance(resolved, discord.Message) and resolved.author.id == bot_id

    async def _backfill_turns(self, channel, bot_name: str, skip_id: int | None) -> list:
        """Recent channel messages as context turns: users as user role, the bot as assistant.

        A fresh session has lost the verbatim turns (reload, restart). The
        Discord history still holds them, so the new context starts with the
        recent exchange instead of only a summary. Only the messages of the
        conversation with the agent are kept: the bot replies, and the user
        messages that mention the bot or reply to it. The scan pages deeper
        into the history until it has BACKFILL_MESSAGES qualifying messages,
        or until BACKFILL_SCAN_MAX raw messages are scanned. Consecutive bot
        messages (pagified replies) merge into one assistant turn. A bot reply
        is a harness notice (API error, content filter timeout), never an
        agent answer: the notice and the message it answers stay out of the
        context. Return an empty list on any failure: the backfill never
        breaks a reply.
        """
        if self.bot.user is None:
            return []
        bot_id = self.bot.user.id
        try:
            prefixes = tuple(
                p for p in await self.bot.get_valid_prefixes(getattr(channel, "guild", None))
                if not p.startswith("<@")
            )
        except Exception:
            prefixes = ()
        try:
            qualifying = []
            skipped = set()
            # Newest first (pinned by oldest_first=False): a reply notice is
            # always seen before the message it answers.
            async for message in channel.history(limit=BACKFILL_SCAN_MAX, oldest_first=False):
                if message.id == skip_id:
                    # The triggering message joins the context as the new user turn.
                    continue
                if message.author.id == bot_id and message.type == discord.MessageType.reply:
                    # A bot reply is a harness notice, never an agent answer: the
                    # agent posts plainly. The notice and the message it answers
                    # stay out of the context.
                    if message.reference is not None and message.reference.message_id is not None:
                        skipped.add(message.reference.message_id)
                    continue
                if message.id in skipped:
                    continue
                if not message.content.strip() or (prefixes and message.content.strip().startswith(prefixes)):
                    continue
                if message.author.id != bot_id:
                    if message.author.bot or not self._participates(message, bot_id):
                        continue
                qualifying.append(message)
                if len(qualifying) >= BACKFILL_MESSAGES:
                    break
        except (discord.Forbidden, discord.HTTPException):
            return []
        turns = []
        for message in reversed(qualifying):
            content = message.content.strip()
            if message.author.id == bot_id:
                if turns and turns[-1]["role"] == "assistant":
                    turns[-1]["content"] += "\n" + content
                else:
                    turns.append({"role": "assistant", "content": content})
                continue
            for form in (f"<@{bot_id}>", f"<@!{bot_id}>"):
                content = content.replace(form, bot_name)
            stamp = f"{message.created_at:{MESSAGE_TIME_FORMAT}}"
            turns.append({"role": "user", "content": f"{stamp} {message.author.display_name} <@{message.author.id}>: {content.strip()}"})
        while turns and sum(len(turn["content"]) for turn in turns) > BACKFILL_MAX_CHARS:
            turns.pop(0)
        return turns

    async def _generate_locked(self, session: Session, session_id: int, channel_id: int, content: str, *, api_key: str, guild_id, user_id, bot_name, user_name, is_owner, message_id) -> str | None:
        """The reply work of generate_reply. The caller holds the session lock."""
        preset = await self._current_preset()
        cache_ttl = (preset.cache_ttl if preset is not None else None) or DEFAULT_CACHE_TTL
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        # Context expiry: idle past the cache lifetime, or a compaction.
        # Only then is the system message rebuilt (prompt, memory, summary).
        # An agent memory update is already in the context as a tool call, so no reload between.
        expired = not session.messages or session.idle() >= cache_ttl
        if session.messages and self.history.needs_compaction(session, cache_ttl, await self._context_length(preset)):
            compact_usage = await self._compact(session_id, session, api_key, preset)
            if compact_usage is not None:
                for key in usage:
                    usage[key] += compact_usage.get(key) or 0
                expired = True
        if expired:
            # Snapshot the shared scopes only.
            # user scope is injected per user before their first message of the context,
            # so the system message never duplicates it.
            memory = await self.memory.recall(guild_id, channel_id, None)
            name = bot_name or (self.bot.user.name if self.bot.user else "Eliza")
            if guild_id is not None:
                rules_block = f"Server rules:\n{await self.config.guild_from_id(guild_id).rules()}"
            elif is_owner:
                rules_block = "You talk to the bot owner. The owner has all rights."
            else:
                rules_block = f"You talk to a limited user. User rules:\n{await self.config.dm_rules()}"
            # A fresh session lost its verbatim turns: the Discord history restores them.
            fresh = not session.messages
            session.start_context(self._system_text(name, memory, rules_block, self._place_block(guild_id, channel_id)))
            if fresh:
                # The persisted summary joins as the compaction exchange, before the backfilled turns it summarizes.
                if session.summary:
                    session.inject_summary()
                channel = self.bot.get_channel(channel_id)
                if channel is not None:
                    for turn in await self._backfill_turns(channel, name, message_id):
                        session.append(turn["role"], turn["content"])
        session.touch()
        speaker = user_name or "User"
        tag = f"{speaker} <@{user_id}>" if user_id is not None else speaker
        additions = []
        if user_id is not None and user_id not in session.seen_users:
            user_memory = await self.memory.read("user", user_id)
            if user_memory:
                additions.append({
                    "role": "user"
                    , "content": f"[memory {speaker}]\n{user_memory}\n[/memory]"
                })
        if message_id is not None:
            # The message snowflake carries the send time of the message.
            stamp = f"{discord.utils.snowflake_time(message_id):{MESSAGE_TIME_FORMAT}}"
        else:
            stamp = f"{datetime.now(timezone.utc):{MESSAGE_TIME_FORMAT}}"
        additions.append({"role": "user", "content": f"{stamp} {tag}: {content}"})
        messages = [*session.messages, *additions]
        tools, routes = await self.mcp.gather_tools()
        # Harness tools come first: their list is stable, MCP tools may vary.
        tools = self.harness_tools.tools() + tools
        payload = {
            "model": await self._model_name(),
            "messages": messages,
            "stream": False,
        }
        if preset is not None:
            # Provider-specific fields, for example prompt_cache_key on Kimi.
            payload.update(preset.extra_payload(session_id))
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        rounds = MCP_TOOL_ROUNDS if tools else 1
        # The tool rounds of this reply: assistant calls and tool results, as
        # the API sent them. The session keeps them until a compaction.
        exchange = []
        for _ in range(rounds):
            try:
                data = await self._chat_request(api_key, payload)
            finally:
                # The idle and cache clock counts from the last provider
                # contact, not from the user message: a long generation or a
                # tool round re-warms the cache when its answer arrives.
                session.touch()
            round_usage = data.get("usage") or {}
            for key in usage:
                usage[key] += round_usage.get(key) or 0
            if round_usage.get("prompt_tokens"):
                # The real prompt size calibrates the compaction trigger.
                session.last_prompt_tokens = round_usage["prompt_tokens"]
            message = self._message_of(data)
            if message is None:
                return f"The API returned an unexpected answer: {str(data)[:500]}"
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                # The session records what the model said. The caller gets the normalized form.
                await self._record_turn(
                    session, additions, exchange, message, guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
                )
                return self._normalize_reply(message)
            # Rebuild the echo instead of reusing the inbound message: most
            # provider-specific fields can be rejected on the next request.
            # Reasoning is the exception: Kimi accepts reasoning_content back,
            # the newer vLLM dialect uses reasoning. Echo the field the
            # provider sent, so the session keeps it until a compaction.
            echo = {"role": "assistant", "content": message.get("content") or ""}
            for key in ("reasoning_content", "reasoning"):
                if message.get(key):
                    echo[key] = message[key]
            echo["tool_calls"] = tool_calls
            messages.append(echo)
            exchange.append(echo)
            for call in tool_calls:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                name = function.get("name", "")
                if name in self._harness_tool_names:
                    result_text = await self.harness_tools.run(
                        name, arguments, guild_id=guild_id, channel_id=channel_id, user_id=user_id
                    )
                else:
                    result_text = await self.mcp.run_tool(name, arguments, routes)
                result = {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": result_text,
                }
                messages.append(result)
                exchange.append(result)
        # The rounds are spent: one last pass without tools, so the model can answer with what it found.
        payload["tool_choice"] = "none"
        try:
            data = await self._chat_request(api_key, payload)
        finally:
            session.touch()
        round_usage = data.get("usage") or {}
        for key in usage:
            usage[key] += round_usage.get(key) or 0
        message = self._message_of(data)
        if message is not None:
            await self._record_turn(
                session, additions, exchange, message, guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
            )
            return self._normalize_reply(message)
        await self.scope_stats.record(
            guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
        )
        return "The agent made too many tool calls in a row. Try a simpler request."

    #
    # Listener
    #

    async def _discord_call(self, action, what: str):
        """Run one Discord API call, retrying transient failures (DNS, connection, 5xx).

        discord.py already retries a 5xx answer a few times: a failure that
        reaches the cog exhausted them, so the retry waits longer. Returns
        None when every attempt fails. Permanent errors (4xx) raise.
        """
        delay = 4
        for attempt in range(4):
            try:
                return await action()
            except (aiohttp.ClientError, asyncio.TimeoutError, discord.DiscordServerError) as e:
                if attempt == 3:
                    log.warning("%s failed after %d attempts: %s", what, attempt + 1, e)
                    return None
                await asyncio.sleep(delay)
                delay *= 2

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message) -> None:
        if self._closed:
            return
        if message.author.bot:
            return
        is_dm = message.guild is None
        # Direct user mention only: mentioned_in also matches @everyone pings.
        mentioned = self.bot.user is not None and self.bot.user in message.mentions
        if not (is_dm or mentioned):
            return
        # Skip real commands only. get_valid_prefixes contains the mention forms
        # with a trailing space, so "@Bot hello" looks like a prefix match. Only
        # get_context knows whether a command name follows the prefix.
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return
        expiry = self._filter_timeouts.get(message.author.id, 0)
        if expiry > time.monotonic():
            # The message is not processed. The reply tells the user and marks
            # the message: the backfill keeps a message the bot answered out
            # of the context.
            left = int((expiry - time.monotonic() + 59) // 60)
            await self._discord_call(
                lambda: message.reply(
                    f"⏳ You are timed out after a content filter rejection. {left} minutes left.",
                    mention_author=False, allowed_mentions=discord.AllowedMentions.none(),
                ),
                "The timeout notice",
            )
            return
        content = message.content
        # The name users address the bot by: the guild nickname when set, else the account name.
        bot_name = self.bot.user.name if self.bot.user else "Eliza"
        if message.guild is not None and message.guild.me is not None:
            bot_name = message.guild.me.display_name
        # Replace the bot mention with the bot name: removing it would leave a hole in the sentence.
        # `<@!id>` is the legacy nickname-mention form. Replace covers every occurrence, not only a leading one.
        bare_mention = False
        if self.bot.user is not None:
            # Test on the raw content: after the replace, a bare mention and a typed bot name look the same.
            bare_mention = content.strip() in (f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>")
            for form in (f"<@{self.bot.user.id}>", f"<@!{self.bot.user.id}>"):
                content = content.replace(form, bot_name)
        content = content.strip()
        if not content or bare_mention:
            # An empty poke still reaches the agent: history and memory give it meaning.
            content = "(poke: the user sent an empty message)"
        guild_id = message.guild.id if message.guild else None
        is_owner = await self.bot.is_owner(message.author)
        # The bot owner is unlimited. Every other user counts against the per-scope limits.
        if not is_owner:
            refused = await self.scope_stats.check_and_count(
                guild_id=guild_id
                , channel_id=message.channel.id
                , user_id=message.author.id
                , limits=await self._rate_limits()
            )
            if refused:
                await self._discord_call(
                    lambda: message.channel.send(refused, allowed_mentions=discord.AllowedMentions.none()), "The rate-limit notice"
                )
                return
        async with contextlib.AsyncExitStack() as stack:
            # The typing indicator is cosmetic: a transient Discord failure
            # on it must not skip the reply. Answer without the indicator then.
            with contextlib.suppress(aiohttp.ClientError, asyncio.TimeoutError, discord.HTTPException):
                await stack.enter_async_context(message.channel.typing())
            try:
                reply = await self.generate_reply(
                    message.channel.id
                    , content
                    , guild_id=guild_id
                    , user_id=message.author.id
                    , bot_name=bot_name
                    , user_name=message.author.display_name
                    , is_owner=is_owner
                    , message_id=message.id
                )
            except ChatError as e:
                # The API error reaches the user as a notice, the raw provider answer in a code block.
                raw = e.raw
                if isinstance(raw, (dict, list)):
                    raw = json.dumps(raw, ensure_ascii=False)
                detail = f"\n```\n{str(raw)[:1500]}\n```" if raw else ""
                notice = f"⚠️ {e}{detail}"
                if e.kind == "content_filter" and not is_owner:
                    now = time.monotonic()
                    self._filter_timeouts = {uid: expiry for uid, expiry in self._filter_timeouts.items() if expiry > now}
                    self._filter_timeouts[message.author.id] = now + FILTER_TIMEOUT
                    notice += f"\nYou are on a {FILTER_TIMEOUT // 60}-minute timeout."
                # The notice answers the message directly: the agent only posts
                # plainly, so a bot reply marks a filtered exchange. The backfill
                # keeps the notice and the message it answers out of the context.
                await self._discord_call(
                    lambda: message.reply(notice, mention_author=False, allowed_mentions=discord.AllowedMentions.none()),
                    "The error notice",
                )
                return
        if reply is None:
            # The agent refused to reply with the no-reply tag.
            return
        for page in pagify(reply):
            # The agent may mention: its answer is the sender's intent.
            try:
                sent = await self._discord_call(
                    lambda: message.channel.send(page, allowed_mentions=discord.AllowedMentions.all()), "The reply send"
                )
            except discord.HTTPException as e:
                log.warning("The reply send failed permanently: %s", e)
                break
            if sent is None:
                break

    #
    # Admin commands
    #

    @commands.group(name="eliza", invoke_without_command=True)
    async def eliza_group(self, ctx: commands.Context) -> None:
        """AgentEliza settings."""
        await ctx.send_help()

    @eliza_group.command(name="setkey")
    @commands.admin()
    async def eliza_setkey(self, ctx: commands.Context, api_key: str) -> None:
        """Set the API key of the chat provider. The command message is deleted to protect the key."""
        await self.config.api_key.set(api_key)
        with contextlib.suppress(discord.HTTPException):
            await ctx.message.delete()
        await ctx.send("The API key has been set. Check it with the `eliza status` command.")

    @eliza_group.command(name="seturl")
    @commands.admin()
    async def eliza_seturl(self, ctx: commands.Context, *, target: str) -> None:
        """Set the chat API: a provider name from `eliza providers`, or a base URL. Use `clear` to reset."""
        if target.lower() == "clear":
            await self.config.base_url.clear()
            await self.config.model_name.clear()
            await ctx.send(f"The provider has been reset to the default: `{DEFAULT_PROVIDER.base_url}`, model `{DEFAULT_PROVIDER.models[0]}`.")
            return
        preset = provider_named(target)
        if preset is not None:
            await self.config.base_url.set(preset.base_url)
            await self.config.model_name.set(preset.models[0])
            await ctx.send(
                f"Provider set to **{preset.name}**: `{preset.base_url}`, model `{preset.models[0]}`. "
                f"Check it with the `eliza status` command."
            )
            return
        base_url = target.rstrip("/")
        await self.config.base_url.set(base_url)
        await ctx.send(f"The API base URL has been set to `{base_url}`. Check it with the `eliza status` command.")

    @eliza_group.command(name="providers")
    @commands.admin()
    async def eliza_providers(self, ctx: commands.Context) -> None:
        """List the known provider presets. The first model of each list is the default."""
        lines = [
            f"**{p.name}** — `{p.base_url}`\nModels: {', '.join(f'`{m}`' for m in p.models)}"
            for p in PROVIDERS
        ]
        embed = discord.Embed(
            title="Providers",
            description="\n".join(lines),
            color=await ctx.embed_colour(),
        )
        await ctx.send(embed=embed)

    @eliza_group.command(name="setmodel")
    @commands.admin()
    async def eliza_setmodel(self, ctx: commands.Context, model_name: str) -> None:
        """Set the model the agent uses. Known names come from the provider list, others are allowed. Use `clear` to reset."""
        if model_name.lower() == "clear":
            await self.config.model_name.clear()
            await ctx.send(f"The model has been reset to the default: `{DEFAULT_PROVIDER.models[0]}`.")
            return
        await self.config.model_name.set(model_name)
        preset = await self._current_preset()
        if preset is not None and model_name not in preset.models:
            known = ", ".join(f"`{m}`" for m in preset.models)
            await ctx.send(
                f"The model has been set to `{model_name}`. It is not in the known list for this provider: {known}. "
                f"The API decides if it works."
            )
            return
        await ctx.send(f"The model has been set to `{model_name}`. Check it with the `eliza status` command.")

    @eliza_group.command(name="status")
    @commands.admin()
    async def eliza_status(self, ctx: commands.Context) -> None:
        """Check the connection to the chat API with the configured key."""
        api_key = await self.config.api_key()
        if not api_key:
            await ctx.send("The API key is not set. Use the `eliza setkey` command first.")
            return
        if self.session is None or self.session.closed:
            self.session = self._new_session()
        base_url = await self._base_url()
        model_name = await self._model_name()
        await ctx.channel.typing()
        try:
            async with self.session.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    data = None
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            embed = discord.Embed(
                title="Chat API status",
                description=f"Connection failed: {e}",
                color=discord.Color.red(),
            )
            await ctx.send(embed=embed)
            return
        if response.status == 200 and data:
            models = [entry.get("id", "?") for entry in data.get("data", [])]
            description = (
                f"Connection OK. {len(models)} models available.\n"
                f"Configured model: `{model_name}` "
                f"({'available' if model_name in models else 'NOT in the model list'})"
            )
            color = discord.Color.green() if model_name in models else discord.Color.orange()
        elif response.status == 401:
            description = "The API key was rejected (401). Set a valid key with the `eliza setkey` command."
            color = discord.Color.red()
        else:
            description = f"Unexpected answer (HTTP {response.status}): {data}"
            color = discord.Color.orange()
        servers = await self.config.mcp_servers()
        description += (
            f"\nMCP servers: {len(servers)} configured, "
            f"{self.mcp.connected_count()} connected, {self.mcp.tool_count()} tools."
        )
        if self._closed:
            description += "\n**The agent is closed.** Reload the cog to start Eliza again."
        preset = await self._current_preset()
        cache_ttl = (preset.cache_ttl if preset is not None else None) or DEFAULT_CACHE_TTL
        active = sum(1 for session in self.history.sessions.values() if session.idle() < cache_ttl)
        description += f"\nActive sessions: {active} of {MAX_SESSIONS}."
        errored = sum(1 for session in self.history.sessions.values() if session.error)
        if errored:
            description += f"\nSessions with a compaction error: {errored}."
        embed = discord.Embed(title="Chat API status", description=description, color=color)
        embed.set_footer(text=f"Endpoint: {base_url} | Model: {model_name}")
        await ctx.send(embed=embed)

    @eliza_group.command(name="usage")
    @commands.admin()
    async def eliza_usage(self, ctx: commands.Context) -> None:
        """Show the remaining quota of the current provider, when it has a usage endpoint."""
        await ctx.channel.typing()
        rows, error = await self._fetch_usage()
        if error:
            await ctx.send(error)
            return
        if not rows:
            await ctx.send("The usage endpoint returned no data.")
            return
        lines = []
        for row in rows:
            if row.get("text"):
                lines.append(f"**{row['name']}** — {row['text']}")
                continue
            parts = []
            if row.get("used") is not None and row.get("limit"):
                parts.append(f"{row['used']:,} / {row['limit']:,}")
            if row.get("percent") is not None:
                parts.append(f"{row['percent']:.1f}% used")
            if row.get("reset"):
                reset = row["reset"]
                parts.append(f"resets <t:{reset}:R>" if isinstance(reset, int) else f"resets {reset}")
            lines.append(f"**{row['name']}** — " + ", ".join(parts) if parts else f"**{row['name']}**")
        embed = discord.Embed(
            title="Provider usage",
            description="\n".join(lines),
            color=await ctx.embed_colour(),
        )
        await ctx.send(embed=embed)

    @eliza_group.command(name="setthreshold")
    @commands.admin()
    async def eliza_setthreshold(self, ctx: commands.Context, percent: int) -> None:
        """Set the usage percent where the cog stops answering. 0 disables the throttle."""
        if not 0 <= percent <= 100:
            await ctx.send("Give a percent between 0 and 100. 0 disables the throttle.")
            return
        await self.config.usage_threshold.set(percent)
        if percent == 0:
            await ctx.send("The usage throttle is disabled.")
        else:
            await ctx.send(f"The cog stops answering when a provider limit reaches {percent}%.")

    @eliza_group.command(name="forgetme")
    async def eliza_forgetme(self, ctx: commands.Context) -> None:
        """Delete your own user memory and summary, and drop your direct-message session.

        Touches only your user scope. Facts about you inside the shared
        server or channel memory stay, and your messages inside a server
        session stay until its next compaction.
        """
        await self.memory.clear("user", ctx.author.id)
        await self.memory.store_summary("user", ctx.author.id, "")
        self.history.sessions.pop(ctx.author.id, None)
        await ctx.send("Your user memory and summary are cleared, and your direct-message session was dropped.")

    @eliza_group.command(name="setrules")
    @commands.admin()
    async def eliza_setrules(self, ctx: commands.Context, *, text: str) -> None:
        """Set the server rules included in the system prompt of this server. `clear` resets to the default.

        The rules load at the start of the next context of the server session.
        """
        if ctx.guild is None:
            await ctx.send("Use this command in a server. The direct-message rules are owner-only: `setdmrules`.")
            return
        if text.lower() == "clear":
            await self.config.guild(ctx.guild).rules.clear()
            await ctx.send("The server rules are reset to the default.")
            return
        if len(text) > RULES_MAX_CHARS:
            await ctx.send(f"The rules are too long: {len(text)} characters, over the {RULES_MAX_CHARS} limit.")
            return
        await self.config.guild(ctx.guild).rules.set(text)
        await ctx.send("The server rules are set. They load at the start of the next context.")

    @eliza_group.command(name="setdmrules")
    @commands.is_owner()
    async def eliza_setdmrules(self, ctx: commands.Context, *, text: str) -> None:
        """Set the user rules of the direct-message system prompt. `clear` resets to the default. Owner only."""
        if text.lower() == "clear":
            await self.config.dm_rules.clear()
            await ctx.send("The direct-message rules are reset to the default.")
            return
        if len(text) > RULES_MAX_CHARS:
            await ctx.send(f"The rules are too long: {len(text)} characters, over the {RULES_MAX_CHARS} limit.")
            return
        await self.config.dm_rules.set(text)
        await ctx.send("The direct-message rules are set. They load at the start of the next context.")

    @eliza_group.command(name="close")
    @commands.admin()
    async def eliza_close(self, ctx: commands.Context) -> None:
        """Close the agent: compact and drop every session, then ignore all messages until the cog is reloaded.

        Every session is summarized and the summary persists, like on a cog
        unload. The agent then answers nothing and starts no new session, so
        the settings can be changed without new activity. A cog reload starts
        the agent again.
        """
        if self._closed:
            await ctx.send("The agent is already closed. Reload the cog to start it again.")
            return
        # Closed first: no new reply starts while the compaction runs.
        self._closed = True
        count = len(self.history.sessions)
        await self._compact_all()
        self.history.sessions.clear()
        await ctx.send(
            f"The agent is closed. {count} session(s) compacted and dropped. "
            "I ignore all messages until the cog is reloaded."
        )

    @eliza_group.group(name="memory", invoke_without_command=True)
    async def eliza_memory(self, ctx: commands.Context) -> None:
        """Inspect and clear the long-term memory and summaries per scope.

        `show` is open to every user: admins see every scope, other users
        see only their own user scope. `clear` is admin-only.
        """
        await ctx.send_help()

    @eliza_memory.command(name="show")
    async def memory_show(self, ctx: commands.Context, scope: str = "all", member: discord.Member = None) -> None:
        """Show the memory and summary of a scope: server, channel, user, or all (the default).

        Admins see every scope and every member. Other users see only their
        own user scope.
        """
        scope = scope.lower()
        if scope not in ("server", "channel", "user", "all"):
            await ctx.send("Give a scope: `server`, `channel`, `user`, or `all`.")
            return
        if isinstance(ctx.author, discord.Member):
            admin = await is_admin_or_superior(self.bot, ctx.author)
        else:
            # A direct message has a User: only the owner sees more than the user scope.
            admin = await self.bot.is_owner(ctx.author)
        if not admin:
            if (member is not None and member != ctx.author) or scope not in ("user", "all"):
                await ctx.send("You can only see your own user memory. Use `eliza memory show user`.")
                return
            scope = "user"
        targets = []
        if scope in ("server", "all") and ctx.guild is not None:
            targets.append(("guild", ctx.guild.id, "Server"))
        if scope in ("channel", "all"):
            targets.append(("channel", ctx.channel.id, "Channel"))
        if scope in ("user", "all"):
            user = member or ctx.author
            targets.append(("user", user.id, f"User {user}"))
        if not targets:
            await ctx.send("No matching scope here. A direct message has only the channel and user scopes.")
            return
        lines = []
        for internal, scope_id, label in targets:
            memory = await self.memory.read(internal, scope_id)
            summary = await self.memory.read_summary(internal, scope_id)
            updated = await self.memory.last_updated(internal, scope_id)
            stamp = f" (updated <t:{updated}:R>)" if updated else ""
            lines.append(f"## {label} memory{stamp}\n{memory or '(empty)'}")
            lines.append(f"## {label} summary\n{summary or '(empty)'}")
        for page in pagify("\n\n".join(lines)):
            await ctx.send(page, allowed_mentions=discord.AllowedMentions.none())

    @eliza_memory.command(name="clear")
    @commands.admin()
    async def memory_clear(self, ctx: commands.Context, scope: str, member: discord.Member = None) -> None:
        """Clear the memory and the summary of one scope, and drop its live session."""
        internal = {"server": "guild", "channel": "channel", "user": "user"}.get(scope.lower())
        if internal is None:
            await ctx.send("Give a scope: `server`, `channel`, or `user`.")
            return
        if internal == "guild":
            if ctx.guild is None:
                await ctx.send("No server in a direct message.")
                return
            scope_id = ctx.guild.id
        elif internal == "channel":
            scope_id = ctx.channel.id
        else:
            scope_id = (member or ctx.author).id
        await self.memory.clear(internal, scope_id)
        await self.memory.store_summary(internal, scope_id, "")
        # Snowflakes are unique across Discord entities: a live session under this id can only
        # belong to this scope. Dropping it starts the next reply with a fresh context.
        self.history.sessions.pop(scope_id, None)
        if internal == "guild":
            # Sessions live per channel: drop every live session of this server.
            for channel in [*ctx.guild.text_channels, *ctx.guild.threads]:
                self.history.sessions.pop(channel.id, None)
        await ctx.send(f"The {scope.lower()} memory and summary are cleared.")

    @eliza_group.command(name="setlimit")
    @commands.admin()
    async def eliza_setlimit(self, ctx: commands.Context, scope: str, count: int) -> None:
        """Set the interaction limit per hour of a scope: user, channel, or server. 0 for unlimited."""
        setting = {
            "user": self.config.limit_user,
            "channel": self.config.limit_channel,
            "server": self.config.limit_server,
        }.get(scope.lower())
        if setting is None:
            await ctx.send("Give a scope: `user`, `channel`, or `server`.")
            return
        if count < 0:
            await ctx.send("Give a count of 0 or more. 0 disables the limit.")
            return
        await setting.set(count)
        if count == 0:
            await ctx.send(f"The {scope.lower()} interaction limit is disabled.")
        else:
            await ctx.send(f"The {scope.lower()} interaction limit is now {count} per hour.")

    @eliza_group.command(name="stats")
    @commands.admin()
    async def eliza_stats(self, ctx: commands.Context, member: discord.Member = None) -> None:
        """Show the usage stats and rate windows of the current server, channel, and user."""
        targets = [("channel", ctx.channel.id, "Channel")]
        if ctx.guild is not None:
            targets.insert(0, ("guild", ctx.guild.id, "Server"))
        user = member or ctx.author
        targets.append(("user", user.id, f"User {user}"))
        limits = await self._rate_limits()
        lines = []
        for internal, scope_id, label in targets:
            stats = await self.scope_stats.get(internal, scope_id)
            rate = await self.scope_stats.rate(internal, scope_id)
            limit = limits.get(internal) or 0
            window = f"{rate.get('count', 0)}/{limit} this hour" if limit else "unlimited"
            lines.append(
                f"**{label}** — {stats.get('messages', 0)} messages, "
                f"{stats.get('prompt_tokens', 0):,} prompt tokens, "
                f"{stats.get('completion_tokens', 0):,} completion tokens, "
                f"{stats.get('cached_tokens', 0):,} cached. Interactions: {window}."
            )
        embed = discord.Embed(
            title="Usage stats",
            description="\n".join(lines),
            color=await ctx.embed_colour(),
        )
        await ctx.send(embed=embed)

    @eliza_group.group(name="mcp", invoke_without_command=True)
    @commands.admin()
    async def eliza_mcp(self, ctx: commands.Context) -> None:
        """Manage the MCP servers the agent can use."""
        await ctx.send_help()

    @eliza_mcp.command(name="add")
    async def mcp_add(self, ctx: commands.Context, name: str, target: str, *args: str) -> None:
        """Add an MCP server. A target in http(s):// form is a remote server. Anything else is a stdio command with optional args."""
        if not self.mcp.available:
            await ctx.send("The `mcp` package is not installed. Reinstall the cog so its requirements are installed.")
            return
        if not MCP_SERVER_NAME_RE.match(name):
            await ctx.send(
                "The name must match `^[a-zA-Z0-9_-]{1,32}$`. "
                "Tools are exposed as `<name>__<tool>` and the API limits tool names."
            )
            return
        if target.startswith(("http://", "https://")):
            spec = {"transport": "http", "url": target, "command": "", "args": []}
        else:
            spec = {"transport": "stdio", "command": target, "args": list(args), "url": ""}
        await self.mcp.close_server(name)
        async with self.config.mcp_servers() as servers:
            servers[name] = spec
        await ctx.send(
            f"MCP server `{name}` saved ({spec['transport']}). "
            f"It connects on first use. Check it with `{ctx.prefix}eliza mcp list`."
        )

    @eliza_mcp.command(name="remove")
    async def mcp_remove(self, ctx: commands.Context, name: str) -> None:
        """Remove an MCP server and close its session."""
        async with self.config.mcp_servers() as servers:
            if name not in servers:
                await ctx.send(f"No MCP server named `{name}`.")
                return
            del servers[name]
        await self.mcp.close_server(name)
        await ctx.send(f"MCP server `{name}` removed.")

    @eliza_mcp.command(name="list")
    async def mcp_list(self, ctx: commands.Context) -> None:
        """List the configured MCP servers and their state."""
        servers = await self.config.mcp_servers()
        if not servers:
            await ctx.send(f"No MCP servers configured. Add one with `{ctx.prefix}eliza mcp add`.")
            return
        lines = []
        for name, spec in servers.items():
            if spec["transport"] == "http":
                target = spec["url"]
            else:
                target = " ".join([spec["command"], *spec.get("args", [])])
            connection = self.mcp.connections.get(name)
            state = connection.state if connection else "idle"
            lines.append(f"**{name}** ({spec['transport']}) `{target}` — {state}")
        embed = discord.Embed(
            title="MCP servers",
            description="\n".join(lines),
            color=await ctx.embed_colour(),
        )
        await ctx.send(embed=embed)
