import asyncio
import contextlib
import io
import json
import logging
import re
import time

import aiohttp
import discord
from redbot.core import commands, Config
from redbot.core.utils.mod import is_admin_or_superior

from .history import DEFAULT_CACHE_TTL, History
from .llm_chat import ChatEngine, ChatError, MAX_SESSIONS
from .llm_compress import Compressor
from .mcp_manager import MCPManager
from .memory import Memory
from .pages import paginate
from .polls import PollManager
from .providers import DEFAULT_PROVIDER, PROVIDERS, provider_for, provider_named
from .stats import ScopeStats
from .tools import HarnessOptions, HarnessTools
from .workspace import Workspace

log = logging.getLogger("red.agenteliza")

USER_AGENT = "RedBot Chat Cog"
USAGE_CACHE_SECONDS = 300
DEFAULT_GUILD_RULES = (
    "Be respectful. Follow the Discord rules. "
    "Do not generate illegal, harmful, or explicit content."
)
DEFAULT_DM_RULES = "Be respectful. Do not generate illegal, harmful, or explicit content."
# Cap of the rules text an admin can set. It joins the system message of every context.
RULES_MAX_CHARS = 4000
# MCP server names become tool names `<name>__<tool>`: the API accepts only these characters.
MCP_SERVER_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,32}$")
# The timeout of a user whose exchange trips the provider content filter.
FILTER_TIMEOUT = 1800
# An extremely long answer must not flood the channel: past
# LONG_REPLY_MAX_PAGES inline pages the rest of the text rides in a file
# on a closing message.
LONG_REPLY_MAX_PAGES = 4


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
            , polls={}
        )
        self.config.register_guild(rules=DEFAULT_GUILD_RULES)
        # Live MCP state lives in the manager, in memory only.
        self.mcp = MCPManager(self.config)
        # Long-term memory lives in Config. The harness tools let the agent control it.
        self.memory = Memory(self.config)
        # The interactive votes: a button view first, a native poll after an idle time.
        self.polls = PollManager(self._get_channel, self._discord_call, self.config)
        # The workspace: one folder per session in the OS temp dir, for the file tools.
        self.workspace = Workspace()
        tool_options = HarnessOptions(memory=self.memory, session_getter=self._get_session)
        tool_options.guild_getter = bot.get_guild
        tool_options.channel_getter = self._get_channel
        tool_options.bot_id_getter = lambda: bot.user.id if bot.user else None
        tool_options.bot_token_getter = lambda: bot.http.token
        tool_options.polls = self.polls
        tool_options.workspace = self.workspace
        self.harness_tools = HarnessTools(tool_options)
        # Usage stats and rate windows per scope, in Config.
        self.scope_stats = ScopeStats(self.config)
        # Conversation sessions: one per guild, one per user in DMs. Verbatim turns in memory only.
        self.history = History(self.memory)
        # Compaction and the reply engine. The cog is their provider surface
        # (chat_request, usage, model getters) until the client splits out.
        self.compactor = Compressor(self.config, self.history, self.memory, self)
        self.engine = ChatEngine(
            bot, self.config, self.history, self.memory, self.mcp, self.harness_tools, self.scope_stats,
            self.compactor, self, polls=self.polls,
        )
        # A poll event without a user message (a vote close, an idle
        # conversion) wakes the agent through the engine.
        self.polls.on_event = self._poll_trigger
        # The majority rule of a guild poll counts the speakers of the session.
        self.polls.participants_getter = self._poll_participants

    async def _poll_participants(self, session_id: int):
        """The active users of a poll session: the speakers of the current context."""
        session = self.history.sessions.get(session_id)
        return set(session.seen_users) if session is not None else set()

    async def cog_load(self) -> None:
        self.session = self._new_session()
        self.mcp.start()
        self.compactor.sweep.start()
        await self.polls.restore(self.bot.add_view)
        # The janitor: folders untouched past the age cap die here and in the sweep.
        await asyncio.to_thread(self.workspace.sweep)

    async def cog_unload(self) -> None:
        self.compactor.sweep.cancel()
        self.polls.close()
        try:
            # Persist every session before the RAM state dies.
            await self.compactor.compact_all()
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
        """Delete the user scope: memory, summary, stats, rate window. Also drops the live DM session, the votes, and the workspace."""
        await self.config.user_from_id(user_id).clear()
        self.history.sessions.pop(user_id, None)
        await self.polls.drop_user(user_id)
        await asyncio.to_thread(self.workspace.drop, user_id)

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

    async def _get_channel(self, channel_id: int):
        """The channel of an id: the cache first, the API on a miss. A direct message channel can be absent from the cache."""
        channel = self.bot.get_channel(channel_id)
        if channel is None:
            with contextlib.suppress(discord.NotFound, discord.Forbidden, discord.HTTPException):
                channel = await self.bot.fetch_channel(channel_id)
        return channel

    async def _rate_limits(self) -> dict:
        """The configured interaction limits per scope, 0 for unlimited."""
        return {
            "user": await self.config.limit_user()
            , "channel": await self.config.limit_channel()
            , "guild": await self.config.limit_server()
        }

    async def _base_url(self) -> str:
        return (await self.config.base_url()) or DEFAULT_PROVIDER.base_url

    async def model_name(self) -> str:
        return (await self.config.model_name()) or DEFAULT_PROVIDER.models[0]

    async def current_preset(self):
        """The provider matching the configured base URL, or None for a custom provider."""
        return provider_for(await self._base_url())

    async def context_length(self, preset) -> int | None:
        """The context size of the configured model, None when unknown."""
        if preset is None:
            return None
        return preset.context_length(await self.model_name())

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

    async def usage_exhausted(self) -> bool:
        """True only on a hard exhaustion report.

        The threshold stops new conversations before the plan is spent.
        Maintenance calls (sweep, unload compaction) keep working past the
        threshold, so sessions still compact cleanly. They stop here.
        """
        rows = await self._usage_rows()
        if not rows:
            return False
        return any(row.get("exhausted") for row in rows)

    async def usage_blocked(self):
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

    async def chat_request(self, api_key: str, payload: dict) -> dict:
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
                    # total caps the whole request: a long generation on a
                    # large context needs several minutes. sock_connect fails
                    # a stalled connect fast so the retry probes again sooner.
                    timeout=aiohttp.ClientTimeout(total=900, sock_connect=15),
                ) as response:
                    body = await response.text()
                    try:
                        data = json.loads(body)
                    except ValueError:
                        # A 200 with a non-JSON body is opaque without the text.
                        log.warning("The API answer is not JSON (HTTP %s): %s", response.status, body[:500])
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
    def message_of(data: dict) -> dict | None:
        """The message of the first choice, or None on a non-standard answer."""
        choices = data.get("choices") or []
        if not choices:
            return None
        return choices[0].get("message")

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
        if self.bot.user is not None and message.author.id == self.bot.user.id:
            return
        is_dm = message.guild is None
        # Direct user mention only: mentioned_in also matches @everyone pings.
        mentioned = self.bot.user is not None and self.bot.user in message.mentions
        if message.author.bot:
            # Another bot reaches the agent in a server only: a direct
            # mention, or a reply to a message of the agent.
            if is_dm:
                return
            resolved = message.reference.resolved if message.type == discord.MessageType.reply and message.reference else None
            answered = (
                isinstance(resolved, discord.Message)
                and self.bot.user is not None
                and resolved.author.id == self.bot.user.id
            )
            if not (mentioned or answered):
                return
        elif not (is_dm or mentioned):
            return
        # Skip real commands only. get_valid_prefixes contains the mention forms
        # with a trailing space, so "@Bot hello" looks like a prefix match. Only
        # get_context knows whether a command name follows the prefix.
        ctx = await self.bot.get_context(message)
        if ctx.valid:
            return
        if self._closed:
            # Closed for maintenance: the messages the agent would answer get
            # the notice as a direct reply, everything else stays ignored.
            await self._discord_call(
                lambda: message.reply(
                    "🛠️ The agent is currently under maintenance. Sorry — it will be back shortly.",
                    mention_author=False, allowed_mentions=discord.AllowedMentions.none(),
                ),
                "The maintenance notice",
            )
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
        attachments = [(a.filename, a.content_type, a.url) for a in message.attachments]
        if (not content or bare_mention) and not attachments:
            # An empty poke still reaches the agent: history and memory give it meaning.
            content = "(poke: the user sent an empty message)"
        guild_id = message.guild.id if message.guild else None
        is_owner = await self.bot.is_owner(message.author)
        # A reply to a bot never pings: a mention-reactive bot would loop with the agent.
        mentions = discord.AllowedMentions.none() if message.author.bot else discord.AllowedMentions.all()
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
                await self._stream_reply(
                    message.channel
                    , self.engine.generate_reply(
                        message.channel.id
                        , content
                        , guild_id=guild_id
                        , user_id=message.author.id
                        , bot_name=bot_name
                        , user_name=message.author.display_name
                        , is_owner=is_owner
                        , message_id=message.id
                        , attachments=attachments
                    )
                    , mentions
                    , tag=str(message.id)
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
            except Exception:
                # An unexpected failure still answers: discord.py only logs a
                # listener exception, the user would see silence otherwise.
                log.exception("The reply failed for message %s:", message.id)
                await self._discord_call(
                    lambda: message.reply(
                        "⚠️ The reply failed with an internal error. The bot log has the traceback.",
                        mention_author=False, allowed_mentions=discord.AllowedMentions.none(),
                    ),
                    "The failure notice",
                )
                return

    async def _sync_pages(self, channel, sent: list, text: str, previous: str, mentions: discord.AllowedMentions, *, final: bool) -> tuple[bool, bool]:
        """Edit or send the pages that changed since previous. Caps the inline pages.

        Returns (alive, overflow): alive False on a permanent send
        failure, overflow True when the text has more pages than the
        inline cap. A sealed page never changes, so in practice only the
        last page is edited and the rest are sends.
        """
        pages = paginate(text, old_content=previous or None, final=final)
        overflow = len(pages) > LONG_REPLY_MAX_PAGES
        for index, page in enumerate(pages[:LONG_REPLY_MAX_PAGES]):
            if not page.updated:
                continue
            # The agent may mention: its answer is the sender's intent.
            try:
                if index < len(sent):
                    result = await self._discord_call(
                        lambda: sent[index].edit(content=page.content, allowed_mentions=mentions),
                        "The reply page edit",
                    )
                else:
                    result = await self._discord_call(
                        lambda: channel.send(page.content, allowed_mentions=mentions),
                        "The reply page send",
                    )
            except discord.HTTPException as e:
                log.warning("The reply page post failed permanently: %s", e)
                return False, overflow
            if result is None:
                return False, overflow
            if index >= len(sent):
                sent.append(result)
        return True, overflow

    async def _stream_reply(self, channel, segments, mentions: discord.AllowedMentions, *, tag: str) -> tuple[bool, bool]:
        """Post the segments of a reply as they arrive: the open page is edited, full pages stay.

        Returns (yielded, posted): yielded when the agent produced any
        text, posted when at least one message reached the channel. A
        permanent send failure stops the posting, but the segments are
        drained to the end: the session record and the lock need the
        full run.
        """
        sent: list[discord.Message] = []
        parts: list[str] = []
        previous = ""
        alive = True
        async for segment in segments:
            parts.append(segment)
            if alive:
                text = "\n\n".join(parts)
                alive, _ = await self._sync_pages(channel, sent, text, previous, mentions, final=False)
                if alive:
                    previous = text
        if not parts:
            return False, False
        if not alive:
            return True, bool(sent)
        full = "\n\n".join(parts)
        alive, overflow = await self._sync_pages(channel, sent, full, previous, mentions, final=True)
        posted = bool(sent)
        if alive and overflow:
            # The inline pages stand as the head: the rest rides in a file.
            file = discord.File(io.BytesIO(full.encode("utf-8")), filename=f"eliza-reply-{tag}.txt")
            try:
                result = await self._discord_call(
                    lambda: channel.send("[...] the answer continues in the attached file", file=file, allowed_mentions=mentions),
                    "The reply file send",
                )
            except discord.HTTPException as e:
                log.warning("The reply file send failed permanently: %s", e)
                result = None
            if result is None:
                with contextlib.suppress(discord.HTTPException):
                    await self._discord_call(
                        lambda: channel.send("[...] the full answer could not be attached: the file upload failed", allowed_mentions=mentions),
                        "The reply page send",
                    )
            else:
                posted = True
        return True, posted

    @commands.Cog.listener()
    async def on_raw_poll_vote_add(self, payload: discord.RawPollVoteActionEvent) -> None:
        """A vote on a native poll: the majority rule can end it early."""
        await self.polls.native_vote(payload.message_id, payload.user_id, True)

    @commands.Cog.listener()
    async def on_raw_poll_vote_remove(self, payload: discord.RawPollVoteActionEvent) -> None:
        """A retracted native poll vote updates the tracked voters."""
        await self.polls.native_vote(payload.message_id, payload.user_id, False)

    async def _poll_trigger(self, session_id: int, channel, harness_text: str) -> None:
        """A poll event without a user message wakes the agent: the harness text in, the reply posted."""
        if self._closed:
            return
        guild = getattr(channel, "guild", None)
        user = getattr(channel, "recipient", None) if guild is None else None
        async with contextlib.AsyncExitStack() as stack:
            # The typing indicator is cosmetic: its failure must not skip the reply.
            with contextlib.suppress(aiohttp.ClientError, asyncio.TimeoutError, discord.HTTPException):
                await stack.enter_async_context(channel.typing())
            try:
                yielded, _ = await self._stream_reply(
                    channel
                    , self.engine.generate_reply(
                        channel.id
                        , harness_text
                        , guild_id=guild.id if guild is not None else None
                        , user_id=user.id if user is not None else None
                        , bot_name=self.bot.user.name if self.bot.user else "Eliza"
                        , user_name=user.display_name if user is not None else None
                        , is_owner=user is not None and await self.bot.is_owner(user)
                    )
                    , discord.AllowedMentions.all()
                    , tag=f"poll-{session_id}"
                )
            except Exception:
                log.exception("The poll trigger reply failed for channel %s:", channel.id)
                return
        if not yielded:
            # The agent chose the no-reply tag on a poll completion.
            log.info("The poll trigger of session %s got a no-reply answer.", session_id)
            return

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
        preset = await self.current_preset()
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
        model_name = await self.model_name()
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
        preset = await self.current_preset()
        cache_ttl = getattr(preset, "cache_ttl", None) or DEFAULT_CACHE_TTL
        active = sum(1 for session in self.history.sessions.values() if session.idle() < cache_ttl)
        description += f"\nActive sessions: {active} of {MAX_SESSIONS}"
        idle = len(self.history.sessions) - active
        if idle:
            description += f", {idle} idle"
        description += "."
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
        await asyncio.to_thread(self.workspace.drop, ctx.author.id)
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

    @eliza_group.group(name="sessions", invoke_without_command=True)
    @commands.admin()
    async def eliza_sessions(self, ctx: commands.Context) -> None:
        """List and close the live conversation sessions."""
        await ctx.send_help()

    @eliza_sessions.command(name="list")
    async def sessions_list(self, ctx: commands.Context) -> None:
        """List the live sessions: server and channel in a guild, the user of a direct message, with the usage stats of the scope."""
        if not self.history.sessions:
            note = "No live sessions."
            if self._closed:
                note += " The agent is closed. Reload the cog to start Eliza again."
            await ctx.send(note)
            return
        preset = await self.current_preset()
        cache_ttl = getattr(preset, "cache_ttl", None) or DEFAULT_CACHE_TTL
        limits = await self._rate_limits()
        lines = []
        recent_first = sorted(self.history.sessions.items(), key=lambda item: item[1].last_active, reverse=True)
        for session_id, session in recent_first:
            if session.scope == "channel":
                channel = await self._get_channel(session_id)
                if channel is not None and channel.guild is not None:
                    label = f"#{channel.name} — {channel.guild.name}"
                else:
                    label = f"channel {session_id}"
                scope = "channel"
            else:
                user = self.bot.get_user(session_id)
                if user is None:
                    with contextlib.suppress(discord.NotFound, discord.Forbidden, discord.HTTPException):
                        user = await self.bot.fetch_user(session_id)
                label = f"DM — {user.display_name}" if user is not None else f"DM — user {session_id}"
                scope = "user"
            idle = session.idle()
            state = "active" if idle < cache_ttl else f"idle {int(idle // 60)} min"
            stats = await self.scope_stats.get(scope, session_id)
            rate = await self.scope_stats.rate(scope, session_id)
            limit = limits.get(scope) or 0
            window = f"{rate.get('count', 0)}/{limit} this hour" if limit else "unlimited"
            lines.append(
                f"**{label}** ({state}) — {stats.get('messages', 0)} messages, "
                f"{stats.get('prompt_tokens', 0):,} prompt tokens, "
                f"{stats.get('completion_tokens', 0):,} completion tokens, "
                f"{stats.get('cached_tokens', 0):,} cached. Interactions: {window}."
            )
        embed = discord.Embed(
            title="Sessions",
            description="\n".join(lines),
            color=await ctx.embed_colour(),
        )
        await ctx.send(embed=embed)

    @eliza_sessions.command(name="close")
    async def sessions_close(self, ctx: commands.Context) -> None:
        """Close the agent: compact and drop every session, then hold for maintenance until the cog is reloaded.

        Every session is summarized and the summary persists, like on a cog
        unload. The agent then starts no new session: a message it would
        answer gets the maintenance notice as a direct reply. A cog reload
        starts the agent again.
        """
        if self._closed:
            await ctx.send("The agent is already closed. Reload the cog to start it again.")
            return
        # Closed first: no new reply starts while the compaction runs.
        self._closed = True
        count = len(self.history.sessions)
        await self.compactor.compact_all()
        self.history.sessions.clear()
        await asyncio.to_thread(self.workspace.drop_all)
        await ctx.send(
            f"The agent is closed. {count} session(s) compacted and dropped. "
            "Messages get the maintenance notice until the cog is reloaded."
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
        for page in paginate("\n\n".join(lines)):
            await ctx.send(page.content, allowed_mentions=discord.AllowedMentions.none())

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
    async def mcp_add(self, ctx: commands.Context, name: str, target: str, *headers: str) -> None:
        """Add an MCP server. The target must be an http(s) URL of a remote server. Local commands are not allowed.
        Extra `Header: value` arguments ride every request to the server. The command message is
        deleted when headers are given, to protect the values."""
        if not self.mcp.available:
            await ctx.send("The `mcp` package is not installed. Reinstall the cog so its requirements are installed.")
            return
        if not MCP_SERVER_NAME_RE.match(name):
            await ctx.send(
                "The name must match `^[a-zA-Z0-9_-]{1,32}$`. "
                "Tools are exposed as `<name>__<tool>` and the API limits tool names."
            )
            return
        if not target.startswith(("http://", "https://")):
            await ctx.send("Only web-based MCP servers are allowed: the target must be an http(s) URL.")
            return
        parsed = {}
        for header in headers:
            head, sep, value = header.partition(":")
            head, value = head.strip(), value.strip()
            if not sep or not head or not value:
                await ctx.send("Each header must be a `Header: value` pair, quoted so it stays one argument.")
                return
            if "\r" in header or "\n" in header:
                await ctx.send("A header must fit on one line.")
                return
            parsed[head] = value
        spec = {"transport": "http", "url": target, "command": "", "args": []}
        if parsed:
            spec["headers"] = parsed
        await self.mcp.close_server(name)
        async with self.config.mcp_servers() as servers:
            servers[name] = spec
        if parsed:
            with contextlib.suppress(discord.HTTPException):
                await ctx.message.delete()
        await ctx.send(
            f"MCP server `{name}` saved ({spec['transport']}"
            + (f", {len(parsed)} headers" if parsed else "")
            + f"). It connects on first use. Check it with `{ctx.prefix}eliza mcp list`."
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
            extra = f", {len(spec['headers'])} headers" if spec.get("headers") else ""
            lines.append(f"**{name}** ({spec['transport']}{extra}) `{target}` — {state}")
        embed = discord.Embed(
            title="MCP servers",
            description="\n".join(lines),
            color=await ctx.embed_colour(),
        )
        await ctx.send(embed=embed)
