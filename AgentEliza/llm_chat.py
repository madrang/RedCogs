"""The reply engine: one user message in, the agent answer out."""

import json
import logging
from datetime import datetime, timezone

import discord

from .history import BACKFILL_MESSAGES, DEFAULT_CACHE_TTL, Session
from .prompt import place_block, system_text
from .tools import MESSAGE_TIME_FORMAT

log = logging.getLogger("red.agenteliza")

MCP_TOOL_ROUNDS = 16
# The system prompt tells the agent a limit of 10 tool calls. The gap gives
# slack when the agent miscounts its own calls.
# Cap of parallel conversation sessions (channels and direct messages
# together). A new session over the cap is refused: every live session is a
# provider cache entry and a summarization load.
MAX_SESSIONS = 3
# Context backfill of a fresh session: recent channel messages from Discord.
# The message count is BACKFILL_MESSAGES from history.py: the compaction
# keeps the same count verbatim. The size cap is in characters, the context
# counts tokens: 128K characters is about 32K tokens at 4 per token.
BACKFILL_MAX_CHARS = 131_072
# Cap of the raw messages scanned to find the qualifying ones. The scan
# pages deeper into the history until it has BACKFILL_MESSAGES messages of
# the conversation with the agent, or this cap.
BACKFILL_SCAN_MAX = 199


class ChatError(Exception):
    """A classified API error. `kind` names the category, `raw` keeps the provider body or the original exception."""

    def __init__(self, kind: str, text: str, raw=None):
        super().__init__(text)
        self.kind = kind
        self.raw = raw


class ChatEngine:
    """One conversation turn: context build, chat request, tool rounds, turn record.

    The api parameter is the provider surface: chat_request, message_of,
    normalize_reply, model_name, current_preset, context_length,
    usage_blocked. The cog plays this role.
    """

    def __init__(self, bot, config, history, memory, mcp, harness_tools, scope_stats, compactor, api, polls=None):
        self.bot = bot
        self.config = config
        self.history = history
        self.memory = memory
        self.mcp = mcp
        self.harness_tools = harness_tools
        self._harness_tool_names = {tool["function"]["name"] for tool in harness_tools.tools()}
        self.scope_stats = scope_stats
        self.compactor = compactor
        self.api = api
        self.polls = polls

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
        blocked = await self.api.usage_blocked()
        if blocked:
            return blocked
        session_id = channel_id if guild_id is not None else user_id
        if session_id not in self.history.sessions:
            preset = await self.api.current_preset()
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

    @staticmethod
    def _poll_result_suffix(message: discord.Message) -> str:
        """The results line of a poll result notification: the embed holds the outcome."""
        for embed in message.embeds:
            if embed.type != "poll_result":
                continue
            fields = {field.name: field.value for field in embed.fields}
            if "poll_question_text" not in fields:
                continue
            # A tie has no victor fields.
            victor = fields.get("victor_answer_text")
            outcome = (
                f"winner: {victor} ({fields.get('victor_answer_votes', '?')} votes)"
                if victor else "no winner: a tie"
            )
            return f"\nPoll results for {fields['poll_question_text']!r}: {outcome}, total votes: {fields.get('total_votes', '?')}."
        return ""

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
                if message.type == discord.MessageType.poll_result:
                    # The poll result notification carries the outcome in its
                    # embed: the agent learns the results of a completed poll.
                    content += self._poll_result_suffix(message)
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
        preset = await self.api.current_preset()
        cache_ttl = (preset.cache_ttl if preset is not None else None) or DEFAULT_CACHE_TTL
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0}
        # Context expiry: idle past the cache lifetime, or a compaction.
        # Only then is the system message rebuilt (prompt, memory, summary).
        # An agent memory update is already in the context as a tool call, so no reload between.
        expired = not session.messages or session.idle() >= cache_ttl
        if session.messages and self.history.needs_compaction(session, cache_ttl, await self.api.context_length(preset)):
            compact_usage = await self.compactor.compact(session_id, session, api_key, preset)
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
                rules_block = (
                    "You talk to the bot owner. The owner has all rights. "
                    "The memory tools reach the user memory and the channel memory of this conversation."
                )
            else:
                rules_block = (
                    f"You talk to a limited user. User rules:\n{await self.config.dm_rules()}\n"
                    "The memory tools reach only the user memory of this conversation."
                )
            # The summary storage follows the scope of the conversation.
            summary_scope = "channel" if guild_id is not None else "user"
            rules_block += f"\nWhenever this conversation becomes idle, you always update the {summary_scope} summary, and the conversation resumes from it later."
            # A fresh session lost its verbatim turns: the Discord history restores them.
            fresh = not session.messages
            session.start_context(system_text(name, memory, rules_block, await place_block(self.bot, guild_id, channel_id)))
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
        if self.polls is not None:
            # The vote status of the session leads the turn: the agent
            # answers with the current counts in sight.
            status = await self.polls.status_text(session_id)
            if status:
                additions.append({"role": "user", "content": f"[harness]\n{status}\n[/harness]"})
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
            "model": await self.api.model_name(),
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
                data = await self.api.chat_request(api_key, payload)
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
            message = self.api.message_of(data)
            if message is None:
                return f"The API returned an unexpected answer: {str(data)[:500]}"
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                # The session records what the model said. The caller gets the normalized form.
                await self._record_turn(
                    session, additions, exchange, message, guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
                )
                return self.api.normalize_reply(message)
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
                try:
                    if name in self._harness_tool_names:
                        result_text = await self.harness_tools.run(
                            name, arguments, guild_id=guild_id, channel_id=channel_id, user_id=user_id,
                            is_owner=is_owner,
                        )
                    else:
                        result_text = await self.mcp.run_tool(name, arguments, routes)
                except Exception as e:
                    # A tool failure must not kill the reply: the model gets the error as the tool result.
                    log.exception("The tool %s failed:", name)
                    result_text = f"Error: the tool {name} failed: {type(e).__name__}: {e}"
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
            data = await self.api.chat_request(api_key, payload)
        finally:
            session.touch()
        round_usage = data.get("usage") or {}
        for key in usage:
            usage[key] += round_usage.get(key) or 0
        message = self.api.message_of(data)
        if message is not None:
            await self._record_turn(
                session, additions, exchange, message, guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
            )
            return self.api.normalize_reply(message)
        await self.scope_stats.record(
            guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
        )
        return "The agent made too many tool calls in a row. Try a simpler request."

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
