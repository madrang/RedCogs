"""The reply engine: one user message in, the agent answer out."""

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
import discord

from .history import BACKFILL_MESSAGES, DEFAULT_CACHE_TTL, Session
from .prompt import place_block, system_text
from .tools import MESSAGE_TIME_FORMAT
from .tools.base import DISCORD_FILE_HOSTS, attachments_text, poll_result_suffix, read_limited
from .tools.files import post_file

log = logging.getLogger("red.agenteliza")

MCP_TOOL_ROUNDS = 16
# A vision chat model reads the images of the conversation directly: the
# images of a message and the images a tool posts join the turns as
# image_url parts. The caps bound one turn (the message images) and one
# note (the posted images): a data URI rides every later request of the
# session, so an unbounded set would bloat the context until a compaction.
VISION_MAX_IMAGES = 4
VISION_IMAGE_BUDGET_BYTES = 8 * 1024 * 1024
# The image mime type of a file extension, for the data URIs of posted
# files (the Discord attachments name theirs through the content type).
IMAGE_EXT_MIMES = {
    "png": "image/png"
  , "jpg": "image/jpeg"
  , "jpeg": "image/jpeg"
  , "gif": "image/gif"
  , "webp": "image/webp"
}
# The system prompt tells the agent a limit of 10 tool calls. The gap gives
# slack when the agent miscounts its own calls.
# Cap of one download of a native provider tool (an image for the vision call).
NATIVE_TOOL_FETCH_MAX_BYTES = 10_485_760
# Cap of parallel conversation sessions (channels and direct messages
# together). A new session over the cap is refused: every live session is a
# provider cache entry and a summarization load.
MAX_SESSIONS = 3
# Context backfill of a fresh session: recent channel messages from Discord.
# The message count is BACKFILL_MESSAGES from history.py: the compaction
# keeps the same count verbatim. The size cap is in characters, the context
# counts tokens: 64000 characters is about 16K tokens at 4 per token. The
# value matches the uniform tool result cap (TOOL_RESULT_MAX_CHARS).
BACKFILL_MAX_CHARS = 64_000
# Cap of the raw messages scanned to find the qualifying ones. The scan
# pages deeper into the history until it has BACKFILL_MESSAGES messages of
# the conversation with the agent, or this cap.
BACKFILL_SCAN_MAX = 99
# The reply sent when the model returns no content at all.
EMPTY_REPLY = "(empty answer)"
# The tag that lets the agent refuse to reply. The harness sends nothing.
NO_REPLY_TAG = "[no-reply]"


class ChatError(Exception):
    """A classified API error. `kind` names the category, `raw` keeps the provider body or the original exception."""

    def __init__(self, kind: str, text: str, raw=None):
        super().__init__(text)
        self.kind = kind
        self.raw = raw


class ChatEngine:
    """One conversation turn: context build, chat request, tool rounds, turn record.

    The api parameter is the provider surface: chat_request, message_of,
    model_name, current_preset, context_length, usage_blocked. The cog
    plays this role.
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

    async def generate_reply(self, channel_id: int, content: str, *, guild_id: int | None = None, user_id: int | None = None, bot_name: str | None = None, user_name: str | None = None, is_owner: bool = False, message_id: int | None = None, attachments: list | None = None) -> AsyncIterator[str]:
        """Send one user message to the chat API and yield the reply in segments.

        The conversation session is the channel, or the user in DMs: the
        history and the provider cache key follow the session, so a channel
        runs one agent context. Each
        stored user turn carries the speaker name as `name: content`.
        The first turn of a user in a context gets their user memory
        injected before it, once per context. Each piece of assistant text
        is yielded as it completes: the notes the model writes alongside
        its tool calls, then the answer. A no-reply answer yields nothing.
        One chat request runs per session at a time: the session lock is
        held for the whole iteration, so the caller must drain it fully.
        """
        api_key = await self.config.api_key()
        if not api_key:
            yield "The API key is not set. An admin can set it with the `eliza setkey` command."
            return
        blocked = await self.api.usage_blocked()
        if blocked:
            yield blocked
            return
        session_id = channel_id if guild_id is not None else user_id
        if session_id not in self.history.sessions:
            preset = await self.api.current_preset()
            cache_ttl = getattr(preset, "cache_ttl", None) or DEFAULT_CACHE_TTL
            active = sum(1 for other in self.history.sessions.values() if other.idle() < cache_ttl)
            if active >= MAX_SESSIONS:
                yield "The agent is already busy in other conversations. Try again later."
                return
        session = await self.history.get(session_id, "channel" if guild_id is not None else "user")
        while True:
            async with session.acquire():
                if self.history.sessions.get(session_id) is not session:
                    # The session was unloaded while this reply waited on its
                    # lock (sweeper unload, close): retry on the live session.
                    session = await self.history.get(session_id, session.scope)
                    continue
                async for segment in self._generate_locked(
                    session, session_id, channel_id, content, api_key=api_key,
                    guild_id=guild_id, user_id=user_id, bot_name=bot_name, user_name=user_name, is_owner=is_owner,
                    message_id=message_id, attachments=attachments,
                ):
                    yield segment
                return

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
            # The bot messages of the window, and the user replies whose
            # reference the API left unresolved, as message id: reference id.
            # After the scan, such a reply qualifies when its target is a
            # bot message of the window.
            bot_messages = set()
            pending = {}
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
                stripped = message.content.strip()
                if message.author.id == bot_id:
                    # A poll result notification has empty content and no
                    # attachments: its outcome rides in the embed, read below.
                    if not stripped and not message.attachments and message.type != discord.MessageType.poll_result:
                        continue
                    bot_messages.add(message.id)
                else:
                    if message.author.bot:
                        continue
                    if not self._participates(message, bot_id):
                        reference = message.reference if message.type == discord.MessageType.reply else None
                        if reference is None or reference.message_id is None or isinstance(reference.resolved, discord.Message):
                            # Not a reply, or provably a reply to someone else.
                            continue
                        pending[message.id] = reference.message_id
                if stripped and prefixes and stripped.startswith(prefixes):
                    continue
                qualifying.append(message)
                if len(qualifying) >= BACKFILL_MESSAGES:
                    break
        except (discord.Forbidden, discord.HTTPException):
            return []
        if pending:
            # An unresolved reference qualifies only against a bot message of
            # the window: the rest was a reply to someone else.
            qualifying = [
                message for message in qualifying
                if message.id not in pending or pending[message.id] in bot_messages
            ]
        turns = []
        for message in reversed(qualifying):
            content = message.content.strip()
            attachment_files = [(a.filename, a.content_type, a.url) for a in message.attachments]
            if message.author.id == bot_id:
                if message.type == discord.MessageType.poll_result:
                    # The poll result notification carries the outcome in its
                    # embed: the agent learns the results of a completed poll.
                    content += poll_result_suffix(message)
                content += attachments_text(attachment_files)
                if not content:
                    # A poll result without a readable embed adds nothing.
                    continue
                if turns and turns[-1]["role"] == "assistant":
                    turns[-1]["content"] += "\n" + content
                else:
                    turns.append({"role": "assistant", "content": content})
                continue
            stamp = f"{message.created_at:{MESSAGE_TIME_FORMAT}}"
            if (not content or content in (f"<@{bot_id}>", f"<@!{bot_id}>")) and not attachment_files:
                # The live path maps an empty message to a poke: the
                # backfill shows the same, so the agent sees the poke.
                turns.append({"role": "user", "content": f"{stamp} {message.author.display_name} <@{message.author.id}>: (poke: the user sent an empty message)"})
                continue
            for form in (f"<@{bot_id}>", f"<@!{bot_id}>"):
                content = content.replace(form, bot_name)
            turns.append({"role": "user", "content": f"{stamp} {message.author.display_name} <@{message.author.id}>: {content.strip()}{attachments_text(attachment_files)}"})
        while turns and sum(len(turn["content"]) for turn in turns) > BACKFILL_MAX_CHARS:
            turns.pop(0)
        return turns

    async def _generate_locked(self, session: Session, session_id: int, channel_id: int, content: str, *, api_key: str, guild_id, user_id, bot_name, user_name, is_owner, message_id, attachments=None) -> AsyncIterator[str]:
        """The reply work of generate_reply. The caller holds the session lock."""
        preset = await self.api.current_preset()
        cache_ttl = getattr(preset, "cache_ttl", None) or DEFAULT_CACHE_TTL
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost": 0.0}
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
            session.start_context(system_text(name, memory, rules_block, await place_block(self.bot, guild_id, channel_id, is_owner=is_owner)))
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
        tools, routes, replaced = await self.mcp.gather_tools(preset, api_key)
        native_routes = {}
        if preset is not None:
            # Native provider tools (for example the Z.AI vision tool) join
            # the MCP tools: a provider tool takes the harness name it reuses.
            for entry in preset.native_tools():
                native_routes[entry["name"]] = entry["handler"]
                replaced.add(entry["name"])
                tools.append({
                    "type": "function"
                    , "function": {
                        "name": entry["name"]
                        , "description": entry["description"]
                        , "parameters": entry["parameters"]
                    }
                })
        # Harness tools come first: their list is stable, MCP tools may vary.
        # A provider tool in the replaced set takes the place of the harness
        # default of the same name.
        tools = [tool for tool in self.harness_tools.tools() if tool["function"]["name"] not in replaced] + tools
        async def channel_nsfw():
            """Whether the current channel sits behind the Discord 18+ gate,
            for the payload build and the native provider tools: the channel
            flag, the flag of the parent channel of a thread, or an
            age-restricted guild. A direct message of the bot owner counts
            as gated: the API reports no user age, and the owner operates
            the bot."""
            if guild_id is None and is_owner:
                return True
            getter = self.harness_tools.channel_getter
            channel = await getter(channel_id) if getter else None
            parent = getattr(channel, "parent", None)
            guild = getattr(channel, "guild", None) or getattr(parent, "guild", None)
            return bool(
                getattr(channel, "nsfw", False)
                or getattr(parent, "nsfw", False)
                or getattr(guild, "nsfw_level", None) == discord.NSFWLevel.age_restricted
            )

        gated = await channel_nsfw()
        request_model = session.model_override or await self.api.model_name()
        # A vision chat model sees the images of the conversation directly.
        # The resolved request model decides, an override included: a preset
        # name maps to its id first, and the provider's vision_models set
        # names the models that accept image parts in the chat contract.
        vision_chat = (
            preset is not None
            and preset.resolve_model(request_model) in preset.vision_models
        )

        async def call_api(tool_payload):
            """One chat-completions call on the active provider, for native provider tools."""
            if preset is not None:
                # The provider payload extras ride this call too, not only the
                # reply: Venice needs venice_parameters on every call, or its
                # default system prompt joins the vision answer.
                tool_payload.update(preset.extra_payload(session_id))
            return await self.api.chat_request(api_key, tool_payload)

        async def fetch_url(url):
            """Download one URL with the shared session, for native provider tools."""
            headers = {}
            if urlparse(url).netloc.lower() in DISCORD_FILE_HOSTS:
                # The Discord file hosts need an authorized request. The bot
                # token goes to these hosts only, never to a foreign URL.
                token = getattr(self.bot.http, "token", None)
                if token:
                    headers["Authorization"] = f"Bot {token}"
            try:
                async with self.api._get_session().get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        log.warning("fetch_url got HTTP %d for %r", response.status, url[:150])
                        return None
                    content_type = response.content_type or "application/octet-stream"
                    body = await read_limited(response, NATIVE_TOOL_FETCH_MAX_BYTES + 1)
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                log.warning("fetch_url failed for %r: %s: %s", url[:150], type(e).__name__, e)
                return None
            if len(body) > NATIVE_TOOL_FETCH_MAX_BYTES:
                log.warning("fetch_url: the body of %r is over the %d bytes cap", url[:150], NATIVE_TOOL_FETCH_MAX_BYTES)
                return None
            return body, content_type

        async def api_post(path, *, json_body=None, data=None, binary=False):
            """One POST to a REST path of the active provider, for native provider tools."""
            return await self.api.provider_post(api_key, path, json_body=json_body, data=data, binary=binary)

        # The images the native tools posted in this reply: a vision chat
        # model sees them through the harness note the tool loop builds.
        posted_images = []

        async def send_file(name, data, caption=None):
            """Post one binary file to the current channel, for native provider tools."""
            getter = self.harness_tools.channel_getter
            channel = await getter(channel_id) if getter else None
            if channel is None:
                return "Error: the current channel is unknown."
            sent = await post_file(channel, data, name, caption)
            if sent.startswith("The file ") and name.rsplit(".", 1)[-1].lower() in IMAGE_EXT_MIMES:
                posted_images.append((name, data))
            return sent

        async def set_conversation_model(model_id):
            """Override the chat model of this conversation, for native
            provider tools. None restores the configured model. The override
            rides the session: it survives a context restart and dies with
            the session. It answers the next and later requests, the
            request that runs the tool keeps its model."""
            session.model_override = model_id

        # The user turn of this message. On a vision chat model the images
        # of the message join as image_url parts (a data URI, fetched with
        # the bot token on the Discord hosts): the model sees them directly.
        # The attachment line stays in the text either way: it names every
        # attachment, and it is the only record a non-vision model keeps.
        text = f"{stamp} {tag}: {content}{attachments_text(attachments)}"
        turn_content = text
        if vision_chat and attachments:
            parts = []
            budget = VISION_IMAGE_BUDGET_BYTES
            for name, content_type, url in attachments:
                if not str(content_type or "").startswith("image/"):
                    continue
                if len(parts) >= VISION_MAX_IMAGES or budget <= 0:
                    break
                fetched = await fetch_url(url)
                if fetched is None:
                    continue
                body, header_type = fetched
                mime = header_type if str(header_type or "").startswith("image/") else ""
                if not mime:
                    continue
                if len(body) > budget:
                    continue
                budget -= len(body)
                parts.append({
                    "type": "image_url"
                  , "image_url": {"url": f"data:{mime};base64,{base64.b64encode(body).decode('ascii')}"}
                })
            if parts:
                turn_content = [{"type": "text", "text": text}, *parts]
        additions.append({"role": "user", "content": turn_content})
        messages = [*session.messages, *additions]
        payload = {
            "model": request_model
            , "messages": messages
            , "stream": False
        }
        if preset is not None:
            # Provider-specific fields, for example prompt_cache_key on Kimi.
            # A provider may also rewrite the model here: Venice resolves a
            # preset name to its id, the 18+ variant behind the gate.
            payload.update(preset.extra_payload(session_id, payload["model"], gated))
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        rounds = MCP_TOOL_ROUNDS if tools else 1

        # A staged overload notice: the wrapper fills it, the reply loop
        # yields it as the leading segment, so the user always sees the
        # error even when the fallback rescues the request.
        overload_notice = []

        async def chat_with_overload_fallback(request_payload):
            """One reply-path chat request with the overload fallback: a
            rate-limit or HTTP error that names model overload moves the
            conversation to the next preset up in cost and retries once on
            it. The move rides
            the session override, so it holds for the conversation and
            dies with it. The error still reaches the user: the notice is
            staged with the fallback name and yields ahead of the answer.
            Any other failure passes through."""
            try:
                return await self.api.chat_request(api_key, request_payload)
            except ChatError as e:
                overloaded = (
                    e.kind in ("rate_limit", "http")
                    and isinstance(e.raw, dict)
                    and "overloaded" in str(e.raw.get("error") or "").lower()
                )
                if preset is None or not overloaded:
                    raise
                fallback = preset.preset_fallback(request_payload.get("model") or "")
                if not fallback:
                    raise
                log.warning(
                    "The model %s is overloaded: the conversation moves to the preset %s for now."
                    , request_payload.get("model"), fallback,
                )
                overload_notice.append(
                    f"⚠️ {e}\n```\n{json.dumps(e.raw, ensure_ascii=False)[:1500]}\n```"
                    f"\nThe preset {fallback} answers this conversation for now."
                )
                await set_conversation_model(fallback)
                request_payload["model"] = fallback
                request_payload.update(preset.extra_payload(session_id, fallback, gated))
                return await self.api.chat_request(api_key, request_payload)

        # The tool rounds of this reply: assistant calls and tool results, as
        # the API sent them. The session keeps them until a compaction.
        exchange = []
        # True once a segment went out: a closing empty answer or no-reply
        # tag must not replace a note the caller already posted.
        emitted = False
        for _ in range(rounds):
            try:
                data = await chat_with_overload_fallback(payload)
            finally:
                # The idle and cache clock counts from the last provider
                # contact, not from the user message: a long generation or a
                # tool round re-warms the cache when its answer arrives.
                session.touch()
            if overload_notice:
                # The overload notice leads the reply, ahead of any model text.
                yield overload_notice.pop(0)
            round_usage = data.get("usage") or {}
            for key in usage:
                usage[key] += round_usage.get(key) or 0
            # The provider's own cost metric of the request, when it names one.
            if preset is not None:
                usage["cost"] += preset.cost_of(data)
            if round_usage.get("prompt_tokens"):
                # The real prompt size calibrates the compaction trigger.
                session.last_prompt_tokens = round_usage["prompt_tokens"]
            message = self.api.message_of(data)
            if message is None:
                yield f"The API returned an unexpected answer: {str(data)[:500]}"
                return
            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                # The session records what the model said.
                await self._record_turn(
                    session, additions, exchange, message, guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
                )
                segment = self._final_segment(message, emitted)
                # One line per reply: a silent model would otherwise leave no
                # trace at all, and a wrong model behind a preset name would
                # stay invisible.
                log.info(
                    "Reply for session %s: model %s, %d tool rounds, closing %s."
                    , session_id, payload.get("model"), len(exchange)
                    , "silent (the no-reply tag or an empty close)" if segment is None else "posted",
                )
                if segment is not None:
                    yield segment
                return
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
            text = (message.get("content") or "").strip()
            if text and text != NO_REPLY_TAG:
                # A note the model wrote alongside its calls: the caller
                # posts it while the tools run.
                emitted = True
                yield message["content"]
            for call in tool_calls:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                name = function.get("name", "")
                try:
                    if name in routes:
                        # Routes win: a provider tool can take a harness name.
                        result_text = await self.mcp.run_tool(name, arguments, routes)
                    elif name in native_routes:
                        result_text = await native_routes[name](arguments, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model)
                    elif name in self._harness_tool_names:
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
                    "role": "tool"
                    , "tool_call_id": call.get("id", "")
                    , "content": result_text,
                }
                messages.append(result)
                exchange.append(result)
            if posted_images and vision_chat:
                # The vision chat model sees what a tool posted: the images
                # join the exchange as a harness note, so the model can
                # judge its own output. The note rides the session like any
                # turn, until a compaction summarizes it. A model without
                # vision keeps the text result of the tool alone.
                parts = [{"type": "text", "text": "[harness] the image a tool posted to the conversation:"}]
                budget = VISION_IMAGE_BUDGET_BYTES
                while posted_images and len(parts) - 1 < VISION_MAX_IMAGES:
                    name, data = posted_images.pop(0)
                    if len(data) > budget:
                        continue
                    budget -= len(data)
                    parts.append({
                        "type": "image_url"
                      , "image_url": {
                            "url": f"data:{IMAGE_EXT_MIMES[name.rsplit('.', 1)[-1].lower()]}"
                                   f";base64,{base64.b64encode(data).decode('ascii')}"
                        }
                    })
                if len(parts) > 1:
                    note = {"role": "user", "content": parts}
                    messages.append(note)
                    exchange.append(note)
                elif posted_images:
                    log.info("The posted images stay unseen: over the vision caps of one note.")
        # The rounds are spent: one last pass without tools, so the model can answer with what it found.
        payload["tool_choice"] = "none"
        try:
            data = await chat_with_overload_fallback(payload)
        finally:
            session.touch()
        if overload_notice:
            # The overload notice leads the reply, ahead of the closing text.
            yield overload_notice.pop(0)
        round_usage = data.get("usage") or {}
        for key in usage:
            usage[key] += round_usage.get(key) or 0
        # The provider's own cost metric of the request, when it names one.
        if preset is not None:
            usage["cost"] += preset.cost_of(data)
        message = self.api.message_of(data)
        if message is not None:
            await self._record_turn(
                session, additions, exchange, message, guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
            )
            segment = self._final_segment(message, emitted)
            # One line per reply: a silent model would otherwise leave no
            # trace at all, and a wrong model behind a preset name would
            # stay invisible.
            log.info(
                "Reply for session %s: model %s, %d tool rounds, closing %s."
                , session_id, payload.get("model"), len(exchange)
                , "silent (the no-reply tag or an empty close)" if segment is None else "posted",
            )
            if segment is not None:
                yield segment
            return
        await self.scope_stats.record(
            guild_id=guild_id, channel_id=channel_id, user_id=user_id, usage=usage
        )
        yield "The agent made too many tool calls in a row. Try a simpler request."

    @staticmethod
    def _final_segment(message: dict, emitted: bool) -> str | None:
        """The postable text of a closing message, or None when the stream ends silent.

        The no-reply tag ends the stream with nothing more. An empty
        message speaks only when no segment went out before: the notice
        would replace a real answer the user already saw. A content array
        (a vision model answering in parts) flattens to its text.
        """
        content = message.get("content") or ""
        if isinstance(content, list):
            content = "".join(part.get("text") or "" for part in content if isinstance(part, dict))
        content = content.strip()
        if content and content != NO_REPLY_TAG:
            # The flattened text, not the raw field: a content array
            # (a vision answer in parts) posts as its text.
            return content
        if not content and not emitted:
            return EMPTY_REPLY
        return None

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
