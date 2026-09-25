"""The reply engine: one user message in, the agent answer out."""

import asyncio
import base64
import json
import logging
import re
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from urllib.parse import urlparse

import aiohttp
import discord

from .history import BACKFILL_MESSAGES, DEFAULT_CACHE_TTL, Session, session_label
from .prompt import place_block, system_text
from .stats import Scope
from .tools import MESSAGE_TIME_FORMAT
from .tools.base import DISCORD_FILE_HOSTS, attachments_text, poll_result_suffix, read_limited, speech_embed_line
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


def collapse_blank_lines(text: str) -> str:
    """A run of blank lines collapses to one blank line. The blank runs at
    both ends drop. The models pad the notes between their tool calls with
    long empty sections."""
    return re.sub(r"(?:[ \t]*\n){3,}", "\n\n", text).strip()


class ChatError(Exception):
    """A classified API error. `kind` names the category, `raw` keeps the provider body or the original exception."""

    def __init__(self, kind: str, text: str, raw=None):
        super().__init__(text)
        self.kind = kind
        self.raw = raw


class IncomingMessage:
    """One user message on its way into the engine: the ids of where it
    happened, the speaker, and the message facts. The derivations the
    engine needs live here: the session of the conversation, the stats
    scope, the speaker tag of the turns, and the send-time stamp. A poll
    event builds one without a user."""

    def __init__(self, *, channel_id: int, content: str, guild_id: int | None = None, user_id: int | None = None, bot_name: str | None = None, user_name: str | None = None, is_owner: bool = False, message_id: int | None = None, attachments: list | None = None):
        self.channel_id = channel_id
        self.content = content
        self.guild_id = guild_id
        self.user_id = user_id
        self.bot_name = bot_name
        self.user_name = user_name
        self.is_owner = is_owner
        self.message_id = message_id
        self.attachments = list(attachments or [])
        self.scope = Scope(channel_id=channel_id, guild_id=guild_id, user_id=user_id)

    @property
    def session_id(self) -> int:
        """The conversation session: the channel in a guild, the user in a direct message."""
        return self.channel_id if self.guild_id is not None else self.user_id

    @property
    def speaker(self) -> str:
        """The name the turns carry: the display name, else a stand-in."""
        return self.user_name or "User"

    @property
    def tag(self) -> str:
        """The speaker tag of the turns: the name and the mention of the user."""
        return f"{self.speaker} <@{self.user_id}>" if self.user_id is not None else self.speaker

    @property
    def stamp(self) -> str:
        """The send time of the message: the snowflake when the id names one, else the moment of the call."""
        moment = discord.utils.snowflake_time(self.message_id) if self.message_id is not None else datetime.now(timezone.utc)
        return f"{moment:{MESSAGE_TIME_FORMAT}}"


class ToolContext:
    """The engine surface one native provider tool runs on: the closures
    of a reply and its facts. The engine builds one per reply; a tool
    handler receives it beside the arguments, and a new capability is a
    field here, not a new argument of every handler."""

    def __init__(self, *, call_api, fetch_url, api_post, send_file, channel_nsfw, set_conversation_model, vision_chat, show_image, request_song=None, media_ceiling=None, add_media_cost=None):
        self.call_api = call_api
        self.fetch_url = fetch_url
        self.api_post = api_post
        self.send_file = send_file
        self.channel_nsfw = channel_nsfw
        self.set_conversation_model = set_conversation_model
        self.vision_chat = vision_chat
        self.show_image = show_image
        self.request_song = request_song
        self.media_ceiling = media_ceiling
        self.add_media_cost = add_media_cost


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

    async def generate_reply(self, message: IncomingMessage) -> AsyncIterator[str]:
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
        session_id = message.session_id
        if session_id not in self.history.sessions:
            preset = await self.api.current_preset()
            cache_ttl = getattr(preset, "cache_ttl", None) or DEFAULT_CACHE_TTL
            active = sum(1 for other in self.history.sessions.values() if other.idle() < cache_ttl)
            if active >= MAX_SESSIONS:
                yield "The agent is already busy in other conversations. Try again later."
                return
        session = await self.history.get(session_id, "channel" if message.guild_id is not None else "user")
        while True:
            async with session.acquire():
                if self.history.sessions.get(session_id) is not session:
                    # The session was unloaded while this reply waited on its
                    # lock (sweeper unload, close): retry on the live session.
                    session = await self.history.get(session_id, session.scope)
                    continue
                async for segment in self._generate_locked(session, message, api_key=api_key):
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
        context. The speech-transcription embed is the one bot reply that
        stays: it reads as the turn of the speaker it names, and the audio
        message it answers still drops. Return an empty list on any failure:
        the backfill never breaks a reply.
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
                    # stay out of the context. A speech-transcription embed is
                    # the one reply that stays: it carries the speech of the
                    # audio message it answers, which drops like any answered
                    # message.
                    if message.reference is not None and message.reference.message_id is not None:
                        skipped.add(message.reference.message_id)
                    if speech_embed_line(message) is None:
                        continue
                if message.id in skipped:
                    continue
                stripped = message.content.strip()
                if message.author.id == bot_id:
                    # A poll result notification or a speech-transcription
                    # embed has empty content and no attachments: its
                    # substance rides in the embed, read below.
                    if (not stripped and not message.attachments
                            and message.type != discord.MessageType.poll_result
                            and speech_embed_line(message) is None):
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
            stamp = f"{message.created_at:{MESSAGE_TIME_FORMAT}}"
            if message.author.id == bot_id:
                speech = speech_embed_line(message)
                if speech is not None:
                    # The transcription embed rides a bot reply, but it is
                    # the turn of the speaker it names, not an agent answer.
                    turns.append({"role": "user", "content": f"{stamp} {speech}"})
                    continue
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

    async def _generate_locked(self, session: Session, message: IncomingMessage, *, api_key: str) -> AsyncIterator[str]:
        """The reply work of generate_reply. The caller holds the session lock."""
        session_id = message.session_id
        channel_id = message.channel_id
        guild_id = message.guild_id
        user_id = message.user_id
        content = message.content
        bot_name = message.bot_name
        is_owner = message.is_owner
        attachments = message.attachments
        preset = await self.api.current_preset()
        cache_ttl = getattr(preset, "cache_ttl", None) or DEFAULT_CACHE_TTL
        usage = {
            "prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0, "cost": 0.0
          , "tool_calls": 0, "images": 0, "inpaints": 0, "music": 0
        }
        # Context expiry: idle past the cache lifetime, or a compaction.
        # Only then is the system message rebuilt (prompt, memory, summary).
        # An agent memory update is already in the context as a tool call, so no reload between.
        expired = not session.messages or session.idle() >= cache_ttl
        # The readable session name of the log lines, built like the sessions
        # list. The channel read stays on the cache: a reply never waits on
        # a fetch for a log line.
        label = session_label(
            session.scope, session_id
            , channel=self.bot.get_channel(channel_id) if session.scope == "channel" else None
            , user_name=message.user_name
        )
        async def channel_nsfw():
            """Whether the current channel sits behind the Discord 18+ gate,
            for the model routing, the payload build, and the native provider
            tools: the channel flag, the flag of the parent channel of a
            thread, or an age-restricted guild. A direct message of the bot
            owner counts as gated: the API reports no user age, and the owner
            operates the bot."""
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
        if not session.messages or session.reroute:
            # A fresh session asks the decision flow of the provider for the
            # capability strengths of the conversation, and a compaction asks
            # again: the context was just rebuilt, so the model choice
            # deserves a re-check. The harness picks the preset in code, and
            # a pick lands on the override with no condense: the fresh
            # summary already fits the target window. A miss (no decision
            # flow, a low bundled credit balance, a failed call) keeps the
            # current model.
            session.reroute = False
            decider = getattr(self.api, "decide_session_model", None)
            if decider is not None:
                state = f"{message.speaker}: {message.content}{attachments_text(attachments)}"
                if session.summary:
                    state = f"Summary of the conversation so far:\n{session.summary}\nNew message: {state}"
                picked = await decider(state, gated)
                if picked:
                    session.model_override = picked
                    log.info(
                        "The decision routing picked the preset %s for session %s%s."
                        , picked, label, "" if not session.messages else " after the compaction",
                    )
        # The conversation's model string, an override included: it decides
        # the compaction budget here and the request model below.
        request_model = session.model_override or await self.api.model_name()
        if session.messages and self.history.needs_compaction(session, cache_ttl, await self.api.context_length(preset, request_model)):
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
                    for turn in await self._backfill_turns(channel, name, message.message_id):
                        session.append(turn["role"], turn["content"])
        session.touch()
        speaker = message.speaker
        tag = message.tag
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
        # The stamp names the send time of the message: the snowflake or
        # the moment of the call.
        stamp = message.stamp
        # A preset may name the only tools its request model may carry (the
        # Qwen pair and Gemma of Venice: the models overuse the wider tool
        # surface). The filter keeps the named tools alone, and the MCP
        # servers never join the reply: their gather is skipped whole.
        # A preset may also bar the MCP servers alone (Gemini of Venice: its
        # backend rejects the schema keywords of the user servers), the
        # harness and provider tools stay.
        allowed = preset.tool_filter(request_model) if preset is not None else None
        mcp_allowed = preset is None or preset.mcp_tools_allowed(request_model)
        if allowed is None and mcp_allowed:
            tools, routes, replaced = await self.mcp.gather_tools(preset, api_key)
        else:
            tools, routes, replaced = [], {}, set()
        native_routes = {}
        media_counts = {}
        if preset is not None:
            # Native provider tools (for example the Z.AI vision tool) join
            # the MCP tools: a provider tool takes the harness name it reuses.
            # A tool that declares "media" names the usage counter a
            # successful call increments (images, inpaints, music).
            # The availability flags of an entry: dm_owner_only joins a
            # direct message only when the bot owner speaks (a guild is not
            # affected), guild_credit_gate hides the entry in a guild while
            # the cog reports the bundled credits under the disable band of
            # the media windows.
            guild_gate = False
            media_windows = None
            if hasattr(self.api, "media_limits"):
                # The windows scale with the credit ratio; an empty dict
                # marks the media tools off (no credits for media). A
                # provider without the policy keeps the full constants.
                media_windows = await self.api.media_limits() or {}
                guild_gate = guild_id is not None and not media_windows
            ceilings = None
            if hasattr(self.api, "media_ceilings"):
                # The render descriptors freeze at the context start: the
                # tool list stays stable inside a context, so the prompt
                # cache holds. The handlers check the live ceiling at call
                # time and answer the temporary-disable error.
                if expired or session.render_ceilings is None:
                    session.render_ceilings = await self.api.media_ceilings()
                ceilings = session.render_ceilings
            for entry in preset.native_tools(ceilings):
                if allowed is not None and entry["name"] not in allowed:
                    # The filter keeps the fixed tool set alone.
                    continue
                if guild_id is None and entry.get("dm_owner_only") and not is_owner:
                    # A direct message of anyone but the owner never sees the tool.
                    continue
                if entry.get("guild_credit_gate") and guild_gate:
                    # The credit balance sits under the floor: the tool leaves the guild list.
                    continue
                native_routes[entry["name"]] = entry["handler"]
                if entry.get("media"):
                    media_counts[entry["name"]] = entry["media"]
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
        if allowed is not None:
            # The filter trims the harness defaults too: the request carries
            # the fixed tool set alone.
            tools = [tool for tool in tools if tool["function"]["name"] in allowed]
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
                # default system prompt joins the vision answer. The model of
                # the tool payload rides along: a raw id passes through the
                # provider untouched, and the output cap of the model applies.
                tool_payload.update(preset.extra_payload(session_id, tool_payload.get("model", "")))
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

        async def api_post(path, *, json_body=None, data=None, binary=False, timeout=120):
            """One POST to a REST path of the active provider, for native provider tools."""
            return await self.api.provider_post(api_key, path, json_body=json_body, data=data, binary=binary, timeout=timeout)

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

        # The images a tool read into this conversation: analyze_image of a
        # vision conversation hands its image over, and the model sees the
        # image itself — no second model describes it between.
        shown_images = []

        async def show_image(name, data, mime):
            """Hand one image to this conversation, for native provider
            tools. True when the conversation model sees images and the
            image joined (a harness note carries it in the next round);
            False when the tool must answer another way."""
            if not vision_chat:
                return False
            shown_images.append((name, data, mime))
            return True

        async def set_conversation_model(model_id):
            """Override the chat model of this conversation, for native
            provider tools. None restores the configured model. Returns
            None when the override stored, an error text when the move
            failed: a target that holds a smaller context window than the
            current model compacts the session first, and a failed
            compaction refuses the move — the turns would not fit the
            next request. The compaction keeps no verbatim turns (keep
            0) and runs on the current model, the one whose window still
            holds the turns, so the new model starts from the summary.
            The override rides the session: it survives a context
            restart and dies with the session. It answers the next and
            later requests, the request that runs the tool keeps its
            model."""
            if preset is not None:
                configured = await self.api.model_name()
                current = session.model_override or configured
                target = model_id if model_id is not None else configured
                current_window = preset.context_length(current)
                target_window = preset.context_length(target)
                if (
                    current_window and target_window and target_window < current_window
                    and len(session.messages) > 1
                ):
                    compact_usage = await self.compactor.compact(session_id, session, api_key, preset, keep=0, reroute=False)
                    if compact_usage is None:
                        name = model_id if model_id is not None else "the default model"
                        reason = f" ({session.error})" if session.error else ""
                        return (
                            f"Error: the condense before the move to {name} failed{reason}. "
                            "The conversation keeps its current model."
                        )
                    for key in usage:
                        usage[key] += compact_usage.get(key) or 0
            session.model_override = model_id
            return None

        api_request_song = getattr(self.api, "request_song", None)

        async def request_song(request):
            """Hand one song request to the approval flow of the cog, for
            native provider tools. The closure answers at once: the vote and
            the generation outcome reach the model as later harness turns."""
            if api_request_song is None:
                return "Error: the song approval is not available here."
            request = dict(request, requester_id=user_id, requester_name=speaker)
            return await api_request_song(session_id, channel_id, request)

        async def media_ceiling(kind: str):
            """The live render price ceiling of the credit ratio for a media
            catalog ("image" or "edit"), None when no ceiling holds. A model
            over it answers the temporary-disable error at call time, long
            after the descriptors froze their list at the context start."""
            if not hasattr(self.api, "media_ceilings"):
                return None
            ceilings = await self.api.media_ceilings()
            return ceilings.get(kind) if isinstance(ceilings, dict) else None

        async def add_media_cost(amount: float):
            """The render price a media tool reports at its answered request
            joins the cost total of the reply: a refusal bills the same, and
            the stats record the sum with the turn."""
            usage["cost"] += float(amount or 0)

        # The surface the native provider tools of this reply run on.
        tool_context = ToolContext(
            call_api=call_api, fetch_url=fetch_url, api_post=api_post
            , send_file=send_file, channel_nsfw=channel_nsfw
            , set_conversation_model=set_conversation_model
            , vision_chat=vision_chat, show_image=show_image
            , request_song=request_song, media_ceiling=media_ceiling
            , add_media_cost=add_media_cost
        )

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
        # The session turns this request was built from. A model switch onto
        # a smaller window compacts the session mid-reply: the identity
        # change marks it, and the overload-fallback retry rebuilds its
        # payload on the compacted context.
        turns_at_request = session.messages
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
            dies with it. A fallback onto a smaller context window
            compacts the session at the move, and the retry rides the
            compacted context. The error still reaches the user: the
            notice is staged with the fallback name and yields ahead of
            the answer. Any other failure passes through."""
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
                error = await set_conversation_model(fallback)
                if error:
                    # The move needed a compaction that failed: the
                    # conversation keeps its model, the original error stands.
                    raise
                if session.messages is not turns_at_request:
                    # The switch compacted the session: the retry rides the
                    # compacted context, with the turns of this reply after
                    # it. additions and exchange name those turns: a second
                    # move of the same reply splices again, and the session
                    # block sits at no fixed offset of messages anymore.
                    # The splice stays in place, so the later rounds and
                    # the closing pass keep appending to the same list.
                    messages[:] = [*session.messages, *additions, *exchange]
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
            answer = self.api.message_of(data)
            if answer is None:
                yield f"The API returned an unexpected answer: {str(data)[:500]}"
                return
            tool_calls = answer.get("tool_calls") or []
            if not tool_calls:
                # The session records what the model said.
                await self._record_turn(
                    session, additions, exchange, answer, scope=message.scope, usage=usage
                )
                segment = self._final_segment(answer, emitted)
                # One line per reply: a silent model would otherwise leave no
                # trace at all, and a wrong model behind a preset name would
                # stay invisible.
                log.info(
                    "Reply for session %s: model %s, %d tool rounds, closing %s."
                    , label, payload.get("model"), len(exchange)
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
            text = collapse_blank_lines(answer.get("content") or "")
            echo = {"role": "assistant", "content": text}
            for key in ("reasoning_content", "reasoning"):
                if answer.get(key):
                    echo[key] = answer[key]
            echo["tool_calls"] = tool_calls
            messages.append(echo)
            exchange.append(echo)
            if text and text != NO_REPLY_TAG:
                # A note the model wrote alongside its calls: the caller
                # posts it while the tools run.
                emitted = True
                yield text
            for call in tool_calls:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                name = function.get("name", "")
                media = media_counts.get(name)
                media_refusal = (
                    # The bot owner bypasses the media windows, like the
                    # interaction limits. A generation still counts: the
                    # channel budget stays honest.
                    await self.scope_stats.media_refusal(message.scope, media_windows)
                    if media and not is_owner else None
                )
                try:
                    if media_refusal is not None:
                        # The hourly media windows are full: the call never
                        # reaches the provider, the model sees the refusal.
                        result_text = media_refusal
                    elif name in routes:
                        # Routes win: a provider tool can take a harness name.
                        result_text = await self.mcp.run_tool(name, arguments, routes)
                    elif name in native_routes:
                        result_text = await native_routes[name](arguments, tool_context)
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
                # The counters of the scope stats: every executed call counts,
                # a media generation counts only when it answers without an
                # error line (the uniform failure prefix of every tool — the
                # window refusal carries it too) and takes one slot of the
                # hourly media windows.
                usage["tool_calls"] += 1
                if media and not result_text.startswith("Error:"):
                    usage[media] = usage.get(media, 0) + 1
                    await self.scope_stats.count_media(message.scope)
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
            if shown_images:
                # A tool read an image into this conversation (analyze_image
                # of a vision conversation): the model sees the image itself
                # in the next round, no other model describes it between.
                # The note rides the session like any turn, until a
                # compaction summarizes it.
                parts = [{"type": "text", "text": "[harness] the image a tool read into this conversation:"}]
                budget = VISION_IMAGE_BUDGET_BYTES
                while shown_images and len(parts) - 1 < VISION_MAX_IMAGES:
                    name, data, mime = shown_images.pop(0)
                    if len(data) > budget:
                        continue
                    budget -= len(data)
                    parts.append({
                        "type": "image_url"
                        , "image_url": {"url": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"}
                    })
                if len(parts) > 1:
                    note = {"role": "user", "content": parts}
                    messages.append(note)
                    exchange.append(note)
                elif shown_images:
                    log.info("The read images stay unseen: over the vision caps of one note.")
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
        answer = self.api.message_of(data)
        if answer is not None:
            await self._record_turn(
                session, additions, exchange, answer, scope=message.scope, usage=usage
            )
            segment = self._final_segment(answer, emitted)
            # One line per reply: a silent model would otherwise leave no
            # trace at all, and a wrong model behind a preset name would
            # stay invisible.
            log.info(
                "Reply for session %s: model %s, %d tool rounds, closing %s."
                , label, payload.get("model"), len(exchange)
                , "silent (the no-reply tag or an empty close)" if segment is None else "posted",
            )
            if segment is not None:
                yield segment
            return
        await self.scope_stats.record(message.scope, usage=usage)
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
        content = collapse_blank_lines(content)
        if content and content != NO_REPLY_TAG:
            # The flattened text, not the raw field: a content array
            # (a vision answer in parts) posts as its text.
            return content
        if not content and not emitted:
            return EMPTY_REPLY
        return None

    async def _record_turn(self, session: Session, additions: list, exchange: list, answer: dict, *, scope: Scope, usage: dict) -> None:
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
        final_content = answer.get("content") or ""
        if isinstance(final_content, str):
            # A content array (a vision answer in parts) stays as the
            # provider sent it: the next request takes the parts back.
            final_content = collapse_blank_lines(final_content)
        final = {"role": "assistant", "content": final_content}
        for key in ("reasoning_content", "reasoning"):
            if answer.get(key):
                final[key] = answer[key]
        session.append_message(final)
        if scope.user_id is not None:
            # The turn landed in the context: the memory note does not repeat.
            session.seen_users.add(scope.user_id)
        await self.scope_stats.record(scope, usage=usage)
