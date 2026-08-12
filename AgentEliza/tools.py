import asyncio
import io
import re
from datetime import datetime, timezone
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, urlparse

import aiohttp
import discord

from .memory import MEMORY_MAX_CHARS, Memory

# Agent-facing scope name -> internal Memory scope.
SCOPE_ALIASES = {
    "server": "guild",
    "channel": "channel",
    "user": "user",
}
# Cap of one web tool result. A giant page can fill the context in one call.
WEB_TOOL_RESULT_MAX_CHARS = 32_000
# Cap of the bytes read from one fetched page.
WEB_FETCH_MAX_BYTES = 1_000_000
WEB_TIMEOUT = aiohttp.ClientTimeout(total=30)
WEB_SEARCH_URL = "https://html.duckduckgo.com/html/?q="
WEB_SEARCH_MAX_RESULTS = 8
# DuckDuckGo answers a non-browser request with an anomaly page (202). The
# check scores the full header set: the User-Agent alone passes only
# sometimes. No brotli in Accept-Encoding: the Brotli package is not a
# dependency.
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate",
    "DNT": "1",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}
# Tags whose content is not page text.
_SKIP_TAGS = {"script", "style", "noscript", "template", "head"}
# Caps of the history_read tool: the raw messages scanned, and the
# qualifying messages returned. Discord has no search endpoint for bots:
# the query parameter filters the scanned window only.
HISTORY_READ_SCAN_MAX = 400
HISTORY_READ_MAX_RESULTS = 64
HISTORY_READ_DEFAULT_RESULTS = 20
# Cap of one message text in a history_read result.
HISTORY_READ_MESSAGE_MAX_CHARS = 1000
# Base Discord upload limit (25 MiB), for direct messages. A guild channel
# reports its own limit in guild.filesize_limit (boosted servers get more).
FILE_SEND_DM_MAX_BYTES = 26_214_400


class _SearchParser(HTMLParser):
    """Collect the results of the DuckDuckGo HTML answer: title, link, snippet."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list = []
        self._current: dict | None = None
        self._field: str | None = None

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        classes = (dict(attrs).get("class") or "").split()
        if "result__a" in classes:
            self._current = {"title": "", "url": _unwrap_duckduckgo(dict(attrs).get("href") or ""), "snippet": ""}
            self._field = "title"
        elif "result__snippet" in classes and self.results:
            # The snippet anchor follows the title anchor of its result.
            self._field = "snippet"

    def handle_endtag(self, tag):
        if tag != "a" or self._field is None:
            return
        if self._field == "title" and self._current is not None:
            self.results.append(self._current)
            self._current = None
        self._field = None

    def handle_data(self, data):
        if self._field == "title" and self._current is not None:
            self._current["title"] += data
        elif self._field == "snippet":
            self.results[-1]["snippet"] += data


def _unwrap_duckduckgo(href: str) -> str:
    """The result links are /l/ redirects: the real URL is the uddg parameter."""
    target = parse_qs(urlparse(href).query).get("uddg")
    return target[0] if target else href


class _TextExtractor(HTMLParser):
    """Strip a page to its readable text: tags and script/style content out."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        elif tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"[ \t]*\n[ \t\n]*", "\n", re.sub(r"[ \t]+", " ", unescape("".join(self.parts)))).strip()


def _cap(text: str) -> str:
    """Truncate a web tool result to the context-friendly cap."""
    if len(text) > WEB_TOOL_RESULT_MAX_CHARS:
        dropped = len(text) - WEB_TOOL_RESULT_MAX_CHARS
        text = text[:WEB_TOOL_RESULT_MAX_CHARS] + f"\n[truncated: {dropped} characters dropped]"
    return text


class HarnessTools:
    """Tools the harness itself provides to the agent, beside the MCP tools.

    Each tool maps to a `_tool_<name>` method. Memory tools resolve the
    scope from the ids of the conversation the agent is answering, so the
    agent can read and update the memory of the current server, channel,
    or user at any time. The optional target parameter selects another
    user or channel of the same server, by id, mention, or name. Names
    resolve through the guild object from guild_getter: the Discord API
    has no user search by name outside a guild.
    """

    def __init__(self, memory: Memory, session_getter, guild_getter=None, channel_getter=None, bot_id_getter=None):
        self.memory = memory
        # Callable returning the shared aiohttp session of the cog, for the web tools.
        self.session_getter = session_getter
        # Callable returning the guild object of an id, for target name resolution.
        self.guild_getter = guild_getter
        # Callable returning the channel object of an id, for the history tool.
        self.channel_getter = channel_getter
        # Callable returning the bot user id, for the involvement filter.
        self.bot_id_getter = bot_id_getter

    def tools(self) -> list:
        """The OpenAI function schemas of the harness tools."""
        scope_property = {
            "type": "string"
            , "enum": list(SCOPE_ALIASES)
            , "description": "The memory scope."
        }
        target_property = {
            "type": "string"
            , "description": (
                "Optional. The channel or user to use instead of the active one. "
                "Formats: a Discord id, a mention (<@id> or <#id>), or an exact name. "
                "A user name is the server display name, the global name, or the username. "
                "A channel name works with or without the #. "
                "Names resolve only inside a server. The server scope ignores this parameter."
            )
        }
        return [
            {
                "type": "function"
                , "function": {
                    "name": "memory_read"
                    , "description": (
                        "Read the long-term memory of one scope: server (facts shared by the whole Discord server), "
                        "channel (facts of the current channel), or user (facts about the person talking to you). "
                        "The optional target selects another channel or user of this server."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {"scope": scope_property, "target": target_property}
                        , "required": ["scope"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "memory_write"
                    , "description": (
                        "Replace the long-term memory of one scope: server, channel, or user. Read the "
                        "scope first and merge when you want to keep the old content. An empty content "
                        f"erases the scope. The harness truncates content over {MEMORY_MAX_CHARS} characters. "
                        "The optional target selects another channel or user of this server."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
                            , "target": target_property
                            , "content": {
                                "type": "string"
                                , "description": "The new memory text."
                            }
                        }
                        , "required": ["scope", "content"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "memory_append"
                    , "description": (
                        "Add text at the end of the long-term memory of one scope: server, channel, or "
                        "user. Use this tool for one new fact. The result warns when the scope is full "
                        "and part of the content did not fit. "
                        "The optional target selects another channel or user of this server."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
                            , "target": target_property
                            , "content": {
                                "type": "string"
                                , "description": "The text to add."
                            }
                        }
                        , "required": ["scope", "content"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "history_read"
                    , "description": (
                        "Read the messages of a channel, a thread, or this direct message. "
                        "The result shows only the conversation with you: your own messages, "
                        "and the messages that mention you or reply to you. The messages arrive "
                        "oldest first. The Discord API has no search endpoint for bots: "
                        "the query parameter filters the scanned messages."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "target": {
                                "type": "string"
                                , "description": (
                                    "Optional. The channel or thread to read instead of the current one. "
                                    "Formats: a Discord id, a mention (<#id>), or an exact name, with or without the #. "
                                    "Resolves only inside the current server. In a direct message, leave it blank."
                                )
                            }
                            , "query": {
                                "type": "string"
                                , "description": "Optional. Keep only the messages that contain this text."
                            }
                            , "limit": {
                                "type": "integer"
                                , "description": (
                                    f"Optional. The maximum number of messages to return, 1 to {HISTORY_READ_MAX_RESULTS}. "
                                    f"Default {HISTORY_READ_DEFAULT_RESULTS}."
                                )
                            }
                            , "after": {
                                "type": "string"
                                , "description": (
                                    "Optional. A UTC date or date-time, for example 2026-08-11 or 2026-08-11 14:30. "
                                    "With before, the messages between the two times. Alone, the messages around that time."
                                )
                            }
                            , "before": {
                                "type": "string"
                                , "description": (
                                    "Optional. A UTC date or date-time, for example 2026-08-11 or 2026-08-11 14:30. "
                                    "With after, the messages between the two times. Alone, the messages around that time."
                                )
                            }
                        }
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "file_send"
                    , "description": (
                        "Send a file to the current conversation. The content is the full text of the file. "
                        "The file can be up to the Discord upload limit of the channel (25 MB on most servers). "
                        "Do not repeat the file content in your answer."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "filename": {
                                "type": "string"
                                , "description": "The file name with its extension, for example notes.md."
                            }
                            , "content": {
                                "type": "string"
                                , "description": "The full text of the file."
                            }
                            , "caption": {
                                "type": "string"
                                , "description": "Optional. A message text sent with the file."
                            }
                        }
                        , "required": ["filename", "content"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "web_search"
                    , "description": (
                        "Search the web with DuckDuckGo. The result is a numbered list. Each entry has "
                        "a title, a URL, and a snippet. Use web_fetch on a result URL to read the full page."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "query": {
                                "type": "string"
                                , "description": "The search query."
                            }
                        }
                        , "required": ["query"]
                    }
                }
            }
            , {
                "type": "function"
                , "function": {
                    "name": "web_fetch"
                    , "description": (
                        "Fetch one web page and return its readable text. The tool strips an HTML page "
                        "to its text. The tool truncates a long page."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "url": {
                                "type": "string"
                                , "description": "The http(s) URL of the page."
                            }
                        }
                        , "required": ["url"]
                    }
                }
            }
        ]

    async def run(self, name: str, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        """Run one harness tool and return its output as text."""
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            return f"Error: unknown harness tool {name}"
        return await handler(arguments, guild_id=guild_id, channel_id=channel_id, user_id=user_id)

    def _scope_ids(self, scope: str, guild_id, channel_id, user_id):
        """Resolve an agent-facing scope name to (scope, scope_id, label), or an error text."""
        internal = SCOPE_ALIASES.get(scope)
        if internal is None:
            return None, None, None, f"Error: unknown scope {scope!r}. Use one of: {', '.join(SCOPE_ALIASES)}."
        scope_id = {
            "guild": guild_id
            , "channel": channel_id
            , "user": user_id
        }[internal]
        if scope_id is None:
            return None, None, None, "Error: there is no server in a direct message."
        return internal, scope_id, Memory.SCOPES[internal][1], None

    def _resolve_target(self, internal: str, target, *, guild_id, channel_id, user_id):
        """Resolve the optional target of a memory tool to (scope_id, name, error text).

        A blank target keeps the active conversation. An id or a mention
        (<@id>, <#id>) resolves directly. A name resolves against the
        members or channels of the current server: the Discord API has no
        user search by name outside a guild. Every target is checked
        against the current server, so a call cannot reach the memory of
        another server. name is None for the active conversation.
        """
        current = {"guild": guild_id, "channel": channel_id, "user": user_id}
        text = str(target or "").strip()
        if not text:
            return current[internal], None, None
        if internal == "guild":
            return None, None, "Error: the server scope is always the current server. Leave the target blank."
        guild = self.guild_getter(guild_id) if self.guild_getter and guild_id is not None else None
        if guild is None:
            return None, None, "Error: a target works only inside a server. In a direct message, leave the target blank."
        mention = re.fullmatch(r"<[@#][!&]?(\d+)>", text)
        if mention:
            text = mention.group(1)
        kind = "channel" if internal == "channel" else "member"
        if text.isdigit():
            wanted = int(text)
            found = guild.get_channel(wanted) if internal == "channel" else guild.get_member(wanted)
            if found is None:
                return None, None, f"Error: no {kind} with the id {wanted} in this server."
            return wanted, self._target_name(internal, found), None
        if internal == "channel":
            name = text.removeprefix("#")
            matches = [
                channel for channel in guild.channels
                if channel.name == name and isinstance(channel, discord.abc.Messageable)
            ]
        else:
            matches = [
                member for member in guild.members
                if text == member.name or text == member.display_name
                or (member.global_name and text == member.global_name)
            ]
        if not matches:
            return None, None, f"Error: no {kind} named {text!r} in this server. Use the Discord id to be sure."
        if len(matches) > 1:
            return None, None, f"Error: {len(matches)} matches for {text!r} in this server. Use the Discord id."
        found = matches[0]
        return found.id, self._target_name(internal, found), None

    @staticmethod
    def _target_name(internal: str, found) -> str:
        """The display name of a resolved target, for the confirmation text."""
        return found.name if internal == "channel" else found.display_name

    async def _tool_memory_read(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        scope_id, name, error = self._resolve_target(scope, arguments.get("target"), guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if error:
            return error
        target = f" for {name}" if name else ""
        text = await self.memory.read(scope, scope_id)
        if not text:
            return f"(no memory stored for the {label.lower()} scope{target})"
        return text

    async def _tool_memory_write(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        scope_id, name, error = self._resolve_target(scope, arguments.get("target"), guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if error:
            return error
        target = f" of {name}" if name else ""
        content = arguments.get("content")
        if not isinstance(content, str):
            return "Error: the content must be a string."
        stored = await self.memory.store(scope, scope_id, content)
        if not stored:
            return f"The {label.lower()} memory{target} has been erased."
        if len(stored) < len(content):
            return (
                f"Warning: the content was truncated from {len(content)} to {len(stored)} characters "
                f"(Config storage limit). The {label.lower()} memory{target} now ends mid-text. "
                "Read it and rewrite it shorter."
            )
        return f"The {label.lower()} memory{target} has been updated ({len(stored)} characters)."

    async def _tool_memory_append(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        scope_id, name, error = self._resolve_target(scope, arguments.get("target"), guild_id=guild_id, channel_id=channel_id, user_id=user_id)
        if error:
            return error
        target = f" of {name}" if name else ""
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            return "Error: the content must be a non-empty string."
        current = await self.memory.read(scope, scope_id)
        combined = f"{current}\n{content}" if current else content
        stored = await self.memory.store(scope, scope_id, combined)
        if len(stored) < len(combined):
            dropped = len(combined) - len(stored)
            return (
                f"Warning: the {label.lower()} memory{target} is full. Only part of the content was added: "
                f"the last {dropped} characters were dropped (Config storage limit). "
                "Read the scope and rewrite it shorter."
            )
        return f"The {label.lower()} memory{target} now holds {len(stored)} characters."

    @staticmethod
    def _involves_bot(message, bot_id: int) -> bool:
        """True when the message belongs to the conversation with the agent.

        The same rule as the context backfill, plus the messages of the bot
        itself: its own messages, the user messages that mention it, the
        replies to it. Other bots stay out. A direct message always
        qualifies.
        """
        if message.author.id == bot_id or message.guild is None:
            return True
        if message.author.bot:
            return False
        if any(user.id == bot_id for user in message.mentions):
            return True
        if message.type != discord.MessageType.reply:
            return False
        resolved = message.reference.resolved if message.reference else None
        return isinstance(resolved, discord.Message) and resolved.author.id == bot_id

    def _resolve_channel(self, target, *, guild_id, channel_id):
        """Resolve the optional target of history_read to (channel, label, error text).

        A blank target keeps the current channel or direct message. An id,
        a <#id> mention, or an exact name resolves inside the current
        server only: a direct message stays private to its own context.
        """
        text = str(target or "").strip()
        if not text:
            channel = self.channel_getter(channel_id) if self.channel_getter else None
            if channel is None:
                return None, None, "Error: the current channel is unknown."
            label = f"#{channel.name}" if isinstance(channel, discord.abc.GuildChannel) else "this direct message"
            return channel, label, None
        guild = self.guild_getter(guild_id) if self.guild_getter and guild_id is not None else None
        if guild is None:
            return None, None, "Error: a target works only inside a server. In a direct message, leave the target blank."
        mention = re.fullmatch(r"<#(\d+)>", text)
        if mention:
            text = mention.group(1)
        if text.isdigit():
            channel = guild.get_channel_or_thread(int(text))
            if channel is None:
                return None, None, f"Error: no channel or thread with the id {text} in this server."
        else:
            name = text.removeprefix("#")
            pool = [
                entry for entry in [*guild.channels, *guild.threads]
                if isinstance(entry, discord.abc.Messageable)
            ]
            matches = [entry for entry in pool if entry.name == name]
            if not matches:
                return None, None, f"Error: no channel or thread named {text!r} in this server. Use the Discord id to be sure."
            if len(matches) > 1:
                return None, None, f"Error: {len(matches)} matches for {text!r} in this server. Use the Discord id."
            channel = matches[0]
        return channel, f"#{channel.name}", None

    @staticmethod
    def _history_time(value, name: str):
        """Parse one UTC date or date-time of the history window, or an error text."""
        if value is None:
            return None, None
        if not isinstance(value, str) or not value.strip():
            return None, f"Error: the {name} time must be a string, for example 2026-08-11 or 2026-08-11 14:30."
        try:
            parsed = datetime.fromisoformat(value.strip())
        except ValueError:
            return None, f"Error: cannot read the {name} time {value!r}. Use a UTC date or date-time, for example 2026-08-11 14:30."
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed, None

    async def _tool_history_read(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        channel, label, error = self._resolve_channel(arguments.get("target"), guild_id=guild_id, channel_id=channel_id)
        if error:
            return error
        bot_id = self.bot_id_getter() if self.bot_id_getter else None
        if bot_id is None:
            return "Error: the bot user is not ready."
        try:
            limit = max(1, min(int(arguments.get("limit") or HISTORY_READ_DEFAULT_RESULTS), HISTORY_READ_MAX_RESULTS))
        except (TypeError, ValueError):
            limit = HISTORY_READ_DEFAULT_RESULTS
        after, error = self._history_time(arguments.get("after"), "after")
        if error:
            return error
        before, error = self._history_time(arguments.get("before"), "before")
        if error:
            return error
        query = arguments.get("query")
        query = query.strip().casefold() if isinstance(query, str) and query.strip() else None
        guild = getattr(channel, "guild", None)
        if guild is not None and guild.me is not None and not channel.permissions_for(guild.me).read_message_history:
            return f"Error: I do not have the permission to read the history of {label}."
        # oldest_first=True alone would scan from the channel start
        # (discord.py: reverse=True starts at OLDEST_OBJECT). The default
        # window scans the newest messages instead, reversed on output.
        bounded = after is not None and before is not None
        window = {"limit": HISTORY_READ_SCAN_MAX}
        if bounded:
            window.update(after=after, before=before, oldest_first=True)
        elif after is not None or before is not None:
            # around accepts a limit of at most 101, capped to 100 by discord.py.
            window.update(around=after if after is not None else before, oldest_first=True, limit=100)
        messages = []
        try:
            async for message in channel.history(**window):
                if not self._involves_bot(message, bot_id):
                    continue
                content = message.content.strip()
                if not content:
                    continue
                if query and query not in content.casefold():
                    continue
                messages.append(message)
                if len(messages) >= limit:
                    break
        except (discord.Forbidden, discord.HTTPException) as e:
            return f"Error: the history read failed: {e}"
        if not bounded and not window.get("oldest_first"):
            messages.reverse()
        if not messages:
            return f"(no matching messages in {label})"
        lines = [f"({len(messages)} messages of {label}, oldest first)"]
        for message in messages:
            content = " ".join(message.content.split())
            if len(content) > HISTORY_READ_MESSAGE_MAX_CHARS:
                content = content[:HISTORY_READ_MESSAGE_MAX_CHARS] + " [...]"
            lines.append(f"{message.created_at:%Y-%m-%d %H:%M} {message.author.display_name} <@{message.author.id}>: {content}")
        return _cap("\n".join(lines))

    async def _tool_file_send(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        filename = arguments.get("filename")
        if not isinstance(filename, str) or not filename.strip():
            return "Error: the filename must be a non-empty string."
        # The name is cosmetic on Discord: keep one clean path-less part.
        name = re.sub(r"[\\/\x00-\x1f]+", "_", filename).strip(" .")[:100]
        if not name:
            return "Error: the filename has no usable characters."
        content = arguments.get("content")
        if not isinstance(content, str) or not content:
            return "Error: the content must be a non-empty string."
        caption = arguments.get("caption")
        if caption is not None and not isinstance(caption, str):
            return "Error: the caption must be a string."
        if caption and len(caption) > 2000:
            return "Error: the caption is over 2000 characters, the Discord message limit."
        channel = self.channel_getter(channel_id) if self.channel_getter else None
        if channel is None:
            return "Error: the current channel is unknown."
        guild = getattr(channel, "guild", None)
        if guild is not None and guild.me is not None and not channel.permissions_for(guild.me).attach_files:
            return "Error: I do not have the permission to attach files in this channel."
        data = content.encode("utf-8")
        limit = guild.filesize_limit if guild is not None else FILE_SEND_DM_MAX_BYTES
        if len(data) > limit:
            return (
                f"Error: the file holds {len(data)} bytes, over the Discord upload limit of this channel "
                f"({limit} bytes). Split the content into smaller files."
            )
        try:
            await channel.send(content=caption or None, file=discord.File(io.BytesIO(data), filename=name))
        except (discord.Forbidden, discord.HTTPException) as e:
            return f"Error: the file send failed: {e}"
        return f"The file {name} ({len(data)} bytes) has been sent."

    async def _tool_web_search(self, arguments: dict, **_scope) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: the query must be a non-empty string."
        try:
            async with self.session_getter().get(
                WEB_SEARCH_URL + quote_plus(query)
                , headers=BROWSER_HEADERS
                , timeout=WEB_TIMEOUT
            ) as response:
                if response.status != 200:
                    return f"Error: the search failed (HTTP {response.status})."
                body = await response.content.read(WEB_FETCH_MAX_BYTES)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return f"Error: the search request failed: {e}"
        parser = _SearchParser()
        parser.feed(body.decode("utf-8", errors="replace"))
        results = [r for r in parser.results if r["url"].startswith(("http://", "https://"))]
        if not results:
            return "(no results)"
        lines = []
        for index, result in enumerate(results[:WEB_SEARCH_MAX_RESULTS], 1):
            title = " ".join(result["title"].split())
            snippet = " ".join(result["snippet"].split())
            lines.append(f"{index}. {title}\n{result['url']}\n{snippet}")
        return _cap("\n\n".join(lines))

    async def _tool_web_fetch(self, arguments: dict, **_scope) -> str:
        url = arguments.get("url")
        if not isinstance(url, str) or not url.strip():
            return "Error: the url must be a non-empty string."
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return "Error: only http and https URLs can be fetched."
        try:
            async with self.session_getter().get(url, timeout=WEB_TIMEOUT) as response:
                if response.status != 200:
                    return f"Error: the page answered HTTP {response.status}."
                content_type = (response.content_type or "").lower()
                body = await response.content.read(WEB_FETCH_MAX_BYTES)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return f"Error: the fetch failed: {e}"
        text = body.decode(response.charset or "utf-8", errors="replace")
        if content_type == "text/html":
            extractor = _TextExtractor()
            extractor.feed(text)
            text = extractor.text()
        elif not content_type.startswith("text/") and content_type not in ("application/json", "application/xml"):
            return f"Error: unsupported content type {content_type or '(unknown)'}. Only web pages and text can be fetched."
        if not text:
            return "(the page has no readable text)"
        return _cap(text)
