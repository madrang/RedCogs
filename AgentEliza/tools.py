import asyncio
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, urlparse

import aiohttp

from .memory import MEMORY_MAX_CHARS, Memory

# Agent-facing scope name -> internal Memory scope.
SCOPE_ALIASES = {
    "server": "guild",
    "channel": "channel",
    "user": "user",
}
# Cap of one web tool result. A giant page can fill the context in one call.
WEB_TOOL_RESULT_MAX_CHARS = 8000
# Cap of the bytes read from one fetched page.
WEB_FETCH_MAX_BYTES = 1_000_000
WEB_TIMEOUT = aiohttp.ClientTimeout(total=30)
WEB_SEARCH_URL = "https://html.duckduckgo.com/html/?q="
WEB_SEARCH_MAX_RESULTS = 8
# DuckDuckGo answers the default aiohttp agent with an anomaly page (202).
BROWSER_USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
# Tags whose content is not page text.
_SKIP_TAGS = {"script", "style", "noscript", "template", "head"}


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
    or user at any time.
    """

    def __init__(self, memory: Memory, session_getter):
        self.memory = memory
        # Callable returning the shared aiohttp session of the cog, for the web tools.
        self.session_getter = session_getter

    def tools(self) -> list:
        """The OpenAI function schemas of the harness tools."""
        scope_property = {
            "type": "string"
            , "enum": list(SCOPE_ALIASES)
            , "description": "The memory scope."
        }
        return [
            {
                "type": "function"
                , "function": {
                    "name": "memory_read"
                    , "description": (
                        "Read the long-term memory of one scope: server (facts shared by the whole Discord server), "
                        "channel (facts of the current channel), or user (facts about the person talking to you)."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {"scope": scope_property}
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
                        f"erases the scope. The harness truncates content over {MEMORY_MAX_CHARS} characters."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
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
                        "and part of the content did not fit."
                    )
                    , "parameters": {
                        "type": "object"
                        , "properties": {
                            "scope": scope_property
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

    async def _tool_memory_read(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        text = await self.memory.read(scope, scope_id)
        if not text:
            return f"(no memory stored for the {label.lower()} scope)"
        return text

    async def _tool_memory_write(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        content = arguments.get("content")
        if not isinstance(content, str):
            return "Error: the content must be a string."
        stored = await self.memory.store(scope, scope_id, content)
        if not stored:
            return f"The {label.lower()} memory has been erased."
        if len(stored) < len(content):
            return (
                f"Warning: the content was truncated from {len(content)} to {len(stored)} characters "
                f"(Config storage limit). The {label.lower()} memory now ends mid-text. "
                "Read it and rewrite it shorter."
            )
        return f"The {label.lower()} memory has been updated ({len(stored)} characters)."

    async def _tool_memory_append(self, arguments: dict, *, guild_id, channel_id, user_id) -> str:
        scope, scope_id, label, error = self._scope_ids(arguments.get("scope", ""), guild_id, channel_id, user_id)
        if error:
            return error
        content = arguments.get("content")
        if not isinstance(content, str) or not content.strip():
            return "Error: the content must be a non-empty string."
        current = await self.memory.read(scope, scope_id)
        combined = f"{current}\n{content}" if current else content
        stored = await self.memory.store(scope, scope_id, combined)
        if len(stored) < len(combined):
            dropped = len(combined) - len(stored)
            return (
                f"Warning: the {label.lower()} memory is full. Only part of the content was added: "
                f"the last {dropped} characters were dropped (Config storage limit). "
                "Read the scope and rewrite it shorter."
            )
        return f"The {label.lower()} memory now holds {len(stored)} characters."

    async def _tool_web_search(self, arguments: dict, **_scope) -> str:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return "Error: the query must be a non-empty string."
        try:
            async with self.session_getter().get(
                WEB_SEARCH_URL + quote_plus(query)
                , headers={"User-Agent": BROWSER_USER_AGENT}
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
