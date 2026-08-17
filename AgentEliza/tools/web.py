"""The web tools: DuckDuckGo search and page fetch to text."""

import asyncio
import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, urlparse

import aiohttp

from .base import DISCORD_FILE_HOSTS, _cap

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


class WebTools:
    """The web_search and web_fetch tools."""

    def web_tools(self) -> list:
        """The OpenAI function schemas of the web tools."""
        return [
            {
                "type": "function"
                , "function": {
                    "name": "web_search"
                    , "description": (
                        "Search the web with DuckDuckGo. The result is a numbered list. "
                        "Each entry has a title, a URL, and a snippet. "
                        "Use web_fetch on a result URL to read the full page."
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
                        "Fetch one web page and return its readable text. "
                        "The tool strips an HTML page to its text and truncates a long page."
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
        headers = {}
        if urlparse(url).netloc.lower() in DISCORD_FILE_HOSTS:
            # The Discord file hosts need an authorized request. The bot
            # token goes to these hosts only, never to a foreign URL.
            token = self.bot_token_getter() if self.bot_token_getter else None
            if token:
                headers["Authorization"] = f"Bot {token}"
        try:
            async with self.session_getter().get(url, headers=headers, timeout=WEB_TIMEOUT) as response:
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
