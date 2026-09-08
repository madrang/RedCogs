import asyncio
import base64
import hashlib
import logging
from urllib.parse import urlparse

import aiohttp

from ..tools import TOOL_RESULT_MAX_CHARS
from ..tools.base import DISCORD_FILE_HOSTS

log = logging.getLogger("red.agenteliza.providers")


async def analyze_image_impl(arguments: dict, call_api, fetch_url, *, model: str, validate=None, vision_chat: bool = False, show_image=None) -> str:
    """The shared analyze_image flow. A conversation model with vision
    takes the image itself: show_image hands its bytes to the engine,
    which joins them as image parts of the next request — no second model
    answers between. Any other conversation gets one vision chat call
    with the image URL, through the provider's vision model.

    validate(body, content_type) returns an error text, or None to accept.
    """
    url = str(arguments.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "Error: the url must be the http(s) URL of an image."
    log.info("analyze_image called with url %r", url[:150])
    question = str(arguments.get("question") or "").strip() or "Describe this image."
    # The bytes download when a flow needs them: a Discord host always
    # (the token fetch feeds the vision call), any host when the
    # conversation takes the image itself (the harness note rides a data
    # URI, so the bytes must exist).
    discord_host = urlparse(url).netloc.lower() in DISCORD_FILE_HOSTS
    body = None
    content_type = None
    if fetch_url is not None and (discord_host or vision_chat):
        fetched = await fetch_url(url)
        if fetched is not None:
            body, content_type = fetched
            if validate is not None:
                error = validate(body, content_type)
                if error:
                    return error
        elif discord_host:
            return "Error: the download of the Discord file failed."
    if vision_chat and show_image is not None and body is not None:
        name = url.rsplit("/", 1)[-1].split("?")[0] or "image"
        if await show_image(name, body, content_type):
            return (
                "The image joined this conversation: a harness note carries it "
                "as an image part beside this result. Answer from the image itself."
            )
    image_part = {"type": "image_url", "image_url": {"url": url}}
    if body is not None:
        image_part["image_url"] = {"url": f"data:{content_type};base64,{base64.b64encode(body).decode('ascii')}"}
    payload = {
        "model": model
        , "messages": [{
            "role": "user"
            , "content": [image_part, {"type": "text", "text": question}]
        }]
        , "stream": False
    }
    # Log the exact shape sent: a live provider error means the wire differs.
    sent = image_part["image_url"]["url"]
    log.info("analyze_image sends an image of %d chars, sha256 %s, prefix %r", len(sent), hashlib.sha256(sent.encode()).hexdigest()[:16], sent[:80])
    data = await call_api(payload)
    choices = data.get("choices") or []
    content = (choices[0].get("message") or {}).get("content") if choices else None
    if isinstance(content, list):
        content = "".join(part.get("text") or "" for part in content if isinstance(part, dict))
    if not content:
        return "Error: the vision model returned no answer."
    if len(content) > TOOL_RESULT_MAX_CHARS:
        dropped = len(content) - TOOL_RESULT_MAX_CHARS
        content = content[:TOOL_RESULT_MAX_CHARS] + f"\n[truncated: {dropped} characters dropped]"
    return content


def analyze_image_tool(model: str, validate=None) -> dict:
    """The analyze_image native tool entry: the schema and the handler."""

    async def handler(arguments, engine):
        return await analyze_image_impl(
            arguments, engine.call_api, engine.fetch_url
            , model=model, validate=validate
            , vision_chat=engine.vision_chat, show_image=engine.show_image
        )

    return {
        "name": "analyze_image"
        , "description": (
            "Analyze an image at an http(s) URL: describe it, read its text, or answer "
            "a question about it. The attachments of a message carry usable URLs. "
            "A conversation model with vision sees the image itself."
        )
        , "parameters": {
            "type": "object"
            , "properties": {
                "url": {"type": "string", "description": "The http(s) URL of the image."}
                , "question": {"type": "string", "default": "Describe this image.", "description": "What to answer about the image."}
            }
            , "required": ["url"]
        }
        , "handler": handler
    }


class Provider:
    """Base class for one chat provider.

    Subclasses set the preset data (name, base_url, models) and override
    the custom behavior: payload extras and usage endpoint parsing.
    A handler normalizes its usage answer into rows of
    {name, used, limit, percent, reset, text, exhausted}.
    """

    name = ""
    base_url = ""
    models = []
    usage_url: str | None = None
    # Documented prompt-cache lifetime in seconds. None: undocumented, the
    # harness assumes DEFAULT_CACHE_TTL from history.py instead.
    cache_ttl: int | None = None
    # Context size in tokens per model. A model without an entry is unknown:
    # the harness falls back to its default budget (CONTEXT_TOKENS in
    # history.py, of which the history uses half).
    context_lengths: dict = {}
    # Model ids that accept image input through the chat contract. The
    # `eliza providers` command marks them.
    vision_models: set = set()

    def context_length(self, model_name: str) -> int | None:
        """The context size of a model in tokens, None when unknown."""
        return self.context_lengths.get(model_name)

    def preset_name(self, model_id: str) -> str | None:
        """The short display name of a chat model id, None to keep the id."""
        return None

    def cost_of(self, data: dict) -> float:
        """The cost of one chat-completions answer, as the provider's own
        generic metric (its currency, or its compute units). 0 when the
        provider names no cost."""
        return 0.0

    def default_model(self) -> str:
        """The model of a cleared configuration: the first entry of the
        list. A provider may return a preset name instead, so the default
        resolves per request like any preset."""
        return self.models[0]

    def preset_fallback(self, model: str) -> str | None:
        """The preset one step up in cost from the preset of the model, for
        an overload fallback. None when the provider ranks no presets or
        the model sits in none."""
        return None

    def resolve_model(self, name: str) -> str:
        """Map a display name to its model id, pass anything else through."""
        return name

    def preset_menu(self) -> list:
        """One line per named chat preset, empty when the provider has none."""
        return []

    def extra_payload(self, session_id: int, model: str = "", nsfw: bool = False) -> dict:
        """Extra fields for the chat completions payload, merged over the
        payload at every request. model carries the request's model string
        (the session override or the configured name): a provider that
        resolves display names to ids may return a "model" key here to
        rewrite it, picking a variant by nsfw (the conversation sits behind
        the 18+ gate). An empty model leaves the payload model alone — the
        native tool payloads set their own model."""
        return {}

    def native_tools(self) -> list:
        """Tool definitions the provider implements itself, live while it is active.

        Each entry: {"name", "description", "parameters", "handler"}. The
        handler is an async callable (arguments, engine) returning text.
        engine is the ToolContext of the reply (llm_chat): the engine
        builds one per reply, and a new capability is a field on it.
        call_api posts one chat-completions payload to the provider.
        fetch_url downloads one URL to (bytes, content_type), or None on
        failure. api_post sends one POST to a REST path of the provider,
        returns the JSON answer and the response headers (case-insensitive)
        as a tuple, and raises ChatError on failure. send_file posts one
        binary file to the current channel and returns the result text.
        channel_nsfw reports whether the current channel sits behind the
        Discord 18+ gate (a direct message of the bot owner counts).
        set_conversation_model overrides the chat model of the current
        conversation (None restores the configured model). vision_chat
        reports whether the conversation model accepts image parts in the
        chat contract. show_image hands one image to the conversation. It
        takes (name, bytes, content type). It answers True when the image
        joined (a harness note carries it in the next round). It answers
        False when the tool must answer another way. A native tool takes
        the place of a harness tool of the same name.
        """
        return []

    def mcp_servers(self, api_key: str) -> dict:
        """MCP server definitions of the provider, live only while the provider is active.

        The shape matches the Config `mcp_servers` entries, plus two
        optional keys: `headers` (HTTP headers of the connection) and
        `replaces` (a harness tool name -> the provider tool that takes
        its place).
        """
        return {}

    async def fetch_usage(self, session: aiohttp.ClientSession, api_key: str):
        """Query the usage endpoint. Return (rows, error_message)."""
        if self.usage_url is None:
            return None, "The current provider has no known usage endpoint."
        try:
            async with session.get(
                self.usage_url,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as response:
                try:
                    data = await response.json(content_type=None)
                except Exception:
                    data = None
                if response.status != 200 or not isinstance(data, dict):
                    return None, f"The usage endpoint returned an error (HTTP {response.status})."
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            return None, f"The connection to the usage endpoint failed: {e}"
        try:
            # A malformed answer degrades to a failed check: a provider parse
            # bug must never break the reply path.
            return self.parse_usage(data), None
        except Exception as e:
            log.warning("The usage answer of %s could not be parsed: %s: %s", self.name or self.usage_url, type(e).__name__, e)
            return None, "The usage endpoint returned an unreadable answer."

    def parse_usage(self, data: dict) -> list:
        raise NotImplementedError

    @staticmethod
    def _num(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None

    @classmethod
    def _fill_percent(cls, rows: list) -> list:
        for row in rows:
            if row["percent"] is None and row["used"] is not None and row["limit"]:
                row["percent"] = row["used"] / row["limit"] * 100
        return rows
