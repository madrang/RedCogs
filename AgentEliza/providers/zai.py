import base64
import logging
from urllib.parse import urlparse

from ..tools import TOOL_RESULT_MAX_CHARS
from ..tools.base import DISCORD_FILE_HOSTS
from .base import Provider

log = logging.getLogger("red.agenteliza.providers")

# The documented vision limits: png and jpg only (no webp), 5 MB per image,
# 6000x6000 pixels.
ZAI_IMAGE_MAX_BYTES = 5_000_000
ZAI_IMAGE_TYPES = ("image/png", "image/jpeg")


def _inline_part(body: bytes, content_type: str):
    """The inline image part, or an error text when the vision model cannot read the image."""
    if content_type not in ZAI_IMAGE_TYPES:
        return None, f"Error: the image type {content_type!r} is not supported. The vision model reads png and jpg only."
    if len(body) > ZAI_IMAGE_MAX_BYTES:
        return None, f"Error: the image is over the 5 MB limit of the vision model ({len(body)} bytes)."
    inline = base64.b64encode(body).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{inline}"}}, None


async def _analyze_image(arguments: dict, call_api, fetch_url=None) -> str:
    """The analyze_image handler: one GLM-4.6V chat call with an image URL."""
    url = str(arguments.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "Error: the url must be the http(s) URL of an image."
    question = str(arguments.get("question") or "").strip() or "Describe this image."
    image_part = {"type": "image_url", "image_url": {"url": url}}
    if fetch_url is not None and urlparse(url).netloc.lower() in DISCORD_FILE_HOSTS:
        fetched = await fetch_url(url)
        if fetched is None:
            return "Error: the download of the Discord file failed."
        image_part, error = _inline_part(*fetched)
        if error:
            return error
    payload = {
        "model": "glm-4.6v"
        , "messages": [{
            "role": "user"
            , "content": [image_part, {"type": "text", "text": question}]
        }]
        , "stream": False
    }
    # Log the exact shape sent: a live 1210 means the wire differs from the probe.
    sent = image_part["image_url"]["url"]
    log.info("analyze_image sends an image of %d chars, prefix %r", len(sent), sent[:80])
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


class ZaiProvider(Provider):
    """Z.AI quota monitor (undocumented, reverse-engineered; may change)."""

    usage_url = "https://api.z.ai/api/monitor/usage/quota/limit"
    # Reported by the owner: glm-5.3 and glm-5.2 run a 1M-token context.
    context_lengths = {
        "glm-5.3": 1_048_576
      , "glm-5.2": 1_048_576
    }

    def native_tools(self) -> list:
        """The vision tool: image analysis through GLM-4.6V, only with a Z.AI provider."""
        return [{
            "name": "analyze_image"
            , "description": (
                "Analyze an image at an http(s) URL: describe it, read its text, or answer "
                "a question about it. The attachments of a message carry usable URLs."
            )
            , "parameters": {
                "type": "object"
                , "properties": {
                    "url": {"type": "string", "description": "The http(s) URL of the image."}
                    , "question": {"type": "string", "description": "What to answer about the image. Default: describe it."}
                }
                , "required": ["url"]
            }
            , "handler": _analyze_image
        }]

    def mcp_servers(self, api_key: str) -> dict:
        """The Z.AI search and reader MCP servers. Search takes the place of the
        harness web_search. The reader joins as zai-reader__webReader: the
        harness web_fetch stays, it reads the Discord file hosts with the bot
        token and the reader cannot."""
        if not api_key:
            return {}
        return {
            "zai-search": {
                "transport": "http"
                , "url": "https://api.z.ai/api/mcp/web_search_prime/mcp"
                , "headers": {"Authorization": f"Bearer {api_key}"}
                , "replaces": {"web_search": "webSearchPrime"}
            }
            , "zai-reader": {
                "transport": "http"
                , "url": "https://api.z.ai/api/mcp/web_reader/mcp"
                , "headers": {"Authorization": f"Bearer {api_key}"}
            }
        }

    def parse_usage(self, data: dict) -> list:
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        rows = []
        for entry in payload.get("limits") or []:
            percent = entry.get("percentage")
            reset_ms = entry.get("nextResetTime")
            rows.append({
                "name": self._window_name(entry),
                "used": self._num(entry.get("currentValue")),
                "limit": self._num(entry.get("usage")),
                "percent": float(percent) if percent is not None else None,
                "reset": self._num(reset_ms / 1000) if isinstance(reset_ms, (int, float)) else None,
                "text": None,
                "exhausted": False,
            })
        return self._fill_percent(rows)

    @staticmethod
    def _window_name(entry: dict) -> str:
        """Name plus window size, from the unit codes: 3 = hours, 5 = monthly, 6 = weekly."""
        name = entry.get("type") or "limit"
        unit = {3: "h", 5: "mo", 6: "wk"}.get(entry.get("unit"))
        number = entry.get("number")
        if unit and number:
            return f"{name} ({number}{unit})"
        return name


class ZaiCodeProvider(ZaiProvider):
    """Z.AI coding subscription endpoint."""

    name = "Z.AI Code"
    base_url = "https://api.z.ai/api/coding/paas/v4"
    models = [
        "glm-5.3"
      , "glm-5.2"
    ]


class ZaiApiProvider(ZaiProvider):
    """Z.AI open platform endpoint."""

    name = "Z.AI API"
    base_url = "https://api.z.ai/api/paas/v4"
    models = [
        "glm-5.3"
      , "glm-5.2"
    ]
