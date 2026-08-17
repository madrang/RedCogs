from ..tools import TOOL_RESULT_MAX_CHARS
from .base import Provider


async def _analyze_image(arguments: dict, call_api) -> str:
    """The analyze_image handler: one GLM-4.6V chat call with an image URL."""
    url = str(arguments.get("url") or "").strip()
    if not url.startswith(("http://", "https://")):
        return "Error: the url must be the http(s) URL of an image."
    question = str(arguments.get("question") or "").strip() or "Describe this image."
    data = await call_api({
        "model": "glm-4.6v"
        , "messages": [{
            "role": "user"
            , "content": [
                {"type": "image_url", "image_url": {"url": url}}
                , {"type": "text", "text": question}
            ]
        }]
        , "stream": False
    })
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
        """The Z.AI search and reader MCP servers, in place of the harness web tools."""
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
                , "replaces": {"web_fetch": "webReader"}
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
