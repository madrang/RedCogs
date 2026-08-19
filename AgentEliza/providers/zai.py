from .base import Provider, analyze_image_tool

# The documented vision limits: png and jpg only (no webp), 5 MB per image,
# 6000x6000 pixels.
ZAI_IMAGE_MAX_BYTES = 5_000_000
ZAI_IMAGE_TYPES = ("image/png", "image/jpeg")
# The vision models known compatible with the image_url contract of
# analyze_image. Keep the list current when a vision model is added.
# glm-5v-turbo needs a plan with access (error 1311 on the coding plan),
# glm-4.6v is the coding-plan model.
ZAI_VISION_MODELS = ("glm-5v-turbo", "glm-4.6v")


def _validate_image(body: bytes, content_type: str):
    """The error text when the Z.AI vision model cannot read the image, else None."""
    if content_type not in ZAI_IMAGE_TYPES:
        return f"Error: the image type {content_type!r} is not supported. The vision model reads png and jpg only."
    if len(body) > ZAI_IMAGE_MAX_BYTES:
        return f"Error: the image is over the 5 MB limit of the vision model ({len(body)} bytes)."
    return None


class ZaiProvider(Provider):
    """Z.AI quota monitor (undocumented, reverse-engineered; may change)."""

    usage_url = "https://api.z.ai/api/monitor/usage/quota/limit"
    # Reported by the owner: glm-5.3 and glm-5.2 run a 1M-token context.
    context_lengths = {
        "glm-5.3": 1_048_576
      , "glm-5.2": 1_048_576
    }
    # The vision model of analyze_image on this provider.
    vision_model = "glm-4.6v"

    def native_tools(self) -> list:
        """The vision tool: image analysis, only with a Z.AI provider."""
        return [analyze_image_tool(self.vision_model, validate=_validate_image)]

    def mcp_servers(self, api_key: str) -> dict:
        """The Z.AI search, reader, and zread MCP servers. Search takes the place
        of the harness web_search. The reader joins as zai-reader__webReader:
        the harness web_fetch stays, it reads the Discord file hosts with the
        bot token and the reader cannot. Zread joins as zai-zread__ tools."""
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
            , "zai-zread": {
                "transport": "http"
                , "url": "https://api.z.ai/api/mcp/zread/mcp"
                , "headers": {"Authorization": f"Bearer {api_key}"}
            }
        }

    def parse_usage(self, data: dict) -> list:
        payload = data.get("data") if isinstance(data.get("data"), dict) else data
        rows = []
        for entry in payload.get("limits") or []:
            if not isinstance(entry, dict):
                continue
            percent = entry.get("percentage")
            reset_ms = entry.get("nextResetTime")
            rows.append({
                "name": self._window_name(entry),
                "used": self._num(entry.get("currentValue")),
                "limit": self._num(entry.get("usage")),
                "percent": self._float(percent),
                "reset": self._num(reset_ms / 1000) if isinstance(reset_ms, (int, float)) else None,
                "text": None,
                "exhausted": False,
            })
        return self._fill_percent(rows)

    @staticmethod
    def _float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
    vision_model = "glm-5v-turbo"
    models = [
        "glm-5.3"
      , "glm-5.2"
    ]
