from .base import Provider, analyze_image_tool

class KimiApiProvider(Provider):
    """Kimi open platform (pay-as-you-go, direct path)."""

    name = "Kimi API"
    base_url = "https://api.moonshot.ai/v1"
    models = ["kimi-k2.6", "kimi-k2.5", "kimi-k3", "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "moonshot-v1-auto"]
    # moonshot-v1-auto routes across the legacy 8k/32k/128k set: 128k worst case.
    context_lengths = {"moonshot-v1-auto": 131_072}
    # Documented balance endpoint.
    usage_url = "https://api.moonshot.ai/v1/users/me/balance"
    # The vision model of analyze_image on this provider: k3 has image_in.
    vision_model = "kimi-k3"
    # The chat models that accept image parts in the chat contract (the
    # same wire as analyze_image): the engine feeds conversation images
    # to a resolved model in this set.
    vision_models = {"kimi-k3"}

    def native_tools(self) -> list:
        """The vision tool: image analysis, only with a Kimi provider."""
        return [analyze_image_tool(self.vision_model)]

    def extra_payload(self, session_id: int, model: str = "", nsfw: bool = False) -> dict:
        # Kimi-specific field: enables context caching per session.
        return {"prompt_cache_key": str(session_id)}

    def parse_usage(self, data: dict) -> list:
        balance = data.get("data") or {}
        available = balance.get("available_balance")
        return [{
            "name": "Balance",
            "used": None,
            "limit": None,
            "percent": None,
            "reset": None,
            "text": (
                f"available ${available}, voucher ${balance.get('voucher_balance')}, "
                f"cash ${balance.get('cash_balance')}"
            ),
            "exhausted": available is not None and available <= 0,
        }]
