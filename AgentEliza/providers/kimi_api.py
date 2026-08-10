from .base import Provider

class KimiApiProvider(Provider):
    """Kimi open platform (pay-as-you-go, direct path)."""

    name = "Kimi API"
    base_url = "https://api.moonshot.ai/v1"
    models = ["kimi-k2.6", "kimi-k2.5", "kimi-k3", "kimi-k2.7-code", "kimi-k2.7-code-highspeed", "moonshot-v1-auto"]
    # Documented balance endpoint.
    usage_url = "https://api.moonshot.ai/v1/users/me/balance"

    def extra_payload(self, session_id: int) -> dict:
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
