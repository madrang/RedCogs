from .base import Provider

class KimiCodeProvider(Provider):
    """Kimi Code subscription endpoint (managed path)."""

    name = "Kimi Code"
    base_url = "https://api.kimi.com/coding/v1"
    models = ["k3-256k", "k3", "kimi-for-coding", "kimi-for-coding-highspeed"]
    # Undocumented endpoint, used by the official CLI.
    usage_url = "https://api.kimi.com/coding/v1/usages"

    def extra_payload(self, session_id: int) -> dict:
        # Kimi-specific field: enables context caching per session.
        return {"prompt_cache_key": str(session_id)}

    def parse_usage(self, data: dict) -> list:
        rows = []
        usage = data.get("usage")
        if isinstance(usage, dict) and usage:
            rows.append({
                "name": usage.get("name") or "Usage",
                "used": self._num(usage.get("used")),
                "limit": self._num(usage.get("limit")),
                "percent": None,
                "reset": usage.get("resetTime"),
                "text": None,
                "exhausted": False,
            })
        for entry in data.get("limits") or []:
            if not isinstance(entry, dict):
                continue
            detail = entry.get("detail") if isinstance(entry.get("detail"), dict) else entry
            rows.append({
                "name": entry.get("name") or detail.get("name") or "Limit",
                "used": self._num(detail.get("used")),
                "limit": self._num(detail.get("limit")),
                "percent": None,
                "reset": detail.get("resetTime"),
                "text": None,
                "exhausted": False,
            })
        entries = data.get("data")
        if isinstance(entries, list):
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                used = self._num(entry.get("used"))
                limit = self._num(entry.get("limit"))
                if limit is None and used is not None:
                    remaining = self._num(entry.get("remaining"))
                    if remaining is not None:
                        limit = used + remaining
                rows.append({
                    "name": entry.get("name") or entry.get("model_name") or "Limit",
                    "used": used,
                    "limit": limit,
                    "percent": None,
                    "reset": entry.get("resetTime"),
                    "text": None,
                    "exhausted": False,
                })
        return self._fill_percent(rows)
