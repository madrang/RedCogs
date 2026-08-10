from .base import Provider

class ZaiProvider(Provider):
    """Z.AI quota monitor (undocumented, reverse-engineered; may change)."""

    usage_url = "https://api.z.ai/api/monitor/usage/quota/limit"

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
    models = ["glm-5.2"]


class ZaiApiProvider(ZaiProvider):
    """Z.AI open platform endpoint."""

    name = "Z.AI API"
    base_url = "https://api.z.ai/api/paas/v4"
    models = ["glm-5.2"]
