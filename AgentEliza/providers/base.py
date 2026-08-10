import asyncio
import aiohttp

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

    def extra_payload(self, session_id: int) -> dict:
        """Extra fields for the chat completions payload."""
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
        return self.parse_usage(data), None

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
