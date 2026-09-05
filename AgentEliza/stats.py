import time

RATE_WINDOW_SECONDS = 3600
# The cost history keeps this many month keys, the current one included.
COST_MONTHS_KEPT = 13

# Default counters registered at each scope.
_STATS_DEFAULT = {
      "messages": 0
    , "prompt_tokens": 0
    , "completion_tokens": 0
    , "cached_tokens": 0
    , "cost": 0.0
    , "cost_months": {}
    , "tool_calls": 0
    , "images": 0
    , "inpaints": 0
    , "music": 0
}
_RATE_DEFAULT = {
      "count": 0
    , "start": 0
}


def month_key() -> str:
    """The UTC month key of the moment, for example 2026-09."""
    return time.strftime("%Y-%m", time.gmtime())

class ScopeStats:
    """Per-scope counters of the AgentEliza cog, stored in Red Config.

    Each scope (guild, channel, user) holds two dicts. `stats` accumulates
    the amounts exchanged with the chat API: one message per interaction,
    plus the prompt, completion, and cached token counts summed over the
    tool-call rounds of the interaction. `rate` is the rolling window of
    the interaction limit: a count and the window start time.
    """

    # Scope name -> Config accessor. Same dispatch as the Memory class.
    SCOPES = {
        "guild": "guild_from_id"
      , "channel": "channel_from_id"
      , "user": "user_from_id"
    }

    def __init__(self, config):
        self.config = config
        for register in (config.register_guild, config.register_channel, config.register_user):
            register(
                stats=dict(_STATS_DEFAULT)
                , rate=dict(_RATE_DEFAULT)
            )

    def _group(self, scope: str, scope_id: int):
        """The Config group of one scope instance."""
        return getattr(self.config, self.SCOPES[scope])(scope_id)

    async def record(self, *, guild_id, channel_id, user_id, usage: dict) -> None:
        """Add one interaction and its token usage to every scope with an id.

        usage holds the accumulated totals of one generate_reply call:
        prompt_tokens, completion_tokens, cached_tokens, and cost — the
        generic cost metric of the provider (its currency or its compute
        units; 0 when the provider names none). The cost lands in the scope
        total and in the bucket of the current UTC month. The integer
        counters ride along: tool_calls (every executed call of the
        interaction) and the media counters a native tool declares
        (images, inpaints, music — a generation counts only when it
        succeeds).
        """
        ids = {"guild": guild_id, "channel": channel_id, "user": user_id}
        cost = float(usage.get("cost") or 0)
        for scope, scope_id in ids.items():
            if scope_id is None:
                continue
            group = self._group(scope, scope_id)
            async with group.stats() as stats:
                stats["messages"] = stats.get("messages", 0) + 1
                for key in (
                    "prompt_tokens", "completion_tokens", "cached_tokens"
                  , "tool_calls", "images", "inpaints", "music"
                ):
                    stats[key] = stats.get(key, 0) + (usage.get(key) or 0)
                if cost:
                    stats["cost"] = stats.get("cost", 0.0) + cost
                    months = dict(stats.get("cost_months") or {})
                    month = month_key()
                    months[month] = months.get(month, 0.0) + cost
                    if len(months) > COST_MONTHS_KEPT:
                        for old in sorted(months)[: len(months) - COST_MONTHS_KEPT]:
                            months.pop(old)
                    stats["cost_months"] = months

    async def check_and_count(self, *, guild_id, channel_id, user_id, limits: dict) -> str | None:
        """Count one interaction against the per-scope limits.

        limits maps scope name to the allowed interactions per window, 0
        for unlimited. Returns a refusal text when a scope is over its
        limit, else increments every scope counter and returns None.
        The check runs before the count, so a refused interaction does
        not consume the allowance of the other scopes.
        """
        ids = {"user": user_id, "channel": channel_id, "guild": guild_id}
        now = int(time.time())
        limited = {
            scope: scope_id
            for scope, scope_id in ids.items()
            if scope_id is not None and limits.get(scope)
        }
        for scope, scope_id in limited.items():
            group = self._group(scope, scope_id)
            async with group.rate() as rate:
                if now - rate.get("start", 0) >= RATE_WINDOW_SECONDS:
                    rate["start"] = now
                    rate["count"] = 0
        for scope, scope_id in limited.items():
            group = self._group(scope, scope_id)
            rate = await group.rate()
            if rate.get("count", 0) >= limits[scope]:
                reset = rate.get("start", now) + RATE_WINDOW_SECONDS
                label = "server" if scope == "guild" else scope
                return (
                    f"The {label} interaction limit is reached ({limits[scope]} per hour). "
                    f"Try again <t:{reset}:R>."
                )
        for scope, scope_id in limited.items():
            group = self._group(scope, scope_id)
            async with group.rate() as rate:
                rate["count"] = rate.get("count", 0) + 1
        return None

    async def get(self, scope: str, scope_id: int) -> dict:
        """The stats dict of one scope instance."""
        return await self._group(scope, scope_id).stats()

    async def top_costs(self, count: int = 5) -> list[tuple[str, int, float]]:
        """The scopes with the highest known cost total, guild and user
        scopes, in descending order. Each entry is (scope, scope_id,
        cost). A scope with no cost stays off the list."""
        entries = []
        for scope, all_data in (
            ("guild", await self.config.all_guilds())
          , ("user", await self.config.all_users())
        ):
            for scope_id, data in (all_data or {}).items():
                cost = float(((data or {}).get("stats") or {}).get("cost") or 0)
                if cost > 0:
                    entries.append((scope, int(scope_id), cost))
        entries.sort(key=lambda entry: entry[2], reverse=True)
        return entries[:count]

    async def rate(self, scope: str, scope_id: int) -> dict:
        """The rate window dict of one scope instance."""
        return await self._group(scope, scope_id).rate()
