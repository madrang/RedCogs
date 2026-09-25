import time

RATE_WINDOW_SECONDS = 3600
# The cost history keeps this many month keys, the current one included.
COST_MONTHS_KEPT = 13
# The hourly allowance of media generations, images and inpaints summed
# (a music tool joins the same windows): per user, and per channel of a
# guild. A direct message knows only the user window.
MEDIA_RATE_USER = 8
MEDIA_RATE_CHANNEL = 32
# The credit ratio bands of the media windows: the full windows hold at and
# above MEDIA_LIMIT_AT, they slide linearly to one generation at
# MEDIA_LIMIT_FLOOR, the single hourly generation holds above
# MEDIA_DISABLE_AT, and under it the media tools stay off.
MEDIA_LIMIT_AT = 1.25
MEDIA_LIMIT_FLOOR = 0.9
MEDIA_DISABLE_AT = 0.75


def media_windows(ratio: float | None) -> dict | None:
    """The hourly media windows of a credit ratio: the full windows at and
    above the top band, a linear slide to one generation at the floor band,
    the single hourly generation above the disable band, and None under it.
    A None ratio keeps the full windows: an unread balance never blocks."""
    if ratio is None or ratio >= MEDIA_LIMIT_AT:
        return {"user": MEDIA_RATE_USER, "channel": MEDIA_RATE_CHANNEL}
    if ratio < MEDIA_DISABLE_AT:
        return None
    if ratio < MEDIA_LIMIT_FLOOR:
        return {"user": 1, "channel": 1}
    part = (ratio - MEDIA_LIMIT_FLOOR) / (MEDIA_LIMIT_AT - MEDIA_LIMIT_FLOOR)
    return {
        "user": 1 + round(part * (MEDIA_RATE_USER - 1))
      , "channel": 1 + round(part * (MEDIA_RATE_CHANNEL - 1))
    }


def cost_ceiling(ratio: float | None, low: float, high: float) -> float | None:
    """The price ceiling of a media catalog at a credit ratio: None (no
    ceiling) at and above the top band or on an unread ratio, a linear slide
    from high to low between the top band and the floor band, and low under
    it. The bands mirror media_windows: the same ratio loosens or tightens
    both."""
    if ratio is None or ratio >= MEDIA_LIMIT_AT:
        return None
    if ratio < MEDIA_LIMIT_FLOOR:
        return low
    part = (ratio - MEDIA_LIMIT_FLOOR) / (MEDIA_LIMIT_AT - MEDIA_LIMIT_FLOOR)
    return low + (high - low) * part


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


def _add_cost(stats: dict, cost: float) -> None:
    """Add one provider cost to a scope stats dict: the total and the
    bucket of the current UTC month, the oldest buckets beyond the kept
    count dropped."""
    stats["cost"] = stats.get("cost", 0.0) + cost
    months = dict(stats.get("cost_months") or {})
    month = month_key()
    months[month] = months.get(month, 0.0) + cost
    if len(months) > COST_MONTHS_KEPT:
        for old in sorted(months)[: len(months) - COST_MONTHS_KEPT]:
            months.pop(old)
    stats["cost_months"] = months


class Scope:
    """Where one interaction happened: a guild channel, or a direct
    message. guild_id None names a direct message, where no guild window
    applies. user_id None names an interaction without a user (a poll
    event). The stats windows read their scope ids from here."""

    def __init__(self, *, channel_id: int, guild_id: int | None = None, user_id: int | None = None):
        if not isinstance(channel_id, int):
            raise ValueError("The scope needs a channel id.")
        if guild_id is not None and not isinstance(guild_id, int):
            raise ValueError("The guild id must be an int or None.")
        if user_id is not None and not isinstance(user_id, int):
            raise ValueError("The user id must be an int or None.")
        self.channel_id = channel_id
        self.guild_id = guild_id
        self.user_id = user_id


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
              , media_rate=dict(_RATE_DEFAULT)
            )

    def _group(self, scope: str, scope_id: int):
        """The Config group of one scope instance."""
        return getattr(self.config, self.SCOPES[scope])(scope_id)

    async def record(self, scope: Scope, usage: dict) -> None:
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
        ids = {"guild": scope.guild_id, "channel": scope.channel_id, "user": scope.user_id}
        cost = float(usage.get("cost") or 0)
        for name, scope_id in ids.items():
            if scope_id is None:
                continue
            group = self._group(name, scope_id)
            async with group.stats() as stats:
                stats["messages"] = stats.get("messages", 0) + 1
                for key in (
                    "prompt_tokens", "completion_tokens", "cached_tokens"
                  , "tool_calls", "images", "inpaints", "music"
                ):
                    stats[key] = stats.get(key, 0) + (usage.get(key) or 0)
                if cost:
                    _add_cost(stats, cost)

    async def record_cost(self, scope: Scope, cost: float) -> None:
        """Add one provider cost without an interaction: a paid media
        generation that ran outside a reply (an approved song). The cost
        lands in the scope totals and the UTC month buckets, no counter
        moves."""
        ids = {"guild": scope.guild_id, "channel": scope.channel_id, "user": scope.user_id}
        for name, scope_id in ids.items():
            if scope_id is None:
                continue
            group = self._group(name, scope_id)
            async with group.stats() as stats:
                _add_cost(stats, float(cost or 0))

    async def check_and_count(self, scope: Scope, limits: dict) -> str | None:
        """Count one interaction against the per-scope limits.

        limits maps scope name to the allowed interactions per window, 0
        for unlimited. Returns a refusal text when a scope is over its
        limit, else increments every scope counter and returns None.
        The check runs before the count, so a refused interaction does
        not consume the allowance of the other scopes.
        """
        ids = {"user": scope.user_id, "channel": scope.channel_id, "guild": scope.guild_id}
        now = int(time.time())
        limited = {
            name: scope_id
            for name, scope_id in ids.items()
            if scope_id is not None and limits.get(name)
        }
        for name, scope_id in limited.items():
            group = self._group(name, scope_id)
            async with group.rate() as rate:
                if now - rate.get("start", 0) >= RATE_WINDOW_SECONDS:
                    rate["start"] = now
                    rate["count"] = 0
        for name, scope_id in limited.items():
            group = self._group(name, scope_id)
            rate = await group.rate()
            if rate.get("count", 0) >= limits[name]:
                reset = rate.get("start", now) + RATE_WINDOW_SECONDS
                label = "server" if name == "guild" else name
                return (
                    f"The {label} interaction limit is reached ({limits[name]} per hour). "
                    f"Try again <t:{reset}:R>."
                )
        for name, scope_id in limited.items():
            group = self._group(name, scope_id)
            async with group.rate() as rate:
                rate["count"] = rate.get("count", 0) + 1
        return None

    async def get(self, scope: str, scope_id: int) -> dict:
        """The stats dict of one scope instance."""
        return await self._group(scope, scope_id).stats()

    async def media_refusal(self, scope: Scope, limits: dict | None = None) -> str | None:
        """The refusal text when one more media generation would pass an
        hourly window, else None. limits names the windows: None keeps the
        full constants, an empty dict marks the media tools off (the bundled
        credits run low). The user window counts everywhere; the channel
        window (MEDIA_RATE_CHANNEL) counts in a guild only. Images and
        inpaints share the windows; the count rises on a successful
        generation only, so the text starts with the uniform error prefix
        and never counts as one."""
        if limits is not None and not limits:
            return "Error: the bundled credits run low, the media tools stay off for now."
        if limits is None:
            limits = {"user": MEDIA_RATE_USER, "channel": MEDIA_RATE_CHANNEL}
        ids = {"user": scope.user_id}
        if scope.guild_id is not None:
            ids["channel"] = scope.channel_id
        limits = {"user": MEDIA_RATE_USER, "channel": MEDIA_RATE_CHANNEL}
        now = int(time.time())
        live = {
            name: scope_id
            for name, scope_id in ids.items()
            if scope_id is not None and limits.get(name)
        }
        for name, scope_id in live.items():
            group = self._group(name, scope_id)
            async with group.media_rate() as rate:
                if now - rate.get("start", 0) >= RATE_WINDOW_SECONDS:
                    rate["start"] = now
                    rate["count"] = 0
        for name, scope_id in live.items():
            group = self._group(name, scope_id)
            rate = await group.media_rate()
            if rate.get("count", 0) >= limits[name]:
                reset = rate.get("start", now) + RATE_WINDOW_SECONDS
                return (
                    f"Error: the {name} generation limit is reached ({limits[name]} per hour). "
                    f"Try again <t:{reset}:R>."
                )
        return None

    async def count_media(self, scope: Scope) -> None:
        """Count one successful media generation into the hourly windows:
        the user window everywhere, the channel window in a guild."""
        ids = {"user": scope.user_id}
        if scope.guild_id is not None:
            ids["channel"] = scope.channel_id
        now = int(time.time())
        for name, scope_id in ids.items():
            if scope_id is None:
                continue
            group = self._group(name, scope_id)
            async with group.media_rate() as rate:
                if now - rate.get("start", 0) >= RATE_WINDOW_SECONDS:
                    rate["start"] = now
                    rate["count"] = 0
                rate["count"] = rate.get("count", 0) + 1

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
