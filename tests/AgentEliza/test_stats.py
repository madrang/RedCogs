"""The media windows of the credit ratio: the bands and the lerped slide,
and the cost writes of the scope stats."""

import pytest

from AgentEliza.stats import (
    MEDIA_RATE_CHANNEL, MEDIA_RATE_USER, Scope, ScopeStats, cost_ceiling, media_windows, month_key,
)
from tests.AgentEliza.fakes import FakeStatsConfig


def test_the_full_windows_hold_at_and_above_the_top_band() -> None:
    full = {"user": MEDIA_RATE_USER, "channel": MEDIA_RATE_CHANNEL}
    # An unread ratio never blocks: the full windows hold.
    assert media_windows(None) == full
    assert media_windows(1.25) == full
    assert media_windows(1.9) == full


def test_the_windows_slide_linearly_to_one_at_the_floor_band() -> None:
    # Two checkpoints of the slide: a fifth and four fifths of the band.
    assert media_windows(0.97) == {"user": 2, "channel": 7}
    assert media_windows(1.18) == {"user": 7, "channel": 26}
    # The floor of the band reads one generation on both windows.
    assert media_windows(0.9) == {"user": 1, "channel": 1}
    assert media_windows(0.91) == {"user": 1, "channel": 2}


def test_one_generation_per_hour_holds_above_the_disable_band() -> None:
    assert media_windows(0.89) == {"user": 1, "channel": 1}
    assert media_windows(0.75) == {"user": 1, "channel": 1}


def test_the_disable_band_closes_the_media_tools() -> None:
    assert media_windows(0.74) is None
    assert media_windows(0.0) is None


def test_the_cost_ceiling_slides_with_the_same_bands() -> None:
    # No ceiling on a healthy balance or an unread ratio: the full catalog.
    assert cost_ceiling(None, 0.01, 0.29) is None
    assert cost_ceiling(1.25, 0.01, 0.29) is None
    # The slide covers the price range between the bands.
    assert cost_ceiling(1.075, 0.01, 0.29) == pytest.approx(0.01 + 0.5 * (0.29 - 0.01))
    # Under the floor band the cheapest entry alone holds.
    assert cost_ceiling(0.9, 0.01, 0.29) == 0.01
    assert cost_ceiling(0.75, 0.01, 0.29) == 0.01


async def test_record_cost_adds_the_price_without_an_interaction() -> None:
    stats = ScopeStats(FakeStatsConfig())
    await stats.record_cost(Scope(channel_id=5, guild_id=9, user_id=7), 0.03)
    for key in ("guild:9", "channel:5", "user:7"):
        scope = stats.config.groups[key].stores["stats"]
        assert scope["cost"] == pytest.approx(0.03)
        assert scope["cost_months"] == {month_key(): pytest.approx(0.03)}
        assert scope["messages"] == 0


async def test_record_buckets_the_cost_of_an_interaction() -> None:
    stats = ScopeStats(FakeStatsConfig())
    await stats.record(
        Scope(channel_id=5, guild_id=9, user_id=7)
        , {"cost": 0.29, "images": 1, "tool_calls": 2}
    )
    scope = stats.config.groups["guild:9"].stores["stats"]
    assert scope["messages"] == 1
    assert scope["images"] == 1
    assert scope["tool_calls"] == 2
    assert scope["cost"] == pytest.approx(0.29)
    assert scope["cost_months"] == {month_key(): pytest.approx(0.29)}
