"""The Venice credit calendar: the refill walk and the guild song gate."""

from datetime import datetime, timezone

import pytest

from AgentEliza.providers.venice import _next_month, next_refill, song_credit_gate


def _moment(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # The day clamps to the end of the shorter month.
        (_moment(2026, 1, 31), _moment(2026, 2, 28)),
        # The clamp sticks: the day stays 28 after February.
        (_moment(2026, 2, 28), _moment(2026, 3, 28)),
        (_moment(2026, 12, 31), _moment(2027, 1, 31)),
        # A leap February keeps its 29th day.
        (_moment(2028, 1, 31), _moment(2028, 2, 29)),
        # The time of day survives.
        (_moment(2026, 9, 3, 13, 13, 35), _moment(2026, 10, 3, 13, 13, 35)),
    ],
)
def test_next_month(raw: datetime, expected: datetime) -> None:
    assert _next_month(raw) == expected


def test_next_refill_walks_to_the_first_month_past_the_moment() -> None:
    seed = _moment(2026, 9, 3)
    assert next_refill(seed.timestamp(), _moment(2026, 9, 10)) == _moment(2026, 10, 3)
    # The refill moment itself belongs to the new cycle.
    assert next_refill(seed.timestamp(), _moment(2026, 10, 3)) == _moment(2026, 11, 3)


def test_song_credit_gate_paces_the_cycle_rest() -> None:
    seed = _moment(2026, 9, 3)
    # Half of the 31-day October cycle is left: the floor is 1.5 x 22500 x 0.5.
    half = _moment(2026, 10, 18, 12)
    floor = 1.5 * 22500 * 0.5
    assert song_credit_gate(seed.timestamp(), floor - 1, half) is True
    assert song_credit_gate(seed.timestamp(), floor, half) is False


def test_song_credit_gate_clamps_the_fraction_at_one_cycle() -> None:
    seed = _moment(2026, 9, 3)
    # A moment before the seed anchor: the fraction clamps at one full cycle.
    before = _moment(2026, 9, 2)
    assert song_credit_gate(seed.timestamp(), 1.5 * 22500 - 1, before) is True
    assert song_credit_gate(seed.timestamp(), 1.5 * 22500, before) is False
