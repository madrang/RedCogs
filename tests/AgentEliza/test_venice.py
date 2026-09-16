"""The Venice credit calendar: the cycle day and the guild song gate."""

from datetime import datetime, timezone

from AgentEliza.providers.venice import next_refill, song_credit_gate


def _moment(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def test_next_refill_walks_to_the_next_cycle_day() -> None:
    assert next_refill(3, _moment(2026, 9, 10)) == _moment(2026, 10, 3)
    # A moment past the midnight of the cycle day waits for the next month.
    assert next_refill(3, _moment(2026, 10, 3, 12)) == _moment(2026, 11, 3)
    # The midnight of the cycle day itself still lies ahead the day before.
    assert next_refill(3, _moment(2026, 10, 2, 23)) == _moment(2026, 10, 3)


def test_next_refill_clamps_the_shorter_months() -> None:
    assert next_refill(31, _moment(2026, 2, 1)) == _moment(2026, 2, 28)
    # A leap February keeps its 29th day.
    assert next_refill(31, _moment(2028, 2, 1)) == _moment(2028, 2, 29)


def test_song_credit_gate_paces_the_cycle_rest() -> None:
    # Half of the 31-day October cycle is left: the floor is 1.5 x 22500 x 0.5.
    half = _moment(2026, 10, 18, 12)
    floor = 1.5 * 22500 * 0.5
    assert song_credit_gate(3, floor - 1, half) is True
    assert song_credit_gate(3, floor, half) is False


def test_song_credit_gate_clamps_the_fraction_at_one_cycle() -> None:
    # The midnight of the refill itself: the whole cycle lies ahead, the floor is 1.5 allowances.
    fresh = _moment(2026, 10, 3)
    assert song_credit_gate(3, 1.5 * 22500 - 1, fresh) is True
    assert song_credit_gate(3, 1.5 * 22500, fresh) is False
