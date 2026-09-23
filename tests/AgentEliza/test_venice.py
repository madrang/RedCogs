"""The Venice credit calendar: the cycle day, the credit gate, and the routing floor."""

from datetime import datetime, timezone

import pytest

from AgentEliza.providers.venice import credit_ratio, next_refill, routing_tier


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


def test_credit_ratio_reads_the_paced_cycle_rest() -> None:
    fresh = _moment(2026, 10, 3)
    # The full buffer right after a refill reads 1.5, the allowance itself 1.0.
    assert credit_ratio(3, 33750, fresh) == 1.5
    assert credit_ratio(3, 22500, fresh) == 1.0
    assert credit_ratio(3, 11250, fresh) == 0.5
    # Half of the cycle is left: the paced rest reads half the allowance, the
    # surplus above it adds its own share.
    assert credit_ratio(3, 22500, _moment(2026, 10, 18, 12)) == 1.5
    assert credit_ratio(3, 16875, _moment(2026, 10, 18, 12)) == 1.25


def test_credit_ratio_counts_the_surplus_above_the_cycle_rest() -> None:
    # The refill day: the allowance reads 1.0, the buffer allowance and a
    # half reads 1.5, and the fresh double reads 2.
    fresh = _moment(2026, 10, 3)
    assert credit_ratio(3, 22500, fresh) == 1.0
    assert credit_ratio(3, 33750, fresh) == 1.5
    assert credit_ratio(3, 45000, fresh) == 2.0
    # The allowance held unused to the eve of the refill: the day of rest
    # left is covered, the surplus reads just under one allowance above 1.
    eve = _moment(2026, 11, 2)
    assert 1.9 < credit_ratio(3, 22500, eve) < 2.0
    # Half the surplus of the first day, held to the last minute of the
    # cycle, still reads 1.5: the minutes of rest left cost almost nothing.
    last_minute = _moment(2026, 11, 2, 23, 59)
    assert credit_ratio(3, 11250, last_minute) == pytest.approx(1.5, abs=0.001)
    # At half the cycle the paced rest is half the allowance: the balance
    # that covers it reads exactly 1.0, the one under it slides under.
    half = _moment(2026, 10, 18, 12)
    assert credit_ratio(3, 11250, half) == 1.0
    assert credit_ratio(3, 11249, half) == pytest.approx(11249 / 11250)


def test_routing_tier_bands_the_balance() -> None:
    fresh = _moment(2026, 10, 3)
    # The buffer band carries the full preset set, the floor band the lite tier.
    assert routing_tier(3, 33750, fresh) == "full"
    assert routing_tier(3, 22500, fresh) == "lite"
    # Under the routing floor the routing stops.
    assert routing_tier(3, 22500 - 1, fresh) == "none"
