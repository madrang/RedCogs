"""The Venice credit calendar: the cycle day, the credit gate, and the routing bands."""

from datetime import datetime, timezone

from AgentEliza.providers.venice import bundled_credit_gate, credit_ratio, next_refill, routing_tier


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


def test_bundled_credit_gate_paces_the_cycle_rest() -> None:
    # Half of the 31-day October cycle is left: the floor is 1.5 x 22500 x 0.5.
    half = _moment(2026, 10, 18, 12)
    floor = 1.5 * 22500 * 0.5
    assert bundled_credit_gate(3, floor - 1, half) is True
    assert bundled_credit_gate(3, floor, half) is False


def test_bundled_credit_gate_clamps_the_fraction_at_one_cycle() -> None:
    # The midnight of the refill itself: the whole cycle lies ahead, the floor is 1.5 allowances.
    fresh = _moment(2026, 10, 3)
    assert bundled_credit_gate(3, 1.5 * 22500 - 1, fresh) is True
    assert bundled_credit_gate(3, 1.5 * 22500, fresh) is False


def test_bundled_credit_gate_keeps_the_minimum_near_the_refill() -> None:
    # Hours before the refill the paced floor falls under half the allowance,
    # the gate still requires it.
    eve = _moment(2026, 10, 2, 23)
    assert bundled_credit_gate(3, 11250 - 1, eve) is True
    assert bundled_credit_gate(3, 11250, eve) is False


def test_credit_ratio_reads_the_paced_cycle_rest() -> None:
    fresh = _moment(2026, 10, 3)
    # The full buffer right after a refill reads 1.5, the allowance itself 1.0.
    assert credit_ratio(3, 33750, fresh) == 1.5
    assert credit_ratio(3, 22500, fresh) == 1.0
    assert credit_ratio(3, 11250, fresh) == 0.5
    # Half of the cycle is left: the balance divides by half the allowance.
    assert credit_ratio(3, 16875, _moment(2026, 10, 18, 12)) == 1.5


def test_routing_tier_bands_the_balance() -> None:
    fresh = _moment(2026, 10, 3)
    # The buffer band carries the full preset set, the floor band the lite tier.
    assert routing_tier(3, 33750, fresh) == "full"
    assert routing_tier(3, 22500, fresh) == "lite"
    # Under the routing floor the routing stops.
    assert routing_tier(3, 22500 - 1, fresh) == "none"
