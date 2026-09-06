"""The Venice credit derivation: the refill calendar and the balance walk."""

from datetime import datetime, timedelta, timezone

import pytest

from AgentEliza.providers.venice import _cycle_spend, _next_month, walk_credits


def _moment(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


def _row(moment: datetime, usd) -> dict:
    return {"date": moment.date().isoformat(), "USD": usd}


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


def test_cycle_spend_sums_the_inclusive_date_range() -> None:
    spend = {"2026-09-01": 1.0, "2026-09-03": 2.0, "2026-09-05": 4.0}
    assert _cycle_spend(spend, _moment(2026, 9, 1), _moment(2026, 9, 3)) == 3.0


def test_cycle_spend_reads_missing_dates_as_zero() -> None:
    assert _cycle_spend({}, _moment(2026, 9, 1), _moment(2026, 9, 30)) == 0.0


def test_walk_returns_none_without_bydate_rows() -> None:
    assert walk_credits({"byDate": []}, 1786000000.0, 22500) is None
    assert walk_credits({}, 1786000000.0, 22500) is None


def test_walk_spend_of_the_open_cycle() -> None:
    seed = datetime.now(timezone.utc) - timedelta(days=5)
    assert walk_credits({"byDate": [_row(seed, 5)]}, seed.timestamp(), 22500) == 22000.0


def test_walk_caps_the_bank_at_three_allowances() -> None:
    # 40 days hold exactly one refill: one month passed, two did not.
    seed = datetime.now(timezone.utc) - timedelta(days=40)
    data = {"byDate": [_row(seed, 0)]}
    assert walk_credits(data, seed.timestamp(), 66000) == 67500.0


def test_walk_accrues_over_two_refills() -> None:
    # 70 days hold exactly two refills, and none of the third.
    seed = datetime.now(timezone.utc) - timedelta(days=70)
    data = {"byDate": [_row(seed, 0)]}
    assert walk_credits(data, seed.timestamp(), 10000) == 55000.0


def test_walk_spend_spans_cycles() -> None:
    seed = datetime.now(timezone.utc) - timedelta(days=70)
    yesterday = datetime.now(timezone.utc) - timedelta(days=1)
    data = {"byDate": [_row(seed, 10), _row(yesterday, 20)]}
    assert walk_credits(data, seed.timestamp(), 10000) == 52000.0


def test_walk_floors_the_balance_at_zero() -> None:
    seed = datetime.now(timezone.utc) - timedelta(days=5)
    assert walk_credits({"byDate": [_row(seed, 50)]}, seed.timestamp(), 10) == 0.0


def test_walk_skips_malformed_rows() -> None:
    seed = datetime.now(timezone.utc) - timedelta(days=5)
    data = {"byDate": [
        {"date": "2026-09-05", "USD": "not a number"},
        {"no": "date"},
        "junk",
        _row(seed, 2),
    ]}
    assert walk_credits(data, seed.timestamp(), 22500) == 22300.0


def test_walk_counts_the_refill_day_in_both_cycles() -> None:
    # The documented low bias: the refill day closes the old cycle and
    # opens the new one, so its spend leaves the pool twice.
    seed = datetime.now(timezone.utc) - timedelta(days=70)
    refill = _next_month(seed)
    on_refill = walk_credits({"byDate": [_row(refill, 1)]}, seed.timestamp(), 20000)
    after_refill = walk_credits({"byDate": [_row(refill + timedelta(days=1), 1)]}, seed.timestamp(), 20000)
    assert on_refill == pytest.approx(after_refill - 100.0)
