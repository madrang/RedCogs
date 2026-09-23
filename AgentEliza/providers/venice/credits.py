# The bundled credit cycle calendar, the guild credit gate, and the routing
# floor: pure date math over the cycle day.
# The gate paces the guild media tools against the cycle rest. The chat
# model routing reads the credit ratio of the balance over the cycle rest.
# The balance itself reads from the rate-limits endpoint of the provider.
# The cycle day lives in Config (`eliza setcycleday`).

import calendar
from datetime import datetime, timezone

from .catalog import (
  VENICE_CREDIT_ALLOWANCE, VENICE_CREDIT_GATE_BUFFER, VENICE_CREDIT_ROUTING_FLOOR,
)


def _cycle_day_of(year: int, month: int, day: int) -> datetime:
    """The cycle day of one month at midnight UTC, clamped to the end of the shorter months."""
    return datetime(year, month, min(day, calendar.monthrange(year, month)[1]), tzinfo=timezone.utc)


def _previous_month(year: int, month: int) -> tuple[int, int]:
    return (year, month - 1) if month > 1 else (year - 1, 12)


def next_refill(cycle_day: int, now: datetime | None = None) -> datetime:
    """The next occurrence of the cycle day after the moment."""
    moment = now if now is not None else datetime.now(timezone.utc)
    candidate = _cycle_day_of(moment.year, moment.month, cycle_day)
    if candidate > moment:
        return candidate
    year, month = (moment.year, moment.month + 1) if moment.month < 12 else (moment.year + 1, 1)
    return _cycle_day_of(year, month, cycle_day)


def _cycle_rest(cycle_day: int, moment: datetime) -> float | None:
    """The fraction of the billing cycle still ahead, between 0 and 1.
    None when the span between the two neighboring refills breaks."""
    upcoming = next_refill(cycle_day, moment)
    this_month = _cycle_day_of(moment.year, moment.month, cycle_day)
    previous = this_month if this_month <= moment else _cycle_day_of(*_previous_month(this_month.year, this_month.month), cycle_day)
    span = (upcoming - previous).total_seconds()
    if span <= 0:
        return None
    return min(max((upcoming - moment).total_seconds() / span, 0.0), 1.0)


def credit_ratio(cycle_day: int, balance: float, now: datetime | None = None) -> float | None:
    """The credit health of the balance over the cycle rest: under 1.0 the
    balance cannot cover the rest of the cycle at the normal rate, at 1.0 it
    covers it exactly, and above 1.0 each full allowance of surplus adds one.
    The allowance held unused to the eve of the refill reads just under 2,
    a fresh double reads 2. None when the cycle span breaks."""
    moment = now if now is not None else datetime.now(timezone.utc)
    remaining = _cycle_rest(cycle_day, moment)
    if remaining is None:
        return None
    paced = VENICE_CREDIT_ALLOWANCE * remaining
    if balance < paced:
        return float(balance) / paced
    return 1.0 + (float(balance) - paced) / VENICE_CREDIT_ALLOWANCE


def routing_tier(cycle_day: int, balance: float, now: datetime | None = None) -> str:
    """The chat routing tier of the balance over the paced cycle rest:
    full at the gate buffer, lite above the routing floor, none under it.
    A broken cycle span reads full: an unread balance never blocks the routing."""
    ratio = credit_ratio(cycle_day, balance, now)
    if ratio is None:
        return "full"
    if ratio < VENICE_CREDIT_ROUTING_FLOOR:
        return "none"
    if ratio < VENICE_CREDIT_GATE_BUFFER:
        return "lite"
    return "full"
