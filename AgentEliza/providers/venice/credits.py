# The bundled credit cycle calendar and the guild song gate: pure date math over the cycle day.
# The balance itself reads from the rate-limits endpoint of the provider.
# The cycle day lives in Config (`eliza setcycleday`).

import calendar
from datetime import datetime, timezone

from .catalog import VENICE_CREDIT_ALLOWANCE, VENICE_CREDIT_GATE_BUFFER


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


def song_credit_gate(cycle_day: int, balance: float, now: datetime | None = None) -> bool:
    """True when the balance sits under the paced floor of the cycle rest:
    the remaining share of the monthly allowance, with VENICE_CREDIT_GATE_BUFFER
    as the safety margin. The guild song tool hides while the gate reads true."""
    moment = now if now is not None else datetime.now(timezone.utc)
    upcoming = next_refill(cycle_day, moment)
    this_month = _cycle_day_of(moment.year, moment.month, cycle_day)
    previous = this_month if this_month <= moment else _cycle_day_of(*_previous_month(this_month.year, this_month.month))
    span = (upcoming - previous).total_seconds()
    if span <= 0:
        return False
    remaining = min(max((upcoming - moment).total_seconds() / span, 0.0), 1.0)
    return float(balance) < VENICE_CREDIT_GATE_BUFFER * VENICE_CREDIT_ALLOWANCE * remaining
