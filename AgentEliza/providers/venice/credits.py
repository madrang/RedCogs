# The refill calendar and the guild song gate: pure date math over the cycle anchor.
# The balance itself reads from the rate-limits endpoint of the provider.
# The anchor lives in Config (eliza setcredit).

import calendar
from datetime import datetime, timezone

from .catalog import VENICE_CREDIT_ALLOWANCE, VENICE_CREDIT_GATE_BUFFER


def _next_month(moment: datetime) -> datetime:
    """The same day and time one calendar month later — the refill cadence of the subscription — clamped to the end of the shorter month."""
    year, month = (moment.year, moment.month + 1) if moment.month < 12 else (moment.year + 1, 1)
    day = min(moment.day, calendar.monthrange(year, month)[1])
    return moment.replace(year=year, month=month, day=day)


def next_refill(cycle_start: float, now: datetime | None = None) -> datetime:
    """The first monthly refill after the moment, from the calendar walk of the cycle seed."""
    anchor = datetime.fromtimestamp(cycle_start, tz=timezone.utc)
    moment = now if now is not None else datetime.now(timezone.utc)
    while True:
        refill = _next_month(anchor)
        if refill > moment:
            return refill
        anchor = refill


def song_credit_gate(cycle_start: float, balance: float, now: datetime | None = None) -> bool:
    """True when the balance sits under the paced floor of the cycle rest:
    the remaining share of the monthly allowance, with VENICE_CREDIT_GATE_BUFFER
    as the safety margin. The guild song tool hides while the gate reads true."""
    anchor = datetime.fromtimestamp(cycle_start, tz=timezone.utc)
    moment = now if now is not None else datetime.now(timezone.utc)
    previous = anchor
    refill = _next_month(anchor)
    while refill <= moment:
        previous = refill
        refill = _next_month(refill)
    span = (refill - previous).total_seconds()
    if span <= 0:
        return False
    remaining = min(max((refill - moment).total_seconds() / span, 0.0), 1.0)
    return float(balance) < VENICE_CREDIT_GATE_BUFFER * VENICE_CREDIT_ALLOWANCE * remaining
