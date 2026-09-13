"""Delivery-day time grid.

The market trades in local calendar days (Europe/Berlin) while the data lives in
UTC. Because of daylight saving time a local delivery day has 23, 24 or 25
hours, which is 92, 96 or 100 quarter-hours. Two rules follow, and
``tests/test_timegrid.py`` pins them (D3):

1. Every timestamp inside ``src/`` is a timezone-aware UTC instant.
2. A period within a delivery day is identified by its ordinal (0, 1, ...),
   never by local wall-clock hour. Local 02:00 does not exist on the spring
   day and happens twice on the autumn day.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

__all__ = [
    "HOUR",
    "delivery_calendar",
    "delivery_periods",
    "ensure_utc_index",
    "expected_periods",
    "local_day_bounds_utc",
]

HOUR = pd.Timedelta(hours=1)
_ZERO = pd.Timedelta(0)


def ensure_utc_index(index: pd.Index) -> pd.DatetimeIndex:
    """Return ``index`` if it is a timezone-aware UTC DatetimeIndex, else raise."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError(f"expected a DatetimeIndex, got {type(index).__name__}")
    if index.tz is None:
        raise ValueError("timestamps must be timezone-aware UTC, got naive timestamps")
    if str(index.tz) != "UTC":
        raise ValueError(f"timestamps must be in UTC, got {index.tz}")
    return index


def local_day_bounds_utc(day: date, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """UTC instants of local midnight at the start and end of ``day``."""
    start = pd.Timestamp(day).tz_localize(tz).tz_convert("UTC")
    end = pd.Timestamp(day + timedelta(days=1)).tz_localize(tz).tz_convert("UTC")
    return start, end


def expected_periods(day: date, tz: str, step: pd.Timedelta = HOUR) -> int:
    """Periods in local ``day``: 23, 24 or 25 hours; 92, 96 or 100 quarter-hours."""
    start, end = local_day_bounds_utc(day, tz)
    span = end - start
    if span % step != _ZERO:
        raise ValueError(f"a local day of {span} is not a whole number of {step}")
    return int(span // step)


def delivery_periods(day: date, tz: str, step: pd.Timedelta = HOUR) -> pd.DatetimeIndex:
    """The UTC start instant of every delivery period in local ``day``."""
    start, end = local_day_bounds_utc(day, tz)
    return pd.date_range(start, end, freq=step, inclusive="left", name="timestamp_utc")


def delivery_calendar(
    index: pd.DatetimeIndex, tz: str, step: pd.Timedelta = HOUR
) -> pd.DataFrame:
    """Map UTC period starts to their local delivery day and period ordinal.

    Returns one row per timestamp with ``delivery_date``, ``period`` (0-based
    ordinal within the day), ``local_hour`` (wall clock, for display only) and
    ``periods_in_day``.
    """
    idx = ensure_utc_index(index)
    local = idx.tz_convert(tz)
    elapsed = idx - local.normalize().tz_convert("UTC")
    off_grid = (elapsed % step) != _ZERO
    if off_grid.any():
        raise ValueError(f"{int(off_grid.sum())} timestamps are not aligned to {step}")
    dates = list(local.date)
    periods_by_date = {d: expected_periods(d, tz, step) for d in set(dates)}
    return pd.DataFrame(
        {
            "delivery_date": dates,
            "period": (elapsed // step).to_numpy(dtype="int64"),
            "local_hour": local.hour.to_numpy(dtype="int64"),
            "periods_in_day": [periods_by_date[d] for d in dates],
        },
        index=idx,
    )
