"""Align quarter-hourly series by local clock time across days and DST changes.

Prices and forecasts repeat by local clock time, so "the same time yesterday"
means the same local clock time, not 96 periods earlier. On the autumn DST day
the repeated hour keeps its first value; on the spring DST day the missing hour
02:00 to 02:59 takes the value of the last earlier clock time. Data gaps stay
NaN.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.timegrid import delivery_periods, ensure_utc_index, expected_periods

__all__ = ["clock_minutes", "clock_table", "lag_by_clock", "local_dates", "values_on"]

_MINUTES_PER_DAY = 24 * 60


def clock_minutes(index: pd.DatetimeIndex, tz: str) -> NDArray[np.int64]:
    """Local clock time of each timestamp, in minutes after local midnight."""
    local = index.tz_convert(tz)
    return np.asarray(local.hour * 60 + local.minute, dtype="int64")


def local_dates(index: pd.DatetimeIndex, tz: str) -> NDArray[np.object_]:
    """The local calendar date of each timestamp."""
    return np.asarray(index.tz_convert(tz).date, dtype=object)


def clock_table(series: pd.Series, tz: str, step: pd.Timedelta) -> pd.DataFrame:
    """Values with local days as rows and local clock minutes as columns."""
    idx = ensure_utc_index(series.index)
    step_minutes = int(step / pd.Timedelta(minutes=1))
    long = pd.DataFrame(
        {
            "day": local_dates(idx, tz),
            "clock": clock_minutes(idx, tz),
            "value": series.to_numpy(dtype="float64"),
        }
    )
    table = long.groupby(["day", "clock"])["value"].first().unstack("clock")
    table = table.reindex(columns=list(range(0, _MINUTES_PER_DAY, step_minutes)))

    periods_per_full_day = _MINUTES_PER_DAY // step_minutes
    observed = long.groupby("day")["clock"].nunique()
    for day in observed.index[observed < periods_per_full_day]:
        if expected_periods(day, tz, step) >= periods_per_full_day:
            continue  # a truncated edge of the history, not a DST day
        existing = clock_minutes(delivery_periods(day, tz, step), tz)
        for column in table.columns:
            minute = int(column)
            if minute not in existing:
                earlier = int(existing[existing < minute].max())
                table.loc[day, minute] = table.loc[day, earlier]
    return table


def values_on(
    table: pd.DataFrame, source_day: date, target_index: pd.DatetimeIndex, tz: str
) -> NDArray[np.float64]:
    """Values of ``source_day`` placed on the target periods by local clock time."""
    if source_day not in table.index:
        return np.full(len(target_index), np.nan)
    row = table.loc[source_day]
    return row.reindex(clock_minutes(target_index, tz)).to_numpy(dtype="float64")


def lag_by_clock(
    series: pd.Series,
    days: int,
    tz: str,
    step: pd.Timedelta,
    table: pd.DataFrame | None = None,
) -> pd.Series:
    """For every period, the value at the same local clock time ``days`` earlier."""
    idx = ensure_utc_index(series.index)
    lookup = (clock_table(series, tz, step) if table is None else table).stack(
        future_stack=True
    )
    shift = timedelta(days=days)
    source_days = [day - shift for day in local_dates(idx, tz)]
    keys = pd.MultiIndex.from_arrays([source_days, clock_minutes(idx, tz)])
    values = lookup.reindex(keys).to_numpy(dtype="float64")
    return pd.Series(values, index=idx, name=f"{series.name}_lag_{days}d")
