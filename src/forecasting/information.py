"""What is known when the daily forecast is issued: the leakage guard (M3).

The forecast for local day D+1 is issued at ``market.forecast_issue_local`` on
day D. ``build_information_set`` returns the dataset exactly as a trader saw it
then: every cell not yet published is removed, using the per-column rules in
``config/settings.yaml`` under ``availability``. Models receive only this view,
so they cannot leak future information even by accident. Columns without a rule
are refused rather than passed through.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import RESOLUTION_STEP, AvailabilityRule, Settings
from src.timegrid import delivery_periods, ensure_utc_index, local_day_bounds_utc

__all__ = [
    "InformationSet",
    "availability_mask",
    "build_information_set",
    "issue_time_utc",
]


@dataclass(frozen=True)
class InformationSet:
    """The data a model may use to forecast ``target_day``."""

    target_day: date
    issue_time_utc: pd.Timestamp
    #: UTC start of every delivery period of the target day.
    target_index: pd.DatetimeIndex
    #: Dataset rows up to the end of the target day, unpublished cells as NaN.
    history: pd.DataFrame
    step: pd.Timedelta
    tz: str


def issue_time_utc(target_day: date, settings: Settings) -> pd.Timestamp:
    """When the forecast for ``target_day`` is issued, on the local day before."""
    hours, minutes = (
        int(part) for part in settings.market.forecast_issue_local.split(":")
    )
    local = pd.Timestamp(target_day - timedelta(days=1)) + pd.Timedelta(
        hours=hours, minutes=minutes
    )
    return local.tz_localize(settings.market.timezone).tz_convert("UTC")


def availability_mask(
    index: pd.DatetimeIndex,
    rule: AvailabilityRule,
    target_day: date,
    settings: Settings,
) -> NDArray[np.bool_]:
    """True where a period's value is published by the issue time."""
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    target_start, target_end = local_day_bounds_utc(
        target_day, settings.market.timezone
    )
    if rule == "before_target_day":
        return np.asarray(index < target_start)
    if rule == "through_target_day":
        return np.asarray(index < target_end)
    if rule == "before_issue_lag":
        known_until = issue_time_utc(target_day, settings) - pd.Timedelta(
            minutes=settings.availability.actuals_lag_minutes
        )
        return np.asarray(index + step <= known_until)
    raise ValueError(f"unknown availability rule {rule!r}")


def build_information_set(
    frame: pd.DataFrame,
    target_day: date,
    settings: Settings,
    lookback_days: int | None = None,
) -> InformationSet:
    """The dataset as it looked at the issue time for ``target_day``.

    ``lookback_days`` limits history to that many days before the target day,
    which keeps cheap models fast; ``None`` keeps everything.
    """
    idx = ensure_utc_index(frame.index)
    if not idx.is_monotonic_increasing or idx.has_duplicates:
        raise ValueError("dataset index must be strictly increasing")
    rules = settings.availability.columns
    unknown = [column for column in frame.columns if column not in rules]
    if unknown:
        raise ValueError(
            f"no availability rule for columns {unknown}; "
            "refusing to pass them to a model"
        )

    tz = settings.market.timezone
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    target_start, target_end = local_day_bounds_utc(target_day, tz)
    stop = int(idx.searchsorted(target_end, side="left"))
    begin = 0
    if lookback_days is not None:
        begin = int(idx.searchsorted(target_start - pd.Timedelta(days=lookback_days)))
    history = frame.iloc[begin:stop].copy()

    history_index = pd.DatetimeIndex(history.index)
    for column in history.columns:
        mask = availability_mask(history_index, rules[column], target_day, settings)
        if not mask.all():
            history[column] = history[column].where(mask)

    return InformationSet(
        target_day=target_day,
        issue_time_utc=issue_time_utc(target_day, settings),
        target_index=delivery_periods(target_day, tz, step),
        history=history,
        step=step,
        tz=tz,
    )
