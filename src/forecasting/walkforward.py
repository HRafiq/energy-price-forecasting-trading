"""Walk-forward forecasting harness shared by every model.

For each target day, in order, the harness builds the information set as of the
issue time on the day before, lets the model refit on its schedule, collects the
forecast and checks it against the contract. Realised prices are joined only
after every forecast has been made. Target days in the hold-out are refused, and
hold-out rows are cut from the data, unless ``allow_holdout=True``, which only
the final Phase 4 report uses.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from itertools import pairwise

import numpy as np
import pandas as pd

from src.config import PRICE_SERIES, PRODUCT_COLUMN, Settings
from src.forecasting.base import Forecaster, ForecastError, quantile_columns
from src.forecasting.information import build_information_set

__all__ = ["HoldoutAccessError", "day_range", "run_walk_forward"]


class HoldoutAccessError(RuntimeError):
    """A run tried to forecast a day in the hold-out period."""


def day_range(first: date, last: date) -> list[date]:
    """Every local day from ``first`` to ``last``, inclusive."""
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def run_walk_forward(
    frame: pd.DataFrame,
    forecaster: Forecaster,
    target_days: Sequence[date],
    settings: Settings,
    refit_every_days: int = 1,
    allow_holdout: bool = False,
) -> pd.DataFrame:
    """Forecast each target day as it would have been forecast at the time.

    Returns one row per delivery period: ``model``, the quantile columns,
    ``target_day``, ``fallback_periods``, ``crossings_repaired``, ``actual`` and
    ``price_product_minutes``.
    """
    days = list(target_days)
    if not days:
        raise ValueError("no target days to forecast")
    if any(later <= earlier for earlier, later in pairwise(days)):
        raise ValueError("target days must be strictly increasing")
    if refit_every_days < 1:
        raise ValueError("refit_every_days must be at least 1")

    holdout = settings.evaluation.holdout_start
    if not allow_holdout:
        inside = [day for day in days if day >= holdout]
        if inside:
            raise HoldoutAccessError(
                f"{len(inside)} target days fall in the hold-out starting {holdout}; "
                "the hold-out is scored once, in Phase 4"
            )
        frame = frame.loc[frame.index < settings.market.local_midnight_utc(holdout)]

    columns = quantile_columns(settings.forecasting.quantiles)
    parts: list[pd.DataFrame] = []
    last_fit: date | None = None
    for day in days:
        if last_fit is None or (day - last_fit).days >= refit_every_days:
            forecaster.fit(
                build_information_set(
                    frame, day, settings, forecaster.fit_lookback_days
                )
            )
            last_fit = day
        info = build_information_set(frame, day, settings, forecaster.lookback_days)
        result = forecaster.forecast(info)
        if (
            result.target_day != day
            or not result.values.index.equals(info.target_index)
            or list(result.values.columns) != columns
        ):
            raise ForecastError(
                f"{forecaster.name}: forecast for {day} breaks the contract"
            )
        array = result.values.to_numpy(dtype="float64")
        if not np.isfinite(array).all():
            raise ForecastError(
                f"{forecaster.name}: forecast for {day} has missing or infinite values"
            )
        if (np.diff(array, axis=1) < 0).any():
            raise ForecastError(
                f"{forecaster.name}: forecast for {day} has crossing quantiles"
            )
        part = result.values.copy()
        part["target_day"] = day
        part["fallback_periods"] = result.fallback_periods
        part["crossings_repaired"] = result.crossings_repaired
        parts.append(part)

    out = pd.concat(parts)
    out.index.name = "timestamp_utc"
    out.insert(0, "model", forecaster.name)
    out["actual"] = frame[PRICE_SERIES].reindex(out.index)
    out[PRODUCT_COLUMN] = frame[PRODUCT_COLUMN].reindex(out.index)
    return out
