"""The forecast contract shared by every model and the walk-forward harness.

A forecast covers every delivery period of one target day with one column per
configured quantile, named ``q05``, ``q10`` and so on. Values must be finite.
Independently trained quantiles can cross; ``make_forecast`` sorts each period's
quantiles and counts how many periods needed it, so crossing stays visible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
import pandas as pd

from src.forecasting.information import InformationSet

__all__ = [
    "ForecastError",
    "Forecaster",
    "QuantileForecast",
    "make_forecast",
    "quantile_column",
    "quantile_columns",
]


class ForecastError(ValueError):
    """A forecast breaks the contract, or a model cannot produce one."""


def quantile_column(q: float) -> str:
    return f"q{round(q * 100):02d}"


def quantile_columns(quantiles: tuple[float, ...]) -> list[str]:
    return [quantile_column(q) for q in quantiles]


@dataclass(frozen=True)
class QuantileForecast:
    model: str
    target_day: date
    issue_time_utc: pd.Timestamp
    #: Index: UTC period starts of the target day. Columns: quantile names.
    values: pd.DataFrame
    #: Periods whose point forecast needed a fallback source.
    fallback_periods: int
    #: Periods whose quantiles crossed and were sorted.
    crossings_repaired: int


class Forecaster(Protocol):
    """What the walk-forward harness needs from a model."""

    @property
    def name(self) -> str: ...

    @property
    def lookback_days(self) -> int | None: ...

    def fit(self, info: InformationSet) -> None: ...

    def forecast(self, info: InformationSet) -> QuantileForecast: ...


def make_forecast(
    model: str,
    info: InformationSet,
    raw: pd.DataFrame,
    quantiles: tuple[float, ...],
    fallback_periods: int = 0,
) -> QuantileForecast:
    """Validate a model's raw quantiles and package them as a forecast."""
    columns = quantile_columns(quantiles)
    missing = [column for column in columns if column not in raw.columns]
    if missing:
        raise ForecastError(f"{model}: missing quantile columns {missing}")
    if not raw.index.equals(info.target_index):
        raise ForecastError(
            f"{model}: forecast index does not match the periods of {info.target_day}"
        )
    array = raw[columns].to_numpy(dtype="float64")
    if not np.isfinite(array).all():
        raise ForecastError(
            f"{model}: forecast for {info.target_day} has missing or infinite values"
        )
    crossed = int((np.diff(array, axis=1) < 0).any(axis=1).sum())
    values = pd.DataFrame(
        np.sort(array, axis=1), index=info.target_index, columns=columns
    )
    return QuantileForecast(
        model=model,
        target_day=info.target_day,
        issue_time_utc=info.issue_time_utc,
        values=values,
        fallback_periods=fallback_periods,
        crossings_repaired=crossed,
    )
