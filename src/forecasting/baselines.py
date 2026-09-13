"""Naive price baselines: the numbers every model must beat, and the fallback.

Both baselines are probabilistic:

* ``naive_previous_day`` forecasts each period with the price at the same local
  clock time on day D, the latest full day published at the 11:40 issue time.
* ``seasonal_naive_previous_week`` uses the same local clock time one week
  before the target day.

Quantiles come from each baseline's own recent errors. For every past day in
``baselines.error_window_days``, the forecast the baseline would have made is
compared with the published price; the error quantiles for each local hour are
added to the point forecast. Hours with too few errors use the quantiles of all
hours. With fewer than ``baselines.min_error_days`` days of errors the baseline
refuses to forecast, rather than show a precise-looking range on thin evidence.

If the source day is missing, the point forecast falls back to a second, longer
lag, then to the last published price. Those periods take their range from the
longer lag's errors, which are wider, or keep the main lag's range when the
longer lag has too little error history, so a few missing periods never stop
the whole forecast. The forecast counts them. The baselines double as the
last-resort fallback for the live pipeline (D5).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import PRICE_SERIES, Settings
from src.features.clock import clock_table, values_on
from src.forecasting.base import (
    ForecastError,
    QuantileForecast,
    make_forecast,
    quantile_columns,
)
from src.forecasting.information import InformationSet

__all__ = ["LaggedPriceBaseline", "build_baselines", "clock_table", "values_on"]


@dataclass(frozen=True)
class LaggedPriceBaseline:
    """Point forecast from an earlier day's prices, quantiles from recent errors."""

    name: str
    lag_days: int
    fallback_lag_days: int
    quantiles: tuple[float, ...]
    error_window_days: int
    min_error_days: int
    min_errors_per_hour: int = 20

    @property
    def lookback_days(self) -> int | None:
        return self.error_window_days + max(self.lag_days, self.fallback_lag_days) + 2

    @property
    def fit_lookback_days(self) -> int | None:
        return self.lookback_days

    def fit(self, info: InformationSet) -> None:
        """Baselines have nothing to learn ahead of time."""

    def forecast(self, info: InformationSet) -> QuantileForecast:
        price = info.history[PRICE_SERIES]
        table = clock_table(price, info.tz, info.step)
        target = info.target_index
        hours = np.asarray(target.tz_convert(info.tz).hour, dtype="int64")

        point = values_on(
            table, info.target_day - timedelta(days=self.lag_days), target, info.tz
        )
        spread = self._spread(table, info.target_day, self.lag_days, hours)

        fallback = np.isnan(point)
        if fallback.any():
            backup = values_on(
                table,
                info.target_day - timedelta(days=self.fallback_lag_days),
                target,
                info.tz,
            )
            if np.isnan(backup[fallback]).any():
                published = price.dropna()
                if published.empty:
                    raise ForecastError(
                        f"{self.name}: no published price before {info.target_day}"
                    )
                backup = np.where(np.isnan(backup), float(published.iloc[-1]), backup)
            point = np.where(fallback, backup, point)
            # An older source is less reliable: size its range with that lag's
            # errors when there is enough history for them.
            try:
                backup_spread = self._spread(
                    table, info.target_day, self.fallback_lag_days, hours
                )
            except ForecastError:
                backup_spread = spread
            spread = np.where(fallback[:, None], backup_spread, spread)

        raw = pd.DataFrame(
            point[:, None] + spread,
            index=target,
            columns=quantile_columns(self.quantiles),
        )
        return make_forecast(
            self.name, info, raw, self.quantiles, fallback_periods=int(fallback.sum())
        )

    def _spread(
        self,
        table: pd.DataFrame,
        target_day: date,
        lag_days: int,
        hours: NDArray[np.int64],
    ) -> NDArray[np.float64]:
        """Error quantile offsets for each target period, one column per quantile."""
        offsets = self._error_offsets(table, target_day, lag_days)
        return np.stack([offsets[q][hours] for q in self.quantiles], axis=1)

    def _error_offsets(
        self, table: pd.DataFrame, target_day: date, lag_days: int
    ) -> dict[float, NDArray[np.float64]]:
        """Quantiles of recent errors, actual minus forecast, for each local hour."""
        days = [
            target_day - timedelta(days=k) for k in range(1, self.error_window_days + 1)
        ]
        sources = [day - timedelta(days=lag_days) for day in days]
        actual = table.reindex(days).to_numpy(dtype="float64")
        predicted = table.reindex(sources).to_numpy(dtype="float64")
        errors = actual - predicted
        ok = np.isfinite(errors)
        error_days = int(ok.any(axis=1).sum())
        if error_days < self.min_error_days:
            raise ForecastError(
                f"{self.name}: only {error_days} days of errors for a {lag_days}-day "
                f"lag before {target_day}; at least {self.min_error_days} are needed"
            )
        clock_hours = np.asarray(table.columns, dtype="int64") // 60
        hours = np.broadcast_to(clock_hours, errors.shape)
        flat = pd.Series(errors[ok])
        by_hour = flat.groupby(hours[ok])
        counts = by_hour.size().reindex(range(24), fill_value=0).to_numpy()
        offsets: dict[float, NDArray[np.float64]] = {}
        for q in self.quantiles:
            pooled = float(flat.quantile(q))
            hourly = by_hour.quantile(q).reindex(range(24)).to_numpy(dtype="float64")
            offsets[q] = np.where(counts >= self.min_errors_per_hour, hourly, pooled)
        return offsets


def build_baselines(settings: Settings) -> list[LaggedPriceBaseline]:
    quantiles = settings.forecasting.quantiles
    window = settings.baselines.error_window_days
    minimum = settings.baselines.min_error_days
    return [
        LaggedPriceBaseline("naive_previous_day", 1, 7, quantiles, window, minimum),
        LaggedPriceBaseline(
            "seasonal_naive_previous_week", 7, 14, quantiles, window, minimum
        ),
    ]
