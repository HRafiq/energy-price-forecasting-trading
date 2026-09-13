"""MSTL: seasonal decomposition of recent prices, from Nixtla's statsforecast.

MSTL splits the last ``history_days`` of prices into daily and weekly seasonal
parts and a trend, repeats the seasonal parts and forecasts the trend with
AutoETS. It uses prices only: the exogenous-variable path reaches only the trend
model. Ranges come from the trend model's
prediction intervals at 50%, 80% and 90%, mapped to the matching quantiles.
The model refits on every forecast, because it is fast and history-driven.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoETS

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import (
    ForecastError,
    QuantileForecast,
    make_forecast,
    quantile_column,
)
from src.forecasting.information import InformationSet

__all__ = ["MSTLModel"]

#: Quantile -> (interval side, interval level in percent).
_INTERVALS = {
    0.05: ("lo", 90),
    0.10: ("lo", 80),
    0.25: ("lo", 50),
    0.75: ("hi", 50),
    0.90: ("hi", 80),
    0.95: ("hi", 90),
}


@dataclass
class MSTLModel:
    settings: Settings
    name: str = "mstl"
    history_days: int = 56
    season_lengths: tuple[int, ...] = (96, 672)
    max_gap_periods: int = 4

    @property
    def lookback_days(self) -> int | None:
        return self.history_days + 2

    @property
    def fit_lookback_days(self) -> int | None:
        return self.lookback_days

    def fit(self, info: InformationSet) -> None:
        """Nothing to train ahead of time; each forecast refits on recent prices."""

    def forecast(self, info: InformationSet) -> QuantileForecast:
        quantiles = self.settings.forecasting.quantiles
        unsupported = [q for q in quantiles if q != 0.5 and q not in _INTERVALS]
        if unsupported:
            raise ForecastError(
                f"{self.name}: no interval maps to quantiles {unsupported}"
            )

        target_start = info.target_index[0]
        start = target_start - pd.Timedelta(days=self.history_days)
        grid = pd.date_range(start, target_start, freq=info.step, inclusive="left")
        prices = info.history[PRICE_SERIES].reindex(grid)
        prices = prices.interpolate(limit=self.max_gap_periods, limit_area="inside")
        if prices.isna().any():
            raise ForecastError(
                f"{self.name}: {int(prices.isna().sum())} price periods missing before "
                f"{info.target_day} after filling gaps of up to "
                f"{self.max_gap_periods} periods"
            )
        frame = pd.DataFrame(
            {
                "unique_id": "DE-LU",
                "ds": grid.tz_localize(None),
                "y": prices.to_numpy(dtype="float64"),
            }
        )
        model = MSTL(
            season_length=list(self.season_lengths),
            trend_forecaster=AutoETS(model="ZZN"),
            alias="m",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = StatsForecast(models=[model], freq="15min", n_jobs=1).forecast(
                df=frame, h=len(info.target_index), level=[50, 80, 90]
            )
        result = result.reset_index(drop=True)
        columns = {quantile_column(0.5): result["m"].to_numpy(dtype="float64")}
        for q in quantiles:
            if q != 0.5:
                side, level = _INTERVALS[q]
                columns[quantile_column(q)] = result[f"m-{side}-{level}"].to_numpy(
                    dtype="float64"
                )
        raw = pd.DataFrame(columns, index=info.target_index)
        return make_forecast(self.name, info, raw, quantiles)
