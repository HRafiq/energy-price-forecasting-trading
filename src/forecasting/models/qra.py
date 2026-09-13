"""Regularized Quantile Regression Averaging over the other models' medians.

QRA (Nowotarski and Weron, 2015) regresses the published price on the median
forecasts of a pool of models, once per quantile, over a recent calibration
window; variants won the GEFCom2014 price track. Plain QRA degrades when members
are nearly collinear, which ours are: the two LightGBM medians correlate at about
0.99, and unpenalized weights grow large and offsetting, so a small disagreement
on the target day moves the forecast by thousands of euros. This implementation
standardizes each member on the calibration window and adds an L1 penalty, the
LASSO-QRA variant of Uniejewski and Weron (2021).

For target day X the calibration uses member forecasts for the
``calibration_days`` days before X, each issued before its own gate, and prices
taken from the information set, so QRA sees nothing a trader could not have had
at 11:40 on day X-1.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import (
    ForecastError,
    QuantileForecast,
    make_forecast,
    quantile_columns,
)
from src.forecasting.information import InformationSet

__all__ = ["QuantileRegressionAveraging"]


@dataclass
class QuantileRegressionAveraging:
    settings: Settings
    #: Member name -> walk-forward output with a ``q50`` column.
    members: Mapping[str, pd.DataFrame]
    name: str = "qra"
    calibration_days: int = 56
    #: L1 penalty on the standardized member weights.
    alpha: float = 0.05
    _models: dict[float, QuantileRegressor] = field(default_factory=dict)
    _means: np.ndarray[Any, Any] | None = None
    _scales: np.ndarray[Any, Any] | None = None

    @property
    def lookback_days(self) -> int | None:
        return self.calibration_days + 2

    @property
    def fit_lookback_days(self) -> int | None:
        return self.lookback_days

    def _medians(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        return pd.DataFrame(
            {name: table["q50"].reindex(index) for name, table in self.members.items()},
            index=index,
        )

    def _standardized(self, values: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
        assert self._means is not None and self._scales is not None
        return np.asarray((values - self._means) / self._scales, dtype="float64")

    def fit(self, info: InformationSet) -> None:
        index = pd.DatetimeIndex(info.history.index)
        first = self.settings.market.local_midnight_utc(
            info.target_day - timedelta(days=self.calibration_days)
        )
        history = info.history.loc[(index >= first) & (index < info.target_index[0])]
        members = self._medians(pd.DatetimeIndex(history.index))
        prices = history[PRICE_SERIES]
        rows = members.notna().all(axis=1) & prices.notna()
        if int(rows.sum()) < 14 * 96:
            raise ForecastError(
                f"{self.name}: only {int(rows.sum())} calibration periods before "
                f"{info.target_day}"
            )
        values = members.loc[rows].to_numpy(dtype="float64")
        self._means = values.mean(axis=0)
        scales = values.std(axis=0)
        self._scales = np.where(scales > 0, scales, 1.0)
        target = prices.loc[rows].to_numpy(dtype="float64")
        self._models = {}
        for q in self.settings.forecasting.quantiles:
            model = QuantileRegressor(quantile=q, alpha=self.alpha, solver="highs")
            model.fit(self._standardized(values), target)
            self._models[q] = model

    def forecast(self, info: InformationSet) -> QuantileForecast:
        members = self._medians(info.target_index)
        if members.isna().any().any():
            missing = [name for name in members if members[name].isna().any()]
            raise ForecastError(
                f"{self.name}: members {missing} lack forecasts for {info.target_day}"
            )
        quantiles = self.settings.forecasting.quantiles
        values = self._standardized(members.to_numpy(dtype="float64"))
        raw = pd.DataFrame(
            {
                column: np.asarray(self._models[q].predict(values), dtype="float64")
                for q, column in zip(
                    quantiles, quantile_columns(quantiles), strict=True
                )
            },
            index=info.target_index,
        )
        return make_forecast(self.name, info, raw, quantiles)
