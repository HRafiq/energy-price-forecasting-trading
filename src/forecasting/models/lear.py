"""LEAR-style linear model: LASSO on arcsinh-scaled prices, empirical quantiles.

Follows the LEAR benchmark of Lago et al. (2021) in spirit: a sparse linear
autoregressive model with exogenous inputs, prices standardised by median and
median absolute deviation and passed through arcsinh to tame spikes, and the
LASSO penalty chosen by AIC. One model serves all periods, with hour-of-day and
weekday indicators instead of 24 separate models, to keep training fast.

Quantiles are split-conformal. A model trained on all but the last
``calibration_days`` days measures its errors on those days per local hour;
then the model is refitted on the whole window, so forecasts do not come from a
model 42 days staler than necessary. The ranges keep the held-out errors, which
may be slightly optimistic for the refitted model; coverage in the comparison
shows whether that matters. Weather features are left out because they only
exist from 2024 and a linear model cannot use partly missing inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import LassoLarsIC

from src.config import Settings
from src.forecasting.base import QuantileForecast, make_forecast, quantile_columns
from src.forecasting.information import InformationSet
from src.forecasting.models.common import (
    FEATURE_HISTORY_DAYS,
    feature_names,
    hourly_offsets,
    local_hours,
    split_calibration,
    target_features,
    training_data,
)

__all__ = ["LearModel"]

GROUPS = (
    "calendar",
    "price_history",
    "load",
    "grid_operator_forecasts",
    "measured",
    "fuels",
)
_CATEGORICAL = ("clock_minute", "weekday")


@dataclass
class LearModel:
    settings: Settings
    name: str = "lear"
    training_days: int = 364
    calibration_days: int = 42
    _names: list[str] = field(default_factory=lambda: feature_names(GROUPS))
    _medians: np.ndarray[Any, Any] | None = None
    _means: np.ndarray[Any, Any] | None = None
    _scales: np.ndarray[Any, Any] | None = None
    _centre: float = 0.0
    _spread: float = 1.0
    _model: LassoLarsIC | None = None
    _offsets: np.ndarray[Any, Any] | None = None

    @property
    def lookback_days(self) -> int | None:
        return FEATURE_HISTORY_DAYS

    @property
    def fit_lookback_days(self) -> int | None:
        return self.training_days + FEATURE_HISTORY_DAYS

    def _numeric(self) -> list[str]:
        return [n for n in self._names if n not in _CATEGORICAL]

    def _design(self, features: pd.DataFrame) -> NDArray[np.float64]:
        assert self._medians is not None and self._means is not None
        assert self._scales is not None
        values = features[self._numeric()].to_numpy(dtype="float64")
        filled = np.where(np.isfinite(values), values, self._medians)
        scaled = (filled - self._means) / self._scales
        hours = features["clock_minute"].to_numpy(dtype="int64") // 60
        weekdays = features["weekday"].to_numpy(dtype="int64")
        return np.hstack([scaled, np.eye(24)[hours], np.eye(7)[weekdays]])

    def _train(self, features: pd.DataFrame, prices: NDArray[np.float64]) -> None:
        """Fit scaling, price transform and LASSO on these rows only."""
        raw = features[self._numeric()].to_numpy(dtype="float64")
        finite = np.isfinite(raw)
        counts = finite.sum(axis=0)
        medians = np.array(
            [
                float(np.median(raw[finite[:, j], j])) if counts[j] else 0.0
                for j in range(raw.shape[1])
            ]
        )
        filled = np.where(finite, raw, medians)
        scales = filled.std(axis=0)
        self._medians = medians
        self._means = filled.mean(axis=0)
        self._scales = np.where(scales > 0, scales, 1.0)
        self._centre = float(np.median(prices))
        self._spread = float(np.median(np.abs(prices - self._centre))) or 1.0
        model = LassoLarsIC(criterion="aic")
        model.fit(
            self._design(features), np.arcsinh((prices - self._centre) / self._spread)
        )
        self._model = model

    def _predict(self, features: pd.DataFrame) -> NDArray[np.float64]:
        assert self._model is not None
        z = self._model.predict(self._design(features))
        return np.asarray(np.sinh(z) * self._spread + self._centre, dtype="float64")

    def fit(self, info: InformationSet) -> None:
        data = training_data(info, self.settings, self._names, self.training_days)
        fit_rows, calibration_rows = split_calibration(data, self.calibration_days)
        prices = data.target.to_numpy(dtype="float64")

        self._train(data.features.loc[fit_rows], prices[fit_rows])
        residuals = prices[calibration_rows] - self._predict(
            data.features.loc[calibration_rows]
        )
        self._offsets = hourly_offsets(
            residuals, data.hours[calibration_rows], self.settings.forecasting.quantiles
        )
        self._train(data.features, prices)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        assert self._offsets is not None
        features = target_features(info, self.settings, self._names)
        point = self._predict(features)
        hours = local_hours(info.target_index, info.tz)
        quantiles = self.settings.forecasting.quantiles
        raw = pd.DataFrame(
            point[:, None] + self._offsets[hours],
            index=info.target_index,
            columns=quantile_columns(quantiles),
        )
        return make_forecast(self.name, info, raw, quantiles)
