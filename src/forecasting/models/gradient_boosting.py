"""LightGBM candidates: quantile regression, and a point model with conformal ranges.

* ``LightGBMQuantileModel`` trains one LightGBM model per quantile with pinball
  loss, the approach most used for probabilistic price forecasting on tabular
  features. Quantiles are trained independently and can cross; the forecast
  contract sorts them and counts it.
* ``LightGBMConformalModel`` trains one median model with absolute-error loss on
  all but the last ``calibration_days`` days, then adds that model's errors on
  those days, per local hour, as the range. Split conformal prediction targets
  coverage directly.

Both use every feature group; LightGBM handles the weather features' missing
values before 2024 natively.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import Settings
from src.features.build import FEATURE_GROUPS
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

__all__ = ["DEFAULT_LGBM_PARAMS", "LightGBMConformalModel", "LightGBMQuantileModel"]

DEFAULT_LGBM_PARAMS: dict[str, Any] = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 7,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}


def _all_features() -> list[str]:
    return feature_names(list(FEATURE_GROUPS))


@dataclass
class LightGBMQuantileModel:
    settings: Settings
    name: str = "lightgbm_quantile"
    training_days: int = 730
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    _names: list[str] = field(default_factory=_all_features)
    _models: dict[float, LGBMRegressor] = field(default_factory=dict)

    @property
    def lookback_days(self) -> int | None:
        return FEATURE_HISTORY_DAYS

    @property
    def fit_lookback_days(self) -> int | None:
        return self.training_days + FEATURE_HISTORY_DAYS

    def fit(self, info: InformationSet) -> None:
        data = training_data(info, self.settings, self._names, self.training_days)
        self._models = {}
        for q in self.settings.forecasting.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **self.params)
            model.fit(data.features, data.target)
            self._models[q] = model

    def forecast(self, info: InformationSet) -> QuantileForecast:
        features = target_features(info, self.settings, self._names)
        quantiles = self.settings.forecasting.quantiles
        raw = pd.DataFrame(
            {
                column: np.asarray(self._models[q].predict(features), dtype="float64")
                for q, column in zip(
                    quantiles, quantile_columns(quantiles), strict=True
                )
            },
            index=info.target_index,
        )
        return make_forecast(self.name, info, raw, quantiles)


@dataclass
class LightGBMConformalModel:
    settings: Settings
    name: str = "lightgbm_conformal"
    training_days: int = 730
    calibration_days: int = 42
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    _names: list[str] = field(default_factory=_all_features)
    _model: LGBMRegressor | None = None
    _offsets: np.ndarray[Any, Any] | None = None

    @property
    def lookback_days(self) -> int | None:
        return FEATURE_HISTORY_DAYS

    @property
    def fit_lookback_days(self) -> int | None:
        return self.training_days + FEATURE_HISTORY_DAYS

    def fit(self, info: InformationSet) -> None:
        data = training_data(info, self.settings, self._names, self.training_days)
        fit_rows, calibration_rows = split_calibration(data, self.calibration_days)
        model = LGBMRegressor(objective="l1", **self.params)
        model.fit(data.features.loc[fit_rows], data.target.loc[fit_rows])
        predicted = np.asarray(
            model.predict(data.features.loc[calibration_rows]), dtype="float64"
        )
        residuals = data.target.to_numpy(dtype="float64")[calibration_rows] - predicted
        self._model = model
        self._offsets = hourly_offsets(
            residuals, data.hours[calibration_rows], self.settings.forecasting.quantiles
        )
        # Refit on the whole window so forecasts are not 42 days staler than needed.
        self._model = LGBMRegressor(objective="l1", **self.params).fit(
            data.features, data.target
        )

    def feature_importance(self) -> pd.Series:
        """Total LightGBM gain per feature of the fitted point model, largest first."""
        if self._model is None:
            raise ValueError("fit the model before asking for its feature importance")
        booster = self._model.booster_
        gains = booster.feature_importance(importance_type="gain")
        return pd.Series(
            gains, index=booster.feature_name(), name="gain", dtype="float64"
        ).sort_values(ascending=False)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        assert self._model is not None and self._offsets is not None
        features = target_features(info, self.settings, self._names)
        point = np.asarray(self._model.predict(features), dtype="float64")
        hours = local_hours(info.target_index, info.tz)
        quantiles = self.settings.forecasting.quantiles
        raw = pd.DataFrame(
            point[:, None] + self._offsets[hours],
            index=info.target_index,
            columns=quantile_columns(quantiles),
        )
        return make_forecast(self.name, info, raw, quantiles)
