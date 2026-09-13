"""Quantile regression forest: one forest, every quantile from its leaves.

A random forest keeps all training prices in its leaves, so a single model gives
any quantile of the conditional price distribution (Meinshausen 2006). Missing
weather values before 2024 are replaced by a sentinel far outside every
feature's range, which trees can split away cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from quantile_forest import RandomForestQuantileRegressor

from src.config import Settings
from src.features.build import FEATURE_GROUPS
from src.forecasting.base import QuantileForecast, make_forecast, quantile_columns
from src.forecasting.information import InformationSet
from src.forecasting.models.common import (
    FEATURE_HISTORY_DAYS,
    feature_names,
    target_features,
    training_data,
)

__all__ = ["QuantileForestModel"]

MISSING_SENTINEL = -1.0e9


@dataclass
class QuantileForestModel:
    settings: Settings
    name: str = "quantile_forest"
    training_days: int = 730
    n_estimators: int = 150
    min_samples_leaf: int = 40
    max_features: float = 0.4
    max_training_rows: int = 60_000
    _names: list[str] = field(
        default_factory=lambda: feature_names(list(FEATURE_GROUPS))
    )
    _model: RandomForestQuantileRegressor | None = None

    @property
    def lookback_days(self) -> int | None:
        return FEATURE_HISTORY_DAYS

    @property
    def fit_lookback_days(self) -> int | None:
        return self.training_days + FEATURE_HISTORY_DAYS

    def fit(self, info: InformationSet) -> None:
        data = training_data(info, self.settings, self._names, self.training_days)
        features = data.features.to_numpy(dtype="float64")
        target = data.target.to_numpy(dtype="float64")
        if len(target) > self.max_training_rows:
            keep = np.random.default_rng(7).choice(
                len(target), self.max_training_rows, replace=False
            )
            features, target = features[keep], target[keep]
        model = RandomForestQuantileRegressor(
            n_estimators=self.n_estimators,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=7,
            n_jobs=-1,
        )
        model.fit(np.where(np.isfinite(features), features, MISSING_SENTINEL), target)
        self._model = model

    def forecast(self, info: InformationSet) -> QuantileForecast:
        assert self._model is not None
        features = target_features(info, self.settings, self._names).to_numpy(
            dtype="float64"
        )
        quantiles = self.settings.forecasting.quantiles
        predicted = np.asarray(
            self._model.predict(
                np.where(np.isfinite(features), features, MISSING_SENTINEL),
                quantiles=list(quantiles),
            ),
            dtype="float64",
        )
        raw = pd.DataFrame(
            predicted, index=info.target_index, columns=quantile_columns(quantiles)
        )
        return make_forecast(self.name, info, raw, quantiles)
