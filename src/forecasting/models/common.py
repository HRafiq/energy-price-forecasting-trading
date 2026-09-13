"""Shared plumbing for the feature-based candidate models.

``fit`` builds features from the training history, keeps the last
``training_days`` days with a published price, and hands features and prices to
the learner. ``forecast`` builds features for the target day from a short
history. Both only ever read an information set.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import PRICE_SERIES, Settings
from src.features.build import FEATURE_GROUPS, build_features
from src.forecasting.base import ForecastError
from src.forecasting.information import InformationSet

__all__ = [
    "FEATURE_HISTORY_DAYS",
    "TrainingData",
    "feature_names",
    "hourly_offsets",
    "local_hours",
    "split_calibration",
    "target_features",
    "training_data",
]

#: History the features need before a target day: seven-day lags, day X-1
#: statistics and the measured-value window, with a margin.
FEATURE_HISTORY_DAYS = 10


def feature_names(groups: Sequence[str]) -> list[str]:
    unknown = [g for g in groups if g not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(f"unknown feature groups {unknown}")
    return [name for group in groups for name in FEATURE_GROUPS[group]]


def local_hours(index: pd.DatetimeIndex, tz: str) -> NDArray[np.int64]:
    return np.asarray(index.tz_convert(tz).hour, dtype="int64")


@dataclass(frozen=True)
class TrainingData:
    features: pd.DataFrame
    target: pd.Series
    days: NDArray[np.object_]
    hours: NDArray[np.int64]


def training_data(
    info: InformationSet,
    settings: Settings,
    names: Sequence[str],
    training_days: int,
) -> TrainingData:
    """Feature rows and published prices for the last ``training_days`` days."""
    features = build_features(info.history, settings)[list(names)]
    target = info.history[PRICE_SERIES]
    index = pd.DatetimeIndex(features.index)
    start = settings.market.local_midnight_utc(
        info.target_day - timedelta(days=training_days)
    )
    rows = (
        target.notna().to_numpy()
        & np.asarray(index >= start)
        & np.asarray(index < info.target_index[0])
    )
    if not rows.any():
        raise ForecastError(f"no training rows before {info.target_day}")
    kept = index[rows]
    return TrainingData(
        features=features.loc[rows],
        target=target.loc[rows],
        days=np.asarray(kept.tz_convert(info.tz).date, dtype=object),
        hours=local_hours(kept, info.tz),
    )


def target_features(
    info: InformationSet, settings: Settings, names: Sequence[str]
) -> pd.DataFrame:
    """Feature rows for every period of the target day."""
    features = build_features(info.history, settings)
    return features.reindex(info.target_index)[list(names)]


def split_calibration(
    data: TrainingData, calibration_days: int
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    """Masks for fitting rows and for the last ``calibration_days`` days."""
    last_day: date = max(data.days)
    first_calibration_day = last_day - timedelta(days=calibration_days - 1)
    calibration = np.asarray(
        [day >= first_calibration_day for day in data.days], dtype=bool
    )
    if calibration.all() or not calibration.any():
        raise ForecastError("training window too short for a calibration split")
    return ~calibration, calibration


def hourly_offsets(
    residuals: NDArray[np.float64],
    hours: NDArray[np.int64],
    quantiles: tuple[float, ...],
    min_per_hour: int = 20,
) -> NDArray[np.float64]:
    """Residual quantiles per local hour, shape (24, quantiles); sparse hours pool."""
    ok = np.isfinite(residuals)
    if not ok.any():
        raise ForecastError("no residuals to calibrate the quantiles")
    series = pd.Series(residuals[ok])
    by_hour = series.groupby(hours[ok])
    counts = by_hour.size().reindex(range(24), fill_value=0).to_numpy()
    offsets = np.empty((24, len(quantiles)))
    for j, q in enumerate(quantiles):
        hourly = by_hour.quantile(q).reindex(range(24)).to_numpy(dtype="float64")
        offsets[:, j] = np.where(
            counts >= min_per_hour, hourly, float(series.quantile(q))
        )
    return offsets
