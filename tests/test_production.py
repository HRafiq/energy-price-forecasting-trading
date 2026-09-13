"""The production path uses the chosen model and keeps every leakage guard."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings, load_settings
from src.forecasting.models.gradient_boosting import (
    DEFAULT_LGBM_PARAMS,
    LightGBMConformalModel,
)
from src.forecasting.production import build_production_model, forecast_day
from src.forecasting.walkforward import HoldoutAccessError
from tests.fakes import corrupt_unpublished, synthetic_market

TZ = "Europe/Berlin"
TARGET = date(2024, 5, 20)
SMALL = {
    **DEFAULT_LGBM_PARAMS,
    "n_estimators": 40,
    "num_leaves": 15,
    "min_child_samples": 20,
}


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    settings = load_settings()
    frame = synthetic_market(settings, date(2024, 2, 1), 130)
    rng = np.random.default_rng(8)
    for column in frame.columns:
        if column not in (PRICE_SERIES, "price_product_minutes"):
            frame[column] = rng.normal(100.0, 20.0, len(frame))
    local = pd.DatetimeIndex(frame.index).tz_convert(TZ)
    clock = np.asarray(local.hour * 60 + local.minute, dtype=float)
    frame[PRICE_SERIES] = (
        60 + 30 * np.sin(2 * np.pi * clock / 1440) + rng.normal(0, 5, len(frame))
    )
    return frame


def _small_model(settings: Settings) -> LightGBMConformalModel:
    return LightGBMConformalModel(
        settings, training_days=60, calibration_days=14, params=dict(SMALL)
    )


def test_the_configured_production_model_is_lightgbm_conformal(
    settings: Settings,
) -> None:
    assert settings.forecasting.production_model == "lightgbm_conformal"
    assert isinstance(build_production_model(settings), LightGBMConformalModel)


def test_production_forecast_covers_the_target_day(
    settings: Settings, market: pd.DataFrame
) -> None:
    result = forecast_day(market, TARGET, settings, model=_small_model(settings))

    assert result.target_day == TARGET
    assert len(result.values) == 96
    assert np.isfinite(result.values.to_numpy()).all()


def test_production_ignores_everything_unpublished(
    settings: Settings, market: pd.DataFrame
) -> None:
    clean = forecast_day(market, TARGET, settings, model=_small_model(settings))
    dirty = forecast_day(
        corrupt_unpublished(market, settings, TARGET),
        TARGET,
        settings,
        model=_small_model(settings),
    )

    pd.testing.assert_frame_equal(clean.values, dirty.values)


def test_production_refuses_hold_out_days(
    settings: Settings, market: pd.DataFrame
) -> None:
    with pytest.raises(HoldoutAccessError):
        forecast_day(
            market,
            settings.evaluation.holdout_start,
            settings,
            model=_small_model(settings),
        )
