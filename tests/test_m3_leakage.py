"""The leakage experiment really leaks, and only in its leaky variants."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import PRICE_SERIES, Settings, load_settings
from src.forecasting.models.gradient_boosting import DEFAULT_LGBM_PARAMS
from src.forecasting.walkforward import day_range, run_walk_forward
from src.health.experiments.m3_leakage import VARIANTS, LeakyQuantileModel
from tests.fakes import synthetic_market

TARGET = date(2024, 5, 20)
SMALL = {
    **DEFAULT_LGBM_PARAMS,
    "n_estimators": 60,
    "num_leaves": 15,
    "min_child_samples": 20,
}


def _market() -> pd.DataFrame:
    settings = load_settings()
    frame = synthetic_market(settings, date(2024, 2, 1), 130)
    rng = np.random.default_rng(4)
    for column in frame.columns:
        if column not in (PRICE_SERIES, "price_product_minutes"):
            frame[column] = rng.normal(100.0, 20.0, len(frame))
    # The price is driven by same-period measured wind, unknowable at the gate.
    wind: NDArray[np.float64] = frame["wind_onshore_actual_mw"].to_numpy(
        dtype="float64"
    )
    frame[PRICE_SERIES] = 200.0 - 1.5 * wind
    return frame


def _mae(settings: Settings, frame: pd.DataFrame, columns: tuple[str, ...]) -> float:
    model = LeakyQuantileModel(
        settings, frame, columns, training_days=60, params=dict(SMALL)
    )
    days = day_range(TARGET, TARGET + timedelta(days=2))
    out = run_walk_forward(frame, model, days, settings, refit_every_days=7)
    return float((out["q50"] - out["actual"]).abs().mean())


def test_actuals_variant_uses_delivery_day_values_that_honest_cannot(
    settings: Settings,
) -> None:
    frame = _market()

    honest = _mae(settings, frame, VARIANTS["honest"])
    leaky = _mae(settings, frame, VARIANTS["actuals"])

    assert leaky < honest / 3
