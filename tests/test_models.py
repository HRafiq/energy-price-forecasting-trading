"""Every candidate model honours the forecast contract and cannot leak."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import Forecaster
from src.forecasting.information import build_information_set
from src.forecasting.models.gradient_boosting import (
    DEFAULT_LGBM_PARAMS,
    LightGBMConformalModel,
    LightGBMQuantileModel,
)
from src.forecasting.models.lear import LearModel
from src.forecasting.models.mstl import MSTLModel
from src.forecasting.models.qra import QuantileRegressionAveraging
from src.forecasting.models.quantile_forest import QuantileForestModel
from src.forecasting.walkforward import day_range, run_walk_forward
from tests.fakes import corrupt_unpublished, synthetic_market

TZ = "Europe/Berlin"
TARGET = date(2024, 5, 20)
SMALL_LGBM = {
    **DEFAULT_LGBM_PARAMS,
    "n_estimators": 30,
    "num_leaves": 15,
    "min_child_samples": 20,
}


def _price(index: pd.DatetimeIndex) -> NDArray[np.float64]:
    local = index.tz_convert(TZ)
    clock = np.asarray(local.hour * 60 + local.minute, dtype=float)
    weekday = np.asarray(local.dayofweek, dtype=float)
    noise = np.random.default_rng(11).normal(0.0, 5.0, len(index))
    return 60.0 + 30.0 * np.sin(2 * np.pi * clock / 1440) - 8.0 * (weekday >= 5) + noise


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    from src.config import load_settings

    settings = load_settings()
    frame = synthetic_market(settings, date(2024, 2, 1), 130, _price)
    rng = np.random.default_rng(5)
    for column in frame.columns:
        if column not in (PRICE_SERIES, "price_product_minutes"):
            frame[column] = rng.normal(100.0, 20.0, len(frame))
    return frame


FACTORIES: dict[str, Callable[[Settings], Forecaster]] = {
    "lear": lambda s: LearModel(s, training_days=60, calibration_days=14),
    "lightgbm_quantile": lambda s: LightGBMQuantileModel(
        s, training_days=60, params=dict(SMALL_LGBM)
    ),
    "lightgbm_conformal": lambda s: LightGBMConformalModel(
        s, training_days=60, calibration_days=14, params=dict(SMALL_LGBM)
    ),
    "quantile_forest": lambda s: QuantileForestModel(
        s, training_days=60, n_estimators=20
    ),
    "mstl": lambda s: MSTLModel(s, history_days=21, season_lengths=(96,)),
}


@pytest.mark.parametrize("name", list(FACTORIES))
def test_model_forecasts_honour_the_contract(
    settings: Settings, market: pd.DataFrame, name: str
) -> None:
    model = FACTORIES[name](settings)
    days = day_range(TARGET, TARGET + timedelta(days=2))

    out = run_walk_forward(market, model, days, settings, refit_every_days=7)

    assert out["target_day"].nunique() == 3
    quantiles = out[[c for c in out.columns if c.startswith("q")]].to_numpy()
    assert np.isfinite(quantiles).all()
    assert (np.diff(quantiles, axis=1) >= 0).all()
    # A sensible model is closer to the price than a flat guess at the mean.
    assert (out["q50"] - out["actual"]).abs().mean() < (
        out["actual"] - out["actual"].mean()
    ).abs().mean()


@pytest.mark.parametrize("name", list(FACTORIES))
def test_models_ignore_everything_unpublished_at_issue_time(
    settings: Settings, market: pd.DataFrame, name: str
) -> None:
    dirty = corrupt_unpublished(market, settings, TARGET)
    results = []
    for frame in (market, dirty):
        model = FACTORIES[name](settings)
        model.fit(
            build_information_set(frame, TARGET, settings, model.fit_lookback_days)
        )
        results.append(
            model.forecast(
                build_information_set(frame, TARGET, settings, model.lookback_days)
            )
        )

    pd.testing.assert_frame_equal(results[0].values, results[1].values)


def _members(market: pd.DataFrame) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(9)
    days = pd.DatetimeIndex(market.index).tz_convert(TZ).date
    members = {}
    for name, noise in (("a", 4.0), ("b", 9.0)):
        members[name] = pd.DataFrame(
            {
                "q50": market[PRICE_SERIES] + rng.normal(0.0, noise, len(market)),
                "target_day": days,
            },
            index=market.index,
        )
    return members


def test_qra_combines_members_within_the_contract(
    settings: Settings, market: pd.DataFrame
) -> None:
    model = QuantileRegressionAveraging(settings, _members(market), calibration_days=28)
    out = run_walk_forward(
        market,
        model,
        day_range(TARGET, TARGET + timedelta(days=2)),
        settings,
        refit_every_days=7,
    )

    quantiles = out[[c for c in out.columns if c.startswith("q")]].to_numpy()
    assert (np.diff(quantiles, axis=1) >= 0).all()
    assert (out["q50"] - out["actual"]).abs().mean() < 8.0


def test_qra_ignores_prices_and_member_rows_after_the_gate(
    settings: Settings, market: pd.DataFrame
) -> None:
    members = _members(market)
    dirty_members = {
        name: table.assign(
            q50=table["q50"].where(
                pd.DatetimeIndex(table.index)
                < settings.market.local_midnight_utc(TARGET + timedelta(days=1)),
                -9.0e9,
            )
        )
        for name, table in members.items()
    }
    dirty_market = corrupt_unpublished(market, settings, TARGET)
    results = []
    for frame, pool in ((market, members), (dirty_market, dirty_members)):
        model = QuantileRegressionAveraging(settings, pool, calibration_days=28)
        model.fit(
            build_information_set(frame, TARGET, settings, model.fit_lookback_days)
        )
        results.append(
            model.forecast(
                build_information_set(frame, TARGET, settings, model.lookback_days)
            )
        )

    pd.testing.assert_frame_equal(results[0].values, results[1].values)


def test_qra_stays_stable_when_members_are_nearly_identical(
    settings: Settings, market: pd.DataFrame
) -> None:
    """The reviewer's case: twin members that disagree only on the target day.

    With twins this collinear, unpenalized weights grow into the hundreds and a
    20 EUR/MWh disagreement moves the forecast far from the price. The L1 penalty
    keeps the weights small and the forecast close. Quantiles may still cross
    when the twins diverge; the forecast contract sorts and counts that.
    """
    rng = np.random.default_rng(12)
    base = market[PRICE_SERIES] + rng.normal(0.0, 4.0, len(market))
    twin = base + rng.normal(0.0, 0.01, len(market))
    target_rows = pd.DatetimeIndex(market.index) >= settings.market.local_midnight_utc(
        TARGET
    )
    twin = twin.where(~target_rows, base + 20.0)
    days = pd.DatetimeIndex(market.index).tz_convert(TZ).date
    members = {
        name: pd.DataFrame({"q50": series, "target_day": days}, index=market.index)
        for name, series in (("base", base), ("twin", twin))
    }
    actual = market[PRICE_SERIES]

    def run(alpha: float) -> tuple[float, float]:
        model = QuantileRegressionAveraging(
            settings, members, calibration_days=28, alpha=alpha
        )
        model.fit(
            build_information_set(market, TARGET, settings, model.fit_lookback_days)
        )
        result = model.forecast(
            build_information_set(market, TARGET, settings, model.lookback_days)
        )
        error = float(
            (result.values["q50"] - actual.reindex(result.values.index)).abs().mean()
        )
        largest_weight = max(
            float(np.abs(m.coef_).max()) for m in model._models.values()
        )
        return error, largest_weight

    plain_error, plain_weight = run(alpha=0.0)
    penalized_error, penalized_weight = run(alpha=0.05)

    assert plain_error > 50.0
    assert penalized_error < 30.0
    assert penalized_weight < plain_weight / 5


def test_qra_penalty_does_not_depend_on_the_price_level(
    settings: Settings, market: pd.DataFrame
) -> None:
    """Members are standardized, so pinball loss and the penalty scale together."""
    members = _members(market)
    results = []
    for factor in (1.0, 10.0):
        scaled_market = market.copy()
        scaled_market[PRICE_SERIES] = market[PRICE_SERIES] * factor
        scaled_members = {
            name: table.assign(q50=table["q50"] * factor)
            for name, table in members.items()
        }
        model = QuantileRegressionAveraging(
            settings, scaled_members, calibration_days=28
        )
        model.fit(
            build_information_set(
                scaled_market, TARGET, settings, model.fit_lookback_days
            )
        )
        forecast = model.forecast(
            build_information_set(scaled_market, TARGET, settings, model.lookback_days)
        )
        results.append(forecast.values)

    np.testing.assert_allclose(
        results[1].to_numpy(), 10.0 * results[0].to_numpy(), rtol=1e-3
    )
