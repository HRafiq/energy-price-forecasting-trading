"""The scheduled refit and the time limit that keep the live model fresh and bounded."""

from __future__ import annotations

import time
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import pytest

from src.config import Settings, load_settings
from src.forecasting.base import QuantileForecast
from src.forecasting.information import InformationSet
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.run_comparison import REFIT_EVERY_DAYS
from src.pipeline import chain
from src.pipeline.model_source import (
    load_model,
    refit_due,
    refresh_production_model,
    served_version,
)
from src.pipeline.timelimit import StepTimeoutError, run_with_time_limit
from tests.test_model_source import _market, _small_model, _use_store
from tests.test_pipeline_chain import FakeModel, _readiness, _why

FIRST = date(2024, 5, 20)
EVERY = 7


def test_the_live_cadence_is_the_one_the_backtests_used() -> None:
    settings = load_settings()
    production = settings.forecasting.production_model
    assert settings.pipeline.refit_every_days == REFIT_EVERY_DAYS[production] == 28


@pytest.mark.parametrize(
    ("forecasts_from", "target", "due"),
    [
        (None, FIRST, True),
        (FIRST, FIRST + timedelta(days=27), False),
        (FIRST, FIRST + timedelta(days=28), True),
        (FIRST, FIRST + timedelta(days=40), True),
    ],
)
def test_a_refit_is_due_as_the_walk_forward_counts_it(
    forecasts_from: date | None, target: date, due: bool
) -> None:
    assert refit_due(forecasts_from, target, 28) is due


def test_a_step_returns_raises_or_is_abandoned_on_time() -> None:
    assert run_with_time_limit(lambda: 7, 1.0, "quick") == 7

    def broken() -> int:
        raise ValueError("bad input")

    with pytest.raises(ValueError, match="bad input"):
        run_with_time_limit(broken, 1.0, "broken")

    started = time.perf_counter()
    with pytest.raises(StepTimeoutError, match="hung still running after 0.2 s"):
        run_with_time_limit(lambda: time.sleep(5), 0.2, "hung")
    assert time.perf_counter() - started < 2.0
    with pytest.raises(ValueError, match="positive"):
        run_with_time_limit(lambda: 1, 0.0, "never")


def test_a_rung_past_its_time_limit_is_abandoned_and_the_chain_moves_on(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Hung(FakeModel):
        def fit(self, info: InformationSet) -> None:
            time.sleep(5)

    monkeypatch.setattr(
        chain,
        "STEPS",
        (
            chain.ChainStep("production", "production model", frozenset(), Hung),
            *chain.STEPS[1:],
        ),
    )
    market = _market(settings, FIRST - timedelta(days=60), 62)

    started = time.perf_counter()
    result = chain.run_chain(market, FIRST, settings, _readiness(), time_limit_s=0.3)

    assert time.perf_counter() - started < 4.0, _why(result)
    assert result.step != "production", _why(result)
    hung = next(a for a in result.attempts if a.step == "production")
    assert not hung.used and "abandoned" in hung.detail


@pytest.fixture
def live(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Settings]:
    """Settings refitting weekly, with MLflow sent to a temporary store."""
    tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    _use_store(monkeypatch, tmp_path)
    base = load_settings()
    yield base.model_copy(
        update={
            "pipeline": base.pipeline.model_copy(
                update={"refit_every_days": EVERY, "step_time_limit_s": 60.0}
            )
        }
    )
    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(registry)


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    return _market(load_settings(), date(2024, 2, 1), 130)


def test_the_first_refit_registers_and_the_next_waits_its_turn(
    live: Settings, market: pd.DataFrame
) -> None:
    first = refresh_production_model(live, FIRST, market, model=_small_model(live))

    assert first.status == "refitted", first.detail
    assert (first.previous_version, first.served_version) == (None, "1")
    assert served_version(live) == ("1", FIRST)
    served, version = load_model(live)
    assert version == "1" and served.name == "lightgbm_conformal"

    early = refresh_production_model(
        live, FIRST + timedelta(days=EVERY - 1), market, model=_small_model(live)
    )
    assert early.status == "not_due"
    assert f"due for delivery day {FIRST + timedelta(days=EVERY)}" in early.detail
    assert served_version(live) == ("1", FIRST)

    due = FIRST + timedelta(days=EVERY)
    waiting = refresh_production_model(
        live, due, market, missing_feeds=("weather",), model=_small_model(live)
    )
    assert waiting.status == "postponed" and "weather" in waiting.detail
    assert served_version(live) == ("1", FIRST)

    second = refresh_production_model(live, due, market, model=_small_model(live))
    assert second.status == "refitted", second.detail
    assert (second.previous_version, second.served_version) == ("1", "2")
    assert served_version(live) == ("2", due)


def test_a_refit_that_hangs_or_forecasts_nonsense_leaves_the_served_version(
    live: Settings, market: pd.DataFrame
) -> None:
    assert (
        refresh_production_model(live, FIRST, market, model=_small_model(live)).status
        == "refitted"
    )
    due = FIRST + timedelta(days=EVERY)

    class Slow(LightGBMConformalModel):
        def fit(self, info: InformationSet) -> None:
            time.sleep(3)

    class Nonsense(LightGBMConformalModel):
        def forecast(self, info: InformationSet) -> QuantileForecast:
            result = super().forecast(info)
            result.values.iloc[:, :] = np.nan
            return result

    quick = live.model_copy(
        update={"pipeline": live.pipeline.model_copy(update={"step_time_limit_s": 0.3})}
    )
    slow = refresh_production_model(
        quick, due, market, model=Slow(live, training_days=60, calibration_days=14)
    )
    small = _small_model(live)
    nonsense = refresh_production_model(
        live,
        due,
        market,
        model=Nonsense(
            live,
            training_days=small.training_days,
            calibration_days=small.calibration_days,
            params=small.params,
        ),
    )

    assert slow.status == "failed" and "abandoned" in slow.detail
    assert nonsense.status == "failed" and "non-finite" in nonsense.detail
    assert served_version(live) == ("1", FIRST)


def test_a_refit_never_fits_a_model_that_would_forecast_the_hold_out(
    live: Settings, market: pd.DataFrame
) -> None:
    holdout = live.evaluation.holdout_start + timedelta(days=3)

    outcome = refresh_production_model(live, holdout, market, model=_small_model(live))

    assert outcome.status == "failed" and "hold-out" in outcome.detail
    assert served_version(live) is None


def test_a_served_version_that_hides_its_age_is_reported(
    live: Settings, market: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    import mlflow.tracking

    import src.pipeline.model_source as model_source

    assert (
        refresh_production_model(live, FIRST, market, model=_small_model(live)).status
        == "refitted"
    )
    real = mlflow.tracking.MlflowClient

    class Forgetful(real):  # type: ignore[misc, valid-type]
        def get_run(self, run_id: str):  # type: ignore[no-untyped-def]
            run = super().get_run(run_id)
            run.data.params.pop("forecasts_from", None)
            return run

    monkeypatch.setattr(model_source, "MlflowClient", Forgetful)

    with pytest.raises(model_source.ModelSourceError, match="age is unknown"):
        served_version(live)
