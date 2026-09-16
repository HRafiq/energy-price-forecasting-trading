"""The registry serves back exactly the model it was given, and guards the hold-out."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import ClassVar

import mlflow
import numpy as np
import pandas as pd
import pytest

import src.pipeline.model_source as model_source
from src.config import PRICE_SERIES, Settings, load_settings
from src.forecasting.information import build_information_set
from src.forecasting.models.gradient_boosting import (
    DEFAULT_LGBM_PARAMS,
    LightGBMConformalModel,
)
from src.pipeline.model_source import (
    ModelSourceError,
    load_model,
    register_model,
    registered_versions,
)
from tests.fakes import synthetic_market

TZ = "Europe/Berlin"
SMALL_LGBM = {
    **DEFAULT_LGBM_PARAMS,
    "n_estimators": 30,
    "num_leaves": 15,
    "min_child_samples": 20,
}
#: Two training days well before the hold-out, so the guard is not what is tested.
FIRST_FITTED_THROUGH = date(2024, 5, 19)


def _market(settings: Settings, first_day: date, days: int) -> pd.DataFrame:
    """A synthetic market with a daily price shape and noisy drivers."""
    frame = synthetic_market(settings, first_day, days)
    rng = np.random.default_rng(3)
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
        settings, training_days=60, calibration_days=14, params=dict(SMALL_LGBM)
    )


def _use_store(patch: pytest.MonkeyPatch, path: Path) -> None:
    """Send the module's MLflow calls to a temporary store, never the repo's."""
    patch.setattr(model_source, "TRACKING_URI", f"sqlite:///{path / 'mlflow.db'}")
    patch.setattr(model_source, "ARTIFACTS", path / "mlruns")


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    _use_store(monkeypatch, tmp_path)
    yield tmp_path
    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(registry)


@dataclass(frozen=True)
class _Registry:
    """A temporary store holding two registered versions, and what built them."""

    settings: Settings
    market: pd.DataFrame
    versions: list[str]
    models: list[LightGBMConformalModel]


@pytest.fixture(scope="module")
def two_versions(tmp_path_factory: pytest.TempPathFactory) -> Iterator[_Registry]:
    settings = load_settings()
    market = _market(settings, date(2024, 2, 1), 130)
    tracking, registry = mlflow.get_tracking_uri(), mlflow.get_registry_uri()
    with pytest.MonkeyPatch.context() as patch:
        _use_store(patch, tmp_path_factory.mktemp("registry"))
        versions, models = [], []
        for offset in (0, 1):
            model = _small_model(settings)
            versions.append(
                register_model(
                    settings,
                    fitted_through=FIRST_FITTED_THROUGH + timedelta(days=offset),
                    frame=market,
                    model=model,
                )
            )
            models.append(model)
        yield _Registry(settings, market, versions, models)
    mlflow.set_tracking_uri(tracking)
    mlflow.set_registry_uri(registry)


def test_the_loaded_model_forecasts_exactly_like_the_model_it_was_logged_from(
    two_versions: _Registry,
) -> None:
    newest = two_versions.models[-1]
    target = FIRST_FITTED_THROUGH + timedelta(days=2)
    info = build_information_set(
        two_versions.market, target, two_versions.settings, newest.lookback_days
    )
    expected = newest.forecast(info)

    loaded, version = load_model(two_versions.settings)
    result = loaded.forecast(info)

    assert version == "2"
    assert result.model == expected.model == "lightgbm_conformal"
    assert result.target_day == target
    assert list(result.values.columns) == list(expected.values.columns)
    assert result.values.index.equals(expected.values.index)
    np.testing.assert_allclose(
        result.values.to_numpy(), expected.values.to_numpy(), rtol=0.0, atol=1e-9
    )


def test_the_alias_moves_to_the_newest_version(two_versions: _Registry) -> None:
    assert two_versions.versions == ["1", "2"]

    rows = registered_versions(two_versions.settings)

    assert [row["version"] for row in rows] == ["2", "1"]
    assert rows[0]["alias"] == "production"
    assert rows[1]["alias"] == ""
    assert load_model(two_versions.settings)[1] == "2"


def test_registered_versions_carry_what_the_dashboard_needs(
    two_versions: _Registry,
) -> None:
    rows = registered_versions(two_versions.settings)

    assert len(rows) == 2
    assert set(rows[0]) == {"version", "alias", "created", "fitted_through", "model"}
    assert rows[0]["fitted_through"] == str(FIRST_FITTED_THROUGH + timedelta(days=1))
    assert rows[1]["fitted_through"] == str(FIRST_FITTED_THROUGH)
    assert rows[0]["model"] == "lightgbm_conformal"
    assert pd.Timestamp(rows[0]["created"]).tzinfo is not None


def test_loading_from_an_empty_registry_says_what_to_run(
    settings: Settings, store: Path
) -> None:
    assert registered_versions(settings) == []
    with pytest.raises(ModelSourceError, match="nothing registered"):
        load_model(settings)


@pytest.mark.parametrize("fitted_through", [date(2026, 5, 31), date(2026, 9, 13)])
def test_a_model_that_would_forecast_a_hold_out_day_is_refused(
    settings: Settings, store: Path, fitted_through: date
) -> None:
    # An empty frame: the guard must refuse before any data is read.
    with pytest.raises(ModelSourceError, match="hold-out"):
        register_model(settings, fitted_through=fitted_through, frame=pd.DataFrame())

    assert registered_versions(settings) == []


def test_the_last_hold_out_day_may_train_the_first_live_model(
    settings: Settings, store: Path
) -> None:
    """Fitting through the scored hold-out is how the first live day is forecast."""
    fitted_through = settings.evaluation.holdout_last_day
    market = _market(settings, fitted_through - timedelta(days=119), 120)

    version = register_model(
        settings,
        fitted_through=fitted_through,
        frame=market,
        model=_small_model(settings),
    )

    rows = registered_versions(settings)
    assert version == "1"
    assert rows[0]["fitted_through"] == str(settings.evaluation.holdout_last_day)
    assert str(settings.evaluation.live_from) == str(fitted_through + timedelta(days=1))


def test_the_cli_lists_an_empty_registry_without_registering_anything(
    store: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert model_source.main(["--list"]) == 0

    assert "nothing registered as price-quantile-gbm" in capsys.readouterr().out


def test_a_wrapper_from_another_process_is_served_by_shape(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cloudpickle rebuilds the wrapper class, so identity cannot be the check.

    Registering through the command line pickles the wrapper class by value, and
    loading it elsewhere rebuilds a different class object with the same shape.
    An isinstance check rejected a perfectly good model; the guard now asks for
    what the adapter uses.
    """

    class RebuiltWrapper:
        """The same shape, rebuilt under another module, as cloudpickle does."""

        model_name = "lightgbm_conformal"
        feature_names: ClassVar[list[str]] = ["price_lag_1d"]
        quantiles = settings.forecasting.quantiles

        def predict(self, context: object, model_input: object) -> None: ...

    class Stranger:
        """Something else entirely."""

    served = _fake_registry(monkeypatch, RebuiltWrapper())
    model, version = model_source.load_model(settings)
    assert version == served and model.name == "lightgbm_conformal"

    _fake_registry(monkeypatch, Stranger())
    with pytest.raises(model_source.ModelSourceError, match="cannot serve"):
        model_source.load_model(settings)


def _fake_registry(monkeypatch: pytest.MonkeyPatch, wrapper: object) -> str:
    """Point load_model at a stand-in registry holding ``wrapper``."""

    class FakeVersion:
        version = "7"

    class FakeClient:
        def get_model_version_by_alias(self, name: str, alias: str) -> FakeVersion:
            return FakeVersion()

    class FakePyfunc:
        def unwrap_python_model(self) -> object:
            return wrapper

    monkeypatch.setattr(model_source, "MlflowClient", FakeClient)
    monkeypatch.setattr(mlflow.pyfunc, "load_model", lambda uri: FakePyfunc())
    monkeypatch.setattr(model_source, "_activate_store", lambda: None)
    return "7"
