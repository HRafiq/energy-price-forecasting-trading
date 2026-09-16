"""The MLflow model registry: where the live pipeline gets its forecaster.

    uv run python -m src.pipeline.model_source --fitted-through 2026-09-14
    uv run python -m src.pipeline.model_source --list

``register_model`` fits a fresh production model on the information set of the
day after ``fitted_through`` -- exactly as ``forecasting.production.forecast_day``
would fit it for that day -- logs it to the local MLflow store under experiment
``price-forecast-production``, registers it as ``price-quantile-gbm`` and moves
the ``production`` alias to the new version. ``load_model`` gives the pipeline
back that version as something the ``Forecaster`` protocol accepts, so the daily
run never trains a model of its own and always says which version it served.

**How the model is serialised.** The fitted ``LightGBMConformalModel`` dataclass
-- one LightGBM booster plus the per-hour conformal offsets -- is stored whole
inside a pyfunc wrapper, which MLflow cloudpickles. Nothing about the model is
re-derived on load, so a loaded version predicts bit-for-bit what the in-process
model predicted; ``tests/test_model_source.py`` holds that to 1e-9. The wrapper's
``predict`` takes a feature frame with the model's feature columns plus
``local_hour`` and returns one column per configured quantile, which is all a
served model needs; ``RegisteredForecaster`` below builds that frame from an
information set and rebuilds a ``QuantileForecast`` from the answer, so the
serving path the pipeline uses is the same one MLflow would serve over HTTP.
The alternative -- saving the booster as a text model and the offsets as JSON --
would be more portable but would re-implement the model's own forecast code in
two places; the wrapper keeps one definition of what the model does. The
artifact is loadable only next to this repository, because unpickling imports
``src.forecasting``; that is what a local single-repo pipeline needs, and the
registry records the source commit so a version can be traced back.

Hold-out safety: the hold-out was scored once, through
``evaluation.holdout_last_day``. A model fit through day D is the model that
forecasts D+1, so registering is refused whenever D+1 falls inside the hold-out
window. Fitting through the last hold-out day is allowed and is exactly what the
first live day, ``evaluation.live_from``, needs.
"""

from __future__ import annotations

import argparse
import subprocess
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PyFuncModel
from mlflow.pyfunc.model import PythonModel
from mlflow.tracking import MlflowClient

from src.config import REPO_ROOT, Settings, load_settings
from src.features.build import FEATURE_GROUPS
from src.forecasting.base import (
    Forecaster,
    QuantileForecast,
    make_forecast,
    quantile_columns,
)
from src.forecasting.information import InformationSet, build_information_set
from src.forecasting.models.common import (
    FEATURE_HISTORY_DAYS,
    local_hours,
    target_features,
)
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.production import build_production_model

__all__ = [
    "ARTIFACTS",
    "EXPERIMENT",
    "HOUR_COLUMN",
    "REGISTERED_NAME",
    "TRACKING_URI",
    "ModelSourceError",
    "RegisteredForecaster",
    "load_model",
    "main",
    "register_model",
    "registered_versions",
]

#: The one registered model of this project: the production quantile forecaster.
REGISTERED_NAME = "price-quantile-gbm"
EXPERIMENT = "price-forecast-production"
#: Local MLflow store: metadata in SQLite, artifacts in mlruns/, both ignored by git.
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACTS = REPO_ROOT / "mlruns"
#: Extra column the pyfunc model needs beside its features: the local hour.
HOUR_COLUMN = "local_hour"
#: Artifact path of the model inside its run.
MODEL_ARTIFACT = "model"
#: Declared rather than inferred: inference reloads the model and reads imports.
PIP_REQUIREMENTS = ["lightgbm", "numpy", "pandas", "pydantic", "scikit-learn"]


class ModelSourceError(RuntimeError):
    """The registry cannot serve, or must not accept, a model."""


class _ConformalPyfunc(PythonModel):
    """The fitted conformal model, cloudpickled whole, behind a pyfunc ``predict``."""

    def __init__(self, model: LightGBMConformalModel) -> None:
        if model._model is None or model._offsets is None:
            raise ModelSourceError("fit the model before registering it")
        self.model = model
        self.model_name = model.name
        self.feature_names = list(model._names)
        self.quantiles = tuple(model.settings.forecasting.quantiles)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame,
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        """Quantiles for every row of a feature frame carrying ``HOUR_COLUMN``."""
        missing = [
            name
            for name in (*self.feature_names, HOUR_COLUMN)
            if name not in model_input.columns
        ]
        if missing:
            raise ModelSourceError(f"the feature frame is missing columns {missing}")
        booster, offsets = self.model._model, self.model._offsets
        if booster is None or offsets is None:
            raise ModelSourceError(f"{self.model_name} was stored unfitted")
        point = np.asarray(
            booster.predict(model_input[self.feature_names]), dtype="float64"
        )
        hours = np.asarray(model_input[HOUR_COLUMN].to_numpy(), dtype="int64")
        return pd.DataFrame(
            point[:, None] + offsets[hours],
            index=model_input.index,
            columns=quantile_columns(self.quantiles),
        )


class RegisteredForecaster:
    """A registered version, adapted to the ``Forecaster`` protocol.

    Forecasting goes through the pyfunc model, so the pipeline runs the same
    serving path a deployed copy would. Training is over: ``fit`` refuses, and a
    new training day means a new registered version.
    """

    def __init__(
        self,
        settings: Settings,
        pyfunc: PyFuncModel,
        wrapper: _ConformalPyfunc,
        version: str,
    ) -> None:
        self.settings = settings
        self.version = version
        self.name = wrapper.model_name
        self.feature_names = list(wrapper.feature_names)
        self._pyfunc = pyfunc

    @property
    def lookback_days(self) -> int | None:
        return FEATURE_HISTORY_DAYS

    @property
    def fit_lookback_days(self) -> int | None:
        return FEATURE_HISTORY_DAYS

    def fit(self, info: InformationSet) -> None:
        raise ModelSourceError(
            f"{REGISTERED_NAME} version {self.version} is already fitted; register a "
            "new version instead of refitting the served one"
        )

    def forecast(self, info: InformationSet) -> QuantileForecast:
        features = target_features(info, self.settings, self.feature_names)
        frame = features.copy()
        frame[HOUR_COLUMN] = local_hours(info.target_index, info.tz)
        predicted = self._pyfunc.predict(frame)
        if not isinstance(predicted, pd.DataFrame):
            raise ModelSourceError(
                f"{REGISTERED_NAME} version {self.version} returned "
                f"{type(predicted).__name__}, not a frame of quantiles"
            )
        raw = pd.DataFrame(
            predicted.to_numpy(dtype="float64"),
            index=info.target_index,
            columns=list(predicted.columns),
        )
        return make_forecast(self.name, info, raw, self.settings.forecasting.quantiles)


def _activate_store() -> None:
    """Point MLflow at the local store and make sure the experiment exists."""
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT)


def _source_commit() -> str:
    """The commit the model was fit from, or ``unknown`` outside a checkout."""
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return done.stdout.strip() or "unknown"


def _refuse_holdout(settings: Settings, fitted_through: date, forecasts: date) -> None:
    evaluation = settings.evaluation
    if evaluation.holdout_start <= forecasts <= evaluation.holdout_last_day:
        raise ModelSourceError(
            f"a model fit through {fitted_through} forecasts {forecasts}, inside the "
            f"hold-out {evaluation.holdout_start} to {evaluation.holdout_last_day}; "
            "the hold-out is scored once, and the live pipeline starts at "
            f"{evaluation.live_from}"
        )


def register_model(
    settings: Settings,
    *,
    fitted_through: date,
    alias: str = "production",
    frame: pd.DataFrame | None = None,
    model: LightGBMConformalModel | None = None,
) -> str:
    """Fit, log and register the production model; return the new version.

    ``fitted_through`` is the last local day whose data the model may use, so it
    is fit exactly as the forecast for the next day would fit it. ``frame``
    defaults to the inputs file and ``model`` to the configured production model;
    both are arguments so tests and backfills can pass their own.
    """
    forecasts = fitted_through + timedelta(days=1)
    _refuse_holdout(settings, fitted_through, forecasts)

    forecaster = model if model is not None else build_production_model(settings)
    if not isinstance(forecaster, LightGBMConformalModel):
        raise ModelSourceError(
            f"the registry serialises {LightGBMConformalModel.__name__} only; "
            f"forecasting.production_model is {settings.forecasting.production_model}"
        )
    inputs = pd.read_parquet(settings.data.inputs_path) if frame is None else frame
    forecaster.fit(
        build_information_set(inputs, forecasts, settings, forecaster.fit_lookback_days)
    )
    wrapper = _ConformalPyfunc(forecaster)

    _activate_store()
    with mlflow.start_run(run_name=f"{forecaster.name}-{fitted_through}"):
        mlflow.set_tags(
            {"model": forecaster.name, "window": "live", "registered_alias": alias}
        )
        mlflow.log_params(
            {
                "model": forecaster.name,
                "registered_name": REGISTERED_NAME,
                "fitted_through": str(fitted_through),
                "forecasts_from": str(forecasts),
                "training_days": str(forecaster.training_days),
                "calibration_days": str(forecaster.calibration_days),
                "feature_groups": ",".join(FEATURE_GROUPS),
                "features": str(len(wrapper.feature_names)),
                "quantiles": ",".join(str(q) for q in wrapper.quantiles),
                "lookback_days": str(forecaster.lookback_days),
                "fit_lookback_days": str(forecaster.fit_lookback_days),
                "source_commit": _source_commit(),
            }
        )
        logged = mlflow.pyfunc.log_model(
            name=MODEL_ARTIFACT,
            python_model=wrapper,
            registered_model_name=REGISTERED_NAME,
            pip_requirements=PIP_REQUIREMENTS,
        )
    if logged.registered_model_version is None:
        raise ModelSourceError(
            f"MLflow logged the model but registered no version of {REGISTERED_NAME}"
        )
    version = str(logged.registered_model_version)
    MlflowClient().set_registered_model_alias(REGISTERED_NAME, alias, version)
    return version


def load_model(
    settings: Settings, *, alias: str = "production"
) -> tuple[Forecaster, str]:
    """The aliased version as a forecaster, with the version it came from."""
    _activate_store()
    client = MlflowClient()
    try:
        registered = client.get_model_version_by_alias(REGISTERED_NAME, alias)
        loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@{alias}")
    except MlflowException as exc:
        raise ModelSourceError(
            f"nothing registered as {REGISTERED_NAME!r} with alias {alias!r} in "
            f"{TRACKING_URI}; run `uv run python -m src.pipeline.model_source "
            "--fitted-through <day>` first"
        ) from exc
    wrapper = loaded.unwrap_python_model()
    # Cloudpickle stores the wrapper class by value, so a version registered in
    # another process rebuilds it under that process's module. The class object
    # then differs from this one and isinstance fails on a perfectly good model,
    # which is why the check is on what the adapter actually needs.
    needed = ("model_name", "feature_names", "quantiles")
    if not all(hasattr(wrapper, name) for name in needed) or not callable(
        getattr(wrapper, "predict", None)
    ):
        raise ModelSourceError(
            f"{REGISTERED_NAME} version {registered.version} holds a "
            f"{type(wrapper).__name__}, which this pipeline cannot serve"
        )
    if tuple(wrapper.quantiles) != tuple(settings.forecasting.quantiles):
        raise ModelSourceError(
            f"{REGISTERED_NAME} version {registered.version} was fit for quantiles "
            f"{list(wrapper.quantiles)}, but forecasting.quantiles is now "
            f"{list(settings.forecasting.quantiles)}; register a new version"
        )
    version = str(registered.version)
    return RegisteredForecaster(settings, loaded, wrapper, version), version


def registered_versions(settings: Settings) -> list[dict[str, str]]:
    """Every registered version, newest first, with what the dashboard shows.

    ``settings`` is taken so every entry point of this module reads the same,
    even though the store's location comes from the repository.
    """
    _activate_store()
    client = MlflowClient()
    try:
        registered = client.get_registered_model(REGISTERED_NAME)
        versions = list(client.search_model_versions(f"name='{REGISTERED_NAME}'"))
    except MlflowException:
        return []
    # Aliases hang off the registered model; search results do not carry them.
    aliases: dict[str, list[str]] = {}
    for alias, number in (registered.aliases or {}).items():
        aliases.setdefault(str(number), []).append(alias)
    rows: list[dict[str, str]] = []
    for version in sorted(versions, key=lambda v: int(v.version), reverse=True):
        params: dict[str, str] = {}
        if version.run_id:
            try:
                params = client.get_run(version.run_id).data.params
            except MlflowException:
                params = {}
        rows.append(
            {
                "version": str(version.version),
                "alias": ",".join(sorted(aliases.get(str(version.version), ()))),
                "created": pd.Timestamp(
                    version.creation_timestamp, unit="ms", tz="UTC"
                ).isoformat(),
                "fitted_through": params.get("fitted_through", ""),
                "model": params.get("model", ""),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register the production model, or list what is registered."
    )
    parser.add_argument(
        "--fitted-through",
        type=date.fromisoformat,
        default=None,
        help="last local day whose data the model may use",
    )
    parser.add_argument("--alias", default="production")
    parser.add_argument(
        "--list", action="store_true", help="print the registered versions and stop"
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    if args.list:
        rows = registered_versions(settings)
        if not rows:
            print(f"nothing registered as {REGISTERED_NAME}")
        for row in rows:
            alias = f" @{row['alias']}" if row["alias"] else ""
            print(
                f"{REGISTERED_NAME} v{row['version']}{alias}: {row['model']} "
                f"fitted through {row['fitted_through']}, created {row['created']}"
            )
        return 0
    if args.fitted_through is None:
        parser.error("pass --fitted-through <day>, or --list")

    version = register_model(
        settings, fitted_through=args.fitted_through, alias=args.alias
    )
    print(
        f"registered {REGISTERED_NAME} version {version} as @{args.alias}, "
        f"fit through {args.fitted_through} for {args.fitted_through + timedelta(1)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
