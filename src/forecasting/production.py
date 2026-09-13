"""The production price forecaster: one model, chosen in Phase 2.

    uv run python -m src.forecasting.production --day 2026-05-31

``forecasting.production_model`` names the model; the choice and its evidence are
in notebooks/model_comparison.ipynb and docs/decisions.md. For a target day the
model trains on the information set as of 11:40 the day before and forecasts
every quarter-hour of that day. The command writes the forecast to
``data/processed/forecasts/production/<day>.parquet``.

Days in the hold-out are refused unless ``allow_holdout`` is set, so no
production run can peek at the final test period before Phase 4.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd

from src.config import Settings, load_settings
from src.forecasting.base import Forecaster, QuantileForecast
from src.forecasting.baselines import build_baselines
from src.forecasting.information import build_information_set
from src.forecasting.models.gradient_boosting import (
    LightGBMConformalModel,
    LightGBMQuantileModel,
)
from src.forecasting.models.lear import LearModel
from src.forecasting.models.mstl import MSTLModel
from src.forecasting.models.quantile_forest import QuantileForestModel
from src.forecasting.walkforward import HoldoutAccessError

__all__ = ["PRODUCTION_MODELS", "build_production_model", "forecast_day", "main"]

PRODUCTION_MODELS: dict[str, Callable[[Settings], Forecaster]] = {
    "lightgbm_conformal": LightGBMConformalModel,
    "lightgbm_quantile": LightGBMQuantileModel,
    "quantile_forest": QuantileForestModel,
    "lear": LearModel,
    "mstl": MSTLModel,
    "naive_previous_day": lambda s: build_baselines(s)[0],
    "seasonal_naive_previous_week": lambda s: build_baselines(s)[1],
}


def build_production_model(settings: Settings) -> Forecaster:
    """A fresh instance of the configured production model."""
    return PRODUCTION_MODELS[settings.forecasting.production_model](settings)


def forecast_day(
    frame: pd.DataFrame,
    target_day: date,
    settings: Settings,
    model: Forecaster | None = None,
    allow_holdout: bool = False,
) -> QuantileForecast:
    """Train on what was known at 11:40 the day before, then forecast ``target_day``."""
    if not allow_holdout and target_day >= settings.evaluation.holdout_start:
        raise HoldoutAccessError(
            f"{target_day} is in the hold-out starting "
            f"{settings.evaluation.holdout_start}"
        )
    forecaster = model or build_production_model(settings)
    forecaster.fit(
        build_information_set(frame, target_day, settings, forecaster.fit_lookback_days)
    )
    return forecaster.forecast(
        build_information_set(frame, target_day, settings, forecaster.lookback_days)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forecast one day with the production model."
    )
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    frame = pd.read_parquet(settings.data.inputs_path)
    result = forecast_day(frame, args.day, settings)

    out_dir = settings.data.processed_path / "forecasts" / "production"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = result.values.copy()
    table.insert(0, "model", result.model)
    table["target_day"] = result.target_day
    table["issue_time_utc"] = result.issue_time_utc
    path = out_dir / f"{args.day.isoformat()}.parquet"
    table.to_parquet(path)
    print(
        f"{result.model}: {len(table)} periods for {args.day}, "
        f"issued {result.issue_time_utc}"
    )
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
