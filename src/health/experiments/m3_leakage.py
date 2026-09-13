"""M3 experiment: how much information from after the gate would flatter a backtest.

    uv run python -m src.health.experiments.m3_leakage

The same LightGBM quantile model is trained walk-forward three times, on the
same days, with three feature sets:

* ``honest``: only features published by 11:40 on the day before delivery, the
  set every candidate model uses;
* ``post_gate_forecasts``: adds the grid operators' onshore wind, offshore wind,
  solar and residual-load forecasts for the delivery day itself, which arrive at
  18:00, six hours after the gate;
* ``actuals``: adds measured wind, solar and load for the delivery day, the
  classic leak.

The leaked columns are read deliberately from the full dataset, outside the
information set, which is exactly what the production harness forbids. The gap
between ``honest`` and the other two is the flattery a leaky backtest would put
on the dashboard. Results go to ``docs/results/m3_leakage.md`` and MLflow.
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.config import REPO_ROOT, Settings, load_settings
from src.forecasting.base import QuantileForecast, make_forecast, quantile_columns
from src.forecasting.evaluate import segment_scores
from src.forecasting.information import InformationSet
from src.forecasting.models.common import (
    FEATURE_HISTORY_DAYS,
    target_features,
    training_data,
)
from src.forecasting.models.gradient_boosting import DEFAULT_LGBM_PARAMS, _all_features
from src.forecasting.walkforward import day_range, run_walk_forward

__all__ = ["VARIANTS", "LeakyQuantileModel", "main"]

VARIANTS: dict[str, tuple[str, ...]] = {
    "honest": (),
    "post_gate_forecasts": (
        "wind_onshore_forecast_mw",
        "wind_offshore_forecast_mw",
        "solar_forecast_mw",
        "residual_load_forecast_mw",
    ),
    "actuals": (
        "wind_onshore_actual_mw",
        "wind_offshore_actual_mw",
        "solar_actual_mw",
        "load_actual_mw",
        "residual_load_actual_mw",
    ),
}
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "m3_leakage.md"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT = "m3-leakage"


@dataclass
class LeakyQuantileModel:
    """LightGBM quantile model with delivery-day columns taken from the full data."""

    settings: Settings
    full_inputs: pd.DataFrame
    leaked_columns: tuple[str, ...]
    name: str = "lightgbm_quantile_leak_test"
    training_days: int = 730
    params: dict[str, Any] = field(default_factory=lambda: dict(DEFAULT_LGBM_PARAMS))
    _models: dict[float, LGBMRegressor] = field(default_factory=dict)

    @property
    def lookback_days(self) -> int | None:
        return FEATURE_HISTORY_DAYS

    @property
    def fit_lookback_days(self) -> int | None:
        return self.training_days + FEATURE_HISTORY_DAYS

    def _with_leak(self, features: pd.DataFrame) -> pd.DataFrame:
        if not self.leaked_columns:
            return features
        leaked = self.full_inputs[list(self.leaked_columns)].reindex(features.index)
        return pd.concat([features, leaked.add_prefix("leaked_")], axis=1)

    def fit(self, info: InformationSet) -> None:
        data = training_data(info, self.settings, _all_features(), self.training_days)
        features = self._with_leak(data.features)
        self._models = {}
        for q in self.settings.forecasting.quantiles:
            model = LGBMRegressor(objective="quantile", alpha=q, **self.params)
            model.fit(features, data.target)
            self._models[q] = model

    def forecast(self, info: InformationSet) -> QuantileForecast:
        features = self._with_leak(
            target_features(info, self.settings, _all_features())
        )
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


def _row(scores: pd.DataFrame) -> dict[str, str]:
    overall = scores.loc["All target days"]
    return {
        "mean pinball": f"{overall['mean pinball']:.2f}",
        "MAE of median": f"{overall['MAE of median']:.2f}",
        "coverage 90%": f"{overall['coverage 90%']:.1%}",
        "spike days, mean pinball": f"{scores.iloc[-1]['mean pinball']:.2f}",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure what leaked features would fake."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--first-day", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    inputs = pd.read_parquet(settings.data.inputs_path)
    last = settings.evaluation.holdout_start - timedelta(days=1)
    first = args.first_day or last - timedelta(days=364)
    days = day_range(first, last)

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(REPO_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)

    rows: dict[str, dict[str, str]] = {}
    for variant, columns in VARIANTS.items():
        model = LeakyQuantileModel(settings, inputs, columns)
        started = time.perf_counter()
        forecasts = run_walk_forward(inputs, model, days, settings, refit_every_days=28)
        seconds = time.perf_counter() - started
        scores = segment_scores(forecasts, settings)
        rows[variant] = _row(scores)
        with mlflow.start_run(run_name=variant):
            mlflow.log_params(
                {"variant": variant, "leaked_columns": ",".join(columns) or "none"}
            )
            mlflow.log_metric(
                "mean_pinball",
                float(
                    np.asarray(
                        scores.loc["All target days", "mean pinball"], dtype="float64"
                    )
                ),
            )
            mlflow.log_metric("run_seconds", seconds)
        print(
            f"{variant}: {rows[variant]['mean pinball']} in {seconds:.0f} s", flush=True
        )

    table = pd.DataFrame(rows).T
    lines = [
        "# M3 leakage experiment",
        "",
        "Generated by `python -m src.health.experiments.m3_leakage`. The same LightGBM",
        "quantile model, walk-forward with a 28-day refit, on three feature sets.",
        f"Target days {days[0]} to {days[-1]}. Prices and errors in €/MWh.",
        "",
        "| feature set | " + " | ".join(table.columns) + " |",
        "|---|" + "---|" * len(table.columns),
        *[
            f"| {name} | " + " | ".join(row.tolist()) + " |"
            for name, row in table.iterrows()
        ],
        "",
    ]
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
