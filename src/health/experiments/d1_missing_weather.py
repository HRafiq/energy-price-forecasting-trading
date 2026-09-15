"""D1 experiment: the weather forecast is missing when the daily forecast is issued.

    uv run python -m src.health.experiments.d1_missing_weather
    uv run python -m src.health.experiments.d1_missing_weather --quick

At 11:40 the production model reads archived weather forecasts for the delivery
day. This experiment measures what it costs when they have not arrived, and what
each step of the fallback chain recovers, on the days of the saved Phase 2
comparison run with its 28-day refit schedule:

* ``full``: the production model as in Phase 2. It is recomputed in the same pass
  as ``weather_missing`` and must reproduce the saved comparison forecasts.
* ``weather_missing``: the same fitted model, with every weather column blanked for
  the delivery day in the information set. No mitigation: this is the damage.
* ``fallback_no_weather``: the production model trained and run without the
  weather group, the first step of the fallback chain.
* ``naive_previous_day`` and ``seasonal_naive_previous_week``: the saved baseline
  forecasts, the last steps.

Arms are scored on validation days and traded with median dispatch against perfect
foresight, on the days every arm can trade. ``--quick`` runs the first 100 days,
which keeps the refit anchor and so the reproduction check, and writes nothing to
``docs/`` or MLflow. No day on or after ``evaluation.holdout_start`` is forecast.

Saved arm forecasts are reused only when they cover exactly the requested days
and carry the expected model name; otherwise the arm is recomputed. A reused arm
keeps the runtime recorded by the run that computed it. Results go to
``docs/results/d1_missing_weather.md``, ``data/processed/experiments/`` and
MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, Settings, load_settings
from src.features.build import FEATURE_GROUPS
from src.forecasting.base import Forecaster, QuantileForecast, quantile_columns
from src.forecasting.evaluate import daily_pinball, segment_scores
from src.forecasting.information import InformationSet
from src.forecasting.models.common import feature_names
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.run_comparison import REFIT_EVERY_DAYS
from src.forecasting.walkforward import HoldoutAccessError, run_walk_forward
from src.trading.run_strategies import run_strategies, select_days
from src.trading.strategies import (
    MEDIAN_FORECAST,
    PERFECT_FORESIGHT,
    PRODUCT,
    REALISED,
    Strategy,
)

__all__ = [
    "ARMS",
    "WeatherOutageModel",
    "blank_weather",
    "cached_forecasts",
    "check_days",
    "common_traded_days",
    "main",
    "matches_days",
    "previous_run_seconds",
    "reproduction_gap",
    "reused_run_seconds",
    "summarise",
    "weather_columns",
]

PRODUCTION = "lightgbm_conformal"
OUTAGE_MODEL = "lightgbm_conformal_weather_missing"
FALLBACK_MODEL = "lightgbm_conformal_no_weather"
BASELINES = ("naive_previous_day", "seasonal_naive_previous_week")
ARMS = ("full", "weather_missing", "fallback_no_weather", *BASELINES)
WEATHER_PREFIX = "wx_"
#: Largest difference, in €/MWh, the recomputed ``full`` arm may show against the
#: saved comparison forecasts before the run stops.
REPRODUCTION_TOLERANCE = 0.01
QUICK_DAYS = 100
EXPERIMENT = "d1-missing-weather"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "d1_missing_weather.md"
STRATEGIES: tuple[Strategy, ...] = (PERFECT_FORESIGHT, MEDIAN_FORECAST)


def check_days(days: Sequence[date], settings: Settings) -> None:
    """Refuse any delivery day in the hold-out."""
    holdout = settings.evaluation.holdout_start
    inside = [day for day in days if day >= holdout]
    if inside:
        raise HoldoutAccessError(
            f"{len(inside)} delivery days fall in the hold-out starting {holdout}; "
            "D1 never reads it"
        )


def weather_columns(frame: pd.DataFrame) -> list[str]:
    """Raw weather forecast columns of the dataset."""
    columns = [c for c in frame.columns if str(c).startswith(WEATHER_PREFIX)]
    if not columns:
        raise ValueError("the dataset has no weather columns to blank")
    return columns


def blank_weather(info: InformationSet) -> InformationSet:
    """The information set with every weather value of the delivery day removed."""
    history = info.history.copy()
    columns = weather_columns(history)
    on_target_day = history.index >= info.target_index[0]
    history.loc[on_target_day, columns] = np.nan
    return replace(info, history=history)


@dataclass
class WeatherOutageModel:
    """The production model, forecasting as if the weather feed had not arrived.

    It fits exactly like the production model. Each forecast is made twice: with
    the information set as published, kept in ``published`` for the reproduction
    check, and with the delivery day's weather blanked, which is returned.
    """

    inner: Forecaster
    name: str = OUTAGE_MODEL
    published: list[pd.DataFrame] = field(default_factory=list)

    @property
    def lookback_days(self) -> int | None:
        return self.inner.lookback_days

    @property
    def fit_lookback_days(self) -> int | None:
        return self.inner.fit_lookback_days

    def fit(self, info: InformationSet) -> None:
        self.inner.fit(info)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        full = self.inner.forecast(info)
        values = full.values.copy()
        values["target_day"] = info.target_day
        self.published.append(values)
        return self.inner.forecast(blank_weather(info))


def reproduction_gap(
    recomputed: pd.DataFrame, saved: pd.DataFrame, quantiles: tuple[float, ...]
) -> float:
    """Largest absolute quantile difference between two forecasts of the same days."""
    columns = quantile_columns(quantiles)
    aligned = saved[columns].reindex(recomputed.index)
    if aligned.isna().any().any():
        raise ValueError("the saved forecasts do not cover every recomputed period")
    diff = recomputed[columns].to_numpy(dtype="float64") - aligned.to_numpy(
        dtype="float64"
    )
    return float(np.abs(diff).max())


def common_traded_days(
    arms: dict[str, pd.DataFrame], days: list[date], settings: Settings
) -> list[date]:
    """Days every arm can trade with median dispatch and perfect foresight."""
    columns = {REALISED, PRODUCT}
    for strategy in STRATEGIES:
        columns |= {strategy.sell_column, strategy.buy_column}
    common: set[date] | None = None
    for frame in arms.values():
        selected, _ = select_days(frame, days[0], days[-1], settings, columns)
        common = set(selected) if common is None else common & set(selected)
    return sorted(common or set())


def _trade(
    frame: pd.DataFrame, days: list[date], settings: Settings, workers: int
) -> pd.DataFrame:
    def solve(chosen: list[date], pool: int) -> tuple[pd.DataFrame, dict[date, str]]:
        _, pnl, failed = run_strategies(
            frame,
            chosen,
            STRATEGIES,
            settings.battery,
            holdout_start=settings.evaluation.holdout_start,
            time_limit_s=settings.trading.solver_time_limit_s,
            workers=pool,
        )
        return pnl, failed

    pnl, failed = solve(days, workers)
    if failed:
        # A pooled solve now and then fails for no reason in the model; the same
        # days solve on their own. Retry them once, one at a time.
        retried, still_failed = solve(sorted(failed), 1)
        if still_failed:
            raise RuntimeError(f"solver failed on {sorted(still_failed)}")
        pnl = pd.concat([pnl, retried], ignore_index=True)
    return pnl.sort_values(["target_day", "strategy"], ignore_index=True)


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype="float64"))


def summarise(
    forecasts: dict[str, pd.DataFrame],
    pnl: dict[str, pd.DataFrame],
    settings: Settings,
) -> pd.DataFrame:
    """One row per arm: forecast scores, profit and the cost against ``full``."""
    rows = []
    for arm, frame in forecasts.items():
        scores = segment_scores(frame, settings)
        overall = scores.loc["All target days"]
        trades = pnl[arm]
        ceiling = trades.loc[trades["strategy"] == PERFECT_FORESIGHT.name, "pnl_eur"]
        median = trades.loc[trades["strategy"] == MEDIAN_FORECAST.name, "pnl_eur"]
        rows.append(
            {
                "arm": arm,
                "mean_pinball": _scalar(overall["mean pinball"]),
                "mae_median": _scalar(overall["MAE of median"]),
                "coverage_90": _scalar(overall["coverage 90%"]),
                "spike_day_pinball": _scalar(scores.iloc[-1]["mean pinball"]),
                "pnl_eur": float(median.sum()),
                "perfect_foresight_pnl_eur": float(ceiling.sum()),
                "capture": float(median.sum() / ceiling.sum()),
            }
        )
    table = pd.DataFrame(rows).set_index("arm")
    full = table.loc["full"]
    table["pinball_vs_full"] = table["mean_pinball"] / full["mean_pinball"] - 1
    table["capture_points_vs_full"] = 100 * (table["capture"] - full["capture"])
    table["pnl_eur_vs_full"] = table["pnl_eur"] - full["pnl_eur"]
    return table


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_parquet(partial)
    os.replace(partial, path)


def matches_days(frame: pd.DataFrame, days: Sequence[date], model: str | None) -> bool:
    """True when ``frame`` covers exactly ``days`` and holds only ``model``.

    ``model=None`` checks the days only, for files without a model column.
    """
    if "target_day" not in frame.columns:
        return False
    if sorted(set(frame["target_day"])) != sorted(days):
        return False
    if model is None:
        return True
    return "model" in frame.columns and set(frame["model"]) == {model}


def cached_forecasts(
    path: Path,
    build: Callable[[], pd.DataFrame],
    days: Sequence[date],
    model: str,
    also_valid: Callable[[], bool] = lambda: True,
) -> tuple[pd.DataFrame, bool]:
    """Saved forecasts when they match the request, else ``build()`` and save.

    Returns the forecasts and whether they were reused. ``also_valid`` lets a
    caller require a companion file to match as well.
    """
    if path.exists():
        frame = pd.read_parquet(path)
        if matches_days(frame, days, model) and also_valid():
            print(f"reusing {path.name}", flush=True)
            return frame, True
        print(f"{path.name} does not match the requested days, recomputing", flush=True)
    frame = build()
    _write_parquet(frame, path)
    return frame, False


def previous_run_seconds(summary_path: Path) -> dict[str, float]:
    """``run_seconds`` from an earlier summary JSON; empty when there is none."""
    if not summary_path.exists():
        return {}
    try:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    recorded = payload.get("run_seconds", {})
    if not isinstance(recorded, dict):
        return {}
    return {str(arm): float(seconds) for arm, seconds in recorded.items()}


def reused_run_seconds(
    reused: Mapping[str, bool], earlier: Mapping[str, float]
) -> dict[str, float]:
    """The recorded runtime of every reused arm that has one."""
    return {
        arm: float(earlier[arm])
        for arm, was_reused in reused.items()
        if was_reused and arm in earlier
    }


def _markdown(
    table: pd.DataFrame, days: list[date], scored: list[date], traded: int, gap: float
) -> str:
    def row(arm: str, r: pd.Series) -> str:
        return (
            f"| {arm} | {r['mean_pinball']:.2f} | {r['pinball_vs_full']:+.1%} | "
            f"{r['coverage_90']:.1%} | {r['mae_median']:.2f} | "
            f"{r['spike_day_pinball']:.2f} | {r['capture']:.1%} | "
            f"{r['capture_points_vs_full']:+.1f} | {r['pnl_eur']:,.0f} | "
            f"{r['pnl_eur_vs_full']:+,.0f} |"
        )

    lines = [
        "# D1 experiment: weather forecast missing at issue time",
        "",
        "Generated by `python -m src.health.experiments.d1_missing_weather`. The",
        "production model walk-forward with the Phase 2 refit schedule, on "
        f"{len(days)} days from {days[0]} to {days[-1]};",
        f"scored on {len(scored)} validation days from {scored[0]} to {scored[-1]},",
        f"traded on the {traded} days every arm can trade, 1 MW / 2 MWh battery,",
        "median dispatch against perfect foresight. Prices and errors in €/MWh,",
        "profit in €.",
        "",
        f"The recomputed full arm reproduces the saved comparison forecasts to "
        f"within {gap:.2f} €/MWh.",
        "",
        "| arm | pinball | vs full | 90% coverage | MAE of median | "
        "spike-day pinball | capture | points vs full | median P&L | € vs full |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[row(str(arm), r) for arm, r in table.iterrows()],
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Measure a missing weather feed and the fallback chain."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument(
        "--quick", action="store_true", help=f"first {QUICK_DAYS} days only"
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    settings = load_settings(args.config)
    quantiles = settings.forecasting.quantiles
    comparison = settings.data.processed_path / "forecasts" / "comparison"
    saved = {
        name: pd.read_parquet(comparison / f"{name}.parquet")
        for name in (PRODUCTION, *BASELINES)
    }
    days = sorted(saved[PRODUCTION]["target_day"].unique())
    if args.quick:
        days = days[:QUICK_DAYS]
    check_days(days, settings)
    scored = [day for day in days if day >= settings.evaluation.validation_start]
    if not scored:
        parser.error("no validation days in the selected span")
    out = settings.data.processed_path / "experiments"
    prefix = "d1_quick" if args.quick else "d1"
    refit = REFIT_EVERY_DAYS[PRODUCTION]
    inputs = pd.read_parquet(settings.data.inputs_path)
    weather_columns(inputs)

    timings: dict[str, float] = {}
    summary_path = out / f"{prefix}_missing_weather.json"
    earlier_timings = previous_run_seconds(summary_path)
    full_path = out / f"{prefix}_forecasts_full.parquet"

    def full_file_matches() -> bool:
        # The published forecasts carry no model column; their days must match.
        return full_path.exists() and matches_days(
            pd.read_parquet(full_path), days, None
        )

    def outage_arm() -> pd.DataFrame:
        model = WeatherOutageModel(LightGBMConformalModel(settings))
        started = time.perf_counter()
        outage = run_walk_forward(inputs, model, days, settings, refit_every_days=refit)
        timings["weather_missing"] = time.perf_counter() - started
        published = pd.concat(model.published)
        published.index.name = "timestamp_utc"
        _write_parquet(published, full_path)
        return outage

    def fallback_arm() -> pd.DataFrame:
        groups = [group for group in FEATURE_GROUPS if group != "weather"]
        model = LightGBMConformalModel(
            settings, name=FALLBACK_MODEL, _names=feature_names(groups)
        )
        started = time.perf_counter()
        frame = run_walk_forward(inputs, model, days, settings, refit_every_days=refit)
        timings["fallback_no_weather"] = time.perf_counter() - started
        return frame

    outage, outage_reused = cached_forecasts(
        out / f"{prefix}_forecasts_weather_missing.parquet",
        outage_arm,
        days,
        OUTAGE_MODEL,
        also_valid=full_file_matches,
    )
    recomputed = pd.read_parquet(full_path)
    gap = reproduction_gap(recomputed, saved[PRODUCTION], quantiles)
    print(f"full arm reproduces the saved forecasts within {gap:.2f} €/MWh", flush=True)
    if gap > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"recomputed full arm differs from the saved comparison by {gap:.4f} "
            f"€/MWh, above {REPRODUCTION_TOLERANCE}; the arms are not comparable"
        )
    fallback, fallback_reused = cached_forecasts(
        out / f"{prefix}_forecasts_fallback_no_weather.parquet",
        fallback_arm,
        days,
        FALLBACK_MODEL,
    )
    timings.update(
        reused_run_seconds(
            {"weather_missing": outage_reused, "fallback_no_weather": fallback_reused},
            earlier_timings,
        )
    )

    def scored_only(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["target_day"].isin(scored)]

    forecasts = {
        "full": scored_only(saved[PRODUCTION]),
        "weather_missing": scored_only(outage),
        "fallback_no_weather": scored_only(fallback),
        **{name: scored_only(saved[name]) for name in BASELINES},
    }
    traded = common_traded_days(forecasts, scored, settings)
    print(f"trading {len(traded)} days per arm", flush=True)
    pnl = {
        arm: _trade(frame, traded, settings, args.workers)
        for arm, frame in forecasts.items()
    }
    table = summarise(forecasts, pnl, settings)
    print(table.round(3).to_string(), flush=True)

    daily = pd.concat(
        [
            pd.DataFrame(
                {
                    "arm": arm,
                    "pinball": daily_pinball(frame, quantiles),
                }
            )
            .rename_axis("target_day")
            .reset_index()
            .merge(
                pnl[arm]
                .pivot(index="target_day", columns="strategy", values="pnl_eur")
                .rename(
                    columns={
                        MEDIAN_FORECAST.name: "pnl_eur",
                        PERFECT_FORESIGHT.name: "perfect_foresight_pnl_eur",
                    }
                )
                .reset_index(),
                on="target_day",
                how="left",
            )
            for arm, frame in forecasts.items()
        ],
        ignore_index=True,
    )
    _write_parquet(daily, out / f"{prefix}_missing_weather_daily.parquet")
    payload = {
        "setup": {
            "days": [str(days[0]), str(days[-1])],
            "scored_days": [str(scored[0]), str(scored[-1])],
            "traded_days": len(traded),
            "refit_every_days": refit,
            "battery": settings.battery.model_dump(),
            "reproduction_gap_eur_mwh": gap,
        },
        "arms": {
            str(arm): {key: float(value) for key, value in row.items()}
            for arm, row in table.iterrows()
        },
        "run_seconds": timings,
    }
    partial = summary_path.with_name(f".{summary_path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, summary_path)

    if args.quick:
        print(f"quick run written to {out}", flush=True)
        return 0

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        _markdown(table, days, scored, len(traded), gap), encoding="utf-8"
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(REPO_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    for arm, row in table.iterrows():
        with mlflow.start_run(run_name=str(arm)):
            mlflow.log_params({"arm": str(arm), "refit_every_days": str(refit)})
            mlflow.log_metrics(
                {
                    "mean_pinball": float(row["mean_pinball"]),
                    "coverage_90": float(row["coverage_90"]),
                    "capture": float(row["capture"]),
                    "pnl_eur": float(row["pnl_eur"]),
                }
            )
            if str(arm) in timings:
                mlflow.log_metric("run_seconds", timings[str(arm)])
    print(f"wrote {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
