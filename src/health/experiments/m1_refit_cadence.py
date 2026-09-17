"""M1 follow-up: refit the production model weekly instead of every 28 days.

    uv run python -m src.health.experiments.m1_refit_cadence
    uv run python -m src.health.experiments.m1_refit_cadence --quick

The regime-shift experiment showed that refitting matters: through the 2021 to 2023
gas crisis a model refitted every 28 days kept 80.6% coverage where a frozen one kept
25.7%. It compared 28 days only against slower cadences. This asks whether faster is
worth it on the validation window, for the production model:

* ``every_28_days``: the Phase 2 cadence, recomputed on the days of the saved
  comparison run. It must reproduce the saved forecasts, which proves both arms read
  the same inputs and differ only in how often they refit.
* ``every_7_days``: the same model on the same days, refitted weekly.

Arms are scored on validation days and traded with median dispatch against perfect
foresight on the days both can trade. Differences are paired by day, with a
moving-block bootstrap over weeks, so a gain has to clear the noise of which weeks
happened to be volatile. ``--quick`` runs the first 100 days and writes nothing to
``docs/`` or MLflow. No day on or after ``evaluation.holdout_start`` is forecast.

Saved arm forecasts are reused only when they cover exactly the requested days and
carry the expected model name. Results go to ``docs/results/m1_refit_cadence.md``,
``data/processed/experiments/`` and MLflow.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, Settings, load_settings
from src.forecasting.evaluate import segment_scores
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.run_comparison import REFIT_EVERY_DAYS
from src.forecasting.walkforward import run_walk_forward
from src.health.experiments.d1_missing_weather import (
    _trade,
    cached_forecasts,
    check_days,
    common_traded_days,
    matches_days,
    reproduction_gap,
)
from src.health.experiments.m1_regime_shift import daily_metrics
from src.trading.backtest import paired_bootstrap_ci
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

__all__ = [
    "CADENCES",
    "CANDIDATE",
    "CONTROL",
    "daily_profit",
    "main",
    "paired_difference",
    "refit_days",
    "results_markdown",
    "summarise",
    "verdict",
]

PRODUCTION = "lightgbm_conformal"
WEEKLY_MODEL = "lightgbm_conformal_weekly_refit"
CONTROL = "every_28_days"
CANDIDATE = "every_7_days"
CADENCES = {CONTROL: REFIT_EVERY_DAYS[PRODUCTION], CANDIDATE: 7}
MODEL_NAMES = {CONTROL: PRODUCTION, CANDIDATE: WEEKLY_MODEL}
REPRODUCTION_TOLERANCE = 0.01
BLOCK_DAYS = 7
BOOTSTRAP_DRAWS = 5000
QUICK_DAYS = 100
EXPERIMENT = "m1-refit-cadence"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "m1_refit_cadence.md"


def refit_days(days: Sequence[date], every: int) -> list[date]:
    """The days the walk-forward refits on, by the harness's own rule.

    A refit happens on the first day and then on any day at least ``every`` calendar
    days after the last refit, so a gap in the days counts as elapsed time.
    """
    fits: list[date] = []
    for day in days:
        if not fits or (day - fits[-1]).days >= every:
            fits.append(day)
    return fits


def paired_difference(
    control: pd.DataFrame,
    candidate: pd.DataFrame,
    column: str,
    *,
    block_days: int = BLOCK_DAYS,
) -> dict[str, float]:
    """Candidate minus control, day by day, with a moving-block bootstrap interval.

    Only days both arms have are paired, so a day one arm could not score cannot
    tilt the comparison.
    """
    left = control.set_index("target_day")[column]
    right = candidate.set_index("target_day")[column]
    both = left.index.intersection(right.index)
    differences = (right.loc[both] - left.loc[both]).dropna().sort_index()
    mean, low, high = paired_bootstrap_ci(
        differences, block_days=block_days, draws=BOOTSTRAP_DRAWS
    )
    return {"mean": mean, "low": low, "high": high, "days": float(len(differences))}


def verdict(result: Mapping[str, float], *, better: str) -> str:
    """Read a paired difference: ``better``, ``worse`` or ``within noise``."""
    if better not in ("lower", "higher"):
        raise ValueError("better must be 'lower' or 'higher'")
    low, high = result["low"], result["high"]
    if better == "lower":
        return "better" if high < 0 else "worse" if low > 0 else "within noise"
    return "better" if low > 0 else "worse" if high < 0 else "within noise"


def daily_profit(pnl: pd.DataFrame) -> pd.DataFrame:
    """Median-dispatch profit and the perfect-foresight ceiling per traded day."""
    # pivot, not pivot_table: a duplicated day or strategy must raise, not be summed.
    wide = pnl.pivot(index="target_day", columns="strategy", values="pnl_eur")
    return pd.DataFrame(
        {
            "target_day": list(wide.index),
            "pnl_eur": wide[MEDIAN_FORECAST.name].to_numpy(dtype="float64"),
            "perfect_foresight_pnl_eur": wide[PERFECT_FORESIGHT.name].to_numpy(
                dtype="float64"
            ),
        }
    )


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype="float64"))


def summarise(
    forecasts: Mapping[str, pd.DataFrame],
    daily: Mapping[str, pd.DataFrame],
    pnl: Mapping[str, pd.DataFrame],
    settings: Settings,
) -> pd.DataFrame:
    """One row per arm: forecast scores, profit and capture."""
    rows = []
    for arm, frame in forecasts.items():
        overall = segment_scores(frame, settings).loc["All target days"]
        profit = daily_profit(pnl[arm])
        earned = float(profit["pnl_eur"].sum())
        ceiling = float(profit["perfect_foresight_pnl_eur"].sum())
        rows.append(
            {
                "arm": arm,
                "refit_every_days": CADENCES[arm],
                "mean_pinball": _scalar(overall["mean pinball"]),
                "coverage_90": _scalar(overall["coverage 90%"]),
                "coverage_50": float(daily[arm]["coverage_50"].mean()),
                "mae_median": _scalar(overall["MAE of median"]),
                "pnl_eur": earned,
                "perfect_foresight_pnl_eur": ceiling,
                "capture": earned / ceiling if ceiling else math.nan,
            }
        )
    return pd.DataFrame(rows).set_index("arm")


def results_markdown(
    table: pd.DataFrame,
    comparison: Mapping[str, Mapping[str, float]],
    *,
    days: Sequence[date],
    scored: Sequence[date],
    traded: int,
    gap: float,
) -> str:
    """The results page: setup, both arms, and the paired differences."""

    def arm_row(arm: str, r: pd.Series) -> str:
        minutes = (
            r["run_seconds"] / 60 if not math.isnan(r["run_seconds"]) else math.nan
        )
        run = f"{minutes:.0f}" if not math.isnan(minutes) else "reused"
        return (
            f"| {arm} | {int(r['refit_every_days'])} | {int(r['fits'])} | "
            f"{r['mean_pinball']:.3f} | {r['coverage_90']:.1%} | "
            f"{r['coverage_50']:.1%} | {r['mae_median']:.2f} | "
            f"{r['capture']:.2%} | {r['pnl_eur']:,.0f} | {run} |"
        )

    pinball, coverage, profit = (
        comparison["pinball"],
        comparison["coverage_90"],
        comparison["pnl_eur"],
    )
    control, candidate = table.loc[CONTROL], table.loc[CANDIDATE]
    total = candidate["pnl_eur"] - control["pnl_eur"]
    points = 100 * (candidate["capture"] - control["capture"])
    ratio = candidate["fits"] / control["fits"]
    lines = [
        "# M1 follow-up: weekly against 28-day refits",
        "",
        "Generated by `python -m src.health.experiments.m1_refit_cadence`. The",
        "production model, LightGBM with conformal ranges, walked forward over the",
        f"{len(days)} days of the saved Phase 2 comparison run "
        f"({days[0]} to {days[-1]}),",
        "once refitted every 28 days and once every 7. Scored on",
        f"{len(scored)} validation days ({scored[0]} to {scored[-1]}), traded on the",
        f"{traded} days both arms can trade: 1 MW / 2 MWh battery, median dispatch",
        "against perfect foresight. Prices and errors in €/MWh, profit in €.",
        "",
        "The recomputed 28-day arm reproduces the saved comparison forecasts "
        + ("exactly" if gap == 0 else f"to within {gap:.2f} €/MWh")
        + ", so both arms read the same inputs and differ only in how often they "
        "refit.",
        "",
        "| arm | refit every (days) | fits | pinball | 90% coverage | 50% coverage | "
        "MAE of median | capture | median P&L | run minutes |",
        "|---|---|---|---|---|---|---|---|---|---|",
        *[arm_row(str(arm), r) for arm, r in table.iterrows()],
        "",
        "## Weekly minus 28 days, paired by day",
        "",
        f"Moving-block bootstrap over {BLOCK_DAYS}-day blocks, "
        f"{BOOTSTRAP_DRAWS:,} draws, 95% interval.",
        "",
        "| measure | mean daily difference | 95% interval | days | reading |",
        "|---|---|---|---|---|",
        f"| pinball, €/MWh (lower is better) | {pinball['mean']:+.3f} | "
        f"{pinball['low']:+.3f} to {pinball['high']:+.3f} | {int(pinball['days'])} | "
        f"{verdict(pinball, better='lower')} |",
        f"| 90% coverage, points | {100 * coverage['mean']:+.2f} | "
        f"{100 * coverage['low']:+.2f} to {100 * coverage['high']:+.2f} | "
        f"{int(coverage['days'])} | not judged: closer to 90% is better |",
        f"| median P&L, € per day (higher is better) | {profit['mean']:+.2f} | "
        f"{profit['low']:+.2f} to {profit['high']:+.2f} | {int(profit['days'])} | "
        f"{verdict(profit, better='higher')} |",
        "",
        f"Over the traded days weekly refitting earned €{abs(total):,.0f} "
        f"{'more' if total >= 0 else 'less'} than refitting every 28 days, "
        f"{points:+.2f} capture points, for {ratio:.1f} times the fits.",
        "",
    ]
    return "\n".join(lines)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(partial, path)


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    frame.to_parquet(partial)
    os.replace(partial, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare weekly against 28-day refits of the production model."
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
    if CADENCES[CONTROL] != 28:
        raise RuntimeError(
            f"the comparison cadence is now {CADENCES[CONTROL]} days, not 28; "
            "rename the control arm before comparing"
        )

    settings = load_settings(args.config)
    quantiles = settings.forecasting.quantiles
    comparison_dir = settings.data.processed_path / "forecasts" / "comparison"
    saved = pd.read_parquet(comparison_dir / f"{PRODUCTION}.parquet")
    days = sorted(saved["target_day"].unique())
    if args.quick:
        days = days[:QUICK_DAYS]
        saved = saved[saved["target_day"].isin(days)]
    check_days(days, settings)
    start = settings.evaluation.validation_start
    scored = [day for day in days if day >= start]
    if not scored:
        parser.error("no validation days in the selected span")

    inputs = pd.read_parquet(settings.data.inputs_path)
    out = settings.data.processed_path / "experiments"
    prefix = "m1_refit_quick" if args.quick else "m1_refit"
    timings: dict[str, float] = {}

    def builder(arm: str) -> Callable[[], pd.DataFrame]:
        def build() -> pd.DataFrame:
            model = (
                LightGBMConformalModel(settings)
                if arm == CONTROL
                else LightGBMConformalModel(settings, name=WEEKLY_MODEL)
            )
            print(f"{arm}: walking forward over {len(days)} days", flush=True)
            started = time.perf_counter()
            frame = run_walk_forward(
                inputs, model, days, settings, refit_every_days=CADENCES[arm]
            )
            timings[arm] = time.perf_counter() - started
            print(f"{arm}: done in {timings[arm] / 60:.1f} min", flush=True)
            return frame

        return build

    paths = {
        arm: out / f"{prefix}_forecasts_{arm}.parquet" for arm in (CONTROL, CANDIDATE)
    }
    candidate_saved = paths[CANDIDATE].exists() and matches_days(
        pd.read_parquet(paths[CANDIDATE]), days, MODEL_NAMES[CANDIDATE]
    )
    if not candidate_saved:
        # Rebuilding one arm on new code or inputs while reusing the other would
        # compare two configurations, so a rebuilt candidate rebuilds the control too.
        paths[CONTROL].unlink(missing_ok=True)

    forecasts: dict[str, pd.DataFrame] = {}
    gap = math.nan
    for arm in (CONTROL, CANDIDATE):
        frame, reused = cached_forecasts(
            paths[arm],
            builder(arm),
            days,
            MODEL_NAMES[arm],
        )
        if reused:
            print(f"{arm}: reused saved forecasts", flush=True)
        forecasts[arm] = frame
        if arm == CONTROL:
            # Checked before the weekly arm runs, so an unfair comparison costs
            # minutes rather than an hour.
            gap = reproduction_gap(frame, saved, quantiles)
            print(f"{CONTROL} reproduces Phase 2 within {gap:.4f} €/MWh", flush=True)
            if gap > REPRODUCTION_TOLERANCE:
                raise RuntimeError(
                    f"recomputed {CONTROL} arm differs from the saved comparison by "
                    f"{gap:.4f} €/MWh, above {REPRODUCTION_TOLERANCE}; the arms "
                    "would not be comparable"
                )

    scored_forecasts = {
        arm: frame[frame["target_day"] >= start] for arm, frame in forecasts.items()
    }
    daily = {arm: daily_metrics(f, settings) for arm, f in scored_forecasts.items()}
    traded = common_traded_days(scored_forecasts, scored, settings)
    print(f"trading {len(traded)} days per arm", flush=True)
    pnl = {
        arm: _trade(frame, traded, settings, args.workers)
        for arm, frame in scored_forecasts.items()
    }
    table = summarise(scored_forecasts, daily, pnl, settings)
    table["fits"] = [len(refit_days(days, CADENCES[arm])) for arm in table.index]
    table["run_seconds"] = [timings.get(str(arm), math.nan) for arm in table.index]
    profit = {arm: daily_profit(frame) for arm, frame in pnl.items()}
    comparison = {
        "pinball": paired_difference(daily[CONTROL], daily[CANDIDATE], "pinball"),
        "coverage_90": paired_difference(
            daily[CONTROL], daily[CANDIDATE], "coverage_90"
        ),
        "pnl_eur": paired_difference(profit[CONTROL], profit[CANDIDATE], "pnl_eur"),
    }

    per_day = pd.concat(
        [
            daily[arm].merge(profit[arm], on="target_day", how="left").assign(arm=arm)
            for arm in (CONTROL, CANDIDATE)
        ],
        ignore_index=True,
    )
    _write_parquet(per_day, out / f"{prefix}_daily.parquet")
    _write_json(
        {
            "setup": {
                "days": [str(days[0]), str(days[-1])],
                "scored_days": [str(scored[0]), str(scored[-1])],
                "traded_days": len(traded),
                "battery": settings.battery.model_dump(),
                "reproduction_gap_eur_mwh": gap,
                "block_days": BLOCK_DAYS,
            },
            "arms": {
                str(arm): {k: float(v) for k, v in row.items()}
                for arm, row in table.iterrows()
            },
            "weekly_minus_28_days": comparison,
            "readings": {
                "pinball": verdict(comparison["pinball"], better="lower"),
                "pnl_eur": verdict(comparison["pnl_eur"], better="higher"),
            },
        },
        out / f"{prefix}_refit_cadence.json",
    )
    print(table.to_string(), flush=True)
    print(json.dumps(comparison, indent=2), flush=True)

    if args.quick:
        print(f"quick run written to {out}", flush=True)
        return 0

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(
        results_markdown(
            table, comparison, days=days, scored=scored, traded=len(traded), gap=gap
        ),
        encoding="utf-8",
    )
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(REPO_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    for name, row in table.iterrows():
        with mlflow.start_run(run_name=str(name)):
            mlflow.log_params(
                {"arm": str(name), "refit_every_days": str(CADENCES[str(name)])}
            )
            mlflow.log_metrics(
                {
                    key: float(row[key])
                    for key in (
                        "mean_pinball",
                        "coverage_90",
                        "coverage_50",
                        "capture",
                        "pnl_eur",
                        "fits",
                    )
                }
            )
            if str(name) in timings:
                mlflow.log_metric("run_seconds", timings[str(name)])
            if name == CANDIDATE:
                for measure, result in comparison.items():
                    mlflow.log_metrics(
                        {f"diff_{measure}_{k}": float(v) for k, v in result.items()}
                    )
    print(f"wrote {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
