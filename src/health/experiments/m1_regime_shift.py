"""M1 experiment: what a frozen model costs when the market changes under it.

    uv run python -m src.health.experiments.m1_regime_shift
    uv run python -m src.health.experiments.m1_regime_shift --arms frozen --workers 4

The production model, LightGBM with conformal ranges, forecasts every day from
2021-01-01 to 2023-12-31, the gas crisis and the two years around it, in four
arms on identical days:

* ``frozen``: fitted once, on the 730 days before the first target day, and never
  refitted, the model nobody retrains;
* ``quarterly``: refitted every 91 days;
* ``monthly``: refitted every 28 days, the production cadence, as a reference;
* ``naive_previous_day``: the previous-day baseline, refitted daily, for context.

The three LightGBM arms use default parameters, 730 training days, 42
calibration days and every feature group except weather, whose archived
forecasts only start in March 2024. Every forecast goes through the walk-forward
harness and so only ever reads the information set at 11:40 on the day before.

Each day is scored on 90% and 50% interval coverage, mean pinball loss and the
MAE of the median, and flagged as a spike day when any price exceeds
``evaluation.spike_threshold_eur_mwh``. The reference battery then trades each
arm's median forecast and, as the ceiling, perfect foresight. Scores and P&L are
aggregated per year, per calendar quarter and over a rolling 28-day window.

Each arm's forecasts are saved as soon as they are complete, so a rerun skips
finished arms. Results go to ``data/processed/experiments/`` (forecasts per arm,
daily scores and the dashboard JSON), ``docs/results/m1_regime_shift.md`` and
MLflow. A run over other days writes beside the main outputs and skips the
report and MLflow. No day on or after ``evaluation.holdout_start`` is read.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.config import (
    GAS_COLUMN,
    PRICE_SERIES,
    PRODUCT_COLUMN,
    REPO_ROOT,
    Settings,
    load_settings,
)
from src.features.build import FEATURE_GROUPS
from src.forecasting.base import Forecaster
from src.forecasting.baselines import build_baselines
from src.forecasting.evaluate import score
from src.forecasting.models.common import feature_names
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.walkforward import HoldoutAccessError, day_range, run_walk_forward
from src.trading.run_strategies import run_strategies
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT, REALISED

__all__ = [
    "ARMS",
    "Arm",
    "build_forecaster",
    "check_days",
    "daily_metrics",
    "dispatch_pnl",
    "experiment_features",
    "experiment_payload",
    "load_checkpoint",
    "main",
    "min_rolling",
    "period_table",
    "quarter_label",
    "refit_interval",
    "rolling_coverage",
    "run_arm",
    "tradable_days",
    "training_range",
    "write_atomically",
]


@dataclass(frozen=True)
class Arm:
    """One way of keeping the model up to date; ``None`` never refits."""

    name: str
    refit_every_days: int | None
    description: str
    baseline: bool = False


ARMS: tuple[Arm, ...] = (
    Arm("frozen", None, "fitted once before the first target day, never refitted"),
    Arm("quarterly", 91, "refitted every 91 days"),
    Arm("monthly", 28, "refitted every 28 days, the production cadence"),
    Arm(
        "naive_previous_day",
        1,
        "previous day's prices with hourly error quantiles, for context",
        baseline=True,
    ),
)
ARM_NAMES = tuple(arm.name for arm in ARMS)
FIRST_DAY = date(2021, 1, 1)
LAST_DAY = date(2023, 12, 31)
TRAINING_DAYS = 730
CALIBRATION_DAYS = 42
EXCLUDED_GROUPS = ("weather",)
ROLLING_DAYS = 28
COVERAGE_TARGET = 0.90
#: Expectation recorded before M1 was run. The report states it and sets the
#: measured coverage against it; nothing in the experiment is tuned towards it.
EXPECTED_FROZEN_COVERAGE = 0.58
EXPECTED_QUARTERLY_RANGE = (0.86, 0.90)

EXPERIMENT = "m1-regime-shift"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "m1_regime_shift.md"
PREFIX = "m1_regime"
DAILY_COLUMNS = (
    "target_day",
    "arm",
    "coverage_90",
    "coverage_50",
    "pinball",
    "mae_q50",
    "spike_day",
    "pnl_eur",
    "perfect_foresight_pnl_eur",
)
METRICS = ("coverage_90", "coverage_50", "pinball", "mae_q50")


# --- arms and forecasts -------------------------------------------------------


def experiment_features() -> list[str]:
    """Every feature group except weather, which has no history before 2024."""
    return feature_names([g for g in FEATURE_GROUPS if g not in EXCLUDED_GROUPS])


def build_forecaster(arm: Arm, settings: Settings) -> Forecaster:
    if arm.baseline:
        baseline = build_baselines(settings)[0]
        if baseline.name != arm.name:
            raise ValueError(f"expected baseline {arm.name}, got {baseline.name}")
        return baseline
    return LightGBMConformalModel(
        settings,
        training_days=TRAINING_DAYS,
        calibration_days=CALIBRATION_DAYS,
        _names=experiment_features(),
    )


def refit_interval(arm: Arm, days: Sequence[date]) -> int:
    """Days between refits; a frozen arm gets an interval longer than the span."""
    if arm.refit_every_days is not None:
        return arm.refit_every_days
    return (days[-1] - days[0]).days + 1


def check_days(days: Sequence[date], settings: Settings) -> None:
    """Refuse any target day in the hold-out."""
    holdout = settings.evaluation.holdout_start
    inside = [day for day in days if day >= holdout]
    if inside:
        raise HoldoutAccessError(
            f"{len(inside)} target days fall in the hold-out starting {holdout}; "
            "M1 never reads it"
        )


def run_arm(
    frame: pd.DataFrame,
    arm: Arm,
    days: Sequence[date],
    settings: Settings,
    forecaster: Forecaster | None = None,
) -> pd.DataFrame:
    """Walk ``arm`` forward over ``days``; the forecasts carry an ``arm`` column."""
    check_days(days, settings)
    model = forecaster if forecaster is not None else build_forecaster(arm, settings)
    holdout = settings.market.local_midnight_utc(settings.evaluation.holdout_start)
    forecasts = run_walk_forward(
        frame.loc[frame.index < holdout],
        model,
        days,
        settings,
        refit_every_days=refit_interval(arm, days),
    )
    forecasts.insert(1, "arm", arm.name)
    return forecasts


def write_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Write to a temporary file beside ``path`` and swap it in."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    write(partial)
    os.replace(partial, path)


def load_checkpoint(path: Path, days: Sequence[date]) -> pd.DataFrame | None:
    """Saved forecasts, only when they cover exactly the requested days."""
    if not path.exists():
        return None
    forecasts = pd.read_parquet(path)
    saved = sorted(set(forecasts["target_day"]))
    return forecasts if saved == list(days) else None


def _forecast_path(out_dir: Path, arm: str) -> Path:
    return out_dir / f"{PREFIX}_forecasts_{arm}.parquet"


def _forecast_job(job: tuple[str, str | None, date, date, str]) -> tuple[str, float]:
    """Pool worker: forecast one arm and save it; returns the arm and seconds."""
    name, config, first, last, out_dir = job
    settings = load_settings(Path(config) if config is not None else None)
    frame = pd.read_parquet(settings.data.inputs_path)
    arm = next(a for a in ARMS if a.name == name)
    started = time.perf_counter()
    forecasts = run_arm(frame, arm, day_range(first, last), settings)
    seconds = time.perf_counter() - started
    path = _forecast_path(Path(out_dir), name)
    write_atomically(path, forecasts.to_parquet)
    return name, seconds


# --- scores -------------------------------------------------------------------


def daily_metrics(forecasts: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Coverage, pinball, MAE of the median and the spike flag per target day."""
    quantiles = settings.forecasting.quantiles
    threshold = settings.evaluation.spike_threshold_eur_mwh
    rows: list[dict[str, object]] = []
    for day, part in forecasts.groupby("target_day", sort=True):
        scores = score(part, quantiles)
        rows.append(
            {
                "target_day": day,
                "coverage_90": scores.get("coverage 90%", math.nan),
                "coverage_50": scores.get("coverage 50%", math.nan),
                "pinball": scores.get("mean pinball", math.nan),
                "mae_q50": scores.get("MAE of median", math.nan),
                "spike_day": bool(part[REALISED].max() > threshold),
            }
        )
    return pd.DataFrame(rows)


def tradable_days(
    forecasts: pd.DataFrame, settings: Settings
) -> tuple[list[date], dict[date, str]]:
    """Days with every period, a finite median and price, and one product length."""
    selected: list[date] = []
    skipped: dict[date, str] = {}
    for key, part in forecasts.groupby("target_day", sort=True):
        if not isinstance(key, date):
            raise TypeError(f"target_day must hold dates, got {key!r}")
        day = key
        expected = pd.date_range(
            settings.market.local_midnight_utc(day),
            settings.market.local_midnight_utc(day + timedelta(days=1)),
            freq="15min",
            inclusive="left",
        )
        values = part[[MEDIAN_FORECAST.sell_column, REALISED]].to_numpy(dtype=float)
        if not part.index.sort_values().equals(expected):
            skipped[day] = "incomplete periods"
        elif not np.isfinite(values).all():
            skipped[day] = "missing price or forecast"
        elif part[PRODUCT_COLUMN].nunique() != 1:
            skipped[day] = "mixed product lengths"
        else:
            selected.append(day)
    return selected, skipped


def dispatch_pnl(
    forecasts: pd.DataFrame,
    days: list[date],
    settings: Settings,
    *,
    workers: int = 1,
    runner: Callable[..., tuple[pd.DataFrame, pd.DataFrame, dict[date, str]]] = (
        run_strategies
    ),
) -> pd.DataFrame:
    """Daily P&L of median dispatch and perfect foresight with the reference battery.

    A pooled solve now and then fails for no reason in the model; failed days are
    retried once, one at a time, before giving up.
    """
    check_days(days, settings)
    strategies = (PERFECT_FORESIGHT, MEDIAN_FORECAST)
    kwargs: dict[str, Any] = {
        "holdout_start": settings.evaluation.holdout_start,
        "time_limit_s": settings.trading.solver_time_limit_s,
    }
    _, pnl, failed = runner(
        forecasts, days, strategies, settings.battery, workers=workers, **kwargs
    )
    if failed:
        _, retried, still_failed = runner(
            forecasts, sorted(failed), strategies, settings.battery, workers=1, **kwargs
        )
        if still_failed:
            raise RuntimeError(f"solver failed on {still_failed}")
        pnl = pd.concat([pnl, retried], ignore_index=True)
    wide = pnl.pivot(index="target_day", columns="strategy", values="pnl_eur")
    return pd.DataFrame(
        {
            "target_day": wide.index,
            "pnl_eur": wide[MEDIAN_FORECAST.name].to_numpy(dtype="float64"),
            "perfect_foresight_pnl_eur": wide[PERFECT_FORESIGHT.name].to_numpy(
                dtype="float64"
            ),
        }
    )


def quarter_label(day: date) -> str:
    return f"{day.year}Q{(day.month - 1) // 3 + 1}"


def rolling_coverage(
    daily: pd.DataFrame, window: int = ROLLING_DAYS, column: str = "coverage_90"
) -> pd.DataFrame:
    """Mean daily ``column`` over the trailing ``window`` calendar days, per arm.

    Indexed by the last day of the window; a window missing any day is NaN.
    """
    first, last = min(daily["target_day"]), max(daily["target_day"])
    calendar = pd.DatetimeIndex(pd.to_datetime(day_range(first, last)))
    series: dict[str, pd.Series] = {}
    for arm, part in daily.groupby("arm", sort=False):
        values = pd.Series(
            part[column].to_numpy(dtype="float64"),
            index=pd.DatetimeIndex(pd.to_datetime(list(part["target_day"]))),
        ).reindex(calendar)
        series[str(arm)] = values.rolling(window, min_periods=window).mean()
    out = pd.DataFrame(series, index=calendar)
    out.index.name = "window_end"
    return out


def min_rolling(rolling: pd.DataFrame) -> dict[str, tuple[float, date]]:
    """The lowest rolling value per arm and the last day of that window."""
    result: dict[str, tuple[float, date]] = {}
    for arm in rolling.columns:
        values = rolling[arm].dropna()
        if values.empty:
            continue
        when = pd.Timestamp(values.idxmin())
        result[str(arm)] = (float(values.min()), when.date())
    return result


def period_table(daily: pd.DataFrame, by: str) -> pd.DataFrame:
    """Scores and P&L per arm and ``quarter``, ``year`` or ``total``.

    Scores are means over days, each day weighted equally; P&L is summed over the
    days that have it and capture is median dispatch P&L over perfect foresight's.
    """
    labels: dict[str, Callable[[date], str]] = {
        "quarter": quarter_label,
        "year": lambda day: str(day.year),
        "total": lambda day: "total",
    }
    if by not in labels:
        raise ValueError(f"by must be one of {sorted(labels)}")
    frame = daily.assign(period=[labels[by](day) for day in daily["target_day"]])
    rows: list[dict[str, object]] = []
    for (arm, period), part in frame.groupby(["arm", "period"], sort=False):
        traded = part.dropna(subset=["pnl_eur", "perfect_foresight_pnl_eur"])
        pnl = float(traded["pnl_eur"].sum())
        ceiling = float(traded["perfect_foresight_pnl_eur"].sum())
        rows.append(
            {
                "arm": arm,
                "period": period,
                "days": len(part),
                **{metric: float(part[metric].mean()) for metric in METRICS},
                "spike_days": int(part["spike_day"].sum()),
                "traded_days": len(traded),
                "pnl_eur": pnl,
                "perfect_foresight_pnl_eur": ceiling,
                "capture_ratio": pnl / ceiling if ceiling > 0 else math.nan,
            }
        )
    table = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(ARM_NAMES)}
    table["arm_rank"] = table["arm"].map(order).fillna(len(order))
    return table.sort_values(["arm_rank", "period"], ignore_index=True).drop(
        columns="arm_rank"
    )


# --- outputs --------------------------------------------------------------------


def _clean(value: object) -> object:
    if isinstance(value, float | np.floating):
        number = float(value)
        return None if not math.isfinite(number) else round(number, 6)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _records(table: pd.DataFrame) -> list[dict[str, object]]:
    return [
        {str(k): _clean(v) for k, v in row.items()}
        for row in table.to_dict(orient="records")
    ]


def setup_description(
    settings: Settings, first: date, last: date, arms: Sequence[Arm]
) -> dict[str, object]:
    battery = settings.battery
    return {
        "summary": (
            "LightGBM with conformal ranges walked forward through the 2021 to 2023 "
            "gas crisis: fitted once and never refitted, versus refitted every 91 "
            "and every 28 days, with the previous-day baseline for context. Same "
            "model, parameters and features in every LightGBM arm; only the refit "
            "schedule differs."
        ),
        "model": "lightgbm_conformal",
        "training_days": TRAINING_DAYS,
        "calibration_days": CALIBRATION_DAYS,
        "feature_groups": [g for g in FEATURE_GROUPS if g not in EXCLUDED_GROUPS],
        "first_target_day": first.isoformat(),
        "last_target_day": last.isoformat(),
        "holdout_start": settings.evaluation.holdout_start.isoformat(),
        "spike_threshold_eur_mwh": settings.evaluation.spike_threshold_eur_mwh,
        "arms": [
            {
                "name": arm.name,
                "refit_every_days": arm.refit_every_days,
                "description": arm.description,
            }
            for arm in arms
        ],
        "battery": {
            "power_mw": battery.power_mw,
            "capacity_mwh": battery.capacity_mwh,
            "round_trip_efficiency": battery.round_trip_efficiency,
            "degradation_eur_per_mwh": battery.degradation_eur_per_mwh,
            "max_cycles_per_day": battery.max_cycles_per_day,
        },
        "strategies": [MEDIAN_FORECAST.name, PERFECT_FORESIGHT.name],
    }


def experiment_payload(
    daily: pd.DataFrame, settings: Settings, setup: dict[str, object]
) -> dict[str, object]:
    """The dashboard's view of M1: rolling coverage, period tables and the setup."""
    rolling = rolling_coverage(daily).dropna(how="all")
    arms = [name for name in ARM_NAMES if name in set(daily["arm"])]
    lowest = min_rolling(rolling)
    return {
        "experiment": "m1_regime_shift",
        "generated_by": "python -m src.health.experiments.m1_regime_shift",
        "setup": setup,
        "coverage_target": COVERAGE_TARGET,
        "arms": arms,
        "rolling_coverage_90": {
            "window_days": ROLLING_DAYS,
            "dates": [ts.date().isoformat() for ts in pd.DatetimeIndex(rolling.index)],
            "arms": {arm: [_clean(v) for v in rolling[arm]] for arm in arms},
        },
        "min_rolling_coverage_90": {
            arm: {"value": round(value, 6), "window_end": when.isoformat()}
            for arm, (value, when) in lowest.items()
        },
        "quarterly": _records(period_table(daily, "quarter")),
        "yearly": _records(period_table(daily, "year")),
        "total": _records(period_table(daily, "total")),
    }


def market_context(
    frame: pd.DataFrame, days: Sequence[date], settings: Settings
) -> pd.DataFrame:
    """Mean and highest price, spike days and mean TTF gas price per quarter."""
    index = pd.DatetimeIndex(frame.index)
    local = np.asarray(index.tz_convert(settings.market.timezone).date, dtype=object)
    wanted = set(days)
    keep = np.asarray([day in wanted for day in local], dtype=bool)
    part = frame.loc[keep, [PRICE_SERIES, GAS_COLUMN]].assign(day=local[keep])
    by_day = part.groupby("day").agg(
        price_mean=(PRICE_SERIES, "mean"),
        price_max=(PRICE_SERIES, "max"),
        gas_mean=(GAS_COLUMN, "mean"),
    )
    threshold = settings.evaluation.spike_threshold_eur_mwh
    by_day["spike"] = by_day["price_max"] > threshold
    by_day["quarter"] = [quarter_label(day) for day in by_day.index]
    grouped = by_day.groupby("quarter", sort=True)
    return pd.DataFrame(
        {
            "mean_price": grouped["price_mean"].mean(),
            "max_price": grouped["price_max"].max(),
            "spike_days": grouped["spike"].sum(),
            "mean_gas": grouped["gas_mean"].mean(),
        }
    )


def training_range(
    forecasts: pd.DataFrame, frame: pd.DataFrame, first: date, settings: Settings
) -> dict[str, float]:
    """Prices a model fitted before ``first`` saw, against what it forecast and met."""
    start = settings.market.local_midnight_utc(first - timedelta(days=TRAINING_DAYS))
    end = settings.market.local_midnight_utc(first)
    index = pd.DatetimeIndex(frame.index)
    prices = frame.loc[(index >= start) & (index < end), PRICE_SERIES].dropna()
    return {
        "training_mean": float(prices.mean()),
        "training_p99": float(prices.quantile(0.99)),
        "training_max": float(prices.max()),
        "q50_max": float(forecasts["q50"].max()),
        "q95_max": float(forecasts["q95"].max()),
        "actual_max": float(forecasts[REALISED].max()),
    }


def _training_range_text(limits: dict[str, float], first: date) -> list[str]:
    start = first - timedelta(days=TRAINING_DAYS)
    return [
        f"Why the frozen arm fell behind: it was fitted once, on prices from {start} "
        f"to {first - timedelta(days=1)} with mean €{limits['training_mean']:.0f}, "
        f"99th percentile €{limits['training_p99']:.0f} and highest "
        f"€{limits['training_max']:.2f}. Over the target days its median forecast "
        f"peaked at €{limits['q50_max']:.2f} and its q95 at "
        f"€{limits['q95_max']:.0f}, while prices reached "
        f"€{limits['actual_max']:.0f}. Tree ensembles such as LightGBM predict from "
        "values learned on their training data and do not extrapolate beyond them, "
        f"and here the conformal ranges add errors from the last {CALIBRATION_DAYS} "
        "training days, which is consistent with this model's forecasts staying "
        "near its training prices while the market moved far above them. The "
        "refitted arms, trained on more recent prices, covered many more of those "
        "periods.",
    ]


def _pct(value: float) -> str:
    return "n/a" if not math.isfinite(value) else f"{value:.1%}"


def _eur(value: float) -> str:
    return f"{value:,.0f}"


def report(
    daily: pd.DataFrame,
    context: pd.DataFrame,
    settings: Settings,
    first: date,
    last: date,
    runtimes: dict[str, float],
    skipped: dict[date, str],
    limits: dict[str, float] | None = None,
) -> str:
    yearly = pd.concat([period_table(daily, "year"), period_table(daily, "total")])
    quarterly = period_table(daily, "quarter")
    lowest = min_rolling(rolling_coverage(daily))
    battery = settings.battery
    lines = [
        "# M1 regime shift experiment",
        "",
        "Generated by `python -m src.health.experiments.m1_regime_shift`. Prices and",
        "errors in €/MWh, P&L in €.",
        "",
        "## Setup",
        "",
        "Frozen before running and not tuned on results.",
        "",
        "* Model: `LightGBMConformalModel`, default parameters, "
        f"{TRAINING_DAYS} training days, {CALIBRATION_DAYS} calibration days, every "
        "feature group except weather (archived weather forecasts start in March "
        "2024).",
        f"* Target days {first} to {last}, {len(day_range(first, last)):,} days, "
        "forecast walk-forward at 11:40 on the day before delivery through the "
        "information set. The hold-out, from "
        f"{settings.evaluation.holdout_start}, is not read.",
        "* Arms, on identical days:",
        *[
            f"  * `{arm.name}`: {arm.description}"
            + (f" ({runtimes[arm.name]:,.0f} s)." if arm.name in runtimes else ".")
            for arm in ARMS
        ],
        "* Scores per day: share of periods inside the 90% (q05 to q95) and 50% "
        "(q25 to q75) central intervals, mean pinball loss over the seven quantiles, "
        "MAE of q50. A spike day has any price above "
        f"€{settings.evaluation.spike_threshold_eur_mwh:.0f}. Period and rolling "
        "figures are means over days, each day weighted equally.",
        f"* Trading: the reference battery, {battery.power_mw:g} MW / "
        f"{battery.capacity_mwh:g} MWh, {battery.round_trip_efficiency:.0%} round "
        f"trip, €{battery.degradation_eur_per_mwh:g} per MWh discharged, at most "
        f"{battery.max_cycles_per_day:g} cycles per day, dispatched on each arm's "
        "median forecast and settled at realised prices. Capture is that P&L over "
        "perfect foresight's, which is the same for every arm.",
        "",
        "## Results per arm",
        "",
    ]
    for arm in ARM_NAMES:
        rows = yearly[yearly["arm"] == arm]
        if rows.empty:
            continue
        lines += [
            f"### {arm}",
            "",
            "| period | days | coverage 90% | coverage 50% | mean pinball | "
            "MAE of q50 | capture | P&L, € |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, row in rows.iterrows():
            lines.append(
                f"| {row['period']} | {int(row['days'])} | "
                f"{_pct(row['coverage_90'])} | {_pct(row['coverage_50'])} | "
                f"{row['pinball']:.2f} | {row['mae_q50']:.2f} | "
                f"{_pct(row['capture_ratio'])} | {_eur(row['pnl_eur'])} |"
            )
        lines.append("")
    total = yearly[yearly["period"] == "total"].set_index("arm")
    if len(total):
        ceiling = float(total["perfect_foresight_pnl_eur"].iloc[0])
        lines += [
            f"Perfect foresight earned €{_eur(ceiling)} over all target days.",
            "",
        ]

    lines += [
        "## Lowest rolling 28-day coverage",
        "",
        "| arm | lowest 90% coverage | window |",
        "|---|---|---|",
    ]
    for arm, (value, when) in lowest.items():
        start = when - timedelta(days=ROLLING_DAYS - 1)
        lines.append(f"| {arm} | {_pct(value)} | {start} to {when} |")
    lines.append("")

    crisis = [q for q in context.index if str(q).startswith("2022")]
    wide = quarterly.pivot(index="period", columns="arm")
    arms = [a for a in ARM_NAMES if a in set(quarterly["arm"])]
    lines += [
        "## Every quarter",
        "",
        "Market context per quarter, then 90% coverage and capture per arm.",
        "",
        "| quarter | mean price | highest price | spike days | mean TTF gas | "
        + " | ".join(f"{a} cov. 90%" for a in arms)
        + " | "
        + " | ".join(f"{a} capture" for a in arms)
        + " |",
        "|---|---|---|---|---|" + "---|" * (2 * len(arms)),
    ]
    for quarter in wide.index:
        ctx = context.loc[quarter] if quarter in context.index else None
        market = (
            f"{ctx['mean_price']:.0f} | {ctx['max_price']:.0f} | "
            f"{int(ctx['spike_days'])} | {ctx['mean_gas']:.0f}"
            if ctx is not None
            else "n/a | n/a | n/a | n/a"
        )
        cov = " | ".join(_pct(wide.at[quarter, ("coverage_90", a)]) for a in arms)
        cap = " | ".join(_pct(wide.at[quarter, ("capture_ratio", a)]) for a in arms)
        lines.append(f"| {quarter} | {market} | {cov} | {cap} |")
    lines.append("")

    lines += ["## The 2022 gas crisis quarters", ""]
    lines += _crisis_text(quarterly, context, crisis)
    if limits is not None:
        lines += ["", *_training_range_text(limits, first)]
    lines += ["", "## Against the expectation recorded before running", ""]
    lines += _expectation_text(yearly, quarterly)
    if skipped:
        lines += [
            "",
            "Days not traded: " + ", ".join(f"{d} ({r})" for d, r in skipped.items()),
        ]
    lines.append("")
    return "\n".join(lines)


def _value(table: pd.DataFrame, arm: str, period: str, column: str) -> float:
    rows = table[(table["arm"] == arm) & (table["period"] == period)]
    return float(rows[column].iloc[0]) if len(rows) else math.nan


def _crisis_text(
    quarterly: pd.DataFrame, context: pd.DataFrame, crisis: list[str]
) -> list[str]:
    lines: list[str] = []
    for quarter in crisis:
        ctx = context.loc[quarter]
        parts = [
            f"* {quarter}: mean price €{ctx['mean_price']:.0f}, highest "
            f"€{ctx['max_price']:.0f}, {int(ctx['spike_days'])} spike days, TTF gas "
            f"€{ctx['mean_gas']:.0f}/MWh."
        ]
        for arm in ARM_NAMES:
            cov = _value(quarterly, arm, quarter, "coverage_90")
            if not math.isfinite(cov):
                continue
            parts.append(
                f"{arm} covered {_pct(cov)} (MAE "
                f"{_value(quarterly, arm, quarter, 'mae_q50'):.0f}, capture "
                f"{_pct(_value(quarterly, arm, quarter, 'capture_ratio'))})"
            )
        lines.append(parts[0] + " " + "; ".join(parts[1:]) + ".")
    frozen = [(q, _value(quarterly, "frozen", q, "coverage_90")) for q in crisis]
    frozen = [(q, v) for q, v in frozen if math.isfinite(v)]
    if frozen:
        worst_q, worst_v = min(frozen, key=lambda item: item[1])
        quarterly_v = _value(quarterly, "quarterly", worst_q, "coverage_90")
        monthly_v = _value(quarterly, "monthly", worst_q, "coverage_90")
        lines += [
            "",
            f"The frozen model's worst 2022 quarter was {worst_q}, at "
            f"{_pct(worst_v)} coverage against the 90% target; in the same quarter "
            f"the quarterly arm covered {_pct(quarterly_v)} and the monthly arm "
            f"{_pct(monthly_v)}.",
        ]
    return lines


def _expectation_text(yearly: pd.DataFrame, quarterly: pd.DataFrame) -> list[str]:
    frozen = _value(yearly, "frozen", "total", "coverage_90")
    frozen_2022 = _value(yearly, "frozen", "2022", "coverage_90")
    refit = _value(yearly, "quarterly", "total", "coverage_90")
    refit_2022 = _value(yearly, "quarterly", "2022", "coverage_90")
    low, high = EXPECTED_QUARTERLY_RANGE
    if not (math.isfinite(frozen) and math.isfinite(refit)):
        return ["Not every arm has finished; no comparison."]
    frozen_close = abs(frozen - EXPECTED_FROZEN_COVERAGE) <= 0.03
    refit_inside = low <= refit <= high
    frozen_quarters = quarterly[quarterly["arm"] == "frozen"]
    worst = frozen_quarters.loc[frozen_quarters["coverage_90"].idxmin()]
    return [
        "The expectation recorded before anything was run was frozen coverage near "
        f"{EXPECTED_FROZEN_COVERAGE:.0%} and quarterly refits recovering to "
        f"{low:.0%} to {high:.0%}. These were expectations, not results.",
        "",
        f"* Measured frozen 90% coverage: {_pct(frozen)} over all target days, "
        f"{_pct(frozen_2022)} in 2022, {_pct(float(worst['coverage_90']))} in its "
        f"lowest quarter, {worst['period']}. "
        + (
            "That is close to the expectation."
            if frozen_close
            else "That differs from the expectation."
        ),
        f"* Measured quarterly 90% coverage: {_pct(refit)} over all target days, "
        f"{_pct(refit_2022)} in 2022. "
        + (
            "That is inside the expected range."
            if refit_inside
            else "That is outside the expected range."
        ),
    ]


def _log_mlflow(
    daily: pd.DataFrame,
    first: date,
    last: date,
    runtimes: dict[str, float],
) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(REPO_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    total = period_table(daily, "total").set_index("arm")
    yearly = period_table(daily, "year")
    lowest = min_rolling(rolling_coverage(daily))
    for arm in ARMS:
        if arm.name not in total.index:
            continue
        row = total.loc[arm.name]
        with mlflow.start_run(run_name=arm.name):
            mlflow.log_params(
                {
                    "arm": arm.name,
                    "model": "naive_previous_day"
                    if arm.baseline
                    else "lightgbm_conformal",
                    "refit_every_days": str(arm.refit_every_days or "never"),
                    "training_days": str(TRAINING_DAYS),
                    "calibration_days": str(CALIBRATION_DAYS),
                    "feature_groups": ",".join(
                        g for g in FEATURE_GROUPS if g not in EXCLUDED_GROUPS
                    ),
                    "first_day": str(first),
                    "last_day": str(last),
                }
            )
            for metric in (*METRICS, "pnl_eur", "capture_ratio"):
                value = float(row[metric])
                if math.isfinite(value):
                    mlflow.log_metric(metric, value)
            for _, year in yearly[yearly["arm"] == arm.name].iterrows():
                mlflow.log_metric(
                    f"coverage_90_{year['period']}", float(year["coverage_90"])
                )
            if arm.name in lowest:
                mlflow.log_metric("min_rolling_28d_coverage_90", lowest[arm.name][0])
            if arm.name in runtimes:
                mlflow.log_metric("run_seconds", runtimes[arm.name])


# --- command line ---------------------------------------------------------------


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return number


def _read_runtimes(path: Path) -> dict[str, float]:
    if not path.exists():
        return {}
    return {str(k): float(v) for k, v in json.loads(path.read_text()).items()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Walk a frozen and a retrained model through the gas crisis."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--arms", nargs="+", choices=ARM_NAMES, default=ARM_NAMES)
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=max(1, (os.cpu_count() or 2) - 1),
        help="processes for arms in parallel and for battery dispatch",
    )
    parser.add_argument("--first-day", type=date.fromisoformat, default=FIRST_DAY)
    parser.add_argument("--last-day", type=date.fromisoformat, default=LAST_DAY)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    first, last = args.first_day, args.last_day
    if first > last:
        parser.error(f"first day {first} is after last day {last}")
    days = day_range(first, last)
    check_days(days, settings)

    full_run = first == FIRST_DAY and last == LAST_DAY and args.config is None
    out_dir = settings.data.processed_path / "experiments"
    if not full_run:
        out_dir = out_dir / f"{PREFIX}_partial_{first}_{last}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runtime_path = out_dir / f"{PREFIX}_runtime_seconds.json"
    runtimes = _read_runtimes(runtime_path)

    pending = [
        name
        for name in args.arms
        if load_checkpoint(_forecast_path(out_dir, name), days) is None
    ]
    for name in args.arms:
        if name not in pending:
            print(f"{name}: forecasts already saved, skipped", flush=True)
    config = str(args.config) if args.config is not None else None
    jobs = [(name, config, first, last, str(out_dir)) for name in pending]

    def finished(name: str, seconds: float) -> None:
        runtimes[name] = seconds
        text = json.dumps(runtimes, indent=2)
        write_atomically(runtime_path, lambda p: p.write_text(text, encoding="utf-8"))
        print(f"{name}: {len(days)} days forecast in {seconds:,.0f} s", flush=True)

    started = time.perf_counter()
    if len(jobs) > 1 and args.workers > 1:
        with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as pool:
            futures = [pool.submit(_forecast_job, job) for job in jobs]
            for future in as_completed(futures):
                finished(*future.result())
    else:
        for job in jobs:
            finished(*_forecast_job(job))

    available = {
        arm.name: forecasts
        for arm in ARMS
        if (forecasts := load_checkpoint(_forecast_path(out_dir, arm.name), days))
        is not None
    }
    missing = [name for name in ARM_NAMES if name not in available]
    if missing:
        print(f"arms still to run: {', '.join(missing)}; summary not written")
        return 0

    parts: list[pd.DataFrame] = []
    skipped_all: dict[date, str] = {}
    for name, forecasts in available.items():
        tradable, skipped = tradable_days(forecasts, settings)
        skipped_all |= skipped
        solve_started = time.perf_counter()
        pnl = dispatch_pnl(forecasts, tradable, settings, workers=args.workers)
        print(
            f"{name}: {len(tradable)} days dispatched in "
            f"{time.perf_counter() - solve_started:,.0f} s",
            flush=True,
        )
        metrics = daily_metrics(forecasts, settings).assign(arm=name)
        parts.append(metrics.merge(pnl, on="target_day", how="left"))
    daily = pd.concat(parts, ignore_index=True)[list(DAILY_COLUMNS)]
    write_atomically(out_dir / f"{PREFIX}_daily.parquet", daily.to_parquet)

    setup = setup_description(settings, first, last, ARMS)
    payload = experiment_payload(daily, settings, setup)
    text = json.dumps(payload, separators=(",", ":"))
    write_atomically(
        out_dir / f"{PREFIX}_experiment.json",
        lambda p: p.write_text(text, encoding="utf-8"),
    )

    frame = pd.read_parquet(settings.data.inputs_path)
    frame = frame.loc[
        frame.index
        < settings.market.local_midnight_utc(settings.evaluation.holdout_start)
    ]
    context = market_context(frame, days, settings)
    limits = training_range(available["frozen"], frame, first, settings)
    summary = pd.concat([period_table(daily, "year"), period_table(daily, "total")])
    print(summary.round(3).to_string())
    print(f"wrote {out_dir} in {time.perf_counter() - started:,.0f} s")
    if full_run:
        _log_mlflow(daily, first, last, runtimes)
        text = report(
            daily, context, settings, first, last, runtimes, skipped_all, limits
        )
        RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.write_text(text, encoding="utf-8")
        print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
