"""Walk-forward backtest of the battery strategies: trading metrics and experiments.

    uv run python -m src.trading.backtest --suite validation
    uv run python -m src.trading.backtest --suite decision-value
    uv run python -m src.trading.backtest --suite synthetic
    uv run python -m src.trading.backtest --suite degradation
    uv run python -m src.trading.backtest --suite holdout --confirm-holdout

Every suite reads saved walk-forward forecasts, issued at 11:40 on the day before
delivery, so no model is retrained here. Each day's schedule is committed before
the gate and settled at the realised price.

- ``validation``: the production model through perfect foresight, median, mean
  and quantile-aware dispatch, with the full trading metrics.
- ``decision-value`` (T2): every comparison model through the same strategies, so
  forecast accuracy can be set against money.
- ``synthetic`` (T2): forecasts built from the realised prices with the same
  average error placed in different periods, plus an evening shifted one hour
  early. Shows that where an error falls matters more than its size.
- ``degradation`` (T3): the optimizer is given different wear prices, but profit
  is always charged the battery's true wear, so greedy cycling shows its cost.
- ``attribution`` (T1): the gap between median or quantile-aware dispatch and
  perfect foresight split into Shapley shares by local hour block and by whether
  the forecast was too high or too low.
- ``holdout`` (T6): the frozen setup, once, on the untouched final period. Needs
  ``--confirm-holdout`` and the forecasts from ``src.forecasting.run_holdout``.

Outputs go to ``data/processed/backtest/<suite>/``; every summary row is logged as
a run in the MLflow experiment ``battery-backtest``, tagged with the forecast run
it consumed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import REPO_ROOT, Settings, load_settings
from src.trading.attribution import DIRECTIONS, HOUR_BLOCKS, attribute_days
from src.trading.battery import Battery
from src.trading.optimizer import DispatchError
from src.trading.run_strategies import run_strategies, select_days, summarise
from src.trading.strategies import (
    MEAN_FORECAST,
    MEDIAN_FORECAST,
    PERFECT_FORESIGHT,
    PRODUCT,
    REALISED,
    Strategy,
    add_mean_forecast,
    dispatch_day,
    quantile_aware,
)

__all__ = [
    "HOLDOUT_MODELS",
    "SUITES",
    "SYNTHETIC_VARIANTS",
    "SuiteResult",
    "attribution_summary",
    "backtest_strategies",
    "block_bootstrap_index",
    "capture_in_months",
    "degradation_table",
    "longest_losing_streak",
    "main",
    "max_drawdown",
    "monthly_capture",
    "paired_bootstrap_ci",
    "pnl_differences",
    "restrict_to_common_days",
    "strategy_table",
    "synthetic_curves",
    "synthetic_table",
    "top_share",
    "trading_metrics",
]

EXPERIMENT = "battery-backtest"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACTS = REPO_ROOT / "mlruns"
SUITES = (
    "validation",
    "decision-value",
    "synthetic",
    "degradation",
    "attribution",
    "holdout",
)
#: Strategies whose gap to perfect foresight is attributed on the validation window.
ATTRIBUTION_STRATEGIES = ("median_forecast", "quantile_q25")
#: The production model, the two other best validation forecasters and naive.
HOLDOUT_MODELS = (
    "lightgbm_conformal",
    "lightgbm_quantile",
    "quantile_forest",
    "naive_previous_day",
)
FORECAST_EXPERIMENTS = {
    "validation": "price-forecast-comparison",
    "holdout": "price-forecast-holdout",
}
SYNTHETIC_VARIANTS = (
    "level_shift",
    "noise_everywhere",
    "noise_idle_periods",
    "noise_traded_periods",
    "peak_one_hour_early",
)
SYNTHETIC_SEED = 20260914
#: Local hours whose prices the timing-error variant pulls one hour earlier.
EVENING_HOURS = range(14, 22)


# --- metrics ------------------------------------------------------------------


def max_drawdown(daily_pnl: pd.Series) -> float:
    """Largest fall of cumulative P&L from its running peak, starting from zero."""
    if daily_pnl.empty:
        return 0.0
    cumulative = daily_pnl.to_numpy(dtype=float).cumsum()
    peak = np.maximum.accumulate(np.maximum(cumulative, 0.0))
    return float((peak - cumulative).max())


def longest_losing_streak(daily_pnl: pd.Series, tolerance_eur: float = 1e-6) -> int:
    """Most consecutive days with a loss."""
    longest = current = 0
    for value in daily_pnl.to_numpy(dtype=float):
        current = current + 1 if value < -tolerance_eur else 0
        longest = max(longest, current)
    return longest


def top_share(daily_pnl: pd.Series, fraction: float = 0.1) -> float:
    """Share of total P&L earned on the best ``fraction`` of days."""
    total = float(daily_pnl.sum())
    if daily_pnl.empty or total <= 0:
        return math.nan
    count = max(1, round(len(daily_pnl) * fraction))
    return float(daily_pnl.nlargest(count).sum()) / total


def trading_metrics(pnl: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Phase 3's summary plus drawdown, losing streak, volatility and concentration."""
    summary = summarise(pnl, order)
    extra: dict[str, dict[str, float]] = {}
    for strategy, group in pnl.sort_values("target_day").groupby("strategy"):
        daily = group["pnl_eur"]
        extra[str(strategy)] = {
            "max_drawdown_eur": max_drawdown(daily),
            "longest_losing_streak_days": float(longest_losing_streak(daily)),
            "daily_pnl_std_eur": float(daily.std(ddof=0)),
            "top_decile_day_share": top_share(daily),
        }
    return summary.join(pd.DataFrame.from_dict(extra, orient="index"))


def monthly_capture(pnl: pd.DataFrame) -> pd.DataFrame:
    """P&L per calendar month and strategy, and each month's capture ratio."""
    month = pd.to_datetime(pnl["target_day"]).dt.to_period("M").astype(str)
    table = pnl.assign(month=month).pivot_table(
        index="month", columns="strategy", values="pnl_eur", aggfunc="sum"
    )
    ceiling = table[PERFECT_FORESIGHT.name]
    capture = table.div(ceiling, axis=0).add_suffix("_capture")
    return table.join(capture).reset_index()


def block_bootstrap_index(
    n: int, *, block_days: int = 7, draws: int = 5000, seed: int = 7
) -> NDArray[np.int64]:
    """Row positions for a moving-block bootstrap, one row of ``n`` per draw.

    Each draw strings together blocks of ``block_days`` consecutive days starting at
    random positions and cuts the result to ``n``, so a statistic recomputed on
    ``values[index]`` keeps the dependence between neighbouring days.
    """
    if n == 0:
        raise ValueError("no days to resample")
    block = min(block_days, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(draws, math.ceil(n / block)))
    index: NDArray[np.int64] = (starts[:, :, None] + np.arange(block)).reshape(
        draws, -1
    )[:, :n]
    return index


def paired_bootstrap_ci(
    differences: pd.Series,
    *,
    block_days: int = 7,
    draws: int = 5000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Mean daily P&L difference and its 95% interval, moving-block bootstrap.

    Daily profits are not independent: a volatile week lifts every strategy for
    several days. Resampling blocks of ``block_days`` consecutive days keeps that
    dependence, so the interval is not too narrow. Returns the mean and the 2.5%
    and 97.5% percentiles of the resampled means.
    """
    values = differences.to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("no differences to bootstrap")
    index = block_bootstrap_index(
        len(values), block_days=block_days, draws=draws, seed=seed
    )
    means = values[index].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def pnl_differences(
    pnl: pd.DataFrame, *, base_model: str, strategy: str
) -> pd.DataFrame:
    """Each model's daily P&L minus the base model's, for one strategy.

    Only days both models traded are paired. One row per other model with the
    mean daily difference, its bootstrap interval and the total over the window.
    """
    chosen = pnl[pnl["strategy"] == strategy]
    wide = chosen.pivot(index="target_day", columns="model", values="pnl_eur")
    rows = []
    for model in wide.columns:
        if model == base_model:
            continue
        paired = (wide[model] - wide[base_model]).dropna().sort_index()
        mean, low, high = paired_bootstrap_ci(paired)
        rows.append(
            {
                "model": model,
                "strategy": strategy,
                "days": len(paired),
                "mean_daily_difference_eur": mean,
                "ci_low_eur": low,
                "ci_high_eur": high,
                "total_difference_eur": float(paired.sum()),
            }
        )
    return pd.DataFrame(rows)


def capture_in_months(pnl: pd.DataFrame, months: set[int]) -> pd.DataFrame:
    """Capture ratio per model and strategy on the days of some calendar months.

    The hold-out covers only part of a year, and summer prices trade differently
    from the year's average, so it is compared with the same calendar months of
    the validation window as well as with the whole window.
    """
    month = pd.to_datetime(pnl["target_day"]).dt.month
    chosen = pnl[month.isin(sorted(months))]
    if "model" not in chosen.columns:
        chosen = chosen.assign(model="")
    rows: list[dict[str, object]] = []
    for model, part in chosen.groupby("model", sort=False):
        totals = part.groupby("strategy", sort=False)["pnl_eur"].sum()
        days = int(part["target_day"].nunique())
        ceiling = float(totals[PERFECT_FORESIGHT.name])
        for strategy, total in totals.items():
            rows.append(
                {
                    "model": model,
                    "strategy": strategy,
                    "days": days,
                    "pnl_per_day_eur": float(total) / days,
                    "capture_ratio": float(total) / ceiling,
                }
            )
    return pd.DataFrame(rows)


def restrict_to_common_days(
    pnl: pd.DataFrame, order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep the days every model traded and recompute each model's metrics.

    Days are chosen before solving; a day whose solve fails for one model must also
    leave the others, or the models would be compared on different days.
    """
    shared = set.intersection(
        *(set(part["target_day"]) for _, part in pnl.groupby("model", sort=False))
    )
    kept = pnl[pnl["target_day"].isin(shared)].reset_index(drop=True)
    parts = []
    for model, part in kept.groupby("model", sort=False):
        metrics = trading_metrics(part.drop(columns="model"), order).reset_index(
            names="strategy"
        )
        metrics.insert(0, "model", model)
        parts.append(metrics)
    return kept, pd.concat(parts, ignore_index=True)


# --- shared plumbing ----------------------------------------------------------


@dataclass
class SuiteResult:
    """Tables written as parquet, one summary logged per row, and run notes."""

    name: str
    window: str
    summary: pd.DataFrame
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    notes: dict[str, object] = field(default_factory=dict)


def backtest_strategies(settings: Settings) -> tuple[Strategy, ...]:
    """Perfect foresight, median, mean and the configured quantile-aware levels."""
    return (
        PERFECT_FORESIGHT,
        MEDIAN_FORECAST,
        MEAN_FORECAST,
        *(quantile_aware(level) for level in settings.trading.dispatch_quantiles),
    )


def _required_columns(strategies: tuple[Strategy, ...]) -> set[str]:
    columns = {REALISED, PRODUCT}
    for strategy in strategies:
        columns |= {strategy.sell_column, strategy.buy_column}
    return columns


def load_window(settings: Settings, window: str, model: str) -> pd.DataFrame:
    """Saved forecasts for one model and window, with the mean column added."""
    folder = {"validation": "comparison", "holdout": "holdout"}[window]
    path = settings.data.processed_path / "forecasts" / folder / f"{model}.parquet"
    if not path.exists():
        source = {
            "validation": "src.forecasting.run_comparison",
            "holdout": "src.forecasting.run_holdout",
        }[window]
        raise FileNotFoundError(f"{path} not found; run python -m {source} first")
    frame = pd.read_parquet(path).sort_index()
    return add_mean_forecast(frame, settings.forecasting.quantiles)


def window_days(
    forecasts: pd.DataFrame,
    settings: Settings,
    window: str,
    strategies: tuple[Strategy, ...],
) -> tuple[list[date], dict[date, str]]:
    """Complete days of the validation window or the hold-out."""
    holdout = settings.evaluation.holdout_start
    if window == "validation":
        first, last = settings.evaluation.validation_start, holdout - timedelta(days=1)
    elif window == "holdout":
        first, last = holdout, max(forecasts["target_day"])
    else:
        raise ValueError(f"unknown window {window!r}")
    return select_days(
        forecasts,
        first,
        last,
        settings,
        _required_columns(strategies),
        allow_holdout=window == "holdout",
    )


def strategy_table(
    forecasts: pd.DataFrame,
    days: list[date],
    strategies: tuple[Strategy, ...],
    battery: Battery,
    settings: Settings,
    *,
    workers: int = 1,
    allow_holdout: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[date, str]]:
    """Dispatch, daily P&L, metrics per strategy, and failed days."""
    dispatch, pnl, failed = run_strategies(
        forecasts,
        days,
        strategies,
        battery,
        holdout_start=settings.evaluation.holdout_start,
        time_limit_s=settings.trading.solver_time_limit_s,
        workers=workers,
        allow_holdout=allow_holdout,
    )
    names = [strategy.name for strategy in strategies]
    metrics = trading_metrics(pnl, names).reset_index(names="strategy")
    return dispatch, pnl, metrics, failed


# --- suites: validation, hold-out, decision value ----------------------------


def run_models(
    settings: Settings,
    battery: Battery,
    window: str,
    models: tuple[str, ...],
    *,
    workers: int,
    keep_dispatch_for: tuple[str, ...] = (),
    common_days: bool = False,
) -> SuiteResult:
    """Every strategy for every model on one window."""
    strategies = backtest_strategies(settings)
    frames = {model: load_window(settings, window, model) for model in models}
    selected = {
        model: window_days(frame, settings, window, strategies)
        for model, frame in frames.items()
    }
    shared = sorted(set.intersection(*(set(days) for days, _ in selected.values())))
    pnl_parts: list[pd.DataFrame] = []
    metric_parts: list[pd.DataFrame] = []
    dispatch_parts: list[pd.DataFrame] = []
    notes: dict[str, object] = {"window": window, "battery": battery.model_dump()}
    for model, frame in frames.items():
        days, skipped = selected[model]
        if common_days:
            days = shared
        dispatch, pnl, metrics, failed = strategy_table(
            frame,
            days,
            strategies,
            battery,
            settings,
            workers=workers,
            allow_holdout=window == "holdout",
        )
        pnl.insert(0, "model", model)
        metrics.insert(0, "model", model)
        pnl_parts.append(pnl)
        metric_parts.append(metrics)
        if model in keep_dispatch_for:
            dispatch.insert(0, "model", model)
            dispatch_parts.append(dispatch)
        dispatched = sorted(pnl["target_day"].unique())
        notes[model] = {
            "days": len(dispatched),
            "first": str(dispatched[0]),
            "last": str(dispatched[-1]),
            "skipped": {str(d): r for d, r in sorted((skipped | failed).items())},
        }
    pnl_all = pd.concat(pnl_parts, ignore_index=True)
    summary = pd.concat(metric_parts, ignore_index=True)
    dispatch_all = pd.concat(dispatch_parts) if dispatch_parts else None
    if common_days:
        pnl_all, summary = restrict_to_common_days(
            pnl_all, [strategy.name for strategy in strategies]
        )
        if dispatch_all is not None:
            traded = set(pnl_all["target_day"])
            dispatch_all = dispatch_all[dispatch_all["target_day"].isin(traded)]
    tables = {"pnl_daily": pnl_all}
    if dispatch_all is not None:
        tables["dispatch"] = dispatch_all
    months = [
        monthly_capture(part.drop(columns="model")).assign(model=model)
        for model, part in pnl_all.groupby("model", sort=False)
    ]
    tables["monthly_capture"] = pd.concat(months, ignore_index=True)
    return SuiteResult(
        name="",
        window=window,
        summary=summary,
        tables=tables,
        notes=notes,
    )


def validation_suite(settings: Settings, battery: Battery, workers: int) -> SuiteResult:
    model = settings.forecasting.production_model
    result = run_models(
        settings,
        battery,
        "validation",
        (model,),
        workers=workers,
        keep_dispatch_for=(model,),
    )
    result.name = "validation"
    return result


def holdout_suite(settings: Settings, battery: Battery, workers: int) -> SuiteResult:
    production = settings.forecasting.production_model
    models = (production, *(m for m in HOLDOUT_MODELS if m != production))
    result = run_models(
        settings,
        battery,
        "holdout",
        models,
        workers=workers,
        keep_dispatch_for=models,
        common_days=True,
    )
    result.name = "holdout"
    costs, per_day, attribution, notes = attribution_tables(
        settings, battery, "holdout", production, ("median_forecast",), workers
    )
    result.tables |= {"attribution_costs": costs, "attribution_days": per_day}
    result.tables["attribution_summary"] = attribution
    result.notes["attribution"] = notes
    return result


def decision_value_suite(
    settings: Settings, battery: Battery, workers: int
) -> SuiteResult:
    scores_path = (
        settings.data.processed_path
        / "forecasts"
        / "comparison"
        / "validation_scores.json"
    )
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    models = tuple(scores)
    result = run_models(
        settings, battery, "validation", models, workers=workers, common_days=True
    )
    accuracy = pd.DataFrame(
        {
            model: {
                "mean_pinball": values["Validation window"]["mean pinball"],
                "mae_of_median": values["Validation window"]["MAE of median"],
                "coverage_90": values["Validation window"]["coverage 90%"],
                "width_90": values["Validation window"]["width 90%"],
            }
            for model, values in scores.items()
        }
    ).T
    result.summary = result.summary.join(accuracy, on="model")
    result.name = "decision-value"
    return result


# --- suite: synthetic forecasts ------------------------------------------------


def synthetic_curves(
    day: pd.DataFrame,
    traded: NDArray[np.bool_],
    target_mae: float,
    rng: np.random.Generator,
    timezone: str,
) -> dict[str, NDArray[np.float64] | None]:
    """Forecast curves built from one day's realised prices.

    - ``level_shift``: every period ``target_mae`` too high. Spreads are intact.
    - ``noise_everywhere``: random errors in every period.
    - ``noise_idle_periods``: the same random draws, only in periods where perfect
      foresight neither charges nor discharges.
    - ``noise_traded_periods``: the same draws, only in periods where it trades.
    The three noise curves are scaled so the day's mean absolute error is exactly
    ``target_mae``; a curve whose periods are empty is ``None``.
    - ``peak_one_hour_early``: from 14:00 to 21:45 local time each period shows the
      price of one hour later, so the evening looks an hour early. Its error is
      whatever that shift produces.
    """
    actual = day[REALISED].to_numpy(dtype=float)
    n = len(actual)
    draws = rng.standard_normal(n)
    curves: dict[str, NDArray[np.float64] | None] = {"level_shift": actual + target_mae}
    for name, mask in (
        ("noise_everywhere", np.ones(n, dtype=bool)),
        ("noise_idle_periods", ~traded),
        ("noise_traded_periods", traded),
    ):
        if not mask.any():
            curves[name] = None
            continue
        scale = target_mae * n / float(np.abs(draws[mask]).sum())
        curves[name] = actual + np.where(mask, draws * scale, 0.0)
    hours = pd.DatetimeIndex(day.index).tz_convert(timezone).hour.to_numpy()
    evening = np.flatnonzero(np.isin(hours, list(EVENING_HOURS)))
    evening = evening[evening + 4 < n]
    shifted = actual.copy()
    shifted[evening] = actual[evening + 4]
    curves["peak_one_hour_early"] = shifted
    return curves


def _synthetic_day(
    job: tuple[pd.DataFrame, Battery, float, int, str, float],
) -> list[dict[str, object]]:
    frame, battery, target_mae, seed, timezone, time_limit_s = job
    day = frame["target_day"].iloc[0]
    actual = frame[REALISED].to_numpy(dtype=float)
    try:
        ceiling = dispatch_day(
            frame, PERFECT_FORESIGHT, battery, time_limit_s=time_limit_s
        )
        schedule = ceiling.schedule
        traded = (
            (schedule["charge_mw"] > 0) | (schedule["discharge_mw"] > 0)
        ).to_numpy()
        rng = np.random.default_rng([seed, day.toordinal()])
        rows: list[dict[str, object]] = []
        curves = synthetic_curves(frame, traded, target_mae, rng, timezone)
        for variant, curve in curves.items():
            if curve is None:
                continue
            result = dispatch_day(
                frame.assign(q50=curve),
                MEDIAN_FORECAST,
                battery,
                time_limit_s=time_limit_s,
            )
            rows.append(
                {
                    "target_day": day,
                    "variant": variant,
                    "mae_eur_mwh": float(np.abs(curve - actual).mean()),
                    "pnl_eur": result.settlement.pnl_eur,
                    "ceiling_pnl_eur": ceiling.settlement.pnl_eur,
                    "cycles": result.settlement.cycles,
                    "traded_periods": int(traded.sum()),
                }
            )
    except DispatchError:
        return []
    return rows


def synthetic_table(
    forecasts: pd.DataFrame,
    days: list[date],
    battery: Battery,
    settings: Settings,
    target_mae: float,
    *,
    workers: int = 1,
    seed: int = SYNTHETIC_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily rows per synthetic variant, and a summary per variant."""
    wanted = set(days)
    jobs = [
        (
            frame.sort_index(),
            battery,
            target_mae,
            seed,
            settings.market.timezone,
            settings.trading.solver_time_limit_s,
        )
        for day, frame in forecasts.groupby("target_day")
        if day in wanted
    ]
    outputs = _map(_synthetic_day, jobs, workers)
    daily = pd.DataFrame([row for rows in outputs for row in rows])
    grouped = daily.groupby("variant", sort=False)
    summary = pd.DataFrame(
        {
            "days": grouped["target_day"].nunique(),
            "mean_mae_eur_mwh": grouped["mae_eur_mwh"].mean(),
            "pnl_eur": grouped["pnl_eur"].sum(),
            "ceiling_pnl_eur": grouped["ceiling_pnl_eur"].sum(),
            "cycles_per_day": grouped["cycles"].mean(),
        }
    )
    summary["capture_ratio"] = summary["pnl_eur"] / summary["ceiling_pnl_eur"]
    summary["lost_eur"] = summary["ceiling_pnl_eur"] - summary["pnl_eur"]
    order = [v for v in SYNTHETIC_VARIANTS if v in summary.index]
    return daily, summary.loc[order].reset_index(names="variant")


def synthetic_suite(settings: Settings, battery: Battery, workers: int) -> SuiteResult:
    model = settings.forecasting.production_model
    strategies = (PERFECT_FORESIGHT, MEDIAN_FORECAST)
    forecasts = load_window(settings, "validation", model)
    days, skipped = window_days(forecasts, settings, "validation", strategies)
    chosen = forecasts[forecasts["target_day"].isin(days)]
    target_mae = float((chosen["q50"] - chosen[REALISED]).abs().mean())
    daily, summary = synthetic_table(
        forecasts, days, battery, settings, target_mae, workers=workers
    )
    return SuiteResult(
        name="synthetic",
        window="validation",
        summary=summary,
        tables={"daily": daily},
        notes={
            "target_mae_eur_mwh": target_mae,
            "mae_source": f"{model} median forecast over the validation window",
            "seed": SYNTHETIC_SEED,
            "skipped": {str(d): r for d, r in skipped.items()},
        },
    )


# --- suite: degradation sweep ---------------------------------------------------


def degradation_table(
    forecasts: pd.DataFrame,
    days: list[date],
    battery: Battery,
    settings: Settings,
    sweep: tuple[float, ...],
    *,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Perfect foresight and median dispatch for each optimizer wear price.

    Profit is recomputed with the battery's true wear per MWh discharged, so a low
    wear price in the optimizer shows up as extra cycling paid for at the true
    cost. Each wear price runs with the configured cycle cap and without a cap.
    """
    true_wear = battery.degradation_eur_per_mwh
    caps: list[float | None] = [battery.max_cycles_per_day]
    if battery.max_cycles_per_day is not None:
        caps.append(None)
    parts: list[pd.DataFrame] = []
    for cap in caps:
        for wear in sweep:
            variant = battery.model_copy(
                update={"degradation_eur_per_mwh": wear, "max_cycles_per_day": cap}
            )
            _, pnl, _ = run_strategies(
                forecasts,
                days,
                (PERFECT_FORESIGHT, MEDIAN_FORECAST),
                variant,
                holdout_start=settings.evaluation.holdout_start,
                time_limit_s=settings.trading.solver_time_limit_s,
                workers=workers,
            )
            pnl["optimizer_wear_eur_per_mwh"] = wear
            pnl["cycle_cap"] = math.nan if cap is None else cap
            pnl["pnl_true_wear_eur"] = (
                pnl["revenue_eur"] - true_wear * pnl["discharged_mwh"]
            )
            parts.append(pnl)
    daily = pd.concat(parts, ignore_index=True)
    keys = ["cycle_cap", "optimizer_wear_eur_per_mwh", "strategy"]
    grouped = daily.groupby(keys, dropna=False, sort=True)
    summary = pd.DataFrame(
        {
            "days": grouped["target_day"].nunique(),
            "revenue_eur": grouped["revenue_eur"].sum(),
            "pnl_true_wear_eur": grouped["pnl_true_wear_eur"].sum(),
            "discharged_mwh": grouped["discharged_mwh"].sum(),
            "cycles_per_day": grouped["cycles"].mean(),
            "losing_days_true_wear": grouped["pnl_true_wear_eur"].agg(
                lambda s: int((s < -1e-6).sum())
            ),
        }
    ).reset_index()
    summary["eur_per_mwh_discharged"] = summary["pnl_true_wear_eur"] / summary[
        "discharged_mwh"
    ].replace(0, np.nan)
    summary["true_wear_eur_per_mwh"] = true_wear
    return daily, summary


def degradation_suite(
    settings: Settings, battery: Battery, workers: int
) -> SuiteResult:
    model = settings.forecasting.production_model
    strategies = (PERFECT_FORESIGHT, MEDIAN_FORECAST)
    forecasts = load_window(settings, "validation", model)
    days, skipped = window_days(forecasts, settings, "validation", strategies)
    daily, summary = degradation_table(
        forecasts,
        days,
        battery,
        settings,
        settings.trading.degradation_sweep_eur_per_mwh,
        workers=workers,
    )
    return SuiteResult(
        name="degradation",
        window="validation",
        summary=summary,
        tables={"daily": daily},
        notes={"model": model, "skipped": {str(d): r for d, r in skipped.items()}},
    )


# --- suite: attribution of the gap to perfect foresight --------------------------


def attribution_summary(costs: pd.DataFrame, days: pd.DataFrame) -> pd.DataFrame:
    """One row per model and strategy: the gap and its Shapley shares.

    Columns ``cost_<block>_<direction>_eur`` hold each group's summed share, and
    ``cost_over_eur`` and ``cost_under_eur`` the totals by direction. The shares of
    a day add up to its gap, so the columns add up to ``gap_eur``.
    """
    rows: list[dict[str, object]] = []
    for (model, strategy), group in days.groupby(["model", "strategy"], sort=False):
        mine = costs[(costs["model"] == model) & (costs["strategy"] == strategy)]
        row: dict[str, object] = {
            "model": model,
            "strategy": strategy,
            "days": int(group["target_day"].nunique()),
            "strategy_pnl_eur": float(group["strategy_pnl_eur"].sum()),
            "ceiling_pnl_eur": float(group["ceiling_pnl_eur"].sum()),
            "gap_eur": float(group["gap_eur"].sum()),
            "attributed_eur": float(group["attributed_eur"].sum()),
            "residual_eur": float(group["residual_eur"].sum()),
        }
        for block, _ in HOUR_BLOCKS:
            for direction in DIRECTIONS:
                cell = mine[(mine["block"] == block) & (mine["direction"] == direction)]
                row[f"cost_{block}_{direction}_eur"] = float(cell["cost_eur"].sum())
        for direction in DIRECTIONS:
            row[f"cost_{direction}_eur"] = float(
                mine.loc[mine["direction"] == direction, "cost_eur"].sum()
            )
        rows.append(row)
    return pd.DataFrame(rows)


def attribution_tables(
    settings: Settings,
    battery: Battery,
    window: str,
    model: str,
    strategy_names: tuple[str, ...],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Per-group costs, per-day gaps, the summary and notes for one model."""
    by_name = {strategy.name: strategy for strategy in backtest_strategies(settings)}
    strategies = tuple(by_name[name] for name in strategy_names)
    forecasts = load_window(settings, window, model)
    days, skipped = window_days(
        forecasts, settings, window, (PERFECT_FORESIGHT, *strategies)
    )
    result = attribute_days(
        forecasts,
        days,
        strategies,
        battery,
        timezone=settings.market.timezone,
        holdout_start=settings.evaluation.holdout_start,
        workers=workers,
        time_limit_s=settings.trading.solver_time_limit_s,
        allow_holdout=window == "holdout",
    )
    costs = result.costs.assign(model=model)
    per_day = result.days.assign(model=model)
    notes: dict[str, object] = {
        "model": model,
        "strategies": list(strategy_names),
        "days": len(days),
        "skipped": {str(d): r for d, r in sorted((skipped | result.failed).items())},
        "orderings_per_day_max": int(per_day["orderings"].max()),
    }
    return costs, per_day, attribution_summary(costs, per_day), notes


def attribution_suite(
    settings: Settings, battery: Battery, workers: int
) -> SuiteResult:
    model = settings.forecasting.production_model
    costs, per_day, summary, notes = attribution_tables(
        settings, battery, "validation", model, ATTRIBUTION_STRATEGIES, workers
    )
    return SuiteResult(
        name="attribution",
        window="validation",
        summary=summary,
        tables={"costs": costs, "days": per_day},
        notes=notes,
    )


# --- runner -------------------------------------------------------------------


def _map(
    function: Callable[
        [tuple[pd.DataFrame, Battery, float, int, str, float]], list[dict[str, object]]
    ],
    jobs: list[tuple[pd.DataFrame, Battery, float, int, str, float]],
    workers: int,
) -> list[list[dict[str, object]]]:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(function, jobs, chunksize=4))
    return [function(job) for job in jobs]


def _forecast_run_id(model: str, window: str) -> str | None:
    experiment = FORECAST_EXPERIMENTS[window]
    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=(
            f"tags.mlflow.runName = '{model}' and attributes.status = 'FINISHED'"
        ),
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return runs[0].info.run_id if runs else None


def log_suite(
    result: SuiteResult, battery: Battery, config_label: str = "default"
) -> int:
    """One MLflow run per summary row; returns the number of runs logged."""
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT)
    keys = [
        c
        for c in (
            "model",
            "strategy",
            "variant",
            "cycle_cap",
            "optimizer_wear_eur_per_mwh",
        )
        if c in result.summary.columns
    ]
    run_ids: dict[str, str | None] = {}
    logged = 0
    for _, row in result.summary.iterrows():
        labels = {key: row[key] for key in keys}
        name = "/".join([result.name, *(str(value) for value in labels.values())])
        with mlflow.start_run(run_name=name):
            tags = {
                "suite": result.name,
                "window": result.window,
                "config": config_label,
            }
            tags |= {key: str(value) for key, value in labels.items()}
            model = labels.get("model")
            if isinstance(model, str):
                if model not in run_ids:
                    run_ids[model] = _forecast_run_id(model, result.window)
                if run_ids[model] is not None:
                    tags["forecast_run_id"] = str(run_ids[model])
            mlflow.set_tags(tags)
            mlflow.log_params(
                {f"battery.{k}": v for k, v in battery.model_dump().items()}
            )
            for column, value in row.items():
                if column in keys or isinstance(value, str):
                    continue
                number = float(value)
                if math.isfinite(number):
                    mlflow.log_metric(str(column), number)
        logged += 1
    return logged


SUITE_RUNNERS: dict[str, Callable[[Settings, Battery, int], SuiteResult]] = {
    "validation": validation_suite,
    "decision-value": decision_value_suite,
    "synthetic": synthetic_suite,
    "degradation": degradation_suite,
    "attribution": attribution_suite,
    "holdout": holdout_suite,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the battery strategies.")
    parser.add_argument("--suite", choices=SUITES, required=True)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--confirm-holdout",
        action="store_true",
        help="required for the hold-out suite: runs the frozen setup once",
    )
    parser.add_argument("--no-mlflow", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="repeat the hold-out suite although it has already run",
    )
    args = parser.parse_args(argv)
    if args.suite == "holdout" and not args.confirm_holdout:
        parser.error("the hold-out suite needs --confirm-holdout")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    settings = load_settings(args.config)
    battery = settings.battery
    out_dir = settings.data.processed_path / "backtest" / args.suite
    if args.config is not None:
        out_dir = out_dir.with_name(f"{args.suite}_{args.config.stem}")
    already_ran = args.suite == "holdout" and (out_dir / "summary.csv").exists()
    if already_ran and not args.overwrite:
        parser.error(
            f"the hold-out backtest has already run ({out_dir.name}); it runs "
            "once. Pass --overwrite only to repeat it on purpose."
        )
    started = time.perf_counter()
    result = SUITE_RUNNERS[args.suite](settings, battery, args.workers)
    elapsed = time.perf_counter() - started

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in result.tables.items():
        table.to_parquet(out_dir / f"{name}.parquet")
    result.summary.to_csv(out_dir / "summary.csv", index=False)
    notes = result.notes | {"runtime_seconds": round(elapsed, 1)}
    (out_dir / "notes.json").write_text(
        json.dumps(notes, indent=2, default=str), encoding="utf-8"
    )
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(result.summary.round(3).to_string(index=False))
    if not args.no_mlflow:
        label = args.config.stem if args.config is not None else "default"
        print(f"logged {log_suite(result, battery, label)} runs to MLflow")
    print(f"{args.suite}: {elapsed:.0f} s, wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
