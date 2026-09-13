"""Run every trading strategy over the validation window.

    uv run python -m src.trading.run_strategies
    uv run python -m src.trading.run_strategies --model quantile_forest --last-days 30

Reads the walk-forward forecasts written by the Phase 2 comparison, so no model is
retrained, and settles every schedule at the realised prices. Each delivery day is
optimized on its own: the battery starts and ends every day at the same state of
charge, so days are independent and run in parallel. Days in the hold-out are
refused; days with a missing period, price or forecast are skipped and listed.

Writes ``data/processed/trading/<model>/dispatch.parquet`` and
``pnl_daily.parquet``. A full run of the production model also writes
``docs/results/phase3_strategies.md``.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Settings, load_settings
from src.forecasting.walkforward import HoldoutAccessError, day_range
from src.trading.battery import Battery
from src.trading.optimizer import DispatchError
from src.trading.strategies import (
    PERFECT_FORESIGHT,
    PRODUCT,
    REALISED,
    DayResult,
    Strategy,
    build_strategies,
    dispatch_day,
)

__all__ = [
    "check_ceiling",
    "load_forecasts",
    "main",
    "run_strategies",
    "select_days",
    "summarise",
]

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "results" / "phase3_strategies.md"
)
PERIOD = pd.Timedelta(minutes=15)
#: A forecast strategy may not beat perfect foresight by more than solver noise.
#: CBC accepts a binary within 1e-6 of whole, worth about a cent at extreme prices.
CEILING_TOLERANCE_EUR = 1e-2


def load_forecasts(settings: Settings, model: str) -> pd.DataFrame:
    path = (
        settings.data.processed_path / "forecasts" / "comparison" / f"{model}.parquet"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run python -m src.forecasting.run_comparison first"
        )
    return pd.read_parquet(path).sort_index()


def select_days(
    forecasts: pd.DataFrame,
    first: date,
    last: date,
    settings: Settings,
    columns: set[str],
) -> tuple[list[date], dict[date, str]]:
    """Complete target days from ``first`` to ``last``, and why others are skipped."""
    holdout = settings.evaluation.holdout_start
    if last >= holdout:
        raise HoldoutAccessError(f"{last} is in the hold-out starting {holdout}")
    if first > last:
        raise ValueError(f"first day {first} is after last day {last}")
    if first < settings.evaluation.validation_start:
        raise ValueError(
            f"first day {first} is before the validation window starting "
            f"{settings.evaluation.validation_start}"
        )
    if settings.data.modeling_resolution != "quarterhour":
        raise NotImplementedError("trading runs on quarter-hour periods only")
    groups = {day: frame for day, frame in forecasts.groupby("target_day")}
    selected: list[date] = []
    skipped: dict[date, str] = {}
    for day in day_range(first, last):
        frame = groups.get(day)
        if frame is None:
            skipped[day] = "no forecast"
            continue
        expected = pd.date_range(
            settings.market.local_midnight_utc(day),
            settings.market.local_midnight_utc(day + timedelta(days=1)),
            freq=PERIOD,
            inclusive="left",
        )
        if len(frame) != len(expected) or not (frame.index == expected).all():
            skipped[day] = "incomplete periods"
            continue
        if not np.isfinite(frame[sorted(columns)].to_numpy(dtype=float)).all():
            skipped[day] = "missing price or forecast"
            continue
        if frame[PRODUCT].nunique() != 1:
            skipped[day] = "mixed product lengths"
            continue
        selected.append(day)
    return selected, skipped


def _run_day(
    job: tuple[pd.DataFrame, tuple[Strategy, ...], Battery, float],
) -> tuple[list[DayResult], str | None]:
    frame, strategies, battery, time_limit_s = job
    try:
        results = [
            dispatch_day(frame, strategy, battery, time_limit_s=time_limit_s)
            for strategy in strategies
        ]
    except DispatchError as exc:
        return [], f"solver failed: {exc}"
    return results, None


def run_strategies(
    forecasts: pd.DataFrame,
    days: list[date],
    strategies: tuple[Strategy, ...],
    battery: Battery,
    *,
    holdout_start: date,
    time_limit_s: float = 60.0,
    workers: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[date, str]]:
    """Dispatch and settle every strategy on every day.

    Returns the per-period dispatch, the per-day P&L and the days whose solve
    failed, with the reason. Raises if a day lies in the hold-out, or if a forecast
    strategy beat perfect foresight on any day, which only a wrong optimizer can do.
    """
    inside = [day for day in days if day >= holdout_start]
    if inside:
        raise HoldoutAccessError(
            f"{len(inside)} days fall in the hold-out starting {holdout_start}"
        )
    wanted = set(days)
    jobs = [
        (frame.sort_index(), strategies, battery, time_limit_s)
        for day, frame in forecasts.groupby("target_day")
        if day in wanted
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_run_day, jobs, chunksize=4))
    else:
        outputs = [_run_day(job) for job in jobs]

    parts: list[pd.DataFrame] = []
    rows: list[dict[str, object]] = []
    failed: dict[date, str] = {}
    for job, (results, failure) in zip(jobs, outputs, strict=True):
        if failure is not None:
            failed[job[0]["target_day"].iloc[0]] = failure
            continue
        for result in results:
            part = result.schedule.copy()
            part.insert(0, "strategy", result.strategy)
            part.insert(0, "target_day", result.target_day)
            parts.append(part)
            s = result.settlement
            rows.append(
                {
                    "target_day": result.target_day,
                    "strategy": result.strategy,
                    "product_minutes": result.product_minutes,
                    "revenue_eur": s.revenue_eur,
                    "degradation_eur": s.degradation_eur,
                    "pnl_eur": s.pnl_eur,
                    "charged_mwh": s.charged_mwh,
                    "discharged_mwh": s.discharged_mwh,
                    "cycles": s.cycles,
                    "planned_value_eur": result.planned_value_eur,
                    "solve_seconds": result.solve_seconds,
                }
            )
    if not rows:
        raise RuntimeError(f"no day could be dispatched: {failed}")
    dispatch = pd.concat(parts)
    dispatch.index.name = "timestamp_utc"
    pnl = pd.DataFrame(rows).sort_values(["target_day", "strategy"], ignore_index=True)
    check_ceiling(pnl)
    return dispatch, pnl, failed


def check_ceiling(
    pnl: pd.DataFrame, tolerance_eur: float = CEILING_TOLERANCE_EUR
) -> None:
    """Raise if any strategy settled above perfect foresight on some day."""
    wide = pnl.pivot(index="target_day", columns="strategy", values="pnl_eur")
    ceiling = PERFECT_FORESIGHT.name
    if ceiling not in wide.columns or wide.shape[1] < 2:
        return
    excess = wide.drop(columns=ceiling).sub(wide[ceiling], axis=0)
    worst = float(excess.max().max())
    if worst > tolerance_eur:
        location = excess.stack().idxmax()
        raise RuntimeError(
            f"a strategy beat perfect foresight by €{worst:.4f} at {location}; "
            "the optimizer did not return an optimum"
        )


def summarise(pnl: pd.DataFrame, order: list[str] | None = None) -> pd.DataFrame:
    """Headline numbers per strategy, in ``order`` or with perfect foresight first."""
    grouped = pnl.groupby("strategy", sort=False)
    summary = pd.DataFrame(
        {
            "days": grouped["target_day"].nunique(),
            "pnl_eur": grouped["pnl_eur"].sum(),
            "discharged_mwh": grouped["discharged_mwh"].sum(),
            "cycles_per_day": grouped["cycles"].mean(),
            "losing_days": grouped["pnl_eur"].agg(lambda s: int((s < -1e-6).sum())),
            "worst_day_eur": grouped["pnl_eur"].min(),
        }
    )
    ceiling = PERFECT_FORESIGHT.name
    if ceiling in summary.index:
        summary["capture_ratio"] = summary["pnl_eur"] / float(
            summary.at[ceiling, "pnl_eur"]
        )
    else:
        summary["capture_ratio"] = np.nan
    preferred = order or [ceiling]
    summary = summary.loc[
        [name for name in preferred if name in summary.index]
        + [name for name in summary.index if name not in preferred]
    ]
    summary["eur_per_mwh_discharged"] = summary["pnl_eur"] / summary[
        "discharged_mwh"
    ].replace(0, np.nan)
    return summary


def _era_table(pnl: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    era = np.where(
        pnl["product_minutes"] == 60, "hourly products", "15-minute products"
    )
    grouped = pnl.assign(era=era).groupby(["era", "strategy"], sort=False)["pnl_eur"]
    table = grouped.agg(days="count", pnl_per_day="mean", total="sum").reset_index()
    ceilings = table[table["strategy"] == PERFECT_FORESIGHT.name].set_index("era")[
        "total"
    ]
    table["capture_ratio"] = table["total"] / table["era"].map(ceilings)
    rank = {name: i for i, name in enumerate(order)}
    table["era_rank"] = (table["era"] == "15-minute products").astype(int)
    table["strategy_rank"] = table["strategy"].map(rank).fillna(len(order))
    return table.sort_values(["era_rank", "strategy_rank"], ignore_index=True)


def _report(
    summary: pd.DataFrame,
    pnl: pd.DataFrame,
    model: str,
    battery: Battery,
    days: list[date],
    skipped: dict[date, str],
    settings: Settings,
) -> str:
    lines = [
        "# Phase 3 trading strategies",
        "",
        "Generated by `python -m src.trading.run_strategies`. Forecasts: "
        f"`{model}`, walk-forward over the validation window, issued at 11:40 on the "
        "day before delivery; no model was retrained. Each schedule is committed "
        "before the 12:00 gate and settled at the realised day-ahead price, with the "
        "battery as a price taker. Before 1 October 2025 schedules follow hourly "
        "products.",
        "",
        f"Battery: {battery.power_mw:g} MW / {battery.capacity_mwh:g} MWh, "
        f"{battery.round_trip_efficiency:.0%} round-trip efficiency, "
        f"€{battery.degradation_eur_per_mwh:g} per MWh discharged, every day starting "
        f"and ending at {battery.initial_soc_fraction:.0%} state of charge"
        + (
            f", at most {battery.max_cycles_per_day:g} cycles per day."
            if battery.max_cycles_per_day is not None
            else ", no cycle cap."
        ),
        "",
        f"Target days {days[0]} to {days[-1]}, {len(days):,} days, "
        f"{len(skipped)} skipped. The hold-out, from "
        f"{settings.evaluation.holdout_start}, is not used.",
        "",
        "| strategy | P&L, € | capture ratio | € per MWh discharged | cycles per day "
        "| losing days | worst day, € |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in summary.iterrows():
        lines.append(
            f"| {name} | {row['pnl_eur']:,.0f} | {row['capture_ratio']:.1%} | "
            f"{row['eur_per_mwh_discharged']:.1f} | {row['cycles_per_day']:.2f} | "
            f"{int(row['losing_days'])} | {row['worst_day_eur']:,.1f} |"
        )
    lines += [
        "",
        "## By product era",
        "",
        "| era | strategy | days | P&L per day, € | capture ratio |",
        "|---|---|---|---|---|",
    ]
    for _, row in _era_table(pnl, list(summary.index)).iterrows():
        lines.append(
            f"| {row['era']} | {row['strategy']} | {int(row['days'])} | "
            f"{row['pnl_per_day']:,.1f} | {row['capture_ratio']:.1%} |"
        )
    if skipped:
        lines += [
            "",
            "Skipped days: " + ", ".join(f"{d} ({r})" for d, r in skipped.items()),
        ]
    lines += [
        "",
        "Capture ratio is a strategy's total P&L divided by perfect foresight's. "
        "Phase 4 adds drawdown, attribution of the gap to perfect foresight by hour "
        "and error direction, the degradation sweep and the other forecast models.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the trading strategies.")
    parser.add_argument(
        "--model", default=None, help="forecast model; default production"
    )
    parser.add_argument("--first", type=date.fromisoformat, default=None)
    parser.add_argument("--last", type=date.fromisoformat, default=None)
    parser.add_argument("--last-days", type=int, default=None)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.last_days is not None and args.last_days < 1:
        parser.error("--last-days must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    settings = load_settings(args.config)
    model = args.model or settings.forecasting.production_model
    last = args.last or settings.evaluation.holdout_start - timedelta(days=1)
    first = args.first or settings.evaluation.validation_start
    if args.last_days is not None:
        first = max(first, last - timedelta(days=args.last_days - 1))

    battery = settings.battery
    strategies = build_strategies(settings)
    columns = {REALISED, PRODUCT}
    columns |= {s.sell_column for s in strategies} | {s.buy_column for s in strategies}
    forecasts = load_forecasts(settings, model)
    days, skipped = select_days(forecasts, first, last, settings, columns)
    if not days:
        print(f"no complete days between {first} and {last}")
        return 1

    started = time.perf_counter()
    dispatch, pnl, failed = run_strategies(
        forecasts,
        days,
        strategies,
        battery,
        holdout_start=settings.evaluation.holdout_start,
        time_limit_s=settings.trading.solver_time_limit_s,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - started
    skipped = dict(sorted((skipped | failed).items()))
    dispatched = sorted(pnl["target_day"].unique())

    # Only the default full run owns the main outputs; any narrower or custom run
    # writes beside them, so the dashboard never reads a partial window.
    full_run = (
        args.first is None
        and args.last is None
        and args.last_days is None
        and args.config is None
    )
    out_dir = settings.data.processed_path / "trading" / model
    if not full_run:
        suffix = f"_{args.config.stem}" if args.config is not None else ""
        out_dir = out_dir / f"partial_{first}_{last}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    dispatch.to_parquet(out_dir / "dispatch.parquet")
    pnl.to_parquet(out_dir / "pnl_daily.parquet")
    summary = summarise(pnl, [strategy.name for strategy in strategies])
    median_ms = pnl["solve_seconds"].median() * 1000
    print(
        f"{model}: {len(dispatched)} days x {len(strategies)} strategies in "
        f"{elapsed:.0f} s, {len(skipped)} skipped; median solve {median_ms:.0f} ms"
    )
    for day, reason in failed.items():
        print(f"  {day}: {reason}")
    print(summary.round(3).to_string())
    print(f"wrote {out_dir}")
    if full_run and model == settings.forecasting.production_model:
        RESULTS_PATH.write_text(
            _report(summary, pnl, model, battery, dispatched, skipped, settings),
            encoding="utf-8",
        )
        print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
