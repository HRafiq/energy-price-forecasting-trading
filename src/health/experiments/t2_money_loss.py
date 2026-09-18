"""T2 follow-up: train the forecaster on the battery's money.

    uv run python -m src.health.experiments.t2_money_loss --tune
    uv run python -m src.health.experiments.t2_money_loss --learning-rate 5 --trees 100

The plan, fixed before any of this was written, is ``docs/plans/
money_trained_forecast_plan.md`` (local). In short: the production model's q50 is
kept as a fixed base, and K extra LightGBM trees are boosted on the SPO+ loss
(Elmachtoub and Grigas, 2022), whose subgradient at quarter-hour t is
``-2 dt (net_real_t - net_shifted_t)``: the difference between the battery
schedule at the real prices and the schedule at prices ``2 p_hat - p``. The
schedules come from :mod:`src.trading.dispatch_lp`, the production optimiser's
linear relaxation, solved with HiGHS. The corrected point values both legs of
dispatch; the forecast ranges are not touched.

``--tune`` runs the learning-rate grid on the tuning window (the forecast days
before ``evaluation.validation_start``) and prints the profit of each setting at
each checkpoint. A validation run takes one setting, walks forward over the
validation window with the production refit cadence, trades both arms and writes
``docs/results/t2_money_loss.md`` (local until adopted). The hold-out is never read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import (
    PRODUCT_COLUMN,
    REPO_ROOT,
    RESOLUTION_STEP,
    Settings,
    load_settings,
)
from src.forecasting.information import build_information_set
from src.forecasting.models.common import (
    local_hours,
    target_features,
    training_data,
)
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.production import build_production_model
from src.health.experiments.t1_mechanism import (
    BLOCKS,
    SUMMER,
    block_cash,
    mean_interval,
    window_schedule,
)
from src.trading.backtest import strategy_table
from src.trading.dispatch_lp import DayProgram, solve_day
from src.trading.optimizer import period_hours, product_blocks
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT, Strategy

__all__ = [
    "CHECKPOINTS",
    "LEARNING_RATES",
    "MONEY",
    "REFIT_EVERY_DAYS",
    "MoneyFit",
    "fit_block",
    "main",
    "refit_days",
    "spo_plus_gradient",
]

PRODUCTION = "lightgbm_conformal"
REFIT_EVERY_DAYS = 28
LEARNING_RATES = (1.0, 5.0, 20.0)
CHECKPOINTS = (25, 100)
MONEY = "money"
CANDIDATE = Strategy("money_point", MONEY, MONEY)
RECONCILE_EUR = 0.01
BASE_TOLERANCE = 1e-6
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t2_money_loss.md"


def refit_days(days: Sequence[date], every: int = REFIT_EVERY_DAYS) -> list[date]:
    """The days a walk-forward refits on, as ``run_walk_forward`` counts them."""
    fits: list[date] = []
    for day in days:
        if not fits or (day - fits[-1]).days >= every:
            fits.append(day)
    return fits


@dataclass
class _DayGroup:
    rows: NDArray[np.int64]
    program: DayProgram
    net_real: NDArray[np.float64]
    prices: NDArray[np.float64]


def _programs_for(
    index: pd.DatetimeIndex,
    minutes: pd.Series,
    battery: Any,
    cache: dict[tuple[int, float, tuple[bool, ...]], DayProgram],
) -> tuple[DayProgram, NDArray[np.int64]]:
    dt = period_hours(index)
    products = product_blocks(index, minutes)
    ties = tuple(bool(products[t] == products[t - 1]) for t in range(1, len(index)))
    key = (len(index), dt, ties)
    if key not in cache:
        cache[key] = DayProgram.build(len(index), dt, battery, products)
    return cache[key], products


def _day_groups(
    index: pd.DatetimeIndex,
    days: NDArray[np.object_],
    prices: NDArray[np.float64],
    minutes: pd.Series,
    settings: Settings,
    programs: dict[tuple[int, float, tuple[bool, ...]], DayProgram],
    real_cache: dict[date, NDArray[np.float64]],
) -> list[_DayGroup]:
    """Training rows grouped into whole delivery days the battery could trade."""
    groups: list[_DayGroup] = []
    order = np.arange(len(index))
    frame = pd.DataFrame({"row": order, "day": days}, index=index)
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    for key, part in frame.groupby("day", sort=True):
        day = cast(date, key)
        rows = part["row"].to_numpy(dtype="int64")
        stamps = pd.DatetimeIndex(part.index)
        expected = settings.market.local_midnight_utc(
            day + timedelta(days=1)
        ) - settings.market.local_midnight_utc(day)
        if len(rows) < 2 or (stamps[1:] - stamps[:-1] != step).any():
            continue
        if len(rows) * step != expected:
            continue
        program, _ = _programs_for(
            stamps, minutes.iloc[rows], settings.battery, programs
        )
        p = prices[rows]
        if day not in real_cache:
            real_cache[day], _ = solve_day(program, p)
        groups.append(_DayGroup(rows, program, real_cache[day], p))
    return groups


def spo_plus_gradient(
    preds: NDArray[np.float64], groups: Iterable[_DayGroup]
) -> tuple[NDArray[np.float64], float]:
    """The SPO+ subgradient per row, and the mean loss per day, for one round."""
    grad = np.zeros_like(preds)
    losses = []
    for group in groups:
        p_hat = preds[group.rows]
        shifted = 2.0 * p_hat - group.prices
        net_shift, value_shift = solve_day(group.program, shifted)
        dt = group.program.dt
        grad[group.rows] = -2.0 * dt * (group.net_real - net_shift)
        wear = group.program.degradation * dt * np.maximum(group.net_real, 0.0).sum()
        value_hat = dt * float(p_hat @ group.net_real) - wear
        value_real = dt * float(group.prices @ group.net_real) - wear
        losses.append(value_shift - 2.0 * value_hat + value_real)
    return grad, float(np.mean(losses)) if losses else 0.0


@dataclass
class MoneyFit:
    """One refit: the production base and the money-trained corrections."""

    fitted_on: date
    base: LightGBMConformalModel
    booster: lgb.Booster
    losses: list[float]
    training_days: int
    seconds: float

    def point(self, info: Any, trees: int) -> NDArray[np.float64]:
        features = target_features(info, self.base.settings, self.base._names)
        assert self.base._model is not None and self.base._offsets is not None
        hours = local_hours(info.target_index, info.tz)
        q50 = self.base.settings.forecasting.quantiles.index(0.5)
        base = np.asarray(self.base._model.predict(features), dtype="float64")
        base = base + self.base._offsets[hours, q50]
        correction = np.asarray(
            self.booster.predict(features, raw_score=True, num_iteration=trees),
            dtype="float64",
        )
        return np.asarray(base + correction, dtype="float64")


def fit_block(
    inputs: pd.DataFrame,
    fit_day: date,
    settings: Settings,
    learning_rate: float,
    trees: int,
    log: Callable[[str], None] = print,
) -> MoneyFit:
    """Fit the production model for ``fit_day`` and boost ``trees`` money trees."""
    started = time.perf_counter()
    base = build_production_model(settings)
    if not isinstance(base, LightGBMConformalModel):
        raise TypeError("the money loss builds on the LightGBM conformal model")
    info = build_information_set(inputs, fit_day, settings, base.fit_lookback_days)
    base.fit(info)
    assert base._model is not None and base._offsets is not None
    data = training_data(info, settings, base._names, base.training_days)
    hours = data.hours
    q50 = settings.forecasting.quantiles.index(0.5)
    init = np.asarray(base._model.predict(data.features), dtype="float64")
    init = init + base._offsets[hours, q50]
    minutes = info.history[PRODUCT_COLUMN].reindex(data.features.index)
    programs: dict[tuple[int, float, tuple[bool, ...]], DayProgram] = {}
    real_cache: dict[date, NDArray[np.float64]] = {}
    groups = _day_groups(
        pd.DatetimeIndex(data.features.index),
        data.days,
        data.target.to_numpy(dtype="float64"),
        minutes,
        settings,
        programs,
        real_cache,
    )
    losses: list[float] = []

    def objective(
        preds: NDArray[np.float64], _: lgb.Dataset
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        grad, loss = spo_plus_gradient(preds, groups)
        losses.append(loss)
        return grad, np.ones_like(preds)

    params = {
        **{k: v for k, v in base.params.items() if k != "n_estimators"},
        "objective": objective,
        "learning_rate": learning_rate,
        "boost_from_average": False,
    }
    dataset = lgb.Dataset(data.features, data.target, init_score=init)
    booster = lgb.train(params, dataset, num_boost_round=trees)
    seconds = time.perf_counter() - started
    log(
        f"fit {fit_day}: {len(groups)} training days, loss {losses[0]:.2f} -> "
        f"{losses[-1]:.2f} over {trees} trees, {seconds / 60:.1f} min"
    )
    return MoneyFit(fit_day, base, booster, losses, len(groups), seconds)


def _forecast_block(
    fit: MoneyFit,
    inputs: pd.DataFrame,
    days: Sequence[date],
    settings: Settings,
    saved: pd.DataFrame,
    checkpoints: Sequence[int],
) -> pd.DataFrame:
    """The money point for every day of one refit block, at each checkpoint."""
    parts = []
    for day in days:
        info = build_information_set(inputs, day, settings, fit.base.lookback_days)
        reproduced = fit.base.forecast(info).values["q50"].to_numpy(dtype="float64")
        stored = saved.loc[saved["target_day"] == day, "q50"].to_numpy(dtype="float64")
        if (
            len(stored) != len(reproduced)
            or np.abs(stored - reproduced).max() > BASE_TOLERANCE
        ):
            raise RuntimeError(
                f"the refit base does not reproduce the saved q50 for {day}; the "
                "arms would not share a base"
            )
        part = pd.DataFrame({"target_day": day}, index=info.target_index)
        for trees in checkpoints:
            part[f"{MONEY}_{trees}"] = fit.point(info, trees)
        parts.append(part)
    return pd.concat(parts)


def _block_job(
    args: tuple[Any, ...],
) -> tuple[date, pd.DataFrame, list[float], int, float]:
    inputs_path, fit_day, block_days, config, lr, trees, checkpoints = args
    settings = load_settings(config)
    inputs = pd.read_parquet(inputs_path)
    saved = pd.read_parquet(
        settings.data.processed_path
        / "forecasts"
        / "comparison"
        / f"{PRODUCTION}.parquet"
    )
    fit = fit_block(
        inputs, fit_day, settings, lr, trees, log=lambda s: print(s, flush=True)
    )
    frame = _forecast_block(fit, inputs, block_days, settings, saved, checkpoints)
    return fit_day, frame, fit.losses, fit.training_days, fit.seconds


def _walk(
    settings: Settings,
    config: Path | None,
    days: Sequence[date],
    learning_rate: float,
    checkpoints: Sequence[int],
    workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Money points for ``days``, refitting as the production walk-forward does."""
    all_days = sorted(
        pd.read_parquet(
            settings.data.processed_path
            / "forecasts"
            / "comparison"
            / f"{PRODUCTION}.parquet"
        )["target_day"].unique()
    )
    fits = refit_days(all_days)
    wanted = set(days)
    blocks = []
    for k, fit_day in enumerate(fits):
        end = fits[k + 1] if k + 1 < len(fits) else all_days[-1] + timedelta(days=1)
        block = [d for d in all_days if fit_day <= d < end and d in wanted]
        if block:
            blocks.append((fit_day, block))
    trees = max(checkpoints)
    jobs = [
        (
            settings.data.inputs_path,
            fit_day,
            block,
            config,
            learning_rate,
            trees,
            tuple(checkpoints),
        )
        for fit_day, block in blocks
    ]
    print(
        f"{len(jobs)} refits over {len(days)} days with {workers} workers", flush=True
    )
    parts, log = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for fit_day, frame, losses, training_days, seconds in pool.map(
            _block_job, jobs
        ):
            parts.append(frame)
            log[str(fit_day)] = {
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "training_days": training_days,
                "seconds": seconds,
            }
    return pd.concat(parts).sort_index(), log


def _trade(
    settings: Settings,
    saved: pd.DataFrame,
    money: pd.DataFrame,
    columns: Sequence[str],
    days: Sequence[date],
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = saved[saved["target_day"].isin(set(days))].copy()
    strategies: list[Strategy] = [PERFECT_FORESIGHT, MEDIAN_FORECAST]
    for column in columns:
        frame[column] = money[column].reindex(frame.index)
        strategies.append(Strategy(column, column, column))
    if frame[list(columns)].isna().any().any():
        raise RuntimeError("a money point is missing for a traded day")
    dispatch, pnl, _, failed = strategy_table(
        frame,
        list(days),
        tuple(strategies),
        settings.battery,
        settings,
        workers=workers,
    )
    if failed:
        raise RuntimeError(f"dispatch failed on {sorted(failed)}")
    return dispatch, pnl


def _tune(settings: Settings, config: Path | None, workers: int) -> None:
    saved = pd.read_parquet(
        settings.data.processed_path
        / "forecasts"
        / "comparison"
        / f"{PRODUCTION}.parquet"
    )
    days = sorted(
        d
        for d in saved["target_day"].unique()
        if d < settings.evaluation.validation_start
    )
    print(f"tuning window: {days[0]} to {days[-1]}, {len(days)} days", flush=True)
    columns, frames = [], []
    for lr in LEARNING_RATES:
        money, log = _walk(settings, config, days, lr, CHECKPOINTS, workers)
        for trees in CHECKPOINTS:
            name = f"{MONEY}_lr{lr:g}_k{trees}"
            columns.append(name)
            frames.append(money[f"{MONEY}_{trees}"].rename(name))
        print(json.dumps({f"lr {lr:g}": log}, indent=1), flush=True)
    money_all = pd.concat(frames, axis=1)
    _, pnl = _trade(settings, saved, money_all, columns, days, workers)
    profit = pnl.groupby("strategy")["pnl_eur"].sum()
    control = profit[MEDIAN_FORECAST.name]
    print(
        f"\ncontrol median_forecast: €{control:,.0f}; perfect foresight "
        f"€{profit[PERFECT_FORESIGHT.name]:,.0f}"
    )
    for name in columns:
        print(f"{name}: €{profit[name]:,.0f} ({profit[name] - control:+,.0f})")
    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    pnl.to_parquet(out / "t2_money_loss_tuning_pnl.parquet")


def _validate(
    settings: Settings, config: Path | None, lr: float, trees: int, workers: int
) -> dict[str, Any]:
    saved = pd.read_parquet(
        settings.data.processed_path
        / "forecasts"
        / "comparison"
        / f"{PRODUCTION}.parquet"
    )
    ev = settings.evaluation
    days = sorted(
        d
        for d in saved["target_day"].unique()
        if ev.validation_start <= d < ev.holdout_start
    )
    money, log = _walk(settings, config, days, lr, (trees,), workers)
    column = f"{MONEY}_{trees}"
    money = money.rename(columns={column: MONEY})
    dispatch, pnl = _trade(settings, saved, money, [MONEY], days, workers)
    profit = pnl.pivot(index="target_day", columns="strategy", values="pnl_eur")
    stored = pd.read_parquet(
        settings.data.processed_path / "trading" / PRODUCTION / "pnl_daily.parquet"
    ).pivot(index="target_day", columns="strategy", values="pnl_eur")
    drift = float(
        (profit[MEDIAN_FORECAST.name] - stored.loc[profit.index, MEDIAN_FORECAST.name])
        .abs()
        .max()
    )
    if drift > RECONCILE_EUR:
        raise RuntimeError(
            f"the control misses the saved validation profit by €{drift:.4f}"
        )
    traded = sorted(profit.index)
    summer = pd.Series(
        pd.to_datetime(pd.Index(traded)).month.isin(sorted(SUMMER)), index=traded
    )
    daily = (profit[MONEY] - profit[MEDIAN_FORECAST.name]).loc[traded]
    tz, wear = settings.market.timezone, float(settings.battery.degradation_eur_per_mwh)
    cycles = pnl.pivot(index="target_day", columns="strategy", values="cycles")
    sold = pnl.pivot(index="target_day", columns="strategy", values="discharged_mwh")
    planned = pnl.pivot(
        index="target_day", columns="strategy", values="planned_value_eur"
    )
    windows, blocks = {}, {}
    for name in (MEDIAN_FORECAST.name, MONEY):
        chosen = dispatch[dispatch["strategy"] == name]
        windows[name], blocks[name] = (
            window_schedule(chosen, tz),
            block_cash(chosen, tz, wear),
        )

    def window_cash(frame: pd.DataFrame) -> pd.Series:
        return (
            frame["discharge_eur"]
            - frame["charge_eur"]
            - wear * frame["discharged_mwh"]
        )

    cash_change = (
        window_cash(windows[MONEY]) - window_cash(windows[MEDIAN_FORECAST.name])
    ).loc[traded]
    block_change = (blocks[MONEY] - blocks[MEDIAN_FORECAST.name]).loc[traded]
    # Accuracy of the point itself, against the production q50.
    scored = saved[saved["target_day"].isin(set(traded))]
    actual = scored["actual"].to_numpy(dtype="float64")
    mae = {
        "median_forecast": float(
            np.abs(scored["q50"].to_numpy(dtype="float64") - actual).mean()
        ),
        MONEY: float(
            np.abs(
                money[MONEY].reindex(scored.index).to_numpy(dtype="float64") - actual
            ).mean()
        ),
    }
    ceiling = float(profit[PERFECT_FORESIGHT.name].sum())
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": PRODUCTION,
        "learning_rate": lr,
        "trees": trees,
        "days": len(traded),
        "first_day": str(traded[0]),
        "last_day": str(traded[-1]),
        "perfect_foresight_pnl_eur": ceiling,
        "arms": {
            name: {
                "pnl_eur": float(profit[name].sum()),
                "capture": float(profit[name].sum()) / ceiling,
                "cycles_per_day": float(cycles[name].mean()),
                "sold_mwh_per_day": float(sold[name].mean()),
                "window_sold_mwh_per_day": float(
                    windows[name]["discharged_mwh"].mean()
                ),
                "plan_minus_settled_eur_per_day": float(
                    (planned[name] - profit[name]).mean()
                ),
                "mae_eur_mwh": mae[name],
            }
            for name in (MEDIAN_FORECAST.name, MONEY)
        },
        "difference": {
            "all": mean_interval(daily),
            "summer": mean_interval(daily, summer),
            "winter": mean_interval(daily, ~summer),
        },
        "window_cash": {
            "all": mean_interval(cash_change),
            "summer": mean_interval(cash_change, summer),
            "winter": mean_interval(cash_change, ~summer),
        },
        "blocks": {name: mean_interval(block_change[name]) for name in BLOCKS},
        "fits": log,
    }
    summary["adopted"] = bool(summary["difference"]["all"]["low"] > 0)
    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    pnl.to_parquet(out / "t2_money_loss_pnl.parquet")
    money.to_parquet(out / "t2_money_loss_points.parquet")
    (out / "t2_money_loss_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    page = results_markdown(summary)
    RESULTS_PATH.write_text(page, encoding="utf-8")
    print(page)
    return summary


def _signed(value: float, places: int = 2) -> str:
    return f"{value:+,.{places}f}"


def _interval(item: dict[str, float]) -> str:
    return (
        f"{_signed(item['mean'])} ({_signed(item['low'])} to {_signed(item['high'])})"
    )


def _eur(value: float) -> str:
    return f"€{value:,.0f}" if value >= 0 else f"-€{-value:,.0f}"


def results_markdown(summary: dict[str, Any]) -> str:
    arms, diff = summary["arms"], summary["difference"]
    adopted = summary["adopted"]
    lines = [
        "# T2 follow-up: a forecaster trained on the money",
        "",
        "Generated by `python -m src.health.experiments.t2_money_loss`. The production "
        f"model's q50 is the base; `money_point` adds {summary['trees']} trees "
        f"boosted on the SPO+ loss at learning rate {summary['learning_rate']:g}, "
        "refit every 28 days as the production walk-forward does, over "
        f"{summary['days']} validation days from {summary['first_day']} to "
        f"{summary['last_day']}. Both arms value both legs of "
        "dispatch at their point and settle at the realised prices for a 1 MW / 2 MWh "
        "battery with €8 wear. The control reproduces the saved validation profit to "
        "within €0.01, and every refit base reproduces the saved q50 exactly.",
        "",
        "Criterion, fixed before the run: adopt `money_point` only if the mean daily "
        "profit difference against `median_forecast` has a 95% moving-block bootstrap "
        "interval (7-day blocks, 5,000 draws) entirely above zero.",
        "",
        "## Arms",
        "",
        "| arm | profit | capture | MAE of the point, €/MWh | cycles a day | "
        "sold, MWh a day | sold 15-21, MWh a day | planned minus settled, € a day |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in (MEDIAN_FORECAST.name, MONEY):
        a = arms[name]
        lines.append(
            f"| {name} | {_eur(a['pnl_eur'])} | {100 * a['capture']:.2f}% | "
            f"{a['mae_eur_mwh']:.2f} | {a['cycles_per_day']:.2f} | "
            f"{a['sold_mwh_per_day']:.2f} | {a['window_sold_mwh_per_day']:.2f} | "
            f"{_signed(a['plan_minus_settled_eur_per_day'])} |"
        )
    lines += [
        "",
        f"Perfect foresight made {_eur(summary['perfect_foresight_pnl_eur'])} on "
        "the same days.",
        "",
        "## money_point minus median_forecast, euros per day",
        "",
        "| measure | all days | June to September | October to May |",
        "|---|---|---|---|",
        f"| daily profit | {_interval(diff['all'])} | {_interval(diff['summer'])} | "
        f"{_interval(diff['winter'])} |",
        f"| cash in 15:00 to 21:00 | {_interval(summary['window_cash']['all'])} | "
        f"{_interval(summary['window_cash']['summer'])} | "
        f"{_interval(summary['window_cash']['winter'])} |",
        "",
        "| block | cash gap |",
        "|---|---|",
    ]
    lines += [f"| {name} | {_interval(summary['blocks'][name])} |" for name in BLOCKS]
    total = diff["all"]["mean"] * diff["all"]["days"]
    verdict = (
        "adopted: the whole interval is above zero"
        if adopted
        else "not adopted: the interval does not lie above zero"
    )
    lines += [
        "",
        "## Training",
        "",
        "| refit | training days | SPO+ loss, first tree | last tree | minutes |",
        "|---|---|---|---|---|",
    ]
    for day, item in summary["fits"].items():
        lines.append(
            f"| {day} | {item['training_days']} | {item['loss_first']:.2f} | "
            f"{item['loss_last']:.2f} | {item['seconds'] / 60:.1f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        f"`money_point` is {verdict}. Over the {summary['days']} days it made "
        f"{_eur(total)} "
        f"against median dispatch, a difference of {_interval(diff['all'])} € a day.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the forecaster on the money.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument(
        "--tune", action="store_true", help="run the grid on the tuning window"
    )
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--trees", type=int, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.tune:
        _tune(settings, args.config, args.workers)
        return 0
    if args.learning_rate is None or args.trees is None:
        parser.error("give --learning-rate and --trees, chosen on the tuning window")
    _validate(settings, args.config, args.learning_rate, args.trees, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
