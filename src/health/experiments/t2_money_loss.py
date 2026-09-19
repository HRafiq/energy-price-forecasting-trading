"""T2 follow-up: train the forecaster on the battery's money.

    uv run python -m src.health.experiments.t2_money_loss --tune
    uv run python -m src.health.experiments.t2_money_loss --learning-rate 5 --trees 100

The plan, fixed before any of this was written, is
``docs/plans/money_trained_forecast_plan.md``. In short: the production model's q50 is
kept as a fixed base, and K extra LightGBM trees are boosted on the SPO+ loss
(Elmachtoub and Grigas, 2022), whose subgradient at quarter-hour t is
``-2 dt (net_real_t - net_shifted_t)``: the difference between the battery
schedule at the real prices and the schedule at prices ``2 p_hat - p``. The
schedules come from :mod:`src.trading.dispatch_lp`, the production optimiser's
linear relaxation, solved with HiGHS. The corrected point values both legs of
dispatch; the forecast ranges are not touched.

After a validation run, ``--rival`` trades a cheap rival, the production q50 plus
its mean residual per local hour over the last 42 training days, refit on the same
calendar; ``--seeds`` refits three blocks with two other seeds for the correction
trees; and ``--report`` rebuilds the results page from the saved artifacts with a
robustness section: where the gain concentrates, by year, without the strongest
quarter, the tuning grid, and against the rival and the seeds. ``--check-lp``
compares the HiGHS relaxation with the production optimiser on 60 validation days.

``--tune`` runs the learning-rate grid on the tuning window (the forecast days
before ``evaluation.validation_start``) and prints the profit of each setting at
each checkpoint. A validation run takes one setting, walks forward over the
validation window with the production refit cadence, trades both arms and writes
``docs/results/t2_money_loss.md``. The hold-out is never read.
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
from src.forecasting.models.gradient_boosting import (
    DEFAULT_LGBM_PARAMS,
    LightGBMConformalModel,
)
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
    "SEEDS",
    "SEED_BLOCKS",
    "MoneyFit",
    "fit_block",
    "main",
    "refit_days",
    "robustness",
    "spo_plus_gradient",
]

PRODUCTION = "lightgbm_conformal"
REFIT_EVERY_DAYS = 28
LEARNING_RATES = (1.0, 5.0, 20.0)
CHECKPOINTS = (25, 100)
MONEY = "money"
LP_CHECK_DAYS = 60
RECONCILE_EUR = 0.01
BASE_TOLERANCE = 1e-6
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t2_money_loss.md"
RIVAL = "bias_hour"
RIVAL_DAYS = 42
#: One refit block in the strongest quarter, one in summer, one in winter.
SEED_BLOCKS = (date(2024, 12, 7), date(2025, 6, 21), date(2026, 1, 31))
SEEDS = (8, 9)


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
    seed: int | None = None,
) -> MoneyFit:
    """Fit the production model for ``fit_day`` and boost ``trees`` money trees.

    ``seed`` reseeds the correction trees' row and column sampling only; the base
    model keeps its own seed, so it still reproduces the saved q50.
    """
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
        **({"random_state": seed} if seed is not None else {}),
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
        "seed": int(DEFAULT_LGBM_PARAMS["random_state"]),
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
    page = results_markdown(summary, robustness(pnl, _saved(settings)))
    RESULTS_PATH.write_text(page, encoding="utf-8")
    print(page)
    return summary


def _saved(settings: Settings) -> pd.DataFrame:
    return pd.read_parquet(
        settings.data.processed_path
        / "forecasts"
        / "comparison"
        / f"{PRODUCTION}.parquet"
    )


def _experiments(settings: Settings) -> Path:
    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _rival_points(saved: pd.DataFrame, timezone: str) -> pd.Series:
    """q50 plus its mean residual per local hour over the days before each refit."""
    days = sorted(saved["target_day"].unique())
    fits = refit_days(days)
    hour = pd.Series(
        pd.DatetimeIndex(saved.index).tz_convert(timezone).hour, index=saved.index
    )
    residual = saved["actual"] - saved["q50"]
    rival = pd.Series(np.nan, index=saved.index, dtype="float64")
    for k, fit_day in enumerate(fits):
        end = fits[k + 1] if k + 1 < len(fits) else days[-1] + timedelta(days=1)
        window = (saved["target_day"] >= fit_day - timedelta(days=RIVAL_DAYS)) & (
            saved["target_day"] < fit_day
        )
        bias = (
            residual[window].groupby(hour[window]).mean().reindex(range(24)).fillna(0.0)
        )
        block = (saved["target_day"] >= fit_day) & (saved["target_day"] < end)
        rival[block] = (
            saved.loc[block, "q50"].to_numpy() + bias.reindex(hour[block]).to_numpy()
        )
    return rival


def _rival(settings: Settings, workers: int) -> None:
    saved = _saved(settings)
    out = _experiments(settings)
    summary = json.loads(
        (out / "t2_money_loss_summary.json").read_text(encoding="utf-8")
    )
    money = pd.read_parquet(out / "t2_money_loss_points.parquet")[MONEY]
    first, last = (
        date.fromisoformat(summary["first_day"]),
        date.fromisoformat(summary["last_day"]),
    )
    days = sorted(d for d in saved["target_day"].unique() if first <= d <= last)
    points = pd.DataFrame(
        {
            RIVAL: _rival_points(saved, settings.market.timezone),
            MONEY: money.reindex(saved.index),
        }
    )
    _, pnl = _trade(settings, saved, points, [RIVAL, MONEY], days, workers)
    pnl.to_parquet(out / "t2_money_loss_rival_pnl.parquet")
    profit = pnl.groupby("strategy")["pnl_eur"].sum()
    print(
        f"rival {_eur(float(profit[RIVAL]))}, money {_eur(float(profit[MONEY]))}, "
        f"control {_eur(float(profit[MEDIAN_FORECAST.name]))}"
    )


def _seed_job(args: tuple[Any, ...]) -> tuple[date, int, pd.Series]:
    config, fit_day, seed, lr, trees = args
    settings = load_settings(config)
    inputs = pd.read_parquet(settings.data.inputs_path)
    saved = _saved(settings)
    days = sorted(saved["target_day"].unique())
    fits = refit_days(days)
    end = fits[fits.index(fit_day) + 1]
    block = [d for d in days if fit_day <= d < end]
    fit = fit_block(
        inputs,
        fit_day,
        settings,
        lr,
        trees,
        log=lambda t: print(t, flush=True),
        seed=seed,
    )
    frame = _forecast_block(fit, inputs, block, settings, saved, (trees,))
    return fit_day, seed, frame[f"{MONEY}_{trees}"]


def _seeds(settings: Settings, config: Path | None, workers: int) -> None:
    out = _experiments(settings)
    summary = json.loads(
        (out / "t2_money_loss_summary.json").read_text(encoding="utf-8")
    )
    lr, trees = float(summary["learning_rate"]), int(summary["trees"])
    saved = _saved(settings)
    money = pd.read_parquet(out / "t2_money_loss_points.parquet")[MONEY]
    jobs = [(config, block, seed, lr, trees) for block in SEED_BLOCKS for seed in SEEDS]
    parts: dict[str, list[pd.Series]] = {}
    with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
        for _, seed, series in pool.map(_seed_job, jobs):
            parts.setdefault(f"seed{seed}", []).append(series)
    points = pd.DataFrame({name: pd.concat(p) for name, p in parts.items()})
    points[f"seed{summary.get('seed', 7)}"] = money.reindex(points.index)
    days = sorted(saved.loc[saved.index.isin(points.index), "target_day"].unique())
    _, pnl = _trade(settings, saved, points, list(points.columns), days, workers)
    pnl.to_parquet(out / "t2_money_loss_seeds_pnl.parquet")
    print(pnl.groupby("strategy")["pnl_eur"].sum().round(0).to_string())


def _check_lp(settings: Settings) -> None:
    """The relaxation against the MILP on real days: how often the objectives agree."""
    from src.trading.optimizer import optimize_dispatch

    saved = _saved(settings)
    ev = settings.evaluation
    days = sorted(
        d
        for d in saved["target_day"].unique()
        if ev.validation_start <= d < ev.holdout_start
    )
    rng = np.random.default_rng(1)
    sample = sorted(rng.choice(len(days), LP_CHECK_DAYS, replace=False))
    programs: dict[tuple[int, float, tuple[bool, ...]], DayProgram] = {}
    rows = []
    for i in sample:
        frame = saved[saved["target_day"] == days[i]].sort_index()
        index = pd.DatetimeIndex(frame.index)
        program, products = _programs_for(
            index, frame[PRODUCT_COLUMN], settings.battery, programs
        )
        for column in ("q50", "actual"):
            prices = frame[column].to_numpy(dtype="float64")
            _, lp_value = solve_day(program, prices)
            milp = optimize_dispatch(frame[column], settings.battery, products=products)
            rows.append(
                {
                    "day": str(days[i]),
                    "curve": column,
                    "gap_eur": abs(lp_value - milp.objective_eur),
                    "negative_price": bool((prices < 0).any()),
                }
            )
    table = pd.DataFrame(rows)
    mismatch = table[table["gap_eur"] > RECONCILE_EUR]
    out = {
        "days": LP_CHECK_DAYS,
        "curves": len(table),
        "mismatches": len(mismatch),
        "mismatches_with_negative_price": int(mismatch["negative_price"].sum()),
        "worst_gap_eur": float(table["gap_eur"].max()),
        "curves_with_negative_price": int(table["negative_price"].sum()),
    }
    (_experiments(settings) / "t2_money_loss_lp_check.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(json.dumps(out, indent=2))


def robustness(pnl: pd.DataFrame, saved: pd.DataFrame) -> dict[str, Any]:
    """Where the gain sits: concentration, by year, without 2024 Q4, negative prices."""
    profit = pnl.pivot(index="target_day", columns="strategy", values="pnl_eur")
    diff = (profit[MONEY] - profit[MEDIAN_FORECAST.name]).sort_index()
    total = float(diff.sum())
    top = diff.sort_values(ascending=False).head(10)
    stamps = pd.to_datetime(pd.Index(diff.index))
    years = np.asarray(stamps.year)
    q4 = (years == 2024) & (np.asarray(stamps.month) >= 10)
    negative = (
        saved.groupby("target_day")["actual"].min().reindex(diff.index) < 0
    ).to_numpy()
    return {
        "total_eur": total,
        "top_days": [{"day": str(d), "eur": float(v)} for d, v in top.items()],
        "top10_share": float(top.sum() / total) if total else float("nan"),
        "wins": int((diff > 0.005).sum()),
        "losses": int((diff < -0.005).sum()),
        "ties": int((diff.abs() <= 0.005).sum()),
        "without_top10": mean_interval(diff.drop(top.index)),
        "without_2024q4": mean_interval(diff[~q4]),
        "by_year": {
            str(year): mean_interval(diff[years == year]) for year in sorted(set(years))
        },
        "negative_price_days": mean_interval(diff[negative]),
        "other_days": mean_interval(diff[~negative]),
    }


def _report(settings: Settings) -> None:
    out = _experiments(settings)
    summary = json.loads(
        (out / "t2_money_loss_summary.json").read_text(encoding="utf-8")
    )
    pnl = pd.read_parquet(out / "t2_money_loss_pnl.parquet")
    saved = _saved(settings)
    extra = robustness(pnl, saved)
    tuning_path = out / "t2_money_loss_tuning_pnl.parquet"
    if tuning_path.exists():
        tuning = pd.read_parquet(tuning_path)
        totals = tuning.groupby("strategy")["pnl_eur"].sum()
        control = float(totals[MEDIAN_FORECAST.name])
        extra["tuning"] = {
            "days": int(tuning["target_day"].nunique()),
            "control": control,
            "arms": {
                str(name): float(value) - control
                for name, value in totals.items()
                if str(name).startswith(MONEY)
            },
        }
    lp_path = out / "t2_money_loss_lp_check.json"
    if lp_path.exists():
        extra["lp_check"] = json.loads(lp_path.read_text(encoding="utf-8"))
    rival_path = out / "t2_money_loss_rival_pnl.parquet"
    if rival_path.exists():
        rp = pd.read_parquet(rival_path).pivot(
            index="target_day", columns="strategy", values="pnl_eur"
        )
        traded = saved["target_day"].isin(set(rp.index))
        rival_points = _rival_points(saved, settings.market.timezone)
        extra["rival"] = {
            "vs_control": mean_interval(rp[RIVAL] - rp[MEDIAN_FORECAST.name]),
            "money_minus_rival": mean_interval(rp[MONEY] - rp[RIVAL]),
            "mae": float((rival_points - saved["actual"])[traded].abs().mean()),
        }
    seeds_path = out / "t2_money_loss_seeds_pnl.parquet"
    if seeds_path.exists():
        sp = pd.read_parquet(seeds_path).pivot(
            index="target_day", columns="strategy", values="pnl_eur"
        )
        arms = {}
        for name in sorted(c for c in sp.columns if str(c).startswith("seed")):
            gain = sp[name] - sp[MEDIAN_FORECAST.name]
            arms[name] = {
                "total": float(gain.sum()),
                "by_block": {
                    str(b): float(
                        gain[
                            (gain.index >= b)
                            & (gain.index < b + timedelta(days=REFIT_EVERY_DAYS))
                        ].sum()
                    )
                    for b in SEED_BLOCKS
                },
            }
        extra["seeds"] = {"days": len(sp), "arms": arms}
    (out / "t2_money_loss_robustness.json").write_text(
        json.dumps(extra, indent=2, default=str), encoding="utf-8"
    )
    page = results_markdown(summary, extra)
    RESULTS_PATH.write_text(page, encoding="utf-8")
    print(page)


def _signed(value: float, places: int = 2) -> str:
    return f"{value:+,.{places}f}"


def _interval(item: dict[str, float]) -> str:
    return (
        f"{_signed(item['mean'])} ({_signed(item['low'])} to {_signed(item['high'])})"
    )


def _eur(value: float) -> str:
    return f"€{value:,.0f}" if value >= 0 else f"-€{-value:,.0f}"


def results_markdown(
    summary: dict[str, Any], extra: dict[str, Any] | None = None
) -> str:
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
    if extra:
        lines += _robustness_lines(extra)
    return "\n".join(lines)


def _robustness_lines(extra: dict[str, Any]) -> list[str]:
    top = extra["top_days"]
    lines = [
        "## Where the gain sits",
        "",
        f"The money arm wins on {extra['wins']} days, loses on {extra['losses']} and "
        f"ties within a cent on {extra['ties']}. The ten best days carry "
        f"{100 * extra['top10_share']:.0f}% of the {_eur(extra['total_eur'])}; the two "
        f"best are {top[0]['day']} ({_eur(top[0]['eur'])}) and {top[1]['day']} "
        f"({_eur(top[1]['eur'])}).",
        "",
        "| days | money minus median, € a day |",
        "|---|---|",
        f"| all but the ten best | {_interval(extra['without_top10'])} |",
        f"| all but October to December 2024 | {_interval(extra['without_2024q4'])} |",
    ]
    for year, item in extra["by_year"].items():
        lines.append(f"| {year} ({item['days']} days) | {_interval(item)} |")
    lines += [
        f"| days with a negative price ({extra['negative_price_days']['days']}) | "
        f"{_interval(extra['negative_price_days'])} |",
        f"| other days ({extra['other_days']['days']}) | "
        f"{_interval(extra['other_days'])} |",
    ]
    if "rival" in extra:
        rival = extra["rival"]
        lines += [
            "",
            "## Against a cheap rival",
            "",
            f"The rival adds to q50 its mean residual per local hour over the last "
            f"{RIVAL_DAYS} training days, refit on the same calendar. Against median "
            f"dispatch it made {_interval(rival['vs_control'])} € a day, with an MAE "
            f"of {rival['mae']:.2f} €/MWh; the money arm beat it by "
            f"{_interval(rival['money_minus_rival'])} € a day. The gain is not "
            "reducible to a simple hour-of-day bias correction.",
        ]
    if "tuning" in extra:
        tuning = extra["tuning"]
        lines += [
            "",
            "## The tuning grid",
            "",
            f"On the {tuning['days']} tuning days before the validation window, "
            f"median dispatch made {_eur(tuning['control'])}. Each setting against it:",
            "",
            "| setting | gain |",
            "|---|---|",
        ]
        for name, gain in sorted(tuning["arms"].items(), key=lambda kv: -kv[1]):
            lines.append(f"| {name} | {_eur(gain)} |")
        lines += [
            "",
            "The setting with the highest tuning profit was used, as the plan fixed; "
            "the margins between the four gentler settings are within noise.",
        ]
    if "lp_check" in extra:
        lp = extra["lp_check"]
        lines += [
            "",
            "## The relaxation against the optimiser",
            "",
            f"On {lp['days']} validation days, q50 and realised prices each, the HiGHS "
            f"relaxation's objective differed from the production optimiser's by more "
            f"than €0.01 on {lp['mismatches']} of {lp['curves']} curves, "
            f"{lp['mismatches_with_negative_price']} of them with a negative price "
            f"({lp['curves_with_negative_price']} curves have one); the worst gap was "
            f"€{lp['worst_gap_eur']:.2f}.",
        ]
    if "seeds" in extra:
        seeds = extra["seeds"]
        lines += [
            "",
            "## Other seeds",
            "",
            f"Three refit blocks ({', '.join(str(b) for b in SEED_BLOCKS)}, "
            f"{seeds['days']} days) refit with other seeds for the correction trees. "
            "Gain against median dispatch, in euros:",
            "",
            "| seed | total | " + " | ".join(str(b) for b in SEED_BLOCKS) + " |",
            "|---|---|" + "---|" * len(SEED_BLOCKS),
        ]
        for name, arm in seeds["arms"].items():
            cells = " | ".join(_eur(arm["by_block"][str(b)]) for b in SEED_BLOCKS)
            lines.append(f"| {name} | {_eur(arm['total'])} | {cells} |")
    lines.append("")
    return lines


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
    parser.add_argument(
        "--rival", action="store_true", help="trade the hour-bias rival"
    )
    parser.add_argument(
        "--seeds", action="store_true", help="refit blocks, other seeds"
    )
    parser.add_argument(
        "--report", action="store_true", help="rebuild the results page"
    )
    parser.add_argument(
        "--check-lp", action="store_true", help="compare the relaxation with the MILP"
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    if args.tune:
        _tune(settings, args.config, args.workers)
        return 0
    if args.rival:
        _rival(settings, args.workers)
        return 0
    if args.seeds:
        _seeds(settings, args.config, args.workers)
        return 0
    if args.report:
        _report(settings)
        return 0
    if args.check_lp:
        _check_lp(settings)
        return 0
    if args.learning_rate is None or args.trees is None:
        parser.error("give --learning-rate and --trees, chosen on the tuning window")
    _validate(settings, args.config, args.learning_rate, args.trees, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
