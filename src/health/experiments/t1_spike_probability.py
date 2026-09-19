"""T1 follow-up: a model of the chance of an evening spike, feeding the dispatch.

    uv run python -m src.health.experiments.t1_spike_probability
    uv run python -m src.health.experiments.t1_spike_probability --quick

The plan, fixed before any of this was written, is
``docs/plans/spike_probability_plan.md``. A LightGBM classifier, refit on the
production calendar, gives each day a probability ``p`` that its evening (17:00 to
20:59) reaches the spike threshold. Three dispatch arms trade the saved production
forecast, settled at real prices:

* ``median``: median dispatch, the control;
* ``blend``: the evening quarter-hours valued at ``(1 - p) q50 + p q90``;
* ``hold_back``: median dispatch, but on days with ``p >= 0.5`` the battery must
  be full at 17:00, so the whole store is there for the evening.

Criterion, fixed in advance and applied to each arm: adopt only if its paired daily
profit difference against ``median`` has a 95% moving-block bootstrap interval
above zero. ``--quick`` runs the first 100 days and writes nothing to ``docs/``.
The hold-out is never read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import PRICE_SERIES, PRODUCT_COLUMN, REPO_ROOT, Settings, load_settings
from src.features.build import EVENING_HOURS, EXTRA_FEATURE_GROUPS, build_features
from src.forecasting.run_comparison import REFIT_EVERY_DAYS
from src.health.experiments.m1_refit_cadence import refit_days
from src.health.experiments.t1_mechanism import (
    BLOCKS,
    SUMMER,
    block_cash,
    mean_interval,
)
from src.trading.battery import Battery
from src.trading.optimizer import optimize_dispatch, product_blocks
from src.trading.settlement import settle

__all__ = [
    "ARMS",
    "CLASSIFIER_PARAMS",
    "DAY_FEATURES",
    "HOLD_BACK_THRESHOLD",
    "day_table",
    "main",
    "reliability",
    "results_markdown",
    "spike_probabilities",
    "trade_day",
]

PRODUCTION = "lightgbm_conformal"
ARMS = ("median", "blend", "hold_back")
HOLD_BACK_THRESHOLD = 0.5
REPORTED_THRESHOLDS = (0.3, 0.5, 0.7)
TRAINING_DAYS = 730
QUICK_DAYS = 100
#: Fixed before the run; nothing here is tuned.
CLASSIFIER_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_child_samples": 20,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "random_state": 7,
    "deterministic": True,
    "force_row_wise": True,
    "verbose": -1,
}
#: Standard features summarised per day as their mean over the day's rows.
DAY_MEANS = (
    "load_forecast_mw",
    "wx_wind_power_onshore",
    "wx_wind_power_offshore",
    "wx_radiation_mean",
    "wx_temperature_mean",
    "residual_load_persistence_mw",
    "ccgt_marginal_cost_eur_mwh",
    "price_prev_day_mean",
    "price_prev_day_max",
    "price_mean_same_clock_7d",
)
DAY_FIRSTS = ("weekday", "is_holiday", "day_of_year_sin", "day_of_year_cos")
DAY_FEATURES = (
    *EXTRA_FEATURE_GROUPS["spike_drivers"],
    *(f"{name}_day_mean" for name in DAY_MEANS),
    *DAY_FIRSTS,
)
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t1_spike_probability.md"


def day_table(inputs: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """One row per local day: the classifier's inputs and the spike label.

    The label is whether the day's evening maximum price reached the threshold;
    it is NaN while the day's prices are not all published. Rows are indexed by
    the day and hold only what was known at that day's issue time, because every
    feature is built from its own row's information set.
    """
    features = build_features(inputs, settings)
    tz = settings.market.timezone
    local = pd.DatetimeIndex(features.index).tz_convert(tz)
    days = pd.Index(local.date)
    table = pd.DataFrame(index=pd.Index(sorted(set(days)), name="target_day"))
    for name in EXTRA_FEATURE_GROUPS["spike_drivers"]:
        table[name] = features[name].groupby(days).first()
    for name in DAY_MEANS:
        table[f"{name}_day_mean"] = features[name].groupby(days).mean()
    for name in DAY_FIRSTS:
        table[name] = features[name].groupby(days).first()
    price = inputs[PRICE_SERIES].reindex(features.index)
    evening = (local.hour >= EVENING_HOURS[0]) & (local.hour <= EVENING_HOURS[1])
    evening_max = price[evening].groupby(days[evening]).max()
    complete = price.notna().groupby(days).all()
    threshold = settings.evaluation.spike_threshold_eur_mwh
    label = (evening_max >= threshold).astype("float64").reindex(table.index)
    table["spike"] = label.where(complete.reindex(table.index).fillna(False))
    return table


def spike_probabilities(
    table: pd.DataFrame, days: Sequence[date], every: int
) -> pd.Series:
    """``p`` for each of ``days``, from classifiers refit on the walk-forward calendar.

    Each refit trains on the ``TRAINING_DAYS`` days before its refit day whose
    label is known, and serves every day up to the next refit.
    """
    fits = refit_days(list(days), every)
    out = pd.Series(np.nan, index=pd.Index(days, name="target_day"), dtype="float64")
    columns = list(DAY_FEATURES)
    for k, fit_day in enumerate(fits):
        end = fits[k + 1] if k + 1 < len(fits) else days[-1] + timedelta(days=1)
        start = fit_day - timedelta(days=TRAINING_DAYS)
        rows = table[(table.index >= start) & (table.index < fit_day)]
        rows = rows[rows["spike"].notna()]
        if rows["spike"].nunique() < 2:
            raise ValueError(f"no spike days to learn from before {fit_day}")
        model = lgb.LGBMClassifier(**CLASSIFIER_PARAMS)
        model.fit(rows[columns], rows["spike"].astype(int))
        block = [d for d in days if fit_day <= d < end]
        served = table.reindex(block)
        scores = np.asarray(model.predict_proba(served[columns]), dtype="float64")
        out.loc[block] = scores[:, 1]
    if out.isna().any():
        raise ValueError("some days received no spike probability")
    return out


def reliability(p: pd.Series, label: pd.Series) -> dict[str, Any]:
    """Brier score against the base rate, log loss, AUC and a reliability table."""
    both = pd.concat([p.rename("p"), label.rename("y")], axis=1).dropna()
    prob = both["p"].to_numpy(dtype="float64")
    y = both["y"].to_numpy(dtype="float64")
    base = float(y.mean())
    eps = 1e-6
    clipped = np.clip(prob, eps, 1 - eps)
    order = np.argsort(prob)
    ranks = np.empty(len(prob))
    ranks[order] = np.arange(1, len(prob) + 1)
    positives = y == 1
    auc = (
        (ranks[positives].sum() - positives.sum() * (positives.sum() + 1) / 2)
        / (positives.sum() * (~positives).sum())
        if 0 < positives.sum() < len(y)
        else float("nan")
    )
    bins = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0001]
    table = []
    for low, high in pairwise(bins):
        inside = (prob >= low) & (prob < high)
        if inside.any():
            table.append(
                {
                    "bin": f"{low:.1f} to {min(high, 1.0):.1f}",
                    "days": int(inside.sum()),
                    "mean_p": float(prob[inside].mean()),
                    "spike_rate": float(y[inside].mean()),
                }
            )
    return {
        "days": len(y),
        "base_rate": base,
        "brier": float(np.mean((prob - y) ** 2)),
        "brier_base_rate": float(np.mean((base - y) ** 2)),
        "log_loss": float(
            -np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped))
        ),
        "auc": float(auc),
        "reliability": table,
    }


def _floor(
    p: float, threshold: float, at: NDArray[np.bool_], battery: Battery
) -> NDArray[np.float64] | None:
    """A full-battery floor at the periods ``at`` when ``p`` reaches ``threshold``."""
    if p < threshold:
        return None
    floor = np.full(len(at), np.nan)
    floor[at] = battery.capacity_mwh
    return floor


def trade_day(
    frame: pd.DataFrame,
    p: float,
    battery: Battery,
    settings: Settings,
    thresholds: Sequence[float] = REPORTED_THRESHOLDS,
) -> list[dict[str, Any]]:
    """Every arm's settled result for one day, plus hold-back at other thresholds."""
    frame = frame.sort_index()
    index = pd.DatetimeIndex(frame.index)
    local = index.tz_convert(settings.market.timezone)
    products = product_blocks(index, frame[PRODUCT_COLUMN])
    evening = (local.hour >= EVENING_HOURS[0]) & (local.hour <= EVENING_HOURS[1])
    q50, q90 = frame["q50"], frame["q90"]
    realised = frame["actual"]
    blend = q50.where(~evening, (1 - p) * q50 + p * q90)
    floor_time = np.asarray(local.hour == EVENING_HOURS[0] - 1)
    curves: dict[str, tuple[pd.Series, NDArray[np.float64] | None]] = {
        "median": (q50, None),
        "blend": (blend, None),
    }
    for threshold in thresholds:
        name = (
            "hold_back"
            if threshold == HOLD_BACK_THRESHOLD
            else f"hold_back_{threshold:g}"
        )
        curves[name] = (q50, _floor(p, threshold, floor_time, battery))
    curves["perfect_foresight"] = (realised, None)
    rows = []
    for name, (curve, floor) in curves.items():
        result = optimize_dispatch(
            curve,
            battery,
            products=products,
            time_limit_s=settings.trading.solver_time_limit_s,
            soc_floor=floor,
        )
        settled = settle(result.schedule, realised, battery)
        schedule = result.schedule.assign(realised_price=realised.to_numpy())
        rows.append(
            {
                "strategy": name,
                "pnl_eur": settled.pnl_eur,
                "planned_value_eur": result.objective_eur,
                "cycles": settled.cycles,
                "discharged_mwh": settled.discharged_mwh,
                "fired": floor is not None,
                "schedule": schedule,
            }
        )
    return rows


def _day_job(args: tuple[Any, ...]) -> tuple[date, list[dict[str, Any]]]:
    frame, p, config = args
    settings = load_settings(config)
    day = frame["target_day"].iloc[0]
    return day, trade_day(frame, p, settings.battery, settings)


def _signed(value: float, places: int = 2) -> str:
    return f"{value:+,.{places}f}"


def _interval(item: Mapping[str, float]) -> str:
    return (
        f"{_signed(item['mean'])} ({_signed(item['low'])} to {_signed(item['high'])})"
    )


def _eur(value: float) -> str:
    return f"€{value:,.0f}" if value >= 0 else f"-€{-value:,.0f}"


def results_markdown(summary: Mapping[str, Any]) -> str:
    arms = summary["arms"]
    cls = summary["classifier"]
    lines = [
        "# T1 follow-up: the chance of an evening spike, fed to the dispatch",
        "",
        "Generated by `python -m src.health.experiments.t1_spike_probability`. A "
        "LightGBM classifier, refit every 28 days on the production calendar and "
        f"trained on the {TRAINING_DAYS} days before each refit, gives each day the "
        "probability that its evening (17:00 to 20:59) reaches "
        f"€{summary['threshold']:.0f}. Three dispatch arms trade the saved production "
        f"forecast on the {summary['days']} validation days ({summary['first_day']} "
        f"to {summary['last_day']}) for a 1 MW / 2 MWh battery with €8 wear, settled "
        "at real prices. `median` values every quarter-hour at q50; `blend` values "
        "the evening at (1 - p) q50 + p q90; `hold_back` is median dispatch with the "
        "battery required to be full at 17:00 when p is at least 0.5.",
        "",
        "Criterion, fixed before the run, for each arm: adopt only if its paired "
        "daily profit difference against `median` has a 95% moving-block bootstrap "
        "interval (7-day blocks, 5,000 draws) entirely above zero.",
        "",
        "## The classifier",
        "",
        f"On the {cls['days']} scored days, {100 * cls['base_rate']:.1f}% of evenings "
        f"spiked. Brier score {cls['brier']:.4f} against {cls['brier_base_rate']:.4f} "
        f"for always saying the base rate; log loss {cls['log_loss']:.4f}; area under "
        f"the ROC curve {cls['auc']:.3f}.",
        "",
        "| predicted p | days | mean p | evenings that spiked |",
        "|---|---|---|---|",
    ]
    for row in cls["reliability"]:
        lines.append(
            f"| {row['bin']} | {row['days']} | {row['mean_p']:.2f} | "
            f"{100 * row['spike_rate']:.1f}% |"
        )
    lines += [
        "",
        "## Arms",
        "",
        "| arm | profit | capture | cycles a day | planned minus settled, € a day | "
        "days the rule fired |",
        "|---|---|---|---|---|---|",
    ]
    for name in (*ARMS, *summary["extra_arms"]):
        a = arms[name]
        lines.append(
            f"| {name} | {_eur(a['pnl_eur'])} | {100 * a['capture']:.2f}% | "
            f"{a['cycles_per_day']:.2f} | {_signed(a['plan_minus_settled'])} | "
            f"{a['fired_days']} |"
        )
    lines += [
        "",
        f"Perfect foresight made {_eur(summary['perfect_foresight_pnl_eur'])}.",
        "",
        "## Each arm minus median, euros per day",
        "",
        "| arm | all days | June to September | October to May | spike evenings | "
        "other evenings | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in (*ARMS[1:], *summary["extra_arms"]):
        d = summary["difference"][name]
        adopted = d["all"]["low"] > 0
        verdict = (
            "adopted"
            if adopted and name in ARMS
            else "reported only"
            if name not in ARMS
            else "not adopted"
        )
        lines.append(
            f"| {name} | {_interval(d['all'])} | {_interval(d['summer'])} | "
            f"{_interval(d['winter'])} | {_interval(d['spike_days'])} | "
            f"{_interval(d['other_days'])} | {verdict} |"
        )
    lines += [
        "",
        "Cash by clock block, each arm minus median, euros per day:",
        "",
        "| block | " + " | ".join(ARMS[1:]) + " |",
        "|---|" + "---|" * len(ARMS[1:]),
    ]
    for block in BLOCKS:
        cells = " | ".join(
            _interval(summary["blocks"][name][block]) for name in ARMS[1:]
        )
        lines.append(f"| {block} | {cells} |")
    both = summary.get("blend_minus_hold_back")
    lines += ["", "## Verdict", ""]
    adopted = [
        name for name in ARMS[1:] if summary["difference"][name]["all"]["low"] > 0
    ]
    if not adopted:
        lines.append(
            "Neither arm is adopted: no interval lies above zero. Blend made "
            f"{_eur(arms['blend']['pnl_eur'] - arms['median']['pnl_eur'])} against "
            "median dispatch, hold-back "
            f"{_eur(arms['hold_back']['pnl_eur'] - arms['median']['pnl_eur'])}."
        )
    else:
        best = max(adopted, key=lambda n: summary["difference"][n]["all"]["mean"])
        lines.append(
            f"Adopted: {', '.join(adopted)}. The higher mean is `{best}`, "
            f"{_interval(summary['difference'][best]['all'])} € a day against median "
            "dispatch."
        )
        if both is not None:
            lines.append(f"Blend minus hold-back: {_interval(both)} € a day.")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A spike probability, fed to dispatch."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument(
        "--quick", action="store_true", help=f"first {QUICK_DAYS} days only"
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    ev = settings.evaluation
    saved = pd.read_parquet(
        settings.data.processed_path
        / "forecasts"
        / "comparison"
        / f"{PRODUCTION}.parquet"
    )
    all_days = sorted(saved["target_day"].unique())
    if all_days[-1] >= ev.holdout_start:
        raise RuntimeError("the saved comparison reaches the hold-out")
    days = [d for d in all_days if d >= ev.validation_start]
    if args.quick:
        days = days[:QUICK_DAYS]
    inputs = pd.read_parquet(settings.data.inputs_path)
    inputs = inputs.loc[
        inputs.index < settings.market.local_midnight_utc(ev.holdout_start)
    ]

    started = time.perf_counter()
    table = day_table(inputs, settings)
    print(
        f"day table: {len(table)} days in {time.perf_counter() - started:.0f} s",
        flush=True,
    )
    every = REFIT_EVERY_DAYS[PRODUCTION]
    # Refits fall on the comparison run's calendar, which starts before validation.
    calendar = [d for d in all_days if d <= days[-1]]
    p_all = spike_probabilities(table, calendar, every)
    p = p_all.reindex(days)
    label = table["spike"].reindex(days)
    classifier = reliability(p, label)
    print(
        f"classifier: brier {classifier['brier']:.4f} vs base "
        f"{classifier['brier_base_rate']:.4f}, auc {classifier['auc']:.3f}",
        flush=True,
    )

    frames = {
        d: part
        for d, part in saved[saved["target_day"].isin(set(days))].groupby("target_day")
    }
    jobs = [(frames[d], float(p.loc[d]), args.config) for d in days]
    rows: list[dict[str, Any]] = []
    schedules: list[pd.DataFrame] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for day, results in pool.map(_day_job, jobs, chunksize=8):
            for r in results:
                schedule = r.pop("schedule").assign(
                    target_day=day, strategy=r["strategy"]
                )
                schedules.append(schedule)
                rows.append({"target_day": day, **r})
    print(
        f"traded {len(days)} days in {(time.perf_counter() - started) / 60:.1f} min",
        flush=True,
    )
    pnl = pd.DataFrame(rows)
    dispatch = pd.concat(schedules)
    profit = pnl.pivot(index="target_day", columns="strategy", values="pnl_eur")
    stored = pd.read_parquet(
        settings.data.processed_path / "trading" / PRODUCTION / "pnl_daily.parquet"
    ).pivot(index="target_day", columns="strategy", values="pnl_eur")
    drift = float(
        (profit["median"] - stored.loc[profit.index, "median_forecast"]).abs().max()
    )
    if drift > 0.01:
        raise RuntimeError(
            f"median dispatch misses the saved validation profit by €{drift:.4f}"
        )

    traded = list(profit.index)
    summer = pd.Series(
        pd.to_datetime(pd.Index(traded)).month.isin(sorted(SUMMER)), index=traded
    )
    spiky = label.reindex(traded).fillna(0.0).astype(bool)
    extra_arms = [
        f"hold_back_{t:g}" for t in REPORTED_THRESHOLDS if t != HOLD_BACK_THRESHOLD
    ]
    ceiling = float(profit["perfect_foresight"].sum())
    wear = float(settings.battery.degradation_eur_per_mwh)
    tz = settings.market.timezone

    def arm_summary(name: str) -> dict[str, Any]:
        part = pnl[pnl["strategy"] == name].set_index("target_day")
        return {
            "pnl_eur": float(part["pnl_eur"].sum()),
            "capture": float(part["pnl_eur"].sum()) / ceiling,
            "cycles_per_day": float(part["cycles"].mean()),
            "plan_minus_settled": float(
                (part["planned_value_eur"] - part["pnl_eur"]).mean()
            ),
            "fired_days": int(part["fired"].sum()),
        }

    def differences(name: str) -> dict[str, Any]:
        d = (profit[name] - profit["median"]).loc[traded]
        return {
            "all": mean_interval(d),
            "summer": mean_interval(d, summer),
            "winter": mean_interval(d, ~summer),
            "spike_days": mean_interval(d, spiky),
            "other_days": mean_interval(d, ~spiky),
        }

    blocks: dict[str, dict[str, Any]] = {}
    cash = {
        name: block_cash(
            dispatch[dispatch["strategy"] == name],
            tz,
            wear,
        )
        for name in ARMS
    }
    for name in ARMS[1:]:
        gap = (cash[name] - cash["median"]).loc[traded]
        blocks[name] = {block: mean_interval(gap[block]) for block in BLOCKS}

    summary: dict[str, Any] = {
        "threshold": settings.evaluation.spike_threshold_eur_mwh,
        "days": len(traded),
        "first_day": str(traded[0]),
        "last_day": str(traded[-1]),
        "perfect_foresight_pnl_eur": ceiling,
        "classifier": classifier,
        "arms": {name: arm_summary(name) for name in (*ARMS, *extra_arms)},
        "extra_arms": extra_arms,
        "difference": {name: differences(name) for name in (*ARMS[1:], *extra_arms)},
        "blocks": blocks,
        "blend_minus_hold_back": mean_interval(
            (profit["blend"] - profit["hold_back"]).loc[traded]
        ),
    }
    summary["adopted"] = [
        n for n in ARMS[1:] if summary["difference"][n]["all"]["low"] > 0
    ]
    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    prefix = "t1_spike_probability_quick" if args.quick else "t1_spike_probability"
    pnl.to_parquet(out / f"{prefix}_pnl.parquet")
    pd.DataFrame({"p": p, "spike": label}).to_parquet(out / f"{prefix}_days.parquet")
    (out / f"{prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    page = results_markdown(summary)
    print(page)
    if not args.quick:
        RESULTS_PATH.write_text(page, encoding="utf-8")
        print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
