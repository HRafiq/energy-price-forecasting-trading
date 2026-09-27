"""T8: how much of the profit is the shape of an ordinary day, not the forecast.

    uv run python -m src.health.experiments.t8_fixed_shape

The plan is ``docs/plans/fixed_shape_plan.md``, committed before this file.

Every baseline in this repository is a forecaster, including the naive one the
comparison presents as its floor. This adds two arms that forecast nothing. For
each delivery day they take the mean price of each local time of day over the
trailing 730 days ending before it, and solve the production optimiser once on
that average profile. The schedule uses nothing whatever about the day it
trades: it is climatology, not prediction.

``fixed_shape`` averages over every trailing day. ``fixed_shape_seasonal``
averages over the same calendar month only, so a June day is dispatched on what
past Junes looked like; that is the fairer version, because the summer and
winter shapes are not the same curve.

Averages are taken by local time of day and mapped onto the target day's own
index, so a 23 or 25 hour day needs no special case. The fixed arms are tied to
the same day-ahead products as production, so on a day that traded hourly
blocks they cannot quietly bid a quarter-hourly shape.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import (
    PRICE_SERIES,
    PRODUCT_COLUMN,
    REPO_ROOT,
    Settings,
    load_settings,
)
from src.health.experiments.t1_mechanism import SUMMER, mean_interval
from src.trading.battery import Battery
from src.trading.dispatch_lp import DayProgram, solve_day_gross
from src.trading.optimizer import period_hours, product_blocks
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

__all__ = [
    "ARMS",
    "TRAILING_DAYS",
    "results_markdown",
    "run",
    "settle",
    "shape_for",
    "trailing_window",
]

PRODUCTION = "lightgbm_conformal"
NAIVE = "naive_previous_day"
#: Matches the production model's training window, so neither side sees more.
TRAILING_DAYS = 730
#: The arms that forecast nothing.
ARMS = ("fixed_shape", "fixed_shape_seasonal")
#: How far the control may drift from the saved backtest profit, in euros.
RECONCILE_EUR = 0.01
OUT = "t8_fixed_shape"

Floats = NDArray[np.float64]


def settle(net: Floats, price: Floats, battery: Battery, dt: float) -> float:
    """What a schedule earned at realised prices, net of wear, as the backtest does."""
    discharge = np.clip(net, 0.0, None)
    wear = battery.degradation_eur_per_mwh
    return float(dt * (price @ net - wear * discharge.sum()))


def trailing_window(
    prices: pd.Series, start: pd.Timestamp, days: int = TRAILING_DAYS
) -> pd.Series:
    """Realised prices strictly before the delivery day, at most ``days`` back.

    This is the leakage guarantee of the whole experiment: the fixed arms see
    nothing stamped at or after the instant their delivery day begins.
    """
    return prices[
        (prices.index < start) & (prices.index >= start - pd.Timedelta(days=days))
    ]


def shape_for(
    history: pd.Series, index: pd.DatetimeIndex, timezone: str, month: int | None
) -> Floats:
    """The average price of each local time of day, mapped onto ``index``.

    ``history`` is realised prices strictly before the target day. With ``month``
    given, only days of that calendar month contribute. Averaging by local time
    of day rather than by position is what makes the 23 and 25 hour days fall
    out correctly: each local clock time takes the mean of the same clock time.
    """
    local = cast(pd.DatetimeIndex, history.index).tz_convert(timezone)
    if month is not None:
        history = history[local.month == month]
        local = cast(pd.DatetimeIndex, history.index).tz_convert(timezone)
    if history.empty:
        raise ValueError("no history to average")
    by_time = history.groupby(local.hour * 60 + local.minute).mean()
    wanted = index.tz_convert(timezone)
    shape = by_time.reindex(wanted.hour * 60 + wanted.minute)
    if bool(shape.isna().any()):
        # A clock time with no history at all: fall back to the window's mean.
        shape = shape.fillna(by_time.mean())
    return np.asarray(shape.to_numpy(), dtype="float64")


def _saved_pnl(settings: Settings, model: str, strategy: str) -> float | None:
    """The profit the committed backtest recorded for one model and strategy."""
    path = settings.data.processed_path / "backtest" / "decision-value" / "summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table[(table["model"] == model) & (table["strategy"] == strategy)]
    return float(rows["pnl_eur"].iloc[0]) if len(rows) else None


def _capture(pnl: float | None, ceiling: float | None) -> float | None:
    """A saved profit as a share of the saved ceiling, when both were recorded."""
    if pnl is None or not ceiling:
        return None
    return pnl / ceiling


def _program_for(
    index: pd.DatetimeIndex,
    minutes: pd.Series,
    battery: Battery,
    cache: dict[tuple[int, float, tuple[bool, ...]], DayProgram],
) -> DayProgram:
    """The day's LP, tied to the same day-ahead products production bid into."""
    dt = period_hours(index)
    products = product_blocks(index, minutes)
    ties = tuple(bool(products[t] == products[t - 1]) for t in range(1, len(index)))
    key = (len(index), dt, ties)
    if key not in cache:
        cache[key] = DayProgram.build(len(index), dt, battery, products)
    return cache[key]


def _day_rows(
    settings: Settings, days_limit: int | None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """One row per delivery day: every arm's profit at realised prices."""
    processed = settings.data.processed_path
    battery, tz = settings.battery, settings.market.timezone
    dispatch = pd.read_parquet(processed / "trading" / PRODUCTION / "dispatch.parquet")
    inputs = pd.read_parquet(
        settings.data.inputs_path, columns=[PRICE_SERIES, PRODUCT_COLUMN]
    )
    prices = inputs[PRICE_SERIES].dropna()

    control = dispatch[dispatch["strategy"] == MEDIAN_FORECAST.name]
    perfect = dispatch[dispatch["strategy"] == PERFECT_FORESIGHT.name]
    days = cast(list[date], sorted(control["target_day"].unique()))
    if days_limit:
        days = days[:days_limit]

    cache: dict[tuple[int, float, tuple[bool, ...]], DayProgram] = {}
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    for day in days:
        group = control[control["target_day"] == day]
        index = pd.DatetimeIndex(group.index)
        realised = group["realised_price"].to_numpy(dtype="float64")
        if not np.isfinite(realised).all():
            skipped["realised price missing"] = (
                skipped.get("realised price missing", 0) + 1
            )
            continue
        dt = period_hours(index)
        start = settings.market.local_midnight_utc(day)
        window = trailing_window(prices, start)
        minutes = inputs[PRODUCT_COLUMN].reindex(index)
        if window.empty or bool(minutes.isna().any()):
            skipped["history missing"] = skipped.get("history missing", 0) + 1
            continue
        program = _program_for(index, minutes, battery, cache)

        pf = perfect[perfect["target_day"] == day]
        row: dict[str, Any] = {
            "target_day": day,
            "median_forecast": settle(
                group["net_mw"].to_numpy(dtype="float64"), realised, battery, dt
            ),
            "perfect_foresight": settle(
                pf["net_mw"].to_numpy(dtype="float64"),
                pf["realised_price"].to_numpy(dtype="float64"),
                battery,
                dt,
            ),
        }
        for arm, month in (("fixed_shape", None), ("fixed_shape_seasonal", day.month)):
            shape = shape_for(window, index, tz, month)
            charge, discharge, _ = solve_day_gross(program, shape)
            # The LP has no binary stopping a period charging and discharging at
            # once, which at a deeply negative price becomes a way to dump energy
            # the production MILP would refuse. An averaged shape is far too
            # smooth to reach there, so the relaxation is tight and this schedule
            # is the MILP's; if a future window ever breaks that, fail loudly
            # rather than quietly report a number production could not have made.
            if bool(((charge > 1e-6) & (discharge > 1e-6)).any()):
                raise RuntimeError(
                    f"{arm} charges and discharges at once on {day}: "
                    "the relaxation is loose and this arm needs the MILP"
                )
            net = discharge - charge
            row[arm] = settle(net, realised, battery, dt)
            row[f"{arm}_cycles"] = float(
                dt * np.clip(net, 0.0, None).sum() / battery.capacity_mwh
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("target_day"), {
        k: float(v) for k, v in skipped.items()
    }


def run(settings: Settings, days_limit: int | None = None) -> dict[str, Any]:
    """Every arm over the validation days, settled at realised prices."""
    daily, skipped = _day_rows(settings, days_limit)
    scored = daily.dropna(subset=list(ARMS))
    summer = pd.Series([d.month in SUMMER for d in scored.index], index=scored.index)
    ceiling = float(scored["perfect_foresight"].sum())
    named = ("perfect_foresight", "median_forecast", *ARMS)

    # The control replays the saved schedule at the saved prices, so it must come
    # back to the committed backtest figure; a drift means the settlement here
    # does not match the settlement there and nothing below can be trusted.
    control_drift = None
    saved = _saved_pnl(settings, PRODUCTION, MEDIAN_FORECAST.name)
    if saved is not None and days_limit is None:
        control_drift = abs(float(scored["median_forecast"].sum()) - saved)
        if control_drift > RECONCILE_EUR:
            raise RuntimeError(
                f"the control is €{control_drift:,.2f} from the saved backtest"
            )

    summary: dict[str, Any] = {
        "days": len(scored),
        "first_day": str(scored.index.min()),
        "last_day": str(scored.index.max()),
        "trailing_days": TRAILING_DAYS,
        "skipped_days": skipped,
        "control_drift_eur": control_drift,
        # Against the saved ceiling, not this run's, so a --days subset cannot
        # divide a full-window profit by a partial one.
        "naive_capture": _capture(
            _saved_pnl(settings, NAIVE, MEDIAN_FORECAST.name),
            _saved_pnl(settings, PRODUCTION, PERFECT_FORESIGHT.name),
        ),
        "arms": {
            arm: {
                "pnl_eur": float(scored[arm].sum()),
                "capture": float(scored[arm].sum() / ceiling),
                "losing_days": int((scored[arm] < 0).sum()),
            }
            for arm in named
        },
        "cycles_per_day": {arm: float(scored[f"{arm}_cycles"].mean()) for arm in ARMS},
        "forecast_buys": {
            arm: {
                "all": mean_interval(scored["median_forecast"] - scored[arm]),
                "summer": mean_interval(
                    scored["median_forecast"] - scored[arm], summer
                ),
                "winter": mean_interval(
                    scored["median_forecast"] - scored[arm], ~summer
                ),
            }
            for arm in ARMS
        },
    }
    return summary | {"daily": daily}


def results_markdown(summary: dict[str, Any]) -> str:
    """The results page, with the decomposition the question asked for."""
    arms, buys = summary["arms"], summary["forecast_buys"]
    ceiling = arms["perfect_foresight"]["pnl_eur"]
    best = max(ARMS, key=lambda a: arms[a]["capture"])
    shape, model = arms[best]["capture"], arms["median_forecast"]["capture"]
    naive = summary.get("naive_capture")
    lines = [
        "# T8: how much of the profit is the shape, not the forecast",
        "",
        f"Generated by `python -m src.health.experiments.{OUT}`. "
        f"{summary['days']} validation days, {summary['first_day']} to "
        f"{summary['last_day']}. The fixed arms average the trailing "
        f"{summary['trailing_days']} days and forecast nothing. Plan: "
        "`docs/plans/fixed_shape_plan.md`, committed before the code.",
        "",
        "## Arms",
        "",
        "| arm | profit | capture | losing days | cycles a day |",
        "|---|---|---|---|---|",
    ]
    for arm in ("perfect_foresight", "median_forecast", *ARMS):
        cycles = summary["cycles_per_day"].get(arm)
        lines.append(
            f"| {arm} | €{arms[arm]['pnl_eur']:,.0f} | {arms[arm]['capture']:.1%} | "
            f"{arms[arm]['losing_days']} | {f'{cycles:.2f}' if cycles else ''} |"
        )
    lines += [
        "",
        "## The decomposition",
        "",
        f"* the shape of an ordinary day, with no forecast at all: **{shape:.1%}**",
        f"* what the production forecast adds on top of it: **{model - shape:+.1%}**",
        f"* what the forecast still leaves on the table: **{1 - model:.1%}**",
        "",
        f"The best fixed arm is `{best}`, at €{arms[best]['pnl_eur']:,.0f} against the "
        f"ceiling's €{ceiling:,.0f}.",
    ]
    if naive is not None:
        lines += [
            "",
            f"For scale, the `{NAIVE}` forecast, which the backtest table presents as "
            f"its floor, captures {naive:.1%}. It is a forecast; these arms are not.",
        ]
    lines += [
        "",
        "## What the forecast buys over each fixed arm, euros per day",
        "",
        "A 95% moving-block bootstrap interval, 7-day blocks, 5,000 draws.",
        "",
        "| arm | all days | summer | winter |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        cells = [
            f"{buys[arm][s]['mean']:+.2f} ({buys[arm][s]['low']:+.2f} to "
            f"{buys[arm][s]['high']:+.2f})"
            for s in ("all", "summer", "winter")
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fixed-shape benchmark.")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--days", type=int, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    summary = run(settings, days_limit=args.days)
    daily = summary.pop("daily")
    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out / f"{OUT}_daily.parquet")
    (out / f"{OUT}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    page = REPO_ROOT / "docs" / "results" / f"{OUT}.md"
    page.write_text(results_markdown(summary), encoding="utf-8")
    print(results_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
