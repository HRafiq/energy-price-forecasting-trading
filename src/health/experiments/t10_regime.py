"""T10: does the forecast-free rule survive a different regime.

    uv run python -m src.health.experiments.t10_regime

The plan is ``docs/plans/regime_test_plan.md``, committed before this file.

T8's rule averages the trailing 730 days of the same calendar month, so it is a
lagging estimator of the shape of a day. It should fail when that shape is
moving, not when prices are merely large: the battery trades spreads, and 2022's
shock was mostly a shock to the level. This runs the same arms over the gas
crisis and over the published validation window, with a trivial forecast
alongside, and asks whether the ordering changes.

Nothing here needs a trained model, which is what makes the crisis window
reachable at all: every arm is solved from realised prices or from averages of
them.
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
from scipy.stats import spearmanr

from src.config import (
    PRICE_SERIES,
    PRODUCT_COLUMN,
    REPO_ROOT,
    Settings,
    load_settings,
)
from src.health.experiments.t1_mechanism import SUMMER
from src.health.experiments.t8_fixed_shape import (
    RECONCILE_EUR,
    settle,
    shape_for,
    trailing_window,
)
from src.trading.battery import Battery
from src.trading.dispatch_lp import DayProgram, solve_day_gross
from src.trading.optimizer import optimize_dispatch, period_hours, product_blocks
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

__all__ = [
    "ARMS",
    "ORDINARY",
    "WINDOWS",
    "results_markdown",
    "run",
    "solve_exactly",
    "window_rows",
]

PRODUCTION = "lightgbm_conformal"
#: Two 730-day windows, each with a full 730 days of price history behind it.
WINDOWS = {
    "crisis": (date(2021, 6, 1), date(2023, 5, 31)),
    "recent": (date(2024, 6, 1), date(2026, 5, 31)),
}
#: Quarter-hourly day-ahead products began here, part-way through ``recent``.
QUARTER_HOUR_FROM = date(2025, 10, 1)
ARMS = ("perfect_foresight", "fixed_shape", "fixed_shape_seasonal", "yesterday")
#: T9's threshold for a day that kept its seasonal ranking of periods.
ORDINARY = 0.90
OUT = "t10_regime"

Floats = NDArray[np.float64]


def solve_exactly(
    program: DayProgram,
    prices: Floats,
    index: pd.DatetimeIndex,
    products: NDArray[np.int64],
    battery: Battery,
) -> tuple[Floats, bool]:
    """The production MILP's schedule, taking the fast LP route where it is tight.

    The LP relaxation has no binary forbidding a period from charging and
    discharging at once. At a sufficiently negative price that becomes a way to
    dump energy, and the relaxation takes it: on about one day in a hundred here,
    and more often in the recent window, which has far more negative prices than
    the crisis window did. Left alone it would inflate one window's ceiling more
    than the other's, so those days are re-solved with the production optimiser
    and everything else keeps the LP's speed.
    """
    charge, discharge, _ = solve_day_gross(program, prices)
    if not bool(((charge > 1e-6) & (discharge > 1e-6)).any()):
        return discharge - charge, False
    exact = optimize_dispatch(
        pd.Series(prices, index=index), battery, products=products, integer=True
    )
    return np.asarray(exact.schedule["net_mw"].to_numpy(), dtype="float64"), True


def _day_index(
    prices: pd.Series, start: pd.Timestamp, end: pd.Timestamp
) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        prices.index[(prices.index >= start) & (prices.index < end)]
    )


def window_rows(
    settings: Settings, first: date, last: date, programs: dict[Any, DayProgram]
) -> pd.DataFrame:
    """Every arm's profit for each delivery day in one window."""
    battery, tz = settings.battery, settings.market.timezone
    inputs = pd.read_parquet(
        settings.data.inputs_path, columns=[PRICE_SERIES, PRODUCT_COLUMN]
    )
    prices = inputs[PRICE_SERIES].dropna()

    rows: list[dict[str, Any]] = []
    for day in pd.date_range(first, last, freq="D").date:
        start = settings.market.local_midnight_utc(day)
        end = settings.market.local_midnight_utc(
            day + pd.Timedelta(days=1).to_pytimedelta()
        )
        index = _day_index(prices, start, end)
        if len(index) < 20:
            continue
        realised = np.asarray(prices.reindex(index).to_numpy(), dtype="float64")
        window = trailing_window(prices, start)
        minutes = inputs[PRODUCT_COLUMN].reindex(index)
        if (
            window.empty
            or bool(minutes.isna().any())
            or not np.isfinite(realised).all()
        ):
            continue
        dt = period_hours(index)
        products = product_blocks(index, minutes)
        ties = tuple(bool(products[t] == products[t - 1]) for t in range(1, len(index)))
        key = (len(index), dt, ties)
        if key not in programs:
            programs[key] = DayProgram.build(len(index), dt, battery, products)
        program = programs[key]

        seasonal = shape_for(window, index, tz, day.month)
        prior = trailing_window(prices, start, days=1)
        shapes: dict[str, Floats] = {
            "perfect_foresight": realised,
            "fixed_shape": shape_for(window, index, tz, None),
            "fixed_shape_seasonal": seasonal,
        }
        if not prior.empty:
            shapes["yesterday"] = shape_for(prior, index, tz, None)

        row: dict[str, Any] = {
            "target_day": day,
            # The product traded, not the resolution of the series: before
            # 2025-10-01 the quarter-hour price series just repeats the hourly
            # value four times, so the period length says nothing about it.
            "quarter_hourly": bool((minutes == 15).all()),
            # T9's description of the day, which needs no dispatch at all.
            "shape_agreement": float(spearmanr(realised, seasonal).statistic),
            "spread": float(realised.max() - realised.min()),
        }
        for arm, shape in shapes.items():
            net, relaxed = solve_exactly(program, shape, index, products, battery)
            row[f"{arm}_milp"] = relaxed
            row[arm] = settle(net, realised, battery, dt)
            row[f"{arm}_cycles"] = float(
                dt * np.clip(net, 0.0, None).sum() / battery.capacity_mwh
            )
        rows.append(row)
    return pd.DataFrame(rows).set_index("target_day")


def _saved(settings: Settings, strategy: str) -> float | None:
    path = settings.data.processed_path / "backtest" / "decision-value" / "summary.csv"
    if not path.exists():
        return None
    table = pd.read_csv(path)
    rows = table[(table["model"] == PRODUCTION) & (table["strategy"] == strategy)]
    return float(rows["pnl_eur"].iloc[0]) if len(rows) else None


def _profile(frame: pd.DataFrame) -> dict[str, Any]:
    """What one window of days looks like, and what each arm made of it."""
    ceiling = float(frame["perfect_foresight"].sum())
    present = [arm for arm in ARMS if arm in frame]
    return {
        "days": len(frame),
        "first_day": str(frame.index.min()),
        "last_day": str(frame.index.max()),
        "mean_price_spread_eur": float(frame["spread"].mean()),
        "ordinary_day_share": float((frame["shape_agreement"] >= ORDINARY).mean()),
        "days_resolved_with_the_milp": {
            arm: int(frame[f"{arm}_milp"].sum()) for arm in present
        },
        "mean_shape_agreement": float(frame["shape_agreement"].mean()),
        "arms": {
            arm: {
                "pnl_eur": float(frame[arm].sum()),
                "capture": float(frame[arm].sum() / ceiling),
                "losing_days": int((frame[arm] < 0).sum()),
                "cycles_per_day": float(frame[f"{arm}_cycles"].mean()),
            }
            for arm in present
        },
    }


def run(settings: Settings) -> dict[str, Any]:
    """Both windows, every arm, and the verdict on each prediction."""
    programs: dict[Any, DayProgram] = {}
    daily = {
        name: window_rows(settings, first, last, programs)
        for name, (first, last) in WINDOWS.items()
    }
    windows = {name: _profile(frame) for name, frame in daily.items()}

    # The ceiling here is solved from realised prices, exactly as the committed
    # backtest solves it, so on the published window the two must agree.
    drift = None
    saved = _saved(settings, PERFECT_FORESIGHT.name)
    if saved is not None:
        drift = abs(windows["recent"]["arms"]["perfect_foresight"]["pnl_eur"] - saved)
        if drift > RECONCILE_EUR:
            raise RuntimeError(f"the ceiling is €{drift:,.2f} from the saved backtest")
    production = _saved(settings, MEDIAN_FORECAST.name)
    if production is not None:
        windows["recent"]["arms"]["median_forecast"] = {
            "pnl_eur": production,
            "capture": production
            / windows["recent"]["arms"]["perfect_foresight"]["pnl_eur"],
            "losing_days": None,
            "cycles_per_day": None,
        }

    # The plan meant to compare the hourly part of `recent` with the wholly
    # hourly `crisis`, to separate the product change from the regime. That
    # comparison cannot be made: quarter-hourly day-ahead began in October 2025
    # and has not yet covered a summer, so its days are all winter days and the
    # split would measure the season, not the product. The season split is
    # reported for both windows instead, which is like-for-like on both counts.
    seasons = {
        name: {
            label: _profile(part)
            for label, part in (
                ("Jun-Sep", frame[[d.month in SUMMER for d in frame.index]]),
                ("Oct-May", frame[[d.month not in SUMMER for d in frame.index]]),
            )
            if len(part)
        }
        for name, frame in daily.items()
    }
    quarter_hourly = daily["recent"]["quarter_hourly"]
    product_overlap = {
        "quarter_hourly_days": int(quarter_hourly.sum()),
        "of_which_summer": int(
            sum(
                d.month in SUMMER
                for d in daily["recent"].index[quarter_hourly.to_numpy(dtype=bool)]
            )
        ),
    }

    fixed = "fixed_shape_seasonal"
    capture_drop = (
        windows["recent"]["arms"][fixed]["capture"]
        - windows["crisis"]["arms"][fixed]["capture"]
    )
    verdicts = {
        "R1_capture_falls_5_points": bool(capture_drop >= 0.05),
        "R2_yesterday_overtakes_the_rule": bool(
            windows["crisis"]["arms"]["yesterday"]["capture"]
            > windows["crisis"]["arms"][fixed]["capture"]
        ),
        "R3_fewer_ordinary_days": bool(
            windows["crisis"]["ordinary_day_share"]
            < windows["recent"]["ordinary_day_share"]
        ),
    }
    return {
        "windows": windows,
        "seasons": seasons,
        "product_overlap": product_overlap,
        "ceiling_drift_eur": drift,
        "capture_drop_points": capture_drop,
        "ordinary_threshold": ORDINARY,
        "yesterday_loses_in_recent": bool(
            windows["recent"]["arms"]["yesterday"]["capture"]
            < windows["recent"]["arms"][fixed]["capture"]
        ),
        "verdicts": verdicts,
        "daily": daily,
    }


def yearly_ordinary_share(settings: Settings) -> pd.DataFrame:
    """The share of days that kept their seasonal ranking, by calendar year.

    Needs only realised prices, so it covers everything the history allows and
    not just the two windows: the mechanism R3 tests, shown in full.
    """
    tz = settings.market.timezone
    prices = pd.read_parquet(settings.data.inputs_path, columns=[PRICE_SERIES])[
        PRICE_SERIES
    ].dropna()
    first = cast(pd.Timestamp, prices.index.min()) + pd.Timedelta(days=730)
    rows: list[dict[str, Any]] = []
    for day in pd.date_range(first.date(), prices.index.max().date(), freq="D").date:
        start = settings.market.local_midnight_utc(day)
        end = settings.market.local_midnight_utc(
            day + pd.Timedelta(days=1).to_pytimedelta()
        )
        index = _day_index(prices, start, end)
        window = trailing_window(prices, start)
        if len(index) < 20 or window.empty:
            continue
        realised = np.asarray(prices.reindex(index).to_numpy(), dtype="float64")
        if not np.isfinite(realised).all():
            continue
        agreement = float(
            spearmanr(realised, shape_for(window, index, tz, day.month)).statistic
        )
        rows.append({"year": day.year, "agreement": agreement})
    frame = pd.DataFrame(rows)
    # Named aggregations on a SeriesGroupBy give back a frame, one row per year.
    return cast(
        pd.DataFrame,
        frame.groupby("year")["agreement"].agg(
            days="size",
            mean_agreement="mean",
            ordinary_share=lambda s: float((s >= ORDINARY).mean()),
        ),
    )


def results_markdown(summary: dict[str, Any], yearly: pd.DataFrame) -> str:
    """The results page, and the verdict on each prediction."""
    w, v = summary["windows"], summary["verdicts"]
    lines = [
        "# T10: does the forecast-free rule survive a different regime",
        "",
        f"Generated by `python -m src.health.experiments.{OUT}`. Two 730-day "
        f"windows: `crisis` {w['crisis']['first_day']} to {w['crisis']['last_day']}, "
        f"`recent` {w['recent']['first_day']} to {w['recent']['last_day']}. "
        "Plan: `docs/plans/regime_test_plan.md`, committed before the code.",
        "",
        "## Verdicts on the predictions, fixed before the run",
        "",
        "| prediction | result |",
        "|---|---|",
        f"| **R1** the rule's capture falls at least 5 points in the crisis | "
        f"{'HOLDS' if v['R1_capture_falls_5_points'] else 'fails'} "
        f"({summary['capture_drop_points'] * 100:+.1f} points) |",
        f"| **R2** a trivial forecast overtakes the rule in the crisis | "
        f"{'HOLDS' if v['R2_yesterday_overtakes_the_rule'] else 'fails'} |",
        f"| **R3** fewer days keep their seasonal ranking in the crisis | "
        f"{'HOLDS' if v['R3_fewer_ordinary_days'] else 'fails'} |",
        "",
        f"`yesterday` loses to the rule in the recent window: "
        f"{summary['yesterday_loses_in_recent']}, which is what makes R2 a "
        "crossover rather than a level comparison.",
        "",
        "## The arms in each window",
        "",
        "Capture is a share of that window's own ceiling. The ceilings differ by "
        "a factor, so the euro columns are not comparable across windows and the "
        "capture columns are the only fair comparison.",
        "",
        "| window | arm | profit | capture | losing days | cycles/day |",
        "|---|---|---|---|---|---|",
    ]
    for name in ("crisis", "recent"):
        for arm, item in w[name]["arms"].items():
            losing = "" if item["losing_days"] is None else str(item["losing_days"])
            cycles = (
                ""
                if item["cycles_per_day"] is None
                else f"{item['cycles_per_day']:.2f}"
            )
            lines.append(
                f"| {name} | {arm} | €{item['pnl_eur']:,.0f} | "
                f"{item['capture']:.1%} | {losing} | {cycles} |"
            )
    lines += [
        "",
        "## What the days themselves looked like",
        "",
        f"An ordinary day is one whose ranking of periods agrees with its seasonal "
        f"shape at {summary['ordinary_threshold']:.2f} or better, T9's threshold.",
        "",
        "| window | mean within-day spread | mean agreement | ordinary days |",
        "|---|---|---|---|",
    ]
    for name in ("crisis", "recent"):
        lines.append(
            f"| {name} | €{w[name]['mean_price_spread_eur']:.0f}/MWh | "
            f"{w[name]['mean_shape_agreement']:.3f} | "
            f"{w[name]['ordinary_day_share']:.1%} |"
        )
    overlap = summary["product_overlap"]
    lines += [
        "",
        "### The confounder named in the plan, and why it cannot be removed",
        "",
        "The plan meant to compare the hourly part of `recent` against the wholly "
        "hourly `crisis`, to tell the product change apart from the regime. That "
        "comparison is not available: of the "
        f"{overlap['quarter_hourly_days']} quarter-hourly days in `recent`, "
        f"{overlap['of_which_summer']} fall in June to September. Quarter-hourly "
        "day-ahead has not yet traded a summer, so splitting on the product would "
        "measure the season instead. It can be revisited after summer 2027.",
        "",
        "The season split is reported instead, which is like-for-like on both "
        "counts and is the comparison the regime question actually needs.",
        "",
        "| window | season | days | fixed_shape_seasonal capture | ordinary days |",
        "|---|---|---|---|---|",
    ]
    for name in ("crisis", "recent"):
        for label, part in summary["seasons"][name].items():
            lines.append(
                f"| {name} | {label} | {part['days']} | "
                f"{part['arms']['fixed_shape_seasonal']['capture']:.1%} | "
                f"{part['ordinary_day_share']:.1%} |"
            )
    lines += [
        "",
        "## R3 in full: how ordinary each year was",
        "",
        "Realised prices only, no dispatch and no model, so this covers every year "
        "the history reaches back far enough to describe.",
        "",
        "| year | days | mean agreement | ordinary days |",
        "|---|---|---|---|",
    ]
    for year, row in yearly.iterrows():
        lines.append(
            f"| {year} | {int(row['days'])} | {row['mean_agreement']:.3f} | "
            f"{row['ordinary_share']:.1%} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="The regime test.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    summary = run(settings)
    daily = summary.pop("daily")
    yearly = yearly_ordinary_share(settings)

    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in daily.items():
        frame.to_parquet(out / f"{OUT}_{name}_daily.parquet")
    yearly.to_parquet(out / f"{OUT}_yearly.parquet")
    (out / f"{OUT}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    page = results_markdown(summary, yearly)
    (REPO_ROOT / "docs" / "results" / f"{OUT}.md").write_text(page, encoding="utf-8")
    print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
