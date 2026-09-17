"""T1 follow-up: do battery losses follow sunset rather than the clock?

    uv run python -m src.health.experiments.t1_sunset
    uv run python -m src.health.experiments.t1_sunset --quick

The T1 attribution splits the production model's gap to perfect foresight by fixed
blocks of local hours. On the validation window its 18:00 to 20:59 block takes 35%
of the gap from June to September but 16% from October to May, while 15:00 to 17:59
moves the other way, 8% against 23%. That is what one would see if the loss sat on
the price ramp as solar output falls away, which comes more than four hours earlier
in December than in June. This experiment tests it: the same Shapley split, with
windows anchored to each day's sunrise and sunset at the centre of Germany.

Windows, in UTC with boundaries rounded to the whole hour, so the four quarter-hours
of an hourly product always share a window:

* ``morning``: sunrise - 1 h to sunrise + 3 h
* ``midday``: sunrise + 3 h to sunset - 3 h
* ``pre_sunset``: sunset - 3 h to sunset
* ``post_sunset``: sunset to sunset + 3 h
* ``night``: everything else

A window cannot cross into the next delivery day, so when sunset rounds to 22:00
local the window after sunset is cut to two hours; the run counts those days.

Two criteria, fixed before the run, must both hold for the hypothesis to stand:

1. the six hours around sunset take a share of the gap that differs by less than
   10 points between June to September and October to May, and
2. over the whole window they take at least the share of the six-hour window
   formed by the 15-17 and 18-20 clock blocks.

After the run, moving-block bootstrap intervals were added for both criteria and
for the seasonal shift of each window, because the criteria alone compare point
estimates. Median dispatch on validation days only. The clock split is read from
the saved T1 attribution; the per-day gaps of the two splits must agree. Results:
``docs/results/t1_sunset.md`` and ``data/processed/experiments/t1_sunset_*``.
``--quick`` runs the first 60 days and writes nothing to ``docs/``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, timedelta
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import REPO_ROOT, load_settings
from src.trading.attribution import Windows, attribute_days
from src.trading.backtest import block_bootstrap_index, load_window, window_days
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

__all__ = [
    "SUNSET",
    "WINDOW_NAMES",
    "judge",
    "main",
    "results_markdown",
    "season_difference_interval",
    "share_difference_interval",
    "share_of_gap",
    "sun_times",
    "sun_windows",
    "window_hours",
]

PRODUCTION = "lightgbm_conformal"
#: The geographic centre of Germany.
LATITUDE = 51.1634
LONGITUDE = 10.4477
WINDOW_NAMES = ("night", "morning", "midday", "pre_sunset", "post_sunset")
SUNSET = ("pre_sunset", "post_sunset")
CLOCK_RAMP = ("15-17", "18-20")
CLOCK_EVENING = ("18-20",)
SUMMER = frozenset({6, 7, 8, 9})
WINTER = frozenset(set(range(1, 13)) - SUMMER)
STABILITY_POINTS = 10.0
GAP_TOLERANCE_EUR = 0.01
BLOCK_DAYS = 7
DRAWS = 5000
QUICK_DAYS = 60
#: What a cached split was computed with; a change here recomputes it.
SCHEME = (
    f"windows {','.join(WINDOW_NAMES)}; centre {LATITUDE}N {LONGITUDE}E; NOAA; "
    "boundaries rounded to the hour; sunrise -1/+3 h, sunset -3/+3 h"
)
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t1_sunset.md"
_HOUR = pd.Timedelta(hours=1)


def sun_times(
    day: date, latitude: float, longitude: float
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Sunrise and sunset on ``day`` in UTC, to within a few minutes.

    The NOAA general solar position equations, with the standard 0.833 degrees for
    refraction and the solar disc.
    """
    gamma = 2 * math.pi / 365 * (day.timetuple().tm_yday - 1)
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.040849 * math.sin(2 * gamma)
    )
    declination = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )
    lat = math.radians(latitude)
    cos_hour_angle = math.cos(math.radians(90.833)) / (
        math.cos(lat) * math.cos(declination)
    ) - math.tan(lat) * math.tan(declination)
    if not -1.0 <= cos_hour_angle <= 1.0:
        raise ValueError(f"the sun does not rise and set on {day} at {latitude}")
    hour_angle = math.degrees(math.acos(cos_hour_angle))
    midnight = pd.Timestamp(day, tz="UTC")
    rise = 720 - 4 * (longitude + hour_angle) - eqtime
    fall = 720 - 4 * (longitude - hour_angle) - eqtime
    return midnight + pd.Timedelta(minutes=rise), midnight + pd.Timedelta(minutes=fall)


def sun_windows(
    index: pd.DatetimeIndex,
    *,
    timezone: str,
    latitude: float = LATITUDE,
    longitude: float = LONGITUDE,
) -> NDArray[np.str_]:
    """The sun-anchored window of every period, by the UTC hour it starts in.

    Sunrise and sunset belong to the period's local delivery day and are rounded to
    the whole hour, so an hourly product is never split between windows.
    """
    if index.tz is None:
        raise ValueError("the index must be timezone-aware")
    hour_start = index.tz_convert("UTC").floor("h")
    local_days = index.tz_convert(timezone).date
    bounds: dict[date, tuple[pd.Timestamp, ...]] = {}
    labels: list[str] = []
    for day, start in zip(local_days, hour_start, strict=True):
        if day not in bounds:
            rise, fall = (t.round("h") for t in sun_times(day, latitude, longitude))
            if fall - 3 * _HOUR < rise + 3 * _HOUR:
                raise ValueError(f"{day} is too short for a midday window")
            bounds[day] = (
                rise - _HOUR,
                rise + 3 * _HOUR,
                fall - 3 * _HOUR,
                fall,
                fall + 3 * _HOUR,
            )
        morning, midday, pre_sunset, post_sunset, night = bounds[day]
        if morning <= start < midday:
            labels.append("morning")
        elif midday <= start < pre_sunset:
            labels.append("midday")
        elif pre_sunset <= start < post_sunset:
            labels.append("pre_sunset")
        elif post_sunset <= start < night:
            labels.append("post_sunset")
        else:
            labels.append("night")
    return np.array(labels, dtype=np.str_)


def window_hours(day: date, timezone: str) -> dict[str, float]:
    """Hours each window holds on one local delivery day."""
    start = pd.Timestamp(day, tz=timezone).tz_convert("UTC")
    end = pd.Timestamp(day + timedelta(days=1), tz=timezone).tz_convert("UTC")
    index = pd.date_range(start, end, freq="15min", inclusive="left")
    labels = sun_windows(pd.DatetimeIndex(index), timezone=timezone)
    return {name: float((labels == name).sum()) / 4 for name in WINDOW_NAMES}


def _months(frame: pd.DataFrame, months: Iterable[int] | None) -> pd.DataFrame:
    if months is None:
        return frame
    return frame[pd.to_datetime(frame["target_day"]).dt.month.isin(list(months))]


def share_of_gap(
    costs: pd.DataFrame,
    days: pd.DataFrame,
    windows: Sequence[str],
    *,
    months: Iterable[int] | None = None,
    direction: str | None = None,
) -> float:
    """The share of the gap to perfect foresight that ``windows`` take."""
    chosen = list(months) if months is not None else None
    selected = _months(costs, chosen)
    if direction is not None:
        selected = selected[selected["direction"] == direction]
    gap = float(_months(days, chosen)["gap_eur"].sum())
    if gap == 0:
        return math.nan
    return float(selected.loc[selected["block"].isin(windows), "cost_eur"].sum()) / gap


def judge(
    *, sunset_summer: float, sunset_winter: float, sunset_all: float, clock_all: float
) -> dict[str, Any]:
    """Apply the two criteria fixed before the run."""
    difference = 100 * abs(sunset_summer - sunset_winter)
    stable = difference < STABILITY_POINTS
    concentrated = sunset_all >= clock_all
    return {
        "season_difference_points": difference,
        "stable": stable,
        "concentrated": concentrated,
        "supported": stable and concentrated,
    }


def share_difference_interval(
    days: pd.DataFrame,
    *,
    block_days: int = BLOCK_DAYS,
    draws: int = DRAWS,
    seed: int = 7,
) -> tuple[float, float, float]:
    """Sunset minus clock share of the gap, in points, with a 95% interval.

    ``days`` has one row per consecutive day with the day's ``gap`` and the costs
    its ``sunset`` and ``clock`` windows take. Blocks of consecutive days are
    resampled and both shares recomputed on every draw.
    """
    values = days[["gap", "sunset", "clock"]].to_numpy(dtype=float)
    index = block_bootstrap_index(
        len(values), block_days=block_days, draws=draws, seed=seed
    )
    gap = values[index, 0].sum(axis=1)
    drawn = 100 * (values[index, 1].sum(axis=1) - values[index, 2].sum(axis=1)) / gap
    point = 100 * (values[:, 1].sum() - values[:, 2].sum()) / values[:, 0].sum()
    low, high = np.percentile(drawn, [2.5, 97.5])
    return float(point), float(low), float(high)


def season_difference_interval(
    days: pd.DataFrame,
    column: str,
    *,
    block_days: int = BLOCK_DAYS,
    draws: int = DRAWS,
    seed: int = 7,
) -> tuple[float, float, float]:
    """A window's share of the gap, June to September minus October to May.

    In points, with a 95% interval. ``days`` has one row per consecutive day with
    ``gap``, the window's cost in ``column`` and a boolean ``summer``. Each draw
    resamples blocks of consecutive days and recomputes both seasons' shares from
    the days it drew.
    """
    values = days[["gap", column]].to_numpy(dtype=float)
    summer = days["summer"].to_numpy(dtype=bool)

    def season_share(rows: Any, cost: Any, in_season: Any) -> Any:
        gap = (rows * in_season).sum(axis=-1)
        return np.divide(
            (cost * in_season).sum(axis=-1),
            gap,
            out=np.full(np.shape(gap), np.nan),
            where=gap != 0,
        )

    index = block_bootstrap_index(
        len(values), block_days=block_days, draws=draws, seed=seed
    )
    gap, cost, in_summer = values[index, 0], values[index, 1], summer[index]
    drawn = 100 * (
        season_share(gap, cost, in_summer) - season_share(gap, cost, ~in_summer)
    )
    point = 100 * (
        season_share(values[:, 0], values[:, 1], summer)
        - season_share(values[:, 0], values[:, 1], ~summer)
    )
    low, high = np.nanpercentile(drawn, [2.5, 97.5])
    return float(point), float(low), float(high)


def _includes_zero(interval: Mapping[str, float]) -> bool:
    return interval["low"] <= 0 <= interval["high"]


def results_markdown(summary: Mapping[str, Any]) -> str:
    """The results page, from the summary the run writes."""
    s = summary
    rows = s["shares"]
    intervals = s["intervals"]

    def pct(value: float | None) -> str:
        return "n/a" if value is None or math.isnan(value) else f"{100 * value:.1f}%"

    def points(r: Mapping[str, float]) -> str:
        return f"{r['mean']:+.1f} | {r['low']:+.1f} to {r['high']:+.1f}"

    def table(scheme: str, names: Sequence[str]) -> list[str]:
        lines = [
            "| window | all days | June to September | October to May | "
            "too low, all days |",
            "|---|---|---|---|---|",
        ]
        for name in names:
            r = rows[scheme][name]
            lines.append(
                f"| {name} | {pct(r['all'])} | {pct(r['summer'])} | "
                f"{pct(r['winter'])} | {pct(r['too_low'])} |"
            )
        return lines

    verdict = s["verdict"]
    truncated = s["truncated"]
    criteria = [
        "| criterion | measured | threshold | holds |",
        "|---|---|---|---|",
        "| 1. sunset six-hour share, June to September against October to May | "
        f"{verdict['season_difference_points']:.1f} points apart | under "
        f"{STABILITY_POINTS:.0f} points | {'yes' if verdict['stable'] else 'no'} |",
        "| 2. sunset six hours against the clock 15-17 and 18-20 blocks, all days | "
        f"{pct(s['sunset_all'])} against {pct(s['clock_all'])} | at least equal | "
        f"{'yes' if verdict['concentrated'] else 'no'} |",
    ]
    labels = {
        "sunset_minus_clock": "sunset six hours minus clock 15:00 to 20:59, all days",
        "sunset_season": "sunset six hours, June to September minus October to May",
        "clock_season": "clock 15:00 to 20:59, June to September minus October to May",
        "pre_sunset_season": "pre_sunset, June to September minus October to May",
        "post_sunset_season": "post_sunset, June to September minus October to May",
        "clock_15_17_season": "clock 15-17, June to September minus October to May",
        "clock_18_20_season": "clock 18-20, June to September minus October to May",
    }
    interval_table = [
        "| share of the gap, points | estimate | 95% interval |",
        "|---|---|---|",
        *[f"| {labels[key]} | {points(intervals[key])} |" for key in labels],
    ]

    if _includes_zero(intervals["sunset_season"]) and _includes_zero(
        intervals["clock_season"]
    ):
        criterion_one = (
            "Criterion 1 cannot separate the hypotheses: the seasonal difference of "
            "the sunset window and of the equally wide clock window both have "
            "intervals that include no difference and large ones."
        )
    else:
        criterion_one = (
            "At least one of the two six-hour windows has a seasonal difference "
            "clear of zero; see the table."
        )
    difference = intervals["sunset_minus_clock"]
    if _includes_zero(difference):
        criterion_two = (
            "Criterion 2 passes on the estimate, but the interval includes zero: the "
            "data cannot tell the sunset window from the clock window."
        )
    else:
        side = "more" if difference["low"] > 0 else "less"
        criterion_two = (
            f"The sunset six hours take {side} of the gap than the clock 15:00 to "
            "20:59 window, beyond sampling noise."
        )
    shifts = [
        f"{labels[key].split(',')[0]} {intervals[key]['mean']:+.1f} points "
        f"({intervals[key]['low']:+.1f} to {intervals[key]['high']:+.1f})"
        for key in (
            "pre_sunset_season",
            "post_sunset_season",
            "clock_15_17_season",
            "clock_18_20_season",
        )
        if not _includes_zero(intervals[key])
    ]

    def clear(key: str) -> bool:
        return not _includes_zero(intervals[key])

    moves = []
    if (
        clear("clock_18_20_season")
        and clear("clock_15_17_season")
        and intervals["clock_18_20_season"]["mean"] > 0
        and intervals["clock_15_17_season"]["mean"] < 0
    ):
        moves.append("later on the clock (from 15:00 to 17:59 into 18:00 to 20:59)")
    if (
        clear("pre_sunset_season")
        and clear("post_sunset_season")
        and intervals["pre_sunset_season"]["mean"] > 0
        and intervals["post_sunset_season"]["mean"] < 0
    ):
        moves.append("earlier against sunset (from after sunset to before it)")
    shift_sentence = (
        "Seasonal shifts clear of zero, June to September minus October to May: "
        + "; ".join(shifts)
        + "."
        + (
            f" In June to September the loss sits {' and '.join(moves)}."
            if moves
            else ""
        )
        if shifts
        else "No single window's seasonal shift is clear of zero."
    )
    months = [
        "| month | gap, € | sunset six hours | clock 15:00 to 20:59 | "
        "clock 18:00 to 20:59 |",
        "|---|---|---|---|---|",
        *[
            f"| {m['month']} | {m['gap_eur']:,.0f} | {pct(m['sunset'])} | "
            f"{pct(m['clock_ramp'])} | {pct(m['clock_evening'])} |"
            for m in s["by_month"]
        ],
    ]
    return "\n".join(
        [
            "# T1 follow-up: do battery losses follow sunset rather than the clock?",
            "",
            "Generated by `python -m src.health.experiments.t1_sunset`. Production "
            f"model, median dispatch, {s['days']} validation days from "
            f"{s['first_day']} to {s['last_day']}. The gap to perfect foresight, "
            f"€{s['gap_eur']:,.0f}, is split into Shapley shares twice: by the saved "
            "T1 clock blocks and by windows anchored to sunrise and sunset at the "
            f"centre of Germany ({LATITUDE}°N, {LONGITUDE}°E), boundaries rounded to "
            "the whole hour. The per-day gaps of the two splits agree to within "
            f"€{s['gap_max_difference_eur']:.2f}, so they divide the same loss. A "
            "Shapley share depends on how the periods are grouped, so the two splits "
            "are two readings of the same loss, not one measured twice. Both sample "
            "fix orders on every day; that noise is independent across days and is "
            "inside the intervals below.",
            "",
            f"On {truncated['days']} days, in months "
            f"{', '.join(str(m) for m in truncated['months'])}, sunset rounds to "
            "22:00 local and the window after it is cut at the end of the delivery "
            "day, so the sunset window holds five hours, not six. Those days carry "
            f"{pct(truncated['summer_gap_share'])} of the June to September gap. The "
            "cut lowers the summer sunset share, which works against the hypothesis.",
            "",
            "## Criteria, fixed before the run",
            "",
            *criteria,
            "",
            "Both criteria hold."
            if verdict["supported"]
            else "The criteria do not both hold.",
            "",
            "## Added after the run: what the criteria can and cannot show",
            "",
            "The criteria compare point estimates. Moving-block bootstrap, "
            f"{BLOCK_DAYS}-day blocks, {DRAWS:,} draws:",
            "",
            *interval_table,
            "",
            criterion_one,
            "",
            criterion_two,
            "",
            shift_sentence,
            "",
            "## Share of the gap by sun-anchored window",
            "",
            *table("sun", WINDOW_NAMES),
            "",
            "## Share of the gap by clock block (saved T1 split)",
            "",
            *table("clock", tuple(rows["clock"])),
            "",
            "## By month",
            "",
            *months,
            "",
        ]
    )


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.partial")
    frame.to_parquet(partial_path)
    os.replace(partial_path, path)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.partial")
    partial_path.write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    os.replace(partial_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Split the gap to perfect foresight by sunrise and sunset."
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
    tz = settings.market.timezone
    forecasts = load_window(settings, "validation", PRODUCTION)
    days, _ = window_days(
        forecasts, settings, "validation", (PERFECT_FORESIGHT, MEDIAN_FORECAST)
    )
    if args.quick:
        days = days[:QUICK_DAYS]
    out = settings.data.processed_path / "experiments"
    prefix = "t1_sunset_quick" if args.quick else "t1_sunset"
    costs_path = out / f"{prefix}_costs.parquet"
    days_path = out / f"{prefix}_days.parquet"
    scheme_path = out / f"{prefix}_scheme.json"

    cached = (
        days_path.exists()
        and costs_path.exists()
        and scheme_path.exists()
        and json.loads(scheme_path.read_text(encoding="utf-8")).get("scheme") == SCHEME
        and set(pd.read_parquet(days_path)["target_day"]) == set(days)
    )
    if cached:
        sun_costs, sun_days = pd.read_parquet(costs_path), pd.read_parquet(days_path)
        print(f"reused the sun-anchored split of {len(days)} days", flush=True)
    else:
        windows = Windows(
            names=WINDOW_NAMES,
            label=partial(
                sun_windows, timezone=tz, latitude=LATITUDE, longitude=LONGITUDE
            ),
        )
        print(f"splitting {len(days)} days with {args.workers} workers", flush=True)
        started = time.perf_counter()
        result = attribute_days(
            forecasts,
            days,
            (MEDIAN_FORECAST,),
            settings.battery,
            timezone=tz,
            holdout_start=settings.evaluation.holdout_start,
            workers=args.workers,
            time_limit_s=settings.trading.solver_time_limit_s,
            windows=windows,
        )
        if result.failed:
            raise RuntimeError(f"solver failed on {sorted(result.failed)}")
        sun_costs, sun_days = result.costs, result.days
        _write_parquet(sun_costs, costs_path)
        _write_parquet(sun_days, days_path)
        _write_json({"scheme": SCHEME}, scheme_path)
        print(f"done in {(time.perf_counter() - started) / 60:.1f} min", flush=True)

    saved = settings.data.processed_path / "backtest" / "attribution"
    wanted = set(days)
    clock_costs = pd.read_parquet(saved / "costs.parquet")
    clock_days = pd.read_parquet(saved / "days.parquet")
    clock_costs = clock_costs[
        (clock_costs["strategy"] == MEDIAN_FORECAST.name)
        & clock_costs["target_day"].isin(wanted)
    ]
    clock_days = clock_days[
        (clock_days["strategy"] == MEDIAN_FORECAST.name)
        & clock_days["target_day"].isin(wanted)
    ]
    if set(clock_days["target_day"]) != set(sun_days["target_day"]):
        raise RuntimeError(
            "the saved clock split covers other days; rerun the backtest"
        )
    gaps = sun_days.set_index("target_day")["gap_eur"].sub(
        clock_days.set_index("target_day")["gap_eur"]
    )
    gap_difference = float(gaps.abs().max())
    if gap_difference > GAP_TOLERANCE_EUR:
        raise RuntimeError(
            f"per-day gaps differ by up to €{gap_difference:.4f}; the splits do not "
            "divide the same loss"
        )

    def shares(
        costs: pd.DataFrame, day_frame: pd.DataFrame, names: Sequence[str]
    ) -> dict[str, Any]:
        return {
            name: {
                "all": share_of_gap(costs, day_frame, [name]),
                "summer": share_of_gap(costs, day_frame, [name], months=SUMMER),
                "winter": share_of_gap(costs, day_frame, [name], months=WINTER),
                "too_low": share_of_gap(costs, day_frame, [name], direction="under"),
            }
            for name in names
        }

    def per_window(costs: pd.DataFrame, blocks: Sequence[str]) -> pd.Series:
        chosen = costs[costs["block"].isin(blocks)]
        return chosen.groupby("target_day")["cost_eur"].sum()

    per_day = (
        pd.DataFrame(
            {
                "gap": sun_days.set_index("target_day")["gap_eur"],
                "sunset": per_window(sun_costs, SUNSET),
                "clock": per_window(clock_costs, CLOCK_RAMP),
                "pre_sunset": per_window(sun_costs, ["pre_sunset"]),
                "post_sunset": per_window(sun_costs, ["post_sunset"]),
                "clock_15_17": per_window(clock_costs, ["15-17"]),
                "clock_18_20": per_window(clock_costs, ["18-20"]),
            }
        )
        .fillna(0.0)
        .sort_index()
    )
    per_day["summer"] = pd.to_datetime(per_day.index).month.isin(sorted(SUMMER))

    def as_interval(values: tuple[float, float, float]) -> dict[str, float]:
        return dict(zip(("mean", "low", "high"), values, strict=True))

    intervals = {"sunset_minus_clock": as_interval(share_difference_interval(per_day))}
    for key, column in (
        ("sunset_season", "sunset"),
        ("clock_season", "clock"),
        ("pre_sunset_season", "pre_sunset"),
        ("post_sunset_season", "post_sunset"),
        ("clock_15_17_season", "clock_15_17"),
        ("clock_18_20_season", "clock_18_20"),
    ):
        intervals[key] = as_interval(season_difference_interval(per_day, column))

    short = [
        day for day in days if sum(window_hours(day, tz)[name] for name in SUNSET) < 6
    ]
    summer_gap = float(per_day.loc[per_day["summer"], "gap"].sum())
    short_gap = float(per_day.loc[per_day.index.isin(short), "gap"].sum())

    clock_names = tuple(dict.fromkeys(clock_costs["block"]))
    sunset_all = share_of_gap(sun_costs, sun_days, SUNSET)
    clock_all = share_of_gap(clock_costs, clock_days, CLOCK_RAMP)
    verdict = judge(
        sunset_summer=share_of_gap(sun_costs, sun_days, SUNSET, months=SUMMER),
        sunset_winter=share_of_gap(sun_costs, sun_days, SUNSET, months=WINTER),
        sunset_all=sunset_all,
        clock_all=clock_all,
    )
    by_month = [
        {
            "month": month,
            "gap_eur": float(_months(sun_days, [month])["gap_eur"].sum()),
            "sunset": share_of_gap(sun_costs, sun_days, SUNSET, months=[month]),
            "clock_ramp": share_of_gap(
                clock_costs, clock_days, CLOCK_RAMP, months=[month]
            ),
            "clock_evening": share_of_gap(
                clock_costs, clock_days, CLOCK_EVENING, months=[month]
            ),
        }
        for month in range(1, 13)
        if not _months(sun_days, [month]).empty
    ]
    summary: dict[str, Any] = {
        "days": len(days),
        "first_day": str(min(days)),
        "last_day": str(max(days)),
        "gap_eur": float(sun_days["gap_eur"].sum()),
        "gap_max_difference_eur": gap_difference,
        "sunset_all": sunset_all,
        "clock_all": clock_all,
        "sunset_summer": share_of_gap(sun_costs, sun_days, SUNSET, months=SUMMER),
        "sunset_winter": share_of_gap(sun_costs, sun_days, SUNSET, months=WINTER),
        "clock_ramp_summer": share_of_gap(
            clock_costs, clock_days, CLOCK_RAMP, months=SUMMER
        ),
        "clock_ramp_winter": share_of_gap(
            clock_costs, clock_days, CLOCK_RAMP, months=WINTER
        ),
        "verdict": verdict,
        "intervals": intervals,
        "truncated": {
            "days": len(short),
            "months": sorted({day.month for day in short}),
            "summer_gap_share": short_gap / summer_gap if summer_gap else math.nan,
        },
        "shares": {
            "sun": shares(sun_costs, sun_days, WINDOW_NAMES),
            "clock": shares(clock_costs, clock_days, clock_names),
        },
        "by_month": by_month,
    }
    _write_json(summary, out / f"{prefix}_summary.json")
    brief = {k: summary[k] for k in ("days", "sunset_all", "clock_all", "verdict")}
    print(
        json.dumps(
            brief | {"intervals": intervals, "truncated": summary["truncated"]},
            indent=2,
        ),
        flush=True,
    )

    if args.quick:
        print(f"quick run written to {out}", flush=True)
        return 0
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(results_markdown(summary), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
