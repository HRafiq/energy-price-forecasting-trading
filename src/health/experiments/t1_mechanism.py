"""T1 follow-up: how the 15:00 to 21:00 window loses money.

    uv run python -m src.health.experiments.t1_mechanism

The T1 split put about 40% of the production model's gap to perfect foresight in
the 15-17 and 18-20 clock blocks on the validation window, and the sunset follow-up
showed the loss is not better placed by the sun. This reads what the two saved
schedules did inside that window, median dispatch against perfect foresight, to see
how the money is lost there. Nothing is re-solved: the schedules and their daily
profit come from ``data/processed/trading/lightgbm_conformal/`` and the Shapley
costs from ``data/processed/backtest/attribution/``, and the run refuses to continue
unless the schedules reproduce the saved profit and the attribution's gaps.

Three readings, in local time:

1. **Window cash, split exactly.** Each schedule discharges ``E`` MWh at an average
   realised price ``P``, charges ``C`` MWh at ``Q`` and pays wear ``w`` on every MWh
   discharged. Perfect foresight's window cash minus the median schedule's splits
   into five parts with midpoint weights, which add up to the difference with no
   remainder: discharge volume ``dE * mean(P)``, discharge price ``dP * mean(E)``,
   charge volume ``-dC * mean(Q)``, charge price ``-dQ * mean(C)`` and wear
   ``-w * dE``. A schedule that does not trade on a side takes the other schedule's
   price for it, so that side's difference is all volume. The same cash gap is
   also given for each of the six clock blocks of the T1 split over the whole day,
   so it shows where the two schedules' cash does part, next to where the Shapley
   split puts the forecast errors' cost.
2. **Day mechanisms, weighted by money.** Each day's Shapley cost in the two window
   blocks is attached to flags whose thresholds were fixed before the data was
   read. 0.25 MWh is about one quarter-hour at the battery's full 1 MW.

   * ``timing``: the two schedules discharge within 0.25 MWh of each other in the
     window, and perfect foresight gets the higher average price.
   * ``emptier``: the median schedule reaches 15:00 with at least 0.25 MWh less
     stored energy than perfect foresight.
   * ``held_back``: it leaves the window at 21:00 with at least 0.25 MWh more.
   * ``fewer_cycles``: it makes at least half a cycle fewer over the whole day.

   A day can carry several flags, so the shares overlap.
3. **Where the forecast put the window's peak.** The hour with the highest mean
   median forecast inside the window, against the hour with the highest mean
   realised price, weighted by the same Shapley cost.

Intervals are 95% moving-block bootstraps over 7-day blocks, 5,000 draws, recomputing
each statistic from the days drawn. The reading at the end of the results page was
written after a read-only first pass of readings 1 and 2 had been seen, so it is a
reading of the evidence, not a test fixed in advance. Validation days only. Results:
``docs/results/t1_mechanism.md`` and ``data/processed/experiments/t1_mechanism_*``.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_settings
from src.trading.attribution import HOUR_BLOCKS
from src.trading.backtest import block_bootstrap_index
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

__all__ = [
    "BLOCKS",
    "EFFECTS",
    "FLAGS",
    "OFFSETS",
    "block_cash",
    "decompose",
    "flag_days",
    "main",
    "mean_interval",
    "peak_hours",
    "peak_offsets",
    "results_markdown",
    "share_interval",
    "window_schedule",
]

PRODUCTION = "lightgbm_conformal"
WINDOW_START_HOUR = 15
WINDOW_END_HOUR = 21
WINDOW_BLOCKS = ("15-17", "18-20")
#: One quarter-hour at the battery's full 1 MW.
MIN_MWH = 0.25
MIN_CYCLES = 0.5
PERIOD_HOURS = 0.25
SUMMER = frozenset({6, 7, 8, 9})
BLOCK_DAYS = 7
DRAWS = 5000
SEED = 7
RECONCILE_EUR = 0.01
FLAGS = ("timing", "emptier", "held_back", "fewer_cycles")
EFFECTS = (
    "discharge_volume",
    "discharge_price",
    "charge_volume",
    "charge_price",
    "wear",
)
OFFSETS = ("same_hour", "one_hour", "two_or_more_hours")
BLOCKS = tuple(name for name, _ in HOUR_BLOCKS)
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t1_mechanism.md"


def window_schedule(dispatch: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """One schedule's day in the window: energy, cash and stored energy at its edges.

    ``dispatch`` holds one strategy's quarter-hours, indexed by UTC timestamp, with
    ``target_day``, ``charge_mw``, ``discharge_mw``, end-of-period ``soc_mwh`` and
    ``realised_price``. Stored energy "at 15:00" is the level at the end of the last
    quarter-hour before it, and "at 21:00" the level at the end of the window.
    """
    frame = dispatch.sort_index()
    hour = pd.DatetimeIndex(frame.index).tz_convert(timezone).hour
    inside = (hour >= WINDOW_START_HOUR) & (hour < WINDOW_END_HOUR)
    before = hour < WINDOW_START_HOUR
    discharged = PERIOD_HOURS * frame["discharge_mw"]
    charged = PERIOD_HOURS * frame["charge_mw"]
    work = pd.DataFrame(
        {
            "target_day": frame["target_day"],
            "discharged_mwh": discharged.where(inside, 0.0),
            "discharge_eur": (discharged * frame["realised_price"]).where(inside, 0.0),
            "charged_mwh": charged.where(inside, 0.0),
            "charge_eur": (charged * frame["realised_price"]).where(inside, 0.0),
        }
    )
    days = work.groupby("target_day").sum()
    days["soc_at_start_mwh"] = frame.loc[before].groupby("target_day")["soc_mwh"].last()
    days["soc_at_end_mwh"] = frame.loc[inside].groupby("target_day")["soc_mwh"].last()
    return days.sort_index()


def block_cash(
    dispatch: pd.DataFrame, timezone: str, wear_eur_per_mwh: float
) -> pd.DataFrame:
    """One schedule's cash per day in each clock block, wear included.

    The blocks are T1's local-hour blocks and cover the whole day, so a day's
    blocks add up to its profit.
    """
    frame = dispatch.sort_index()
    hour = pd.DatetimeIndex(frame.index).tz_convert(timezone).hour
    block = pd.Series(hour, index=frame.index).map(
        {hour: name for name, hours in HOUR_BLOCKS for hour in hours}
    )
    discharged = PERIOD_HOURS * frame["discharge_mw"]
    charged = PERIOD_HOURS * frame["charge_mw"]
    cash = (discharged - charged) * frame["realised_price"] - (
        wear_eur_per_mwh * discharged
    )
    table = cash.groupby([frame["target_day"], block]).sum().unstack(fill_value=0)
    return table.reindex(columns=list(BLOCKS), fill_value=0.0).sort_index()


def _average_price(eur: pd.Series, mwh: pd.Series) -> pd.Series:
    return eur / mwh.where(mwh > 0)


def decompose(
    median: pd.DataFrame, foresight: pd.DataFrame, wear_eur_per_mwh: float
) -> pd.DataFrame:
    """Perfect foresight's window cash minus the median schedule's, in five parts.

    Both frames come from :func:`window_schedule` and share their days. The parts
    add up to ``cash_gap_eur`` exactly; see the module docstring for the formula.
    """
    if not median.index.equals(foresight.index):
        raise ValueError("the two schedules cover different days")
    rows = {}
    for side, energy, eur in (
        ("discharge", "discharged_mwh", "discharge_eur"),
        ("charge", "charged_mwh", "charge_eur"),
    ):
        e_med, e_pf = median[energy], foresight[energy]
        p_med = _average_price(median[eur], e_med)
        p_pf = _average_price(foresight[eur], e_pf)
        # A schedule that did not trade this side has no price: take the other's,
        # so the difference is all volume; with neither trading, both terms are 0.
        p_med, p_pf = p_med.fillna(p_pf).fillna(0.0), p_pf.fillna(p_med).fillna(0.0)
        sign = 1.0 if side == "discharge" else -1.0
        rows[f"{side}_volume"] = sign * (e_pf - e_med) * (p_pf + p_med) / 2
        rows[f"{side}_price"] = sign * (p_pf - p_med) * (e_pf + e_med) / 2
    rows["wear"] = -wear_eur_per_mwh * (
        foresight["discharged_mwh"] - median["discharged_mwh"]
    )

    def cash(schedule: pd.DataFrame) -> pd.Series:
        return (
            schedule["discharge_eur"]
            - schedule["charge_eur"]
            - wear_eur_per_mwh * schedule["discharged_mwh"]
        )

    out = pd.DataFrame(rows)[list(EFFECTS)]
    out["cash_gap_eur"] = cash(foresight) - cash(median)
    return out


def flag_days(
    median: pd.DataFrame,
    foresight: pd.DataFrame,
    cycles_median: pd.Series,
    cycles_foresight: pd.Series,
) -> pd.DataFrame:
    """The four mechanism flags for each day, with the thresholds fixed in advance."""
    price_med = _average_price(median["discharge_eur"], median["discharged_mwh"])
    price_pf = _average_price(foresight["discharge_eur"], foresight["discharged_mwh"])
    energy_gap = foresight["discharged_mwh"] - median["discharged_mwh"]
    flags = pd.DataFrame(
        {
            "timing": (energy_gap.abs() < MIN_MWH) & (price_pf > price_med),
            "emptier": foresight["soc_at_start_mwh"] - median["soc_at_start_mwh"]
            >= MIN_MWH,
            "held_back": median["soc_at_end_mwh"] - foresight["soc_at_end_mwh"]
            >= MIN_MWH,
            "fewer_cycles": cycles_foresight.reindex(median.index)
            - cycles_median.reindex(median.index)
            >= MIN_CYCLES,
        },
        index=median.index,
    )
    return flags.fillna(False).astype(bool)


def peak_hours(dispatch: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Each day's window hour with the highest median forecast and realised price.

    ``dispatch`` is the median schedule, whose ``sell_price`` is the median forecast.
    Each hour is averaged over its quarter-hours first, so an hourly product is one
    price. A tie goes to the earlier hour.
    """
    frame = dispatch.sort_index()
    local = pd.DatetimeIndex(frame.index).tz_convert(timezone)
    inside = (local.hour >= WINDOW_START_HOUR) & (local.hour < WINDOW_END_HOUR)
    hourly = (
        pd.DataFrame(
            {
                "target_day": frame["target_day"].to_numpy(),
                "hour": local.hour,
                "forecast": frame["sell_price"].to_numpy(),
                "realised": frame["realised_price"].to_numpy(),
            }
        )
        .loc[inside]
        .groupby(["target_day", "hour"])
        .mean()
        .reset_index()
    )
    by_day = hourly.groupby("target_day")
    return pd.DataFrame(
        {
            f"{name}_hour": hourly.loc[by_day[name].idxmax()].set_index("target_day")[
                "hour"
            ]
            for name in ("forecast", "realised")
        }
    )


def peak_offsets(dispatch: pd.DataFrame, timezone: str) -> pd.Series:
    """Hours between the forecast's window peak and the realised one.

    Positive means the forecast put the peak later than it came; see
    :func:`peak_hours`.
    """
    hours = peak_hours(dispatch, timezone)
    return (hours["forecast_hour"] - hours["realised_hour"]).rename("offset_hours")


def offset_bucket(offset: pd.Series) -> pd.Series:
    """``same_hour``, ``one_hour`` or ``two_or_more_hours``, either way."""
    if offset.isna().any():
        raise ValueError("a day has no peak offset")
    size = offset.abs()
    return pd.Series(
        np.select(
            [size == 0, size == 1], ["same_hour", "one_hour"], "two_or_more_hours"
        ),
        index=offset.index,
    )


def share_interval(
    cost: pd.Series,
    mask: pd.Series,
    *,
    block_days: int = BLOCK_DAYS,
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, float]:
    """The share of ``cost`` on the days in ``mask``, in percent, with an interval.

    ``cost`` and ``mask`` run over consecutive days in order. Each draw resamples
    blocks of days and recomputes both sums.
    """
    values = cost.to_numpy(dtype=float)
    chosen = mask.reindex(cost.index).fillna(False).to_numpy(dtype=bool)
    index = block_bootstrap_index(
        len(values), block_days=block_days, draws=draws, seed=seed
    )
    drawn = (
        100 * (values[index] * chosen[index]).sum(axis=1) / values[index].sum(axis=1)
    )
    low, high = np.percentile(drawn, [2.5, 97.5])
    return {
        "share_pct": float(100 * values[chosen].sum() / values.sum()),
        "low": float(low),
        "high": float(high),
        "days": int(chosen.sum()),
        "eur": float(values[chosen].sum()),
    }


def _cost_share_interval(
    total: pd.Series,
    part: pd.Series,
    *,
    block_days: int = BLOCK_DAYS,
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, float]:
    """``part`` as a percentage of ``total``, both per consecutive day."""
    whole, piece = total.to_numpy(dtype=float), part.to_numpy(dtype=float)
    index = block_bootstrap_index(
        len(whole), block_days=block_days, draws=draws, seed=seed
    )
    drawn = 100 * piece[index].sum(axis=1) / whole[index].sum(axis=1)
    low, high = np.percentile(drawn, [2.5, 97.5])
    return {
        "share_pct": float(100 * piece.sum() / whole.sum()),
        "low": float(low),
        "high": float(high),
        "eur": float(piece.sum()),
    }


def mean_interval(
    values: pd.Series,
    in_scope: pd.Series | None = None,
    *,
    block_days: int = BLOCK_DAYS,
    draws: int = DRAWS,
    seed: int = SEED,
) -> dict[str, float]:
    """The mean per day over the days ``in_scope``, with an interval.

    Blocks are drawn over every consecutive day, so a season keeps its neighbours'
    dependence, and each draw averages only the in-scope days it contains.
    """
    data = values.to_numpy(dtype=float)
    scope = (
        np.ones(len(data), dtype=bool)
        if in_scope is None
        else in_scope.reindex(values.index).fillna(False).to_numpy(dtype=bool)
    )
    index = block_bootstrap_index(
        len(data), block_days=block_days, draws=draws, seed=seed
    )
    counts = scope[index].sum(axis=1)
    sums = (data[index] * scope[index]).sum(axis=1)
    drawn = np.divide(sums, counts, out=np.full(len(sums), np.nan), where=counts > 0)
    low, high = np.nanpercentile(drawn, [2.5, 97.5])
    return {
        "mean": float(data[scope].mean()),
        "low": float(low),
        "high": float(high),
        "days": int(scope.sum()),
        "total": float(data[scope].sum()),
    }


def _excludes_zero(interval: Mapping[str, float]) -> bool:
    return bool(interval["low"] > 0 or interval["high"] < 0)


def _eur(value: float) -> str:
    return f"€{value:,.0f}" if value >= 0 else f"-€{-value:,.0f}"


def _signed(value: float, places: int = 2) -> str:
    return f"{value:+,.{places}f}"


def _interval(item: Mapping[str, float], key: str, places: int = 2) -> str:
    return (
        f"{_signed(item[key], places)} ({_signed(item['low'], places)} to "
        f"{_signed(item['high'], places)})"
    )


def _share(item: Mapping[str, float]) -> str:
    return f"{item['share_pct']:.1f}% ({item['low']:.1f} to {item['high']:.1f})"


def results_markdown(summary: Mapping[str, Any]) -> str:
    """The results page, with every conclusion sentence built from the intervals."""
    labels = {
        "discharge_volume": "discharge volume",
        "discharge_price": "discharge price",
        "charge_volume": "charge volume",
        "charge_price": "charge price",
        "wear": "wear",
    }
    flag_labels = {
        "timing": "timing: similar energy sold, at a worse average price",
        "emptier": "emptier: at least 0.25 MWh less stored at 15:00",
        "held_back": "held back: at least 0.25 MWh more left at 21:00",
        "fewer_cycles": "fewer cycles: at least half a cycle fewer that day",
        "no_flag": "no flag",
    }
    offset_labels = {
        "same_hour": "same hour",
        "one_hour": "one hour apart",
        "two_or_more_hours": "two hours or more apart",
    }
    decomposition = summary["decomposition"]
    lines = [
        "# T1 follow-up: how 15:00 to 21:00 loses money",
        "",
        "Generated by `python -m src.health.experiments.t1_mechanism` from the saved "
        f"validation schedules of the production model, {summary['model']}: median "
        "dispatch against perfect foresight for a 1 MW / 2 MWh battery with "
        f"€{summary['wear_eur_per_mwh']:g} wear per MWh discharged, over "
        f"{summary['days']} days from {summary['first_day']} to "
        f'{summary["last_day"]}. "The window" is 15:00 to 20:59 local time. The '
        "schedules reproduce the saved daily profit, and the attribution's gaps, to "
        "within €0.01. Intervals are 95% moving-block bootstraps over 7-day blocks, "
        "5,000 draws.",
        "",
        "## The money",
        "",
        f"Over the {summary['days']} days the gap to perfect foresight is "
        f"{_eur(summary['gap_eur'])}. The Shapley split puts "
        f"{_eur(summary['window_cost']['eur'])} of it in the 15-17 and 18-20 blocks, "
        f"{_share(summary['window_cost'])} of the gap. Forecasts above the realised "
        f"price carry {_share(summary['window_direction']['over'])} of that window "
        "cost and forecasts below it "
        f"{_share(summary['window_direction']['under'])}. Sections 2 and 3 share out "
        "that window cost. A Shapley cost can be negative, on "
        f"{summary['negative_window_days']} days here, so a share and its interval "
        "can fall outside 0 to 100%.",
        "",
        "## 1. Window cash, perfect foresight minus median",
        "",
        "Euros per day, positive where perfect foresight comes out ahead. The five "
        "parts add up to the window cash gap exactly. This is cash traded inside the "
        "window, which is not the same as the window's Shapley cost: the Shapley cost "
        "is what the forecast errors in these hours cost the whole day. Wear follows "
        "the same energy difference as discharge volume without the price weighting, "
        "so one can clear zero when the other does not.",
        "",
        "| part | all days | June to September | October to May |",
        "|---|---|---|---|",
    ]
    for name in (*EFFECTS, "cash_gap_eur"):
        label = labels.get(name, "window cash gap")
        row = decomposition[name]
        lines.append(
            f"| {label} | {_interval(row['all'], 'mean')} | "
            f"{_interval(row['summer'], 'mean')} | {_interval(row['winter'], 'mean')} |"
        )
    totals = {name: decomposition[name]["all"]["total"] for name in EFFECTS}
    lines += [
        "",
        "Over all days: "
        + ", ".join(f"{labels[name]} {_eur(totals[name])}" for name in EFFECTS)
        + f"; window cash gap {_eur(decomposition['cash_gap_eur']['all']['total'])}.",
        "",
        "## Cash by clock block over the whole day",
        "",
        "Perfect foresight minus median, euros per day. The cash gaps add up to the "
        "daily gap, and so do the Shapley costs, the T1 split's cost on each "
        "block's forecast errors, also per day and split by whether the forecast was "
        "above or below the outcome. They are two accountings of the same gap, so a "
        "block's cash does not show where its own forecast errors cost.",
        "",
        "| block | cash gap | Shapley cost | forecast above | forecast below |",
        "|---|---|---|---|---|",
    ]
    for name in BLOCKS:
        item = summary["blocks"][name]
        lines.append(
            f"| {name} | {_interval(item['cash'], 'mean')} | "
            f"{_signed(item['shapley_eur_per_day'])} | "
            f"{_signed(item['over_eur_per_day'])} | "
            f"{_signed(item['under_eur_per_day'])} |"
        )
    lines += [
        "",
        "## Stored energy at the window's edges",
        "",
        "MWh, the mean over days, and perfect foresight minus median.",
        "",
        "| edge | median | perfect foresight | difference |",
        "|---|---|---|---|",
    ]
    for edge, label in (("start", "15:00"), ("end", "21:00")):
        item = summary["stored"][edge]
        lines.append(
            f"| {label} | {item['median']:.2f} | {item['foresight']:.2f} | "
            f"{_interval(item['difference'], 'mean')} |"
        )
    lines += [
        "",
        "## 2. Day mechanisms",
        "",
        "Share of the window's Shapley cost on the days carrying each flag. "
        "Thresholds were fixed before the data was read; a day can carry several "
        "flags.",
        "",
        "| flag | days | cost | share of window cost |",
        "|---|---|---|---|",
    ]
    for name in (*FLAGS, "no_flag"):
        item = summary["flags"][name]
        lines.append(
            f"| {flag_labels[name]} | {item['days']} | {_eur(item['eur'])} | "
            f"{_share(item)} |"
        )
    lines += [
        "",
        "## 3. Where the forecast put the window's peak",
        "",
        "The hour with the highest median forecast in the window against the hour "
        "with the highest realised price, weighted by the window's Shapley cost.",
        "",
        "| forecast peak | days | cost | share of window cost |",
        "|---|---|---|---|",
    ]
    for name in OFFSETS:
        item = summary["offsets"][name]
        lines.append(
            f"| {offset_labels[name]} | {item['days']} | {_eur(item['eur'])} | "
            f"{_share(item)} |"
        )
    early, late = (
        summary["offset_direction"]["early"],
        summary["offset_direction"]["late"],
    )
    lines += [
        "",
        f"Days with the peak an hour or more out carry "
        f"{_share(summary['offsets']['off_by_an_hour_or_more'])} of the window's "
        f"cost. Where the hours differ, the forecast peak came early on {early} days "
        f"and late on {late}. For comparison, the realised peak fell in the same hour "
        f"as the day before's on {summary['persistence']['same_hour']} of "
        f"{summary['persistence']['days']} days. The comparison is by whole hour, so "
        "it says nothing about timing within the hour.",
        "",
        "## Reading",
        "",
    ]
    lines += [f"- {sentence}" for sentence in reading(summary)]
    lines.append("")
    return "\n".join(lines)


#: What each part says about perfect foresight, and the direction that decides it:
#: a volume or wear part by the sign of the energy difference, which its price
#: weighting could flip, a price part by its own sign.
_PART_MEANING = {
    "discharge_volume": (
        "discharged",
        "sells more energy in the window",
        "sells less energy in the window",
    ),
    "discharge_price": (
        None,
        "sells its window energy at better average prices",
        "sells its window energy at worse average prices",
    ),
    "charge_volume": (
        "charged",
        "buys more energy in the window",
        "buys less energy in the window",
    ),
    "charge_price": (
        None,
        "buys its window energy at lower average prices",
        "pays higher average prices for its window energy",
    ),
    "wear": (
        "discharged",
        "discharges more in the window, paying more wear",
        "discharges less in the window, saving wear",
    ),
}
_SEASONS = (("summer", "June to September"), ("winter", "October to May"))


def _per_day(item: Mapping[str, float], unit: str = "€", per: str = "a day") -> str:
    return (
        f"{_signed(item['mean'])} {unit} {per} (95% interval {_signed(item['low'])} "
        f"to {_signed(item['high'])})"
    )


def _season_clause(season_parts: Mapping[str, Any], positive: bool) -> str:
    same = [
        label
        for season, label in _SEASONS
        if _excludes_zero(season_parts[season])
        and (season_parts[season]["mean"] > 0) == positive
    ]
    other = [
        label
        for season, label in _SEASONS
        if _excludes_zero(season_parts[season])
        and (season_parts[season]["mean"] > 0) != positive
    ]
    if len(same) == len(_SEASONS):
        clause = ", clear of zero in both seasons"
    elif same:
        clause = f", clear of zero {same[0]}"
    else:
        clause = ", though neither season alone clears zero in that direction"
    if other:
        clause += f", and {other[0]} clears it the other way"
    return clause


def reading(summary: Mapping[str, Any]) -> list[str]:
    """Conclusion sentences, each stated only as far as its interval allows."""
    parts = summary["decomposition"]
    sentences = []
    gap = parts["cash_gap_eur"]["all"]
    window_cost = summary["window_cost"]["eur"]
    if _excludes_zero(gap):
        ahead = "more" if gap["mean"] > 0 else "less"
        sentences.append(
            f"Inside the window perfect foresight's cash comes to {ahead} than the "
            f"median schedule's: {_per_day(gap)}."
        )
    else:
        sentence = (
            "Inside the window perfect foresight's cash cannot be told from the median "
            f"schedule's: {_per_day(gap)}."
        )
        per_day = window_cost / summary["days"]
        if gap["high"] < per_day:
            sentence += (
                f" Even its upper bound is below the {_eur(window_cost)}, "
                f"{per_day:.2f} € a day, the Shapley split puts on these hours' "
                "forecast errors, so that cost does not show up as cash traded in them."
            )
        sentences.append(sentence)
    for side, label in (("under", "below"), ("over", "above")):
        item = summary["window_direction"][side]
        if item["low"] > 50:
            sentences.append(
                f"Most of the window's cost comes from forecasts {label} the realised "
                f"price: {_share(item)}."
            )
    blocks = summary["blocks"]
    below = [
        n
        for n in BLOCKS
        if blocks[n]["under_eur_per_day"] > blocks[n]["over_eur_per_day"]
    ]
    above = [
        n
        for n in BLOCKS
        if blocks[n]["over_eur_per_day"] > blocks[n]["under_eur_per_day"]
    ]
    if below and above:
        sentences.append(
            "Across the day, on point estimates, forecasts below the outcome carry "
            f"more of the cost in {', '.join(below)} and forecasts above it in "
            f"{', '.join(above)}."
        )
    unclear = []
    for name in EFFECTS:
        item = parts[name]["all"]
        if not _excludes_zero(item):
            unclear.append(name.replace("_", " "))
            continue
        energy, up, down = _PART_MEANING[name]
        positive = item["mean"] > 0
        more = summary["energy"][energy]["mean"] > 0 if energy else positive
        sentences.append(
            f"Perfect foresight {up if more else down}: the {name.replace('_', ' ')} "
            f"part is {_per_day(item)}{_season_clause(parts[name], positive)}."
        )
    if unclear:
        sentences.append(
            "These parts cannot be told from zero over all days: "
            + ", ".join(unclear)
            + "."
        )
    for edge, label in (("start", "15:00"), ("end", "21:00")):
        difference = summary["stored"][edge]["difference"]
        if _excludes_zero(difference):
            fuller = "more" if difference["mean"] > 0 else "less"
            sentences.append(
                f"Perfect foresight has {fuller} energy stored at {label} than the "
                f"median schedule: {_per_day(difference, 'MWh', 'on average')}."
            )
        else:
            sentences.append(
                f"Stored energy at {label} does not differ beyond noise: "
                f"{_per_day(difference, 'MWh', 'on average')}."
            )
    for side, chosen in (
        ("ahead", [name for name in BLOCKS if blocks[name]["cash"]["low"] > 0]),
        ("behind", [name for name in BLOCKS if blocks[name]["cash"]["high"] < 0]),
    ):
        if chosen:
            sentences.append(
                f"Over the whole day perfect foresight's cash comes out {side} beyond "
                "noise in "
                + "; ".join(
                    f"{name}, {_per_day(blocks[name]['cash'])}" for name in chosen
                )
                + "."
            )
    flags = summary["flags"]
    ranked = sorted(FLAGS, key=lambda name: flags[name]["share_pct"], reverse=True)
    top, rest = ranked[0], ranked[1:]
    overlapping = [name for name in rest if flags[top]["low"] <= flags[name]["high"]]
    if not overlapping:
        sentences.append(
            f"The {top.replace('_', ' ')} flag carries more of the window's cost than "
            f"any other, {_share(flags[top])}, and its interval clears every other "
            "flag's."
        )
    else:
        sentences.append(
            f"The {top.replace('_', ' ')} flag carries the largest share of the "
            f"window's cost, {_share(flags[top])}, but its interval overlaps "
            + " and ".join(name.replace("_", " ") for name in overlapping)
            + ", so no single mechanism can be named as the main one."
        )
    offsets = summary["offsets"]
    same, off = offsets["same_hour"], offsets["off_by_an_hour_or_more"]
    persistence = summary["persistence"]
    sentences.append(
        f"The forecast put the window's peak in the right hour on {same['days']} of "
        f"{summary['days']} days, against {persistence['same_hour']} of "
        f"{persistence['days']} for the day before's realised peak hour. Days with "
        f"the peak an hour or more out carry {_share(off)} of the window's cost."
    )
    return sentences


def _write(path: Path, write: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = path.with_name(f".{path.name}.partial")
    write(partial_path)
    os.replace(partial_path, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read how the 15:00 to 21:00 window loses money."
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    tz = settings.market.timezone
    wear = float(settings.battery.degradation_eur_per_mwh)
    trading = settings.data.processed_path / "trading" / PRODUCTION
    dispatch = pd.read_parquet(trading / "dispatch.parquet")
    pnl = pd.read_parquet(trading / "pnl_daily.parquet")
    saved = settings.data.processed_path / "backtest" / "attribution"
    costs = pd.read_parquet(saved / "costs.parquet")
    attribution = pd.read_parquet(saved / "days.parquet")

    names = (MEDIAN_FORECAST.name, PERFECT_FORESIGHT.name)
    costs = costs[
        (costs["strategy"] == MEDIAN_FORECAST.name) & (costs["model"] == PRODUCTION)
    ]
    attribution = attribution[
        (attribution["strategy"] == MEDIAN_FORECAST.name)
        & (attribution["model"] == PRODUCTION)
    ].set_index("target_day")
    days = sorted(attribution.index)
    holdout = settings.evaluation.holdout_start
    if days[-1] >= holdout:
        raise RuntimeError(f"attribution reaches the hold-out from {holdout}")
    if len(days) != (days[-1] - days[0]).days + 1:
        raise RuntimeError("the attribution days are not consecutive")
    if not set(costs["block"]) >= set(WINDOW_BLOCKS):
        raise RuntimeError(f"the attribution has no {WINDOW_BLOCKS} blocks")

    schedules = {}
    by_block: dict[str, pd.DataFrame] = {}
    profit = pnl.pivot(index="target_day", columns="strategy", values="pnl_eur")
    for name in names:
        chosen = dispatch[
            (dispatch["strategy"] == name) & dispatch["target_day"].isin(days)
        ]
        if set(chosen["target_day"]) != set(days):
            raise RuntimeError(
                f"the {name} schedule does not cover the attributed days"
            )
        discharged = PERIOD_HOURS * chosen["discharge_mw"]
        charged = PERIOD_HOURS * chosen["charge_mw"]
        quarter_cash = (discharged - charged) * chosen["realised_price"]
        cash = (quarter_cash - wear * discharged).groupby(chosen["target_day"]).sum()
        drift = float((cash - profit.loc[days, name]).abs().max())
        if drift > RECONCILE_EUR:
            raise RuntimeError(
                f"the {name} schedule misses its saved profit by up to €{drift:.4f}"
            )
        schedules[name] = (chosen, window_schedule(chosen, tz))
        by_block[name] = block_cash(chosen, tz, wear)
    gap_drift = float(
        (
            attribution["gap_eur"]
            - (
                profit.loc[days, PERFECT_FORESIGHT.name]
                - profit.loc[days, MEDIAN_FORECAST.name]
            )
        )
        .abs()
        .max()
    )
    if gap_drift > RECONCILE_EUR:
        raise RuntimeError(
            f"attribution gaps differ from the profit by €{gap_drift:.4f}"
        )

    median_dispatch, median = schedules[MEDIAN_FORECAST.name]
    _, foresight = schedules[PERFECT_FORESIGHT.name]
    parts = decompose(median, foresight, wear)
    cycles = pnl.pivot(index="target_day", columns="strategy", values="cycles")
    flags = flag_days(
        median, foresight, cycles[MEDIAN_FORECAST.name], cycles[PERFECT_FORESIGHT.name]
    )
    in_window = costs[costs["block"].isin(WINDOW_BLOCKS)]
    window_cost = (
        in_window.groupby("target_day")["cost_eur"].sum().reindex(days, fill_value=0.0)
    )
    over_cost = (
        in_window[in_window["direction"] == "over"]
        .groupby("target_day")["cost_eur"]
        .sum()
        .reindex(days, fill_value=0.0)
    )
    gap = attribution["gap_eur"].reindex(days)
    hours = peak_hours(median_dispatch, tz).reindex(days)
    offsets = (hours["forecast_hour"] - hours["realised_hour"]).rename("offset_hours")
    yesterday = hours["realised_hour"].shift(1)
    buckets = offset_bucket(offsets)
    summer = pd.Series(
        pd.to_datetime(pd.Index(days)).month.isin(sorted(SUMMER)), index=days
    )

    per_day = pd.concat(
        [
            parts,
            flags,
            window_cost.rename("window_cost_eur"),
            gap.rename("gap_eur"),
            offsets,
            summer.rename("summer"),
            median.add_prefix("median_"),
            foresight.add_prefix("foresight_"),
        ],
        axis=1,
    ).loc[days]

    block_gap = (by_block[PERFECT_FORESIGHT.name] - by_block[MEDIAN_FORECAST.name]).loc[
        days
    ]
    block_drift = float((block_gap.sum(axis=1) - gap).abs().max())
    if block_drift > RECONCILE_EUR:
        raise RuntimeError(f"the blocks miss the daily gap by €{block_drift:.4f}")
    total_gap = float(gap.sum())
    index = block_bootstrap_index(
        len(days), block_days=BLOCK_DAYS, draws=DRAWS, seed=SEED
    )
    drawn_share = (
        100
        * window_cost.to_numpy()[index].sum(axis=1)
        / gap.to_numpy()[index].sum(axis=1)
    )
    low, high = np.percentile(drawn_share, [2.5, 97.5])
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": PRODUCTION,
        "wear_eur_per_mwh": wear,
        "days": len(days),
        "first_day": str(days[0]),
        "last_day": str(days[-1]),
        "gap_eur": total_gap,
        "window_cost": {
            "eur": float(window_cost.sum()),
            "share_pct": float(100 * window_cost.sum() / total_gap),
            "low": float(low),
            "high": float(high),
        },
        "decomposition": {
            name: {
                "all": mean_interval(per_day[name]),
                "summer": mean_interval(per_day[name], summer),
                "winter": mean_interval(per_day[name], ~summer),
            }
            for name in (*EFFECTS, "cash_gap_eur")
        },
        "stored": {
            edge: {
                "median": float(median[column].mean()),
                "foresight": float(foresight[column].mean()),
                "difference": mean_interval(foresight[column] - median[column]),
            }
            for edge, column in (
                ("start", "soc_at_start_mwh"),
                ("end", "soc_at_end_mwh"),
            )
        },
        "negative_window_days": int((window_cost < 0).sum()),
        "energy": {
            name: mean_interval(foresight[column] - median[column])
            for name, column in (
                ("discharged", "discharged_mwh"),
                ("charged", "charged_mwh"),
            )
        },
        "persistence": {
            "same_hour": int((hours["realised_hour"] == yesterday).sum()),
            "days": int(yesterday.notna().sum()),
        },
        "window_direction": {
            "over": _cost_share_interval(window_cost, over_cost),
            "under": _cost_share_interval(window_cost, window_cost - over_cost),
        },
        "blocks": {
            name: {
                "cash": mean_interval(block_gap[name]),
                "shapley_eur_per_day": float(
                    costs.loc[costs["block"] == name, "cost_eur"].sum() / len(days)
                ),
                **{
                    f"{direction}_eur_per_day": float(
                        costs.loc[
                            (costs["block"] == name)
                            & (costs["direction"] == direction),
                            "cost_eur",
                        ].sum()
                        / len(days)
                    )
                    for direction in ("over", "under")
                },
            }
            for name in BLOCKS
        },
        "flags": {
            **{name: share_interval(window_cost, flags[name]) for name in FLAGS},
            "no_flag": share_interval(window_cost, ~flags.any(axis=1)),
        },
        "offsets": {
            **{name: share_interval(window_cost, buckets == name) for name in OFFSETS},
            "off_by_an_hour_or_more": share_interval(
                window_cost, buckets != "same_hour"
            ),
        },
        "offset_direction": {
            "early": int((offsets < 0).sum()),
            "late": int((offsets > 0).sum()),
        },
    }
    totals = parts[list(EFFECTS)].sum(axis=1)
    remainder = float((totals - parts["cash_gap_eur"]).abs().max())
    if remainder > 1e-6:
        raise RuntimeError(f"the five parts miss the cash gap by €{remainder:.6f}")

    out = settings.data.processed_path / "experiments"
    _write(out / "t1_mechanism_days.parquet", per_day.to_parquet)
    _write(
        out / "t1_mechanism_summary.json",
        lambda path: path.write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        ),
    )
    _write(
        RESULTS_PATH,
        lambda path: path.write_text(results_markdown(summary), encoding="utf-8"),
    )
    print(results_markdown(summary))
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
