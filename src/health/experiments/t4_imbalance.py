"""T4 experiment: what an outage costs once the day-ahead schedule is sold.

    uv run python -m src.health.experiments.t4_imbalance

The Phase 3 backtest settles every committed schedule as if it were delivered,
which is safe because the schedules are feasible by construction. This experiment
breaks that assumption on purpose. For the saved validation dispatch of the
production model, one outage per day makes the battery unavailable for a window
of quarter-hours. Outside the window the battery follows its committed schedule
as far as physics allows: a charge that would overfill the cells and a discharge
that would drain them below empty are delivered only in part. Nothing is
re-optimized, because the day-ahead position is fixed; intraday re-trading is out
of scope.

Every MW of deviation, actual minus committed net power, settles at the single
German imbalance price reBAP: imbalance cash is dt * deviation * reBAP, so a
shortfall pays reBAP and a surplus receives it, and a negative reBAP flips both.
Day-ahead revenue stays as committed. Wear is charged on the energy actually
discharged.

An outage leaves the battery at a different state of charge at midnight than the
schedule planned. That gap is valued as if it were closed by imbalance over the
next day's first two hours, at their mean reBAP: missing energy is bought before
charging losses; surplus energy is sold after discharge losses and wear, but never
at a loss, because the battery can simply keep it. The same cost with the gap
valued at zero is kept as a sensitivity.

Scenarios, for median dispatch and perfect foresight as a reference: a random
2-hour outage per day (seeded), the worst 2-hour window per day (all starts
searched) and a random 4-hour outage. Results go to
``docs/results/t4_imbalance.md`` and
``data/processed/experiments/t4_imbalance_daily.parquet``. Only validation days
before ``evaluation.holdout_start`` are read.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from src.config import REPO_ROOT, Settings, load_settings
from src.trading.battery import Battery
from src.trading.optimizer import period_hours

__all__ = [
    "SCENARIOS",
    "DayOutcome",
    "Delivery",
    "Scenario",
    "costliest_days",
    "evaluate_day",
    "main",
    "outage_windows",
    "random_start",
    "run_experiment",
    "simulate_delivery",
    "summarise",
    "value_soc_gap",
    "worst_start",
]

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

MODEL = "lightgbm_conformal"
STRATEGIES = ("median_forecast", "perfect_foresight")
SEED = 20260914
IMBALANCE_COLUMN = "imbalance_price_excess_eur_mwh"
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t4_imbalance.md"
#: A committed MW within this of the feasible power counts as fully delivered, so
#: solver noise in the saved schedules does not show up as a deviation.
FEASIBILITY_TOLERANCE_MW = 1e-6
#: The end-of-day gap is priced at the mean reBAP of this many hours after
#: midnight: closing a full 2 MWh gap at 1 MW takes two hours, and one extreme
#: quarter-hour should not decide the value.
GAP_HOURS = 2.0


@dataclass(frozen=True)
class Scenario:
    """One outage per day of ``hours`` hours, at a random or the worst start."""

    name: str
    hours: float
    placement: str  # "random" or "worst"


SCENARIOS: tuple[Scenario, ...] = (
    Scenario("random_2h", 2.0, "random"),
    Scenario("worst_2h", 2.0, "worst"),
    Scenario("random_4h", 4.0, "random"),
)


@dataclass(frozen=True)
class Delivery:
    """What the battery actually did, one row per outage scenario.

    ``actual_net_mw`` and ``discharge_mw`` are power at the grid connection and
    ``soc_mwh`` the energy in the cells at the end of each period.
    """

    actual_net_mw: FloatArray
    discharge_mw: FloatArray
    soc_mwh: FloatArray


def simulate_delivery(
    committed_net_mw: ArrayLike,
    outage: ArrayLike,
    battery: Battery,
    dt_hours: float,
) -> Delivery:
    """Follow a committed schedule through outages without re-optimizing.

    ``outage`` is boolean, shape (scenarios, periods) or (periods,); True marks a
    period in which the battery is unavailable and delivers 0 MW. In every other
    period the committed charge or discharge is delivered in full if the cells
    allow it, otherwise only the part that fits between empty and full.
    """
    net = np.asarray(committed_net_mw, dtype=np.float64)
    mask = np.atleast_2d(np.asarray(outage, dtype=bool))
    if net.ndim != 1 or mask.ndim != 2 or mask.shape[1] != net.size:
        raise ValueError("outage must have one column per committed period")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be positive")
    scenarios, periods = mask.shape
    want_charge = np.clip(-net, 0.0, battery.power_mw)
    want_discharge = np.clip(net, 0.0, battery.power_mw)
    eta_c = battery.charge_efficiency
    eta_d = battery.discharge_efficiency
    capacity = battery.capacity_mwh

    soc: FloatArray = np.full(scenarios, battery.initial_soc_mwh, dtype=np.float64)
    actual = np.zeros((scenarios, periods), dtype=np.float64)
    discharge = np.zeros((scenarios, periods), dtype=np.float64)
    soc_path = np.zeros((scenarios, periods), dtype=np.float64)
    for t in range(periods):
        up = ~mask[:, t]
        charge = np.where(up, want_charge[t], 0.0)
        room = np.maximum(capacity - soc, 0.0) / (eta_c * dt_hours)
        charge = np.where(charge <= room + FEASIBILITY_TOLERANCE_MW, charge, room)
        out = np.where(up, want_discharge[t], 0.0)
        stored = np.maximum(soc, 0.0) * eta_d / dt_hours
        out = np.where(out <= stored + FEASIBILITY_TOLERANCE_MW, out, stored)
        soc = np.clip(
            soc + eta_c * charge * dt_hours - out * dt_hours / eta_d, 0.0, capacity
        )
        actual[:, t] = out - charge
        discharge[:, t] = out
        soc_path[:, t] = soc
    return Delivery(actual_net_mw=actual, discharge_mw=discharge, soc_mwh=soc_path)


def value_soc_gap(
    soc_end_mwh: ArrayLike, battery: Battery, price_eur_mwh: float
) -> FloatArray:
    """Money from closing the end-of-day state of charge gap at one price.

    Energy below the planned state of charge is bought and loses the charging
    efficiency on the way in; at a negative price buying it earns money. Energy
    above plan would be sold after discharge losses and wear, but its value is
    floored at zero: when the price does not cover wear the battery keeps the
    energy instead of paying to dump it.
    """
    gap = np.asarray(soc_end_mwh, dtype=np.float64) - battery.initial_soc_mwh
    surplus = np.maximum(gap, 0.0) * battery.discharge_efficiency
    deficit = np.minimum(gap, 0.0) / battery.charge_efficiency
    margin = max(price_eur_mwh - battery.degradation_eur_per_mwh, 0.0)
    return np.asarray(surplus * margin + deficit * price_eur_mwh, dtype=np.float64)


@dataclass(frozen=True)
class DayOutcome:
    """Money and energy of one day under each outage scenario (arrays per row)."""

    committed_pnl_eur: float
    day_ahead_revenue_eur: float
    deviation_mw: FloatArray
    imbalance_cash_eur: FloatArray
    wear_eur: FloatArray
    soc_end_mwh: FloatArray
    soc_value_eur: FloatArray
    pnl_zero_soc_eur: FloatArray
    pnl_eur: FloatArray
    cost_eur: FloatArray
    cost_zero_soc_eur: FloatArray
    undelivered_mwh: FloatArray


def evaluate_day(
    committed_net_mw: ArrayLike,
    day_ahead_price: ArrayLike,
    imbalance_price: ArrayLike,
    outage: ArrayLike,
    battery: Battery,
    dt_hours: float,
    terminal_price: float,
) -> DayOutcome:
    """Settle one committed day with and without each outage.

    Committed P&L is the Phase 3 settlement: dt * sum(price * net) minus wear on
    the committed discharge. With an outage, P&L is the same day-ahead revenue,
    plus imbalance cash dt * sum(deviation * reBAP), minus wear on the actual
    discharge, plus the end-of-day state of charge gap valued at
    ``terminal_price`` (see ``value_soc_gap``). Cost is committed minus outage
    P&L.
    """
    net = np.asarray(committed_net_mw, dtype=np.float64)
    price = np.asarray(day_ahead_price, dtype=np.float64)
    rebap = np.asarray(imbalance_price, dtype=np.float64)
    if not (net.shape == price.shape == rebap.shape) or net.ndim != 1:
        raise ValueError("schedule and prices must be one-dimensional and aligned")
    values = np.concatenate([net, price, rebap, [terminal_price]])
    if not np.isfinite(values).all():
        raise ValueError("schedule and prices must be finite")
    delivery = simulate_delivery(net, outage, battery, dt_hours)
    wear_rate = battery.degradation_eur_per_mwh

    revenue = float(dt_hours * np.sum(price * net))
    committed_wear = wear_rate * dt_hours * float(np.clip(net, 0.0, None).sum())
    committed_pnl = revenue - committed_wear

    deviation = delivery.actual_net_mw - net
    imbalance = dt_hours * (deviation * rebap).sum(axis=1)
    wear = wear_rate * dt_hours * delivery.discharge_mw.sum(axis=1)
    soc_end = delivery.soc_mwh[:, -1]
    soc_value = value_soc_gap(soc_end, battery, terminal_price)
    pnl_zero = revenue + imbalance - wear
    pnl = pnl_zero + soc_value
    return DayOutcome(
        committed_pnl_eur=committed_pnl,
        day_ahead_revenue_eur=revenue,
        deviation_mw=deviation,
        imbalance_cash_eur=imbalance,
        wear_eur=wear,
        soc_end_mwh=soc_end,
        soc_value_eur=soc_value,
        pnl_zero_soc_eur=pnl_zero,
        pnl_eur=pnl,
        cost_eur=committed_pnl - pnl,
        cost_zero_soc_eur=committed_pnl - pnl_zero,
        undelivered_mwh=dt_hours * np.abs(deviation).sum(axis=1),
    )


def outage_windows(n_periods: int, length: int, starts: Sequence[int]) -> BoolArray:
    """Boolean outage masks, one row per start, each ``length`` periods long."""
    first = np.asarray(starts, dtype=np.int64).reshape(-1)
    if length < 1 or length > n_periods:
        raise ValueError("outage length must be between 1 and the number of periods")
    if (first < 0).any() or (first + length > n_periods).any():
        raise ValueError("every outage window must lie inside the day")
    periods = np.arange(n_periods)
    return np.asarray(
        (periods[None, :] >= first[:, None])
        & (periods[None, :] < first[:, None] + length),
        dtype=bool,
    )


def random_start(day: date, n_periods: int, length: int, seed: int) -> int:
    """A uniformly random start for a window that fits inside the day.

    The generator is seeded by the seed, the day and the window length only, so
    every strategy meets the same outage and reruns reproduce it.
    """
    rng = np.random.default_rng([seed, day.toordinal(), length])
    return int(rng.integers(0, n_periods - length + 1))


def worst_start(
    committed_net_mw: ArrayLike,
    day_ahead_price: ArrayLike,
    imbalance_price: ArrayLike,
    battery: Battery,
    dt_hours: float,
    terminal_price: float,
    length: int,
) -> int:
    """The start of the window that costs most; the earliest wins a tie."""
    n_periods = np.asarray(committed_net_mw).size
    starts = list(range(n_periods - length + 1))
    outcome = evaluate_day(
        committed_net_mw,
        day_ahead_price,
        imbalance_price,
        outage_windows(n_periods, length, starts),
        battery,
        dt_hours,
        terminal_price,
    )
    return starts[int(np.argmax(outcome.cost_eur))]


def _terminal_prices(reference: pd.DataFrame, dt_hours: float) -> dict[date, float]:
    """Mean reBAP of the next day's first ``GAP_HOURS``, per target day.

    ``reference`` is one strategy's dispatch with a ``rebap`` column, sorted by
    time. The last validation day uses its own last ``GAP_HOURS`` instead,
    because the hours after it belong to the hold-out.
    """
    periods = round(GAP_HOURS / dt_hours)
    grouped = reference.groupby("target_day", sort=True)["rebap"]
    heads = grouped.apply(lambda day: float(day.iloc[:periods].mean()))
    tails = grouped.apply(lambda day: float(day.iloc[-periods:].mean()))
    known = set(heads.index)
    return {
        day: float(
            heads[day + timedelta(days=1)]
            if day + timedelta(days=1) in known
            else tails[day]
        )
        for day in heads.index
    }


def run_experiment(
    dispatch: pd.DataFrame,
    imbalance_price: pd.Series,
    battery: Battery,
    *,
    strategies: Sequence[str] = STRATEGIES,
    scenarios: Sequence[Scenario] = SCENARIOS,
    seed: int = SEED,
    timezone: str = "Europe/Berlin",
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Daily outcomes per scenario and strategy, plus consistency checks.

    ``dispatch`` is the saved Phase 3 dispatch (UTC index, columns target_day,
    strategy, net_mw, realised_price); ``imbalance_price`` is reBAP on the same
    UTC quarter-hours. The checks report the largest deviation and cost when no
    outage happens, which should be solver noise.
    """
    frame = dispatch[dispatch["strategy"].isin(list(strategies))].sort_index()
    rebap_all = imbalance_price.reindex(frame.index)
    if rebap_all.isna().any():
        raise ValueError("reBAP is missing for some dispatched quarter-hours")
    frame = frame.assign(rebap=rebap_all.to_numpy(dtype=float))

    reference = frame[frame["strategy"] == strategies[0]]
    terminal = _terminal_prices(
        reference, period_hours(pd.DatetimeIndex(reference.index[:2]))
    )

    rows: list[dict[str, Any]] = []
    max_deviation = 0.0
    max_cost = 0.0
    for (strategy, day), group in frame.groupby(["strategy", "target_day"], sort=True):
        index = pd.DatetimeIndex(group.index)
        dt = period_hours(index)
        net = group["net_mw"].to_numpy(dtype=float)
        price = group["realised_price"].to_numpy(dtype=float)
        rebap = group["rebap"].to_numpy(dtype=float)
        n_periods = net.size

        baseline = evaluate_day(
            net,
            price,
            rebap,
            np.zeros((1, n_periods), dtype=bool),
            battery,
            dt,
            terminal[day],
        )
        max_deviation = max(max_deviation, float(np.abs(baseline.deviation_mw).max()))
        max_cost = max(max_cost, float(np.abs(baseline.cost_eur).max()))

        for scenario in scenarios:
            length = round(scenario.hours / dt)
            if scenario.placement == "random":
                start = random_start(day, n_periods, length, seed)
            elif scenario.placement == "worst":
                start = worst_start(
                    net, price, rebap, battery, dt, terminal[day], length
                )
            else:
                raise ValueError(f"unknown placement {scenario.placement!r}")
            mask = outage_windows(n_periods, length, [start])
            outcome = evaluate_day(net, price, rebap, mask, battery, dt, terminal[day])
            window = slice(start, start + length)
            cash = dt * outcome.deviation_mw[0] * rebap
            worst = int(np.argmin(cash))
            rows.append(
                {
                    "target_day": day,
                    "strategy": strategy,
                    "scenario": scenario.name,
                    "outage_hours": scenario.hours,
                    "outage_start_utc": index[start],
                    "outage_start_local": index[start]
                    .tz_convert(timezone)
                    .strftime("%H:%M"),
                    "committed_mwh_in_window": dt * float(np.abs(net[window]).sum()),
                    "committed_pnl_eur": outcome.committed_pnl_eur,
                    "day_ahead_revenue_eur": outcome.day_ahead_revenue_eur,
                    "imbalance_cash_eur": float(outcome.imbalance_cash_eur[0]),
                    "wear_eur": float(outcome.wear_eur[0]),
                    "soc_end_mwh": float(outcome.soc_end_mwh[0]),
                    "soc_value_eur": float(outcome.soc_value_eur[0]),
                    "pnl_zero_soc_eur": float(outcome.pnl_zero_soc_eur[0]),
                    "pnl_eur": float(outcome.pnl_eur[0]),
                    "cost_eur": float(outcome.cost_eur[0]),
                    "cost_zero_soc_eur": float(outcome.cost_zero_soc_eur[0]),
                    "undelivered_mwh": float(outcome.undelivered_mwh[0]),
                    "window_rebap_min": float(rebap[window].min()),
                    "window_rebap_mean": float(rebap[window].mean()),
                    "window_rebap_max": float(rebap[window].max()),
                    "window_day_ahead_mean": float(price[window].mean()),
                    "terminal_rebap": terminal[day],
                    "imbalance_cash_outside_window_eur": float(
                        cash.sum() - cash[window].sum()
                    ),
                    "costliest_quarter_hour_local": index[worst]
                    .tz_convert(timezone)
                    .strftime("%H:%M"),
                    "costliest_quarter_hour_rebap": float(rebap[worst]),
                }
            )
    checks = {
        "no_outage_max_deviation_mw": max_deviation,
        "no_outage_max_cost_eur": max_cost,
    }
    return pd.DataFrame(rows), checks


def summarise(daily: pd.DataFrame) -> pd.DataFrame:
    """One row per scenario and strategy: cost distribution in context."""
    rows: list[dict[str, Any]] = []
    for (scenario, strategy), group in daily.groupby(
        ["scenario", "strategy"], sort=False
    ):
        cost = group["cost_eur"]
        mean_pnl = float(group["committed_pnl_eur"].mean())
        mean_cost = float(cost.mean())
        rows.append(
            {
                "scenario": scenario,
                "strategy": strategy,
                "days": len(group),
                "mean committed P&L": mean_pnl,
                "mean cost": mean_cost,
                "median cost": float(cost.median()),
                "P95 cost": float(cost.quantile(0.95)),
                "max cost": float(cost.max()),
                "share cost > day P&L": float(
                    (cost > group["committed_pnl_eur"]).mean()
                ),
                "undelivered MWh": float(group["undelivered_mwh"].sum()),
                "mean cost, SoC gap at zero": float(group["cost_zero_soc_eur"].mean()),
                "days of P&L per outage": mean_cost / mean_pnl
                if mean_pnl != 0
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def costliest_days(
    daily: pd.DataFrame, scenario: str, strategy: str, n: int = 5
) -> pd.DataFrame:
    """The ``n`` most expensive outage days of one scenario and strategy."""
    group = daily[(daily["scenario"] == scenario) & (daily["strategy"] == strategy)]
    mean_pnl = float(group["committed_pnl_eur"].mean())
    top = group.nlargest(n, "cost_eur").copy()
    top["days of P&L"] = top["cost_eur"] / mean_pnl if mean_pnl else float("nan")
    return top


def _cell(value: object, digits: int = 2) -> str:
    if isinstance(value, float | np.floating):
        return f"{float(value):,.{digits}f}"
    return str(value)


def _markdown(frame: pd.DataFrame, formats: dict[str, str] | None = None) -> list[str]:
    formats = formats or {}
    header = "| " + " | ".join(frame.columns) + " |"
    rule = (
        "|"
        + "|".join("---" if frame[c].dtype == object else "---:" for c in frame.columns)
        + "|"
    )
    body = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if column in formats:
                cells.append(format(value, formats[column]))
            else:
                cells.append(_cell(value))
        body.append("| " + " | ".join(cells) + " |")
    return [header, rule, *body]


def _report(
    daily: pd.DataFrame,
    checks: dict[str, float],
    battery: Battery,
    first: date,
    last: date,
    settlement_gap_eur: float | None,
    seed: int,
) -> str:
    summary = summarise(daily)
    random_median = daily[
        (daily["scenario"] == "random_2h") & (daily["strategy"] == "median_forecast")
    ]
    # A cent of tolerance: days the outage barely touches show float noise.
    gain_share = float((random_median["cost_eur"] < -0.01).mean())
    shown = summary.copy()
    shown["share cost > day P&L"] = shown["share cost > day P&L"].map(
        lambda v: f"{v:.1%}"
    )
    lines = [
        "# T4 imbalance risk: the cost of an undelivered schedule",
        "",
        "Generated by `uv run python -m src.health.experiments.t4_imbalance` from the",
        f"saved validation dispatch of `{MODEL}` and the German imbalance price",
        f"(reBAP, ENTSO-E). Target days {first} to {last}; no hold-out day is read.",
        "Money in EUR, energy in MWh.",
        "",
        "## Question",
        "",
        "The Phase 3 backtest assumes every committed day-ahead schedule is",
        "delivered. That holds for a healthy battery, because the optimizer only",
        "commits feasible schedules. What does one day cost when the battery",
        "cannot deliver, for example after an inverter trip or a grid fault?",
        "",
        "## Method",
        "",
        f"* Battery: {battery.power_mw:g} MW / {battery.capacity_mwh:g} MWh, "
        f"{battery.round_trip_efficiency:.0%} round trip split evenly between "
        f"charging and discharging, wear {battery.degradation_eur_per_mwh:g} EUR "
        f"per MWh discharged, every day planned from and back to "
        f"{battery.initial_soc_mwh:g} MWh.",
        "* An outage makes the battery deliver 0 MW for a window of quarter-hours.",
        "  Outside the window it follows the committed schedule as far as the",
        "  cells allow: a committed charge that would overfill them or a committed",
        "  discharge that would drain them below empty is delivered only in part.",
        "  A missed charge therefore also shrinks the discharge that follows it.",
        "* Deviation is actual minus committed net power. Germany settles it at",
        "  one imbalance price, reBAP: imbalance cash = 0.25 h * deviation * reBAP.",
        "  A shortfall pays reBAP, a surplus receives it; a negative reBAP turns",
        "  both around.",
        "* Outage P&L = committed day-ahead revenue + imbalance cash - wear on the",
        "  energy actually discharged + value of the end-of-day state of charge",
        "  gap. Cost = committed P&L (the Phase 3 settlement) - outage P&L.",
        "* **End-of-day state of charge.** An outage leaves the battery at a",
        "  different state of charge at midnight than planned. Valuing that gap at",
        "  zero would be wrong in both directions: a missed discharge would count",
        "  the whole shortfall as lost although the energy is still in the cells,",
        "  and a missed charge would look cheap although the battery starts the",
        "  next day short. The gap is instead valued as if closed by imbalance",
        "  over the next day's first two hours, at their mean reBAP (a full 2 MWh",
        "  gap takes two hours at 1 MW, and a single extreme quarter-hour, such",
        "  as -1,755 EUR/MWh at midnight on 2025-10-04, should not decide it).",
        "  Missing energy is bought before charging losses. Surplus energy is",
        "  sold after discharge losses and wear, but never at a loss, because the",
        "  battery can keep it. For the last validation day the hours after it",
        "  belong to the hold-out, so that day's own last two hours are used. The",
        "  cost with the gap valued at zero is shown as a sensitivity.",
        f"* Scenarios, each day on its own: a 2-hour outage at a uniformly random "
        f"start (seed {seed}, the same start for every strategy), the worst 2-hour",
        "  window of the day (every start searched) and a 4-hour outage at a",
        "  random start. Windows lie inside the delivery day.",
        "",
        "## Simplifications",
        "",
        "* No re-optimization and no intraday re-trading. A real desk would try",
        "  to close the position on the intraday market once the outage is known,",
        "  usually at a better price than reBAP; this is out of scope. The costs",
        "  describe a standalone battery that simply fails. They are not an upper",
        "  bound: an outage can also gain, because a missed charge is paid at",
        f"  reBAP. {gain_share:.0%} of random 2-hour outage days for median dispatch",
        "  have a negative cost.",
        "* The battery is a price taker in imbalance too, and it is its own",
        "  balancing group: no netting against a portfolio, no extra penalties or",
        "  fees beyond the published reBAP.",
        "* Each day is independent: the next day's schedule is assumed to start",
        "  from its planned state of charge, with the gap settled as described,",
        "  rather than carrying the outage into the next day's schedule.",
        "* Before 1 October 2025 the day-ahead products were hourly, but deviations",
        "  are settled per quarter-hour, the German imbalance settlement period.",
        "* Outages are placed on every day to measure the cost per outage day,",
        "  not their frequency, which depends on the asset.",
        "",
        "## Consistency checks",
        "",
        "* Without an outage the simulation delivers the committed schedule: the",
        "  largest deviation is "
        f"{checks['no_outage_max_deviation_mw']:.1e} MW and the largest cost "
        f"{checks['no_outage_max_cost_eur']:.1e} EUR.",
    ]
    if settlement_gap_eur is not None:
        lines.append(
            "* Committed P&L matches the saved Phase 3 settlement to within "
            f"{settlement_gap_eur:.1e} EUR per day."
        )
    lines += [
        "",
        "## Cost per outage day",
        "",
        "Cost is EUR per outage day. Share cost > day P&L is the share of days on",
        "which the outage wiped out more than that day's committed profit. Days of",
        "P&L per outage is mean cost over mean committed daily P&L.",
        "",
        *_markdown(shown, {"days": "d"}),
        "",
        "## Costliest days",
        "",
        "Start is the local start of the outage. reBAP mean and max are over the",
        "outage window; DA mean is the day-ahead price over the same window.",
        "Cash outside is the imbalance cash settled after the window, when a",
        "missed charge left too little energy for a committed discharge (or a",
        "missed discharge left too little room for a charge). Costliest QH is",
        "the quarter-hour with the largest imbalance payment and its reBAP. Gap",
        "reBAP is the price used for the end-of-day state of charge gap.",
    ]
    columns = {
        "target_day": "day",
        "outage_start_local": "start",
        "committed_pnl_eur": "committed P&L",
        "cost_eur": "cost",
        "cost_zero_soc_eur": "cost, gap at zero",
        "undelivered_mwh": "undelivered MWh",
        "window_rebap_mean": "reBAP mean",
        "window_rebap_max": "reBAP max",
        "window_day_ahead_mean": "DA mean",
        "imbalance_cash_outside_window_eur": "cash outside",
        "costliest_quarter_hour_local": "costliest QH",
        "costliest_quarter_hour_rebap": "reBAP there",
        "terminal_rebap": "gap reBAP",
        "days of P&L": "days of P&L",
    }
    for scenario in daily["scenario"].unique():
        for strategy in daily["strategy"].unique():
            top = costliest_days(daily, scenario, strategy)
            table = top[list(columns)].rename(columns=columns)
            lines += ["", f"### {scenario}, {strategy}", "", *_markdown(table)]

    median = summary[summary["strategy"] == "median_forecast"].set_index("scenario")
    if {"random_2h", "worst_2h"} <= set(median.index):
        mean_pnl = median["mean committed P&L"].astype(float).to_dict()
        mean_cost = median["mean cost"].astype(float).to_dict()
        pnl = float(mean_pnl["random_2h"])
        random_cost = float(mean_cost["random_2h"])
        worst_cost = float(mean_cost["worst_2h"])
        lines += [
            "",
            "## In context",
            "",
            f"* Median dispatch earns {pnl:,.2f} EUR on an average day, about "
            f"{365 * pnl:,.0f} EUR a year.",
            f"* A random 2-hour outage costs {random_cost:,.2f} EUR on average, "
            f"{random_cost / pnl:.2f} days of P&L. Twelve such outages a year cost "
            f"{12 * random_cost:,.0f} EUR, "
            f"{12 * random_cost / (365 * pnl):.1%} of annual P&L.",
            f"* The worst 2-hour window costs {worst_cost:,.2f} EUR on average, "
            f"{worst_cost / pnl:.2f} days of P&L. The worst window is often not "
            "the one with the highest reBAP: missing a midday charge empties the "
            "evening discharge, and the shortfall is settled at evening reBAP. "
            "The damage is concentrated in "
            "the committed charge and discharge blocks and on days when reBAP "
            "spikes.",
        ]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Price the imbalance cost of undelivered day-ahead schedules."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args(argv)
    started = time.perf_counter()

    settings: Settings = load_settings(args.config)
    battery = settings.battery
    first = settings.evaluation.validation_start
    last = settings.evaluation.holdout_start - timedelta(days=1)
    window_start = settings.market.local_midnight_utc(first)
    window_end = settings.market.local_midnight_utc(settings.evaluation.holdout_start)

    trading = settings.data.processed_path / "trading" / MODEL
    dispatch = pd.read_parquet(trading / "dispatch.parquet")
    dispatch = dispatch[
        (dispatch.index >= window_start)
        & (dispatch.index < window_end)
        & (dispatch["target_day"] >= first)
        & (dispatch["target_day"] <= last)
    ]
    imbalance = pd.read_parquet(
        settings.data.processed_path / "entsoe" / "imbalance_prices_de.parquet",
        columns=[IMBALANCE_COLUMN],
    )[IMBALANCE_COLUMN]
    imbalance = imbalance[
        (imbalance.index >= window_start) & (imbalance.index < window_end)
    ]

    daily, checks = run_experiment(
        dispatch,
        imbalance,
        battery,
        seed=args.seed,
        timezone=settings.market.timezone,
    )

    settlement_gap: float | None = None
    pnl_path = trading / "pnl_daily.parquet"
    if pnl_path.exists():
        saved = pd.read_parquet(pnl_path, columns=["target_day", "strategy", "pnl_eur"])
        merged = daily.merge(saved, on=["target_day", "strategy"], how="left")
        if merged["pnl_eur_y"].isna().any():
            raise RuntimeError("the saved Phase 3 P&L is missing for some days")
        settlement_gap = float(
            (merged["committed_pnl_eur"] - merged["pnl_eur_y"]).abs().max()
        )

    out = settings.data.processed_path / "experiments" / "t4_imbalance_daily.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out)
    days = sorted(set(daily["target_day"]))
    report = _report(
        daily, checks, battery, days[0], days[-1], settlement_gap, args.seed
    )
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(report, encoding="utf-8")
    seconds = time.perf_counter() - started
    print(summarise(daily).to_string(index=False))
    print(
        f"wrote {RESULTS_PATH.relative_to(REPO_ROOT)} and "
        f"{out.relative_to(REPO_ROOT)} in {seconds:.1f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
