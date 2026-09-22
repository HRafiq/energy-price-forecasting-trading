"""T5 follow-up: bid curves from the quantile fan instead of fixed volumes.

    uv run python -m src.health.experiments.t5_bid_curves

The plan, fixed before any of this was written, is ``docs/plans/bid_curves_plan.md``.
The production schedule's volumes are kept; each sale gets a limit price from a
lower quantile of the forecast and each purchase from the mirrored upper one. An
order executes only if the realised price reaches its limit. The day is then
replayed: a sale the store cannot cover is a delivery failure settled at the
German imbalance price, as the T4 outage experiment settles deviations, and the
end-of-day state of charge gap is valued at the day's terminal prices as T4
values it. Nothing is re-optimised.

Arms: ``volumes`` (the control, every order executes), ``limits_q25`` (limits q25
for sales and q75 for purchases) and ``limits_q10`` (q10 and q90). Criterion,
fixed in advance and applied to each limit arm: adopt only if its paired daily
profit difference against ``volumes``, imbalance and gap included, has a 95%
moving-block bootstrap interval above zero. Validation days only.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import REPO_ROOT, Settings, load_settings
from src.health.experiments.t1_mechanism import SUMMER, mean_interval
from src.health.experiments.t4_imbalance import (
    IMBALANCE_COLUMN,
    _terminal_prices,
    simulate_delivery,
    value_soc_gap,
)
from src.trading.battery import Battery
from src.trading.optimizer import period_hours
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

__all__ = ["ARMS", "LEVELS", "DayResult", "main", "replay_day", "results_markdown"]

PRODUCTION = "lightgbm_conformal"
CONTROL = "volumes"
LEVELS = {"limits_q25": 0.25, "limits_q10": 0.10}
ARMS = (CONTROL, *LEVELS)
RECONCILE_EUR = 0.01
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t5_bid_curves.md"


def _quantile_name(level: float) -> str:
    return f"q{round(level * 100):02d}"


class DayResult(dict[str, Any]):
    """One arm's settled day: money and how many orders were withheld."""


def replay_day(
    net_mw: NDArray[np.float64],
    price: NDArray[np.float64],
    rebap: NDArray[np.float64],
    sell_limit: NDArray[np.float64] | None,
    buy_limit: NDArray[np.float64] | None,
    battery: Battery,
    dt: float,
    terminal_price: float,
) -> DayResult:
    """Settle one day's schedule under limit orders.

    ``net_mw`` is the planned net power (discharge positive). With limits, a sale
    at period t is committed only if ``price[t] >= sell_limit[t]`` and a purchase
    only if ``price[t] <= buy_limit[t]``; without limits everything is committed.
    The committed schedule is replayed through the store: what it cannot deliver
    settles at ``rebap``, and the end-of-day gap at ``terminal_price``.
    """
    net = np.asarray(net_mw, dtype="float64")
    selling, buying = net > 0, net < 0
    executes: NDArray[np.bool_]
    if sell_limit is None or buy_limit is None:
        executes = np.ones(len(net), dtype=bool)
    else:
        executes = np.where(
            selling, price >= sell_limit, np.where(buying, price <= buy_limit, True)
        )
    committed = np.where(executes, net, 0.0)
    delivery = simulate_delivery(committed, np.zeros(len(net), dtype=bool), battery, dt)
    actual = delivery.actual_net_mw[0]
    wear_rate = battery.degradation_eur_per_mwh
    revenue = float(dt * np.sum(price * committed))
    deviation = actual - committed
    imbalance = float(dt * np.sum(deviation * rebap))
    wear = float(wear_rate * dt * delivery.discharge_mw[0].sum())
    soc_end = float(delivery.soc_mwh[0, -1])
    soc_value = float(value_soc_gap(np.array([soc_end]), battery, terminal_price)[0])
    return DayResult(
        revenue_eur=revenue,
        imbalance_eur=imbalance,
        wear_eur=wear,
        soc_end_mwh=soc_end,
        soc_value_eur=soc_value,
        pnl_no_costs_eur=revenue - wear,
        pnl_eur=revenue + imbalance - wear + soc_value,
        withheld_sales=int((selling & ~executes).sum()),
        withheld_purchases=int((buying & ~executes).sum()),
        undelivered_mwh=float(dt * np.abs(deviation).sum()),
        cycles=float(
            dt
            * delivery.discharge_mw[0].sum()
            / battery.discharge_efficiency
            / battery.capacity_mwh
        ),
    )


def _load(settings: Settings) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    processed = settings.data.processed_path
    dispatch = pd.read_parquet(processed / "trading" / PRODUCTION / "dispatch.parquet")
    dispatch = dispatch[dispatch["strategy"] == MEDIAN_FORECAST.name].sort_index()
    forecasts = pd.read_parquet(
        processed / "forecasts" / "comparison" / f"{PRODUCTION}.parquet"
    )
    rebap = pd.read_parquet(processed / "entsoe" / "imbalance_prices_de.parquet")[
        IMBALANCE_COLUMN
    ]
    return dispatch, rebap, forecasts


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
    lines = [
        "# T5 follow-up: bid curves from the quantile fan",
        "",
        "Generated by `python -m src.health.experiments.t5_bid_curves`. The production "
        f"schedule's volumes on the {summary['days']} validation days "
        f"({summary['first_day']} to {summary['last_day']}), bid three ways for a "
        "1 MW / 2 MWh battery with €8 wear: as fixed volumes at the clearing price "
        "(`volumes`, the backtest as it stands), and with a limit on every order from "
        "the forecast's fan: sales at q25 and purchases at q75 (`limits_q25`), or at "
        "q10 and q90 (`limits_q10`). An order executes only if the realised price "
        "reaches its limit. The committed orders are replayed through the store: a "
        "sale the store cannot cover settles at the German imbalance price, and the "
        "end-of-day state of charge gap at the day's terminal prices, as T4 does. The "
        "control reproduces the saved validation profit to within €0.01.",
        "",
        "Criterion, fixed before the run, for each limit arm: adopt only if its paired "
        "daily profit difference against `volumes`, imbalance and gap included, has a "
        "95% moving-block bootstrap interval (7-day blocks, 5,000 draws) entirely "
        "above zero.",
        "",
        "## Arms",
        "",
        "| arm | profit | capture | withheld sales | withheld purchases | "
        "undelivered MWh | imbalance cash | end-of-day gap value | cycles a day |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name in ARMS:
        a = arms[name]
        lines.append(
            f"| {name} | {_eur(a['pnl_eur'])} | {100 * a['capture']:.2f}% | "
            f"{a['withheld_sales']} | {a['withheld_purchases']} | "
            f"{a['undelivered_mwh']:.1f} | {_eur(a['imbalance_eur'])} | "
            f"{_eur(a['soc_value_eur'])} | {a['cycles_per_day']:.2f} |"
        )
    lines += [
        "",
        f"Perfect foresight made {_eur(summary['perfect_foresight_pnl_eur'])} on the "
        "same days. Withheld counts are quarter-hours.",
        "",
        "## Each limit arm minus volumes, euros per day",
        "",
        "| arm | all days | June to September | October to May | spike days | "
        "other days | without imbalance and gap | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in LEVELS:
        d = summary["difference"][name]
        verdict = "adopted" if d["all"]["low"] > 0 else "not adopted"
        lines.append(
            f"| {name} | {_interval(d['all'])} | {_interval(d['summer'])} | "
            f"{_interval(d['winter'])} | {_interval(d['spike_days'])} | "
            f"{_interval(d['other_days'])} | {_interval(d['no_costs'])} | {verdict} |"
        )
    lines += ["", "## Verdict", ""]
    adopted = [n for n in LEVELS if summary["difference"][n]["all"]["low"] > 0]
    if adopted:
        best = max(adopted, key=lambda n: summary["difference"][n]["all"]["mean"])
        lines.append(
            f"Adopted: {', '.join(adopted)}; the higher mean is `{best}`, "
            f"{_interval(summary['difference'][best]['all'])} € a day."
        )
    else:
        parts = ", ".join(
            f"{n} {_eur(arms[n]['pnl_eur'] - arms[CONTROL]['pnl_eur'])}" for n in LEVELS
        )
        lines.append(f"Neither arm is adopted: no interval lies above zero ({parts}).")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Bid curves from the quantile fan.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    battery = settings.battery
    ev = settings.evaluation
    dispatch, rebap, forecasts = _load(settings)
    days = sorted(
        d
        for d in dispatch["target_day"].unique()
        if ev.validation_start <= d < ev.holdout_start
    )
    dispatch = dispatch[dispatch["target_day"].isin(set(days))]
    dt = period_hours(pd.DatetimeIndex(dispatch.index[:96]))
    aligned = dispatch.assign(rebap=rebap.reindex(dispatch.index))
    if aligned["rebap"].isna().any():
        missing = aligned[aligned["rebap"].isna()]["target_day"].unique()
        raise RuntimeError(
            f"no imbalance price on {len(missing)} days, e.g. {missing[:3]}"
        )
    terminal = _terminal_prices(aligned, dt)
    quantiles = forecasts.reindex(dispatch.index)
    spike_days = (
        forecasts.groupby("target_day")["actual"].max() >= ev.spike_threshold_eur_mwh
    )

    rows = []
    for key, part in aligned.groupby("target_day", sort=True):
        day = cast(date, key)
        net = part["net_mw"].to_numpy(dtype="float64")
        price = part["realised_price"].to_numpy(dtype="float64")
        day_rebap = part["rebap"].to_numpy(dtype="float64")
        fan = quantiles.loc[part.index]
        for name in ARMS:
            if name == CONTROL:
                sell_limit = buy_limit = None
            else:
                level = LEVELS[name]
                sell_limit = fan[_quantile_name(level)].to_numpy(dtype="float64")
                buy_limit = fan[_quantile_name(1 - level)].to_numpy(dtype="float64")
            result = replay_day(
                net, price, day_rebap, sell_limit, buy_limit, battery, dt, terminal[day]
            )
            rows.append({"target_day": day, "arm": name, **result})
    table = pd.DataFrame(rows)
    profit = table.pivot(index="target_day", columns="arm", values="pnl_eur")
    saved = pd.read_parquet(
        settings.data.processed_path / "trading" / PRODUCTION / "pnl_daily.parquet"
    ).pivot(index="target_day", columns="strategy", values="pnl_eur")
    drift = float(
        (profit[CONTROL] - saved.loc[profit.index, MEDIAN_FORECAST.name]).abs().max()
    )
    if drift > RECONCILE_EUR:
        raise RuntimeError(
            f"the control misses the saved validation profit by €{drift:.4f}"
        )
    ceiling = float(saved.loc[profit.index, PERFECT_FORESIGHT.name].sum())
    traded = list(profit.index)
    summer = pd.Series(
        pd.to_datetime(pd.Index(traded)).month.isin(sorted(SUMMER)), index=traded
    )
    spiky = spike_days.reindex(traded).fillna(False).astype(bool)
    no_costs = table.pivot(index="target_day", columns="arm", values="pnl_no_costs_eur")

    def arm_summary(name: str) -> dict[str, Any]:
        part = table[table["arm"] == name]
        return {
            "pnl_eur": float(part["pnl_eur"].sum()),
            "capture": float(part["pnl_eur"].sum()) / ceiling,
            "withheld_sales": int(part["withheld_sales"].sum()),
            "withheld_purchases": int(part["withheld_purchases"].sum()),
            "withheld_on_spike_days": int(
                part.loc[
                    part["target_day"].map(spiky).astype(bool),
                    ["withheld_sales", "withheld_purchases"],
                ]
                .to_numpy()
                .sum()
            ),
            "undelivered_mwh": float(part["undelivered_mwh"].sum()),
            "imbalance_eur": float(part["imbalance_eur"].sum()),
            "soc_value_eur": float(part["soc_value_eur"].sum()),
            "cycles_per_day": float(part["cycles"].mean()),
        }

    def differences(name: str) -> dict[str, Any]:
        d = (profit[name] - profit[CONTROL]).loc[traded]
        return {
            "all": mean_interval(d),
            "summer": mean_interval(d, summer),
            "winter": mean_interval(d, ~summer),
            "spike_days": mean_interval(d, spiky),
            "other_days": mean_interval(d, ~spiky),
            "no_costs": mean_interval((no_costs[name] - no_costs[CONTROL]).loc[traded]),
        }

    summary: dict[str, Any] = {
        "days": len(traded),
        "first_day": str(traded[0]),
        "last_day": str(traded[-1]),
        "perfect_foresight_pnl_eur": ceiling,
        "arms": {name: arm_summary(name) for name in ARMS},
        "difference": {name: differences(name) for name in LEVELS},
    }
    summary["adopted"] = [
        n for n in LEVELS if summary["difference"][n]["all"]["low"] > 0
    ]
    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out / "t5_bid_curves_daily.parquet")
    (out / "t5_bid_curves_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    page = results_markdown(summary)
    RESULTS_PATH.write_text(page, encoding="utf-8")
    print(page)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
