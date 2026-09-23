"""T7: dispatching on coherent price paths instead of one point forecast.

    uv run python -m src.health.experiments.t7_scenario_dispatch

The plan is ``docs/plans/scenario_dispatch_plan.md``, committed before this file.

The optimiser's objective is linear in price and none of its constraints depend
on price, so the expected profit of a fixed schedule equals its profit at the
mean price. A risk-neutral schedule over any scenario set is therefore the
schedule at the scenario mean, and dispatching on paths cannot differ from
dispatching on their average. The ``mean`` arm below exists to prove that the
scenario machinery reproduces exactly that, and the run stops if it does not.

What can differ is a risk-sensitive objective. The CVaR arms give up expected
value to lift the worst scenarios, which is the only way a scenario set can
change what the battery does here.

Scenarios come from an empirical copula. For every past delivery day the realised
price of each quarter-hour is located inside that day's own forecast quantiles,
giving a rank in [0, 1] per period: a 96-vector of how that day landed within its
forecast. A scenario for the target day is one such vector drawn from days
strictly before it, read back through the target day's quantiles. The marginals
are the model's own; the dependence across periods is one real day's.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, vstack

from src.config import REPO_ROOT, Settings, load_settings
from src.forecasting.base import quantile_column
from src.health.experiments.t1_mechanism import SUMMER, mean_interval
from src.trading.battery import Battery
from src.trading.dispatch_lp import DayProgram
from src.trading.optimizer import period_hours
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

__all__ = [
    "ARMS",
    "CVAR_ALPHA",
    "SCENARIOS",
    "Tails",
    "day_ranks",
    "results_markdown",
    "scenario_prices",
    "solve_cvar",
]

PRODUCTION = "lightgbm_conformal"
#: Scenarios drawn per delivery day, from the most recent eligible days.
SCENARIOS = 200
#: The tail CVaR averages over: the worst fifth of scenarios.
CVAR_ALPHA = 0.2
#: Weight on the CVaR term; 0.0 is the risk-neutral reference.
ARMS: dict[str, float] = {"mean": 0.0, "cvar_25": 0.25, "cvar_50": 0.5}
LEVELS = ("cvar_25", "cvar_50")
OUT = "t7_scenario_dispatch"


#: Ranks are kept strictly inside (0, 1) so the tail transform stays invertible.
EDGE = 1e-9


@dataclass(frozen=True)
class Tails:
    """How far prices run past the ends of the fan, as a share of its half-width.

    14.4% of validation periods land outside q05 to q95, and clamping them to the
    edge throws away how far outside they went: on the 7.3% above the fan the
    realised price averaged 138.7 EUR/MWh against a fan top of 116.9, so flat
    tails understate those periods by about 22 EUR/MWh. Scenarios built that way
    cannot contain a real spike, and a hedge tested against them is being asked
    to insure a calmer world than the one that exists.

    So each tail is given an exponential shape whose scale is a multiple of the
    day's own spread, fitted on past days only. ``lower`` and ``upper`` are those
    multiples; a day with a wide fan gets a proportionally longer tail.
    """

    lower: float
    upper: float

    @classmethod
    def flat(cls) -> Tails:
        """Tails of zero length: the clamped behaviour, kept for comparison."""
        return cls(lower=0.0, upper=0.0)


Edges = tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
]


def _spreads(quantiles: NDArray[np.float64], levels: NDArray[np.float64]) -> Edges:
    """The fan's edges and half-widths, floored so a flat day cannot divide by zero."""
    mid = int(np.argmin(np.abs(levels - 0.5)))
    low, high = quantiles[:, 0], quantiles[:, -1]
    below = np.maximum(quantiles[:, mid] - low, 1e-6)
    above = np.maximum(high - quantiles[:, mid], 1e-6)
    return low, high, below, above


def fit_tails(
    quantiles: NDArray[np.float64],
    levels: NDArray[np.float64],
    actual: NDArray[np.float64],
) -> Tails:
    """Mean normalised overshoot beyond each end of the fan, over the days given."""
    low, high, below, above = _spreads(quantiles, levels)
    under, over = actual < low, actual > high
    lower = float(((low - actual)[under] / below[under]).mean()) if under.any() else 0.0
    upper = float(((actual - high)[over] / above[over]).mean()) if over.any() else 0.0
    return Tails(lower=max(lower, 0.0), upper=max(upper, 0.0))


def day_ranks(
    quantiles: NDArray[np.float64],
    levels: NDArray[np.float64],
    actual: NDArray[np.float64],
    tails: Tails | None = None,
) -> NDArray[np.float64]:
    """Where each realised price sits inside its own forecast, in (0, 1).

    Linear interpolation between the quantile levels. Outside the fan the rank
    continues into the tail rather than stopping at the edge, so a price far
    above q95 is recorded as further out than one just above it.
    """
    tails = tails or Tails.flat()
    n = quantiles.shape[0]
    out = np.empty(n, dtype="float64")
    for t in range(n):
        out[t] = float(np.interp(actual[t], quantiles[t], levels))
    low, high, below, above = _spreads(quantiles, levels)
    if tails.upper > 0:
        over = actual > high
        excess = (actual[over] - high[over]) / above[over]
        out[over] = 1.0 - (1.0 - levels[-1]) * np.exp(-excess / tails.upper)
    if tails.lower > 0:
        under = actual < low
        deficit = (low[under] - actual[under]) / below[under]
        out[under] = levels[0] * np.exp(-deficit / tails.lower)
    return np.clip(out, EDGE, 1.0 - EDGE)


def scenario_prices(
    quantiles: NDArray[np.float64],
    levels: NDArray[np.float64],
    ranks: NDArray[np.float64],
    tails: Tails | None = None,
) -> NDArray[np.float64]:
    """Read a day's rank vectors back through another day's quantiles.

    The exact inverse of :func:`day_ranks`, tails included, so a rank recorded
    beyond the fan comes back as a price beyond the fan.
    """
    tails = tails or Tails.flat()
    draws, n = ranks.shape
    prices = np.empty((draws, n), dtype="float64")
    for t in range(n):
        prices[:, t] = np.interp(ranks[:, t], levels, quantiles[t])
    low, high, below, above = _spreads(quantiles, levels)
    if tails.upper > 0:
        excess = -tails.upper * np.log(
            np.clip((1.0 - ranks) / (1.0 - levels[-1]), EDGE, None)
        )
        prices[ranks > levels[-1]] = (high + excess * above)[ranks > levels[-1]]
    if tails.lower > 0:
        deficit = -tails.lower * np.log(np.clip(ranks / levels[0], EDGE, None))
        prices[ranks < levels[0]] = (low - deficit * below)[ranks < levels[0]]
    return prices


def solve_cvar(
    program: DayProgram, prices: NDArray[np.float64], weight: float
) -> NDArray[np.float64]:
    """Net power per period, maximising the mean blended with the tail.

    ``prices`` is scenarios by periods. With ``weight`` zero this is the
    risk-neutral problem, which by the argument in the module docstring has the
    same solution as dispatching at the scenario mean.
    """
    n, dt, wear = program.n, program.dt, program.degradation
    draws = prices.shape[0]
    bounds: list[tuple[float | None, float | None]]
    # profit of schedule x under scenario s is gain[s] @ x, x = [charge, discharge]
    gain = np.hstack([-dt * prices, dt * (prices - wear)])
    if weight == 0.0:
        cost = -gain.mean(axis=0)
        a_ub, b_ub = program.a_ub, program.b_ub
        a_eq, b_eq = program.a_eq, program.b_eq
        bounds = list(program.bounds)
    else:
        # z = [x (2n), var (1), u (draws)]; u_s >= var - profit_s, u_s >= 0.
        mean_gain = gain.mean(axis=0)
        cost = np.concatenate(
            [
                -(1 - weight) * mean_gain,
                [-weight],
                np.full(draws, weight / (CVAR_ALPHA * draws)),
            ]
        )
        tail = hstack(
            [
                csr_matrix(-gain),
                csr_matrix(np.ones((draws, 1))),
                csr_matrix(-np.eye(draws)),
            ]
        )
        pad = csr_matrix((program.a_ub.shape[0], 1 + draws))
        a_ub = vstack([hstack([program.a_ub, pad]), tail]).tocsr()
        b_ub = np.concatenate([program.b_ub, np.zeros(draws)])
        a_eq = hstack(
            [program.a_eq, csr_matrix((program.a_eq.shape[0], 1 + draws))]
        ).tocsr()
        b_eq = program.b_eq
        bounds = list(program.bounds)
        bounds.append((None, None))
        bounds += [(0.0, None)] * draws
    result = linprog(
        cost, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs"
    )
    if not result.success:
        raise RuntimeError(f"HiGHS did not solve the day: {result.message}")
    x: NDArray[np.float64] = np.asarray(result.x, dtype="float64")
    net: NDArray[np.float64] = x[n : 2 * n] - x[:n]
    return net


def settle(
    net: NDArray[np.float64], price: NDArray[np.float64], battery: Battery, dt: float
) -> float:
    """What a schedule earned at realised prices, net of wear, as the backtest does."""
    discharge = np.clip(net, 0.0, None)
    wear = battery.degradation_eur_per_mwh
    return float(dt * (price @ net - wear * discharge.sum()))


def _load(settings: Settings) -> tuple[pd.DataFrame, pd.DataFrame]:
    processed = settings.data.processed_path
    dispatch = pd.read_parquet(processed / "trading" / PRODUCTION / "dispatch.parquet")
    forecasts = pd.read_parquet(
        processed / "forecasts" / "comparison" / f"{PRODUCTION}.parquet"
    )
    return dispatch, forecasts


def run(settings: Settings, days_limit: int | None = None) -> dict[str, Any]:
    """Every arm over the validation days, settled at realised prices."""
    quantile_levels = np.asarray(settings.forecasting.quantiles, dtype="float64")
    columns = [quantile_column(q) for q in settings.forecasting.quantiles]
    dispatch, forecasts = _load(settings)
    battery = settings.battery

    evaluation = settings.evaluation
    in_window = (forecasts["target_day"] >= evaluation.validation_start) & (
        forecasts["target_day"] < evaluation.holdout_start
    )
    forecasts = forecasts.loc[in_window]
    days = sorted(forecasts["target_day"].unique())
    if days_limit:
        days = days[:days_limit]

    control = dispatch[dispatch["strategy"] == MEDIAN_FORECAST.name]
    control_net = {d: g["net_mw"].to_numpy() for d, g in control.groupby("target_day")}
    perfect = dispatch[dispatch["strategy"] == PERFECT_FORESIGHT.name]

    history: list[NDArray[np.float64]] = []
    # The tails are fitted on past days only and refitted as the pool moves, so
    # no day contributes to the shape of the scenarios it is scored against.
    seen_q: list[NDArray[np.float64]] = []
    seen_a: list[NDArray[np.float64]] = []
    tails = Tails.flat()
    rows: list[dict[str, Any]] = []
    for day in days:
        frame = forecasts[forecasts["target_day"] == day]
        quantiles = frame[columns].to_numpy(dtype="float64")
        actual = frame["actual"].to_numpy(dtype="float64")
        if not np.isfinite(actual).all() or day not in control_net:
            continue
        dt = period_hours(pd.DatetimeIndex(frame.index))
        program = DayProgram.build(len(frame), dt, battery)

        row: dict[str, Any] = {"target_day": day}
        row["median"] = settle(control_net[day], actual, battery, dt)
        # A Berlin day has 92, 96 or 100 quarter-hours, so a rank vector only
        # applies to a day of its own shape. Spring and autumn DST days therefore
        # draw on their own kind, which is why the pool is filtered rather than
        # simply the most recent days.
        same_shape = [v for v in history if v.shape[1] == quantiles.shape[0]]
        if len(same_shape) >= SCENARIOS:
            pool = np.vstack(same_shape[-SCENARIOS:])
            prices = scenario_prices(quantiles, quantile_levels, pool, tails)
            for arm, weight in ARMS.items():
                net = solve_cvar(program, prices, weight)
                row[arm] = settle(net, actual, battery, dt)
                row[f"{arm}_expected"] = float(
                    np.mean([settle(net, p, battery, dt) for p in prices])
                )
        seen_q.append(quantiles)
        seen_a.append(actual)
        tails = fit_tails(
            np.vstack(seen_q), quantile_levels, np.concatenate(seen_a)
        )
        history.append(
            day_ranks(quantiles, quantile_levels, actual, tails).reshape(1, -1)
        )
        rows.append(row)

    daily = pd.DataFrame(rows).set_index("target_day")
    scored = daily.dropna(subset=list(ARMS))
    summary: dict[str, Any] = {
        "days": len(scored),
        "first_day": str(scored.index.min()) if len(scored) else None,
        "last_day": str(scored.index.max()) if len(scored) else None,
        "scenarios": SCENARIOS,
        "cvar_alpha": CVAR_ALPHA,
        "tails": {"lower": tails.lower, "upper": tails.upper},
        "perfect_foresight_pnl_eur": _perfect_total(perfect, scored.index, battery),
        "arms": {
            arm: {
                "pnl_eur": float(scored[arm].sum()),
                "expected_at_scenario_mean_eur": float(scored[f"{arm}_expected"].sum()),
            }
            for arm in ARMS
        }
        | {"median": {"pnl_eur": float(scored["median"].sum())}},
        "difference": {
            arm: {"all": mean_interval(scored[arm] - scored["median"])}
            for arm in ARMS
        },
    }
    summer = pd.Series(
        [d.month in SUMMER for d in scored.index], index=scored.index
    )
    for arm in ARMS:
        diff = scored[arm] - scored["median"]
        summary["difference"][arm]["summer"] = mean_interval(diff, summer)
        summary["difference"][arm]["winter"] = mean_interval(diff, ~summer)
    return summary | {"daily": daily}


def _perfect_total(
    perfect: pd.DataFrame, days: pd.Index, battery: Battery
) -> float:
    """What perfect foresight made on the same days, settled the same way."""
    total = 0.0
    for _, group in perfect[perfect["target_day"].isin(days)].groupby("target_day"):
        dt = period_hours(pd.DatetimeIndex(group.index))
        total += settle(
            group["net_mw"].to_numpy(dtype="float64"),
            group["realised_price"].to_numpy(dtype="float64"),
            battery,
            dt,
        )
    return float(total)


def results_markdown(summary: dict[str, Any]) -> str:
    """The results page, with the verdict read off the intervals."""
    arms, diff = summary["arms"], summary["difference"]
    lines = [
        "# T7: dispatching on price paths instead of point forecasts",
        "",
        f"Generated by `python -m src.health.experiments.{OUT}`. "
        f"{summary['days']} validation days, {summary['first_day']} to "
        f"{summary['last_day']}, {summary['scenarios']} scenarios a day, CVaR over "
        f"the worst {summary['cvar_alpha']:.0%} of them. Plan: "
        "`docs/plans/scenario_dispatch_plan.md`, committed before the code.",
        "",
        "## Arms",
        "",
        "| arm | profit | expected profit at its own scenario mean |",
        "|---|---|---|",
        f"| median (control) | €{arms['median']['pnl_eur']:,.0f} | |",
    ]
    for arm in ARMS:
        lines.append(
            f"| {arm} | €{arms[arm]['pnl_eur']:,.0f} | "
            f"€{arms[arm]['expected_at_scenario_mean_eur']:,.0f} |"
        )
    lines += ["", "## Each arm minus the control, euros per day", "",
              "| arm | all days | summer | winter |", "|---|---|---|---|"]
    for arm in ARMS:
        cells = []
        for scope in ("all", "summer", "winter"):
            d = diff[arm][scope]
            cells.append(f"{d['mean']:+.2f} ({d['low']:+.2f} to {d['high']:+.2f})")
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    adopted = [a for a in LEVELS if diff[a]["all"]["low"] > 0]
    verdict = (
        f"Adopted: {max(adopted, key=lambda a: diff[a]['all']['mean'])}."
        if adopted
        else "Neither risk-sensitive arm is adopted: no interval lies above zero."
    )
    lines += ["", "## Verdict", "", verdict]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--days", type=int, default=None, help="first N days only")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    summary = run(settings, days_limit=args.days)
    daily = summary.pop("daily")
    processed = settings.data.processed_path / "experiments"
    processed.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(processed / f"{OUT}_daily.parquet")
    (processed / f"{OUT}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    page = REPO_ROOT / "docs" / "results" / f"{OUT}.md"
    page.write_text(results_markdown(summary), encoding="utf-8")
    print(results_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
