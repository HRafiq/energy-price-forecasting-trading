"""Day-ahead battery dispatch as a mixed-integer linear program, in PuLP with CBC.

For the periods t of one delivery day, each Δt hours long:

    maximise    Σ_t Δt · (sell_t · d_t - buy_t · c_t - degradation · d_t)

    subject to  0 ≤ c_t ≤ P · u_t             charge only in charging periods
                0 ≤ d_t ≤ P · (1 - u_t)       discharge only in the others
                SoC_t = SoC_(t-1) + η_c · c_t · Δt - d_t · Δt / η_d
                0 ≤ SoC_t ≤ E
                SoC before the first period = SoC after the last = initial SoC
                c and d constant within one day-ahead product
                Σ_t d_t · Δt / η_d ≤ cycle cap · E, when a cap is set

c_t and d_t are charge and discharge power in MW at the grid connection, SoC_t is
the energy in the cells in MWh at the end of period t, and u_t is binary. Without
the binary, a negative price makes the linear relaxation charge and discharge at
once, burning energy to be paid for it; ``integer=False`` solves that relaxation
so tests can show it.

Sell and buy prices are one curve for perfect foresight and median dispatch.
Quantile-aware dispatch values the two legs at different quantiles. Ending every
day at its starting state of charge makes days independent and comparable. Before
1 October 2025 the auction traded hourly products, so the quarter-hours of one
hour must carry the same schedule; ``products`` labels which periods belong
together.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
import pulp
from numpy.typing import NDArray

from src.trading.battery import Battery

__all__ = [
    "DispatchError",
    "DispatchResult",
    "optimize_dispatch",
    "period_hours",
    "product_blocks",
]

#: Solver noise below this is rounded to zero in the returned schedule.
_TOLERANCE = 1e-7


class DispatchError(RuntimeError):
    """The solver did not prove an optimal schedule."""


@dataclass(frozen=True)
class DispatchResult:
    """An optimal schedule and its value on the prices it was optimized against.

    ``schedule`` has the price index and columns ``charge_mw``, ``discharge_mw``,
    ``net_mw`` (discharge minus charge, so positive means selling) and
    ``soc_mwh`` (state of charge at the end of each period).
    """

    schedule: pd.DataFrame
    objective_eur: float
    solve_seconds: float


def period_hours(index: pd.DatetimeIndex) -> float:
    """The common period length of a regular, increasing timestamp index."""
    if len(index) < 2:
        raise ValueError("at least two periods are needed to infer the period length")
    steps = index[1:] - index[:-1]
    if steps[0] <= pd.Timedelta(0) or not (steps == steps[0]).all():
        raise ValueError("timestamps must be increasing and evenly spaced")
    return float(steps[0] / pd.Timedelta(hours=1))


def product_blocks(
    index: pd.DatetimeIndex, product_minutes: pd.Series | NDArray[np.int64]
) -> NDArray[np.int64]:
    """Label each period with the day-ahead product that contains it.

    Periods of a 60-minute product share the label of their UTC hour; German
    local hours are whole UTC hours, including on DST days.
    """
    if index.tz is None:
        raise ValueError("the index must be timezone-aware")
    minutes = np.asarray(product_minutes, dtype="int64")
    if len(minutes) != len(index):
        raise ValueError("product_minutes must have one value per period")
    utc = index.tz_convert("UTC")
    labels = np.empty(len(index), dtype="int64")
    for length in np.unique(minutes):
        if length <= 0:
            raise ValueError(f"invalid product length {length} minutes")
        mask = minutes == length
        labels[mask] = utc[mask].floor(f"{int(length)}min").asi8
    return labels


def optimize_dispatch(
    sell_prices: pd.Series,
    battery: Battery,
    *,
    buy_prices: pd.Series | None = None,
    products: NDArray[np.int64] | None = None,
    integer: bool = True,
    time_limit_s: float = 60.0,
    soc_floor: NDArray[np.float64] | None = None,
) -> DispatchResult:
    """Solve one delivery day's schedule.

    ``sell_prices`` values discharging and ``buy_prices`` values charging; both
    are €/MWh on the same regular index. ``products`` optionally labels periods
    of one day-ahead product, which must be contiguous. ``soc_floor`` optionally
    gives a lower bound in MWh on the state of charge at the end of each period,
    NaN where there is none; a rule that holds charge back for the evening is
    expressed this way.
    """
    index = pd.DatetimeIndex(sell_prices.index)
    if buy_prices is not None and not buy_prices.index.equals(sell_prices.index):
        raise ValueError("buy_prices must have the same index as sell_prices")
    sell = sell_prices.to_numpy(dtype=float)
    buy = sell if buy_prices is None else buy_prices.to_numpy(dtype=float)
    if not (np.isfinite(sell).all() and np.isfinite(buy).all()):
        raise ValueError("prices must be finite")
    n = len(sell)
    dt = period_hours(index)
    _check_products(products, n)
    if soc_floor is not None:
        if len(soc_floor) != n:
            raise ValueError("soc_floor must have one value per period")
        if np.nanmax(soc_floor, initial=0.0) > battery.capacity_mwh + 1e-9:
            raise ValueError("soc_floor cannot exceed the battery's capacity")

    problem = pulp.LpProblem("battery_dispatch", pulp.LpMaximize)
    charge = [
        problem.add_variable(f"charge_{t}", lowBound=0, upBound=battery.power_mw)
        for t in range(n)
    ]
    discharge = [
        problem.add_variable(f"discharge_{t}", lowBound=0, upBound=battery.power_mw)
        for t in range(n)
    ]
    soc = [
        problem.add_variable(f"soc_{t}", lowBound=0, upBound=battery.capacity_mwh)
        for t in range(n)
    ]
    degradation = battery.degradation_eur_per_mwh
    problem += pulp.lpSum(
        dt * ((sell[t] - degradation) * discharge[t] - buy[t] * charge[t])
        for t in range(n)
    )

    if integer:
        for t in range(n):
            charging = problem.add_variable(f"charging_{t}", cat=pulp.LpBinary)
            problem += charge[t] <= battery.power_mw * charging, f"charge_mode_{t}"
            problem += (
                discharge[t] <= battery.power_mw * (1 - charging),
                f"discharge_mode_{t}",
            )

    eta_c, eta_d = battery.charge_efficiency, battery.discharge_efficiency
    previous: pulp.LpVariable | float = battery.initial_soc_mwh
    for t in range(n):
        problem += (
            soc[t] == previous + eta_c * dt * charge[t] - (dt / eta_d) * discharge[t],
            f"soc_balance_{t}",
        )
        previous = soc[t]
    problem += soc[n - 1] == battery.initial_soc_mwh, "end_at_start_soc"
    if soc_floor is not None:
        for t in range(n):
            if np.isfinite(soc_floor[t]):
                problem += soc[t] >= float(soc_floor[t]), f"soc_floor_{t}"

    if products is not None:
        for t in range(1, n):
            if products[t] == products[t - 1]:
                problem += charge[t] == charge[t - 1], f"product_charge_{t}"
                problem += discharge[t] == discharge[t - 1], f"product_discharge_{t}"

    if battery.max_cycles_per_day is not None:
        problem += (
            pulp.lpSum(discharge) * (dt / eta_d)
            <= battery.max_cycles_per_day * battery.capacity_mwh,
            "cycle_cap",
        )

    started = time.perf_counter()
    problem.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit_s))
    elapsed = time.perf_counter() - started
    if problem.sol_status != pulp.LpSolutionOptimal:
        raise DispatchError(
            f"no proven optimum for {index[0]} ({n} periods): solver status "
            f"{pulp.LpStatus[problem.status]}, solution "
            f"{pulp.LpSolution[problem.sol_status]}"
        )

    values = {
        "charge_mw": _values(charge),
        "discharge_mw": _values(discharge),
        "soc_mwh": _values(soc),
    }
    schedule = pd.DataFrame(
        {
            "charge_mw": values["charge_mw"],
            "discharge_mw": values["discharge_mw"],
            "net_mw": values["discharge_mw"] - values["charge_mw"],
            "soc_mwh": values["soc_mwh"],
        },
        index=sell_prices.index,
    )
    objective = float(
        dt
        * np.sum(
            (sell - degradation) * values["discharge_mw"] - buy * values["charge_mw"]
        )
    )
    return DispatchResult(
        schedule=schedule, objective_eur=objective, solve_seconds=elapsed
    )


def _check_products(products: NDArray[np.int64] | None, n: int) -> None:
    if products is None:
        return
    if len(products) != n:
        raise ValueError("products must have one label per period")
    starts = [products[0]] + [b for a, b in pairwise(products) if a != b]
    if len(starts) != len(set(starts)):
        raise ValueError("periods of one product must be contiguous")


def _values(variables: list[pulp.LpVariable]) -> NDArray[np.float64]:
    raw = [variable.varValue for variable in variables]
    if any(value is None for value in raw):
        raise DispatchError("the solver returned a schedule with missing values")
    values = np.asarray(raw, dtype=float)
    values[np.abs(values) < _TOLERANCE] = 0.0
    return values
