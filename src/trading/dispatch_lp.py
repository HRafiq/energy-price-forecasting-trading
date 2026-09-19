"""The battery day as a linear program, solved with HiGHS, for training loops.

The production optimiser (:mod:`src.trading.optimizer`) is a mixed-integer program
in PuLP with CBC and takes about 50 ms a day. A loss that needs a schedule for
every training day on every boosting round solves the same problem tens of
thousands of times, so this module states it as a plain linear program, drops the
charge-or-discharge binary, and hands it to HiGHS through SciPy in a few
milliseconds. Everything else is the same: power and capacity limits, charge and
discharge efficiencies, the day ending at its starting state of charge, wear per
MWh discharged, the cycle cap, and one schedule per day-ahead product. With
non-negative prices and efficiencies below one the relaxation never charges and
discharges at once, so it matches the integer optimum; a negative price can make it
do both, which is the one case where the two can differ.

Variables per period t: charge c_t and discharge d_t in MW. The state of charge is
a running sum, so it needs no variables of its own:

    SoC_t = SoC_0 + Σ_{s<=t} (eta_c dt c_s - dt / eta_d d_s), 0 <= SoC_t <= E,
    SoC_n = SoC_0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, vstack

from src.trading.battery import Battery

__all__ = ["DayProgram", "solve_day"]


@dataclass(frozen=True)
class DayProgram:
    """Everything about one day's program except the prices.

    Built once per day shape (number of periods, their length and the product
    labels) and reused for every price vector.
    """

    n: int
    dt: float
    degradation: float
    a_ub: csr_matrix
    b_ub: NDArray[np.float64]
    a_eq: csr_matrix
    b_eq: NDArray[np.float64]
    bounds: list[tuple[float, float]]

    @classmethod
    def build(
        cls,
        n: int,
        dt: float,
        battery: Battery,
        products: NDArray[np.int64] | None = None,
    ) -> DayProgram:
        eta_c, eta_d = battery.charge_efficiency, battery.discharge_efficiency
        lower = np.tril(np.ones((n, n)))
        # Cumulative state of charge in terms of c (first n columns) and d.
        soc = np.hstack([eta_c * dt * lower, -(dt / eta_d) * lower])
        s0, cap = battery.initial_soc_mwh, battery.capacity_mwh
        ub_rows: list[NDArray[np.float64]] = [soc, -soc]
        ub_rhs: list[NDArray[np.float64]] = [np.full(n, cap - s0), np.full(n, s0)]
        if battery.max_cycles_per_day is not None:
            cycle = np.hstack([np.zeros(n), np.full(n, dt / eta_d)]).reshape(1, -1)
            ub_rows.append(cycle)
            ub_rhs.append(np.array([battery.max_cycles_per_day * cap]))
        eq_rows: list[NDArray[np.float64]] = [soc[-1:, :]]
        eq_rhs: list[NDArray[np.float64]] = [np.array([0.0])]
        if products is not None:
            ties = [t for t in range(1, n) if products[t] == products[t - 1]]
            if ties:
                tie = np.zeros((2 * len(ties), 2 * n))
                for k, t in enumerate(ties):
                    tie[2 * k, t], tie[2 * k, t - 1] = 1.0, -1.0
                    tie[2 * k + 1, n + t], tie[2 * k + 1, n + t - 1] = 1.0, -1.0
                eq_rows.append(tie)
                eq_rhs.append(np.zeros(2 * len(ties)))
        return cls(
            n=n,
            dt=dt,
            degradation=battery.degradation_eur_per_mwh,
            a_ub=csr_matrix(vstack([csr_matrix(r) for r in ub_rows])),
            b_ub=np.concatenate(ub_rhs),
            a_eq=csr_matrix(vstack([csr_matrix(r) for r in eq_rows])),
            b_eq=np.concatenate(eq_rhs),
            bounds=[(0.0, battery.power_mw)] * (2 * n),
        )


def solve_day(
    program: DayProgram,
    sell: NDArray[np.float64],
    buy: NDArray[np.float64] | None = None,
) -> tuple[NDArray[np.float64], float]:
    """Net power (discharge minus charge) per period, and the objective in euros.

    ``sell`` values discharging and ``buy`` charging, both in €/MWh; ``buy``
    defaults to ``sell``. The objective is the schedule's value on those prices,
    net of wear, exactly as the production optimiser defines it.
    """
    prices_buy = sell if buy is None else buy
    if len(sell) != program.n or len(prices_buy) != program.n:
        raise ValueError(f"expected {program.n} prices")
    # linprog minimises: negate the profit.
    cost = np.concatenate(
        [program.dt * prices_buy, -program.dt * (sell - program.degradation)]
    )
    result = linprog(
        cost,
        A_ub=program.a_ub,
        b_ub=program.b_ub,
        A_eq=program.a_eq,
        b_eq=program.b_eq,
        bounds=program.bounds,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"HiGHS did not solve the day: {result.message}")
    x = np.asarray(result.x, dtype="float64")
    charge, discharge = x[: program.n], x[program.n :]
    return discharge - charge, float(-result.fun)
