"""Settle a committed day-ahead schedule at the realised clearing prices.

The battery is a price taker: its volumes do not move the auction, and every
committed MW clears at the published price. Revenue is the sum over periods of
Δt · price · net power, so charging at a negative price earns money. Degradation
is charged per MWh discharged to the grid. Cycles count the energy drawn from the
cells in units of capacity, so one full discharge from full to empty is one cycle.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.trading.battery import Battery
from src.trading.optimizer import period_hours

__all__ = ["Settlement", "settle"]


@dataclass(frozen=True)
class Settlement:
    """Money and energy of one settled schedule."""

    revenue_eur: float
    degradation_eur: float
    pnl_eur: float
    charged_mwh: float
    discharged_mwh: float
    cycles: float


def settle(schedule: pd.DataFrame, prices: pd.Series, battery: Battery) -> Settlement:
    """Value ``schedule`` at ``prices``, which must share its index."""
    if not prices.index.equals(schedule.index):
        raise ValueError("prices must have the same index as the schedule")
    price = prices.to_numpy(dtype=float)
    if not np.isfinite(price).all():
        raise ValueError("settlement prices must be finite")
    dt = period_hours(pd.DatetimeIndex(schedule.index))
    charge = schedule["charge_mw"].to_numpy(dtype=float)
    discharge = schedule["discharge_mw"].to_numpy(dtype=float)
    revenue = float(dt * np.sum(price * (discharge - charge)))
    discharged = float(dt * discharge.sum())
    degradation = battery.degradation_eur_per_mwh * discharged
    return Settlement(
        revenue_eur=revenue,
        degradation_eur=degradation,
        pnl_eur=revenue - degradation,
        charged_mwh=float(dt * charge.sum()),
        discharged_mwh=discharged,
        cycles=discharged / battery.discharge_efficiency / battery.capacity_mwh,
    )
