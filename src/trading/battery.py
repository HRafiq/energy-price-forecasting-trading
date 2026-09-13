"""The battery as a trading asset: power, energy, efficiency and wear.

A 1 MW / 2 MWh battery is a two-hour battery: at full power it fills or empties in
two hours. Round-trip efficiency is split evenly between the two directions, so
charging and discharging each keep its square root. Degradation is priced per MWh
discharged to the grid, which stops the optimizer cycling on spreads too thin to
pay for the wear.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["Battery"]


class Battery(BaseModel):
    """Physical and economic parameters of one battery."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    power_mw: float = Field(gt=0)
    capacity_mwh: float = Field(gt=0)
    round_trip_efficiency: float = Field(gt=0, le=1)
    degradation_eur_per_mwh: float = Field(ge=0)
    #: Every delivery day starts and ends at this share of capacity.
    initial_soc_fraction: float = Field(default=0.5, ge=0, le=1)
    #: Optional cap on equivalent full cycles per delivery day.
    max_cycles_per_day: float | None = Field(default=None, gt=0)

    @property
    def duration_hours(self) -> float:
        """Hours to fill the battery from empty at full power, ignoring losses."""
        return self.capacity_mwh / self.power_mw

    @property
    def charge_efficiency(self) -> float:
        """Share of the energy bought that reaches the cells."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def discharge_efficiency(self) -> float:
        """Share of the energy drawn from the cells that reaches the grid."""
        return math.sqrt(self.round_trip_efficiency)

    @property
    def initial_soc_mwh(self) -> float:
        """State of charge at the start and end of every delivery day."""
        return self.initial_soc_fraction * self.capacity_mwh
