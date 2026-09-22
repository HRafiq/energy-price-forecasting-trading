"""Limit orders replayed through the store: what executes, and what it costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import Settings
from src.health.experiments import t5_bid_curves as bc

Floats = NDArray[np.float64]


def _day(settings: Settings) -> tuple[Floats, Floats, Floats]:
    """Buy 1 MW for an hour in the night, sell 0.9 MW for an hour in the evening.

    With a 90% round trip, 1 MWh bought stores 0.949 MWh and 0.9 MWh sold draws
    0.949, so the day ends where it started and nothing is curtailed unless an
    order is withheld.
    """
    net = np.zeros(96)
    net[8:12] = -1.0  # 02:00 to 02:59 charge
    net[76:80] = 0.9  # 19:00 to 19:59 discharge
    price = np.full(96, 50.0)
    price[8:12] = 20.0
    price[76:80] = 150.0
    rebap = np.full(96, 100.0)
    return net, price, rebap


def test_without_limits_the_replay_is_the_plain_settlement(settings: Settings) -> None:
    net, price, rebap = _day(settings)
    battery = settings.battery

    result = bc.replay_day(net, price, rebap, None, None, battery, 0.25, 60.0)

    bought, sold = 1.0, 0.9
    wear = battery.degradation_eur_per_mwh * sold
    assert result["revenue_eur"] == pytest.approx(150.0 * sold - 20.0 * bought)
    assert result["imbalance_eur"] == 0.0 and result["undelivered_mwh"] == 0.0
    assert result["soc_value_eur"] == pytest.approx(0.0, abs=1e-9)
    assert result["pnl_eur"] == pytest.approx(result["revenue_eur"] - wear)
    assert result["withheld_sales"] == 0 and result["withheld_purchases"] == 0


def test_a_withheld_purchase_starves_the_sale_and_pays_imbalance(
    settings: Settings,
) -> None:
    net, price, rebap = _day(settings)
    net[76:84] = 0.9  # two evening hours, 1.8 MWh, more than a half-full store holds
    price[80:84] = 150.0
    battery = settings.battery
    # Purchases allowed only up to 15 EUR: none execute. Sales allowed from 100.
    buy_limit = np.full(96, 15.0)
    sell_limit = np.full(96, 100.0)

    result = bc.replay_day(
        net, price, rebap, sell_limit, buy_limit, battery, 0.25, 60.0
    )

    assert result["withheld_purchases"] == 4 and result["withheld_sales"] == 0
    # The store starts half full (1 MWh); with losses it covers under 1 MWh of the
    # 1.8 MWh sold, and the rest is a delivery failure paid at reBAP.
    deliverable = battery.initial_soc_mwh * battery.discharge_efficiency
    assert result["undelivered_mwh"] == pytest.approx(1.8 - deliverable, abs=1e-6)
    assert result["imbalance_eur"] == pytest.approx(
        -100.0 * (1.8 - deliverable), abs=1e-6
    )
    assert result["soc_end_mwh"] == pytest.approx(0.0, abs=1e-9)
    # An empty store at the end of the day is a deficit against the start level.
    assert result["soc_value_eur"] < 0
    assert result["pnl_eur"] < result["pnl_no_costs_eur"]


def test_a_withheld_sale_keeps_the_energy_and_values_it_at_the_terminal_price(
    settings: Settings,
) -> None:
    net, price, rebap = _day(settings)
    battery = settings.battery
    sell_limit = np.full(96, 200.0)  # the evening price of 150 never reaches it
    buy_limit = np.full(96, 30.0)

    result = bc.replay_day(
        net, price, rebap, sell_limit, buy_limit, battery, 0.25, 60.0
    )

    assert result["withheld_sales"] == 4 and result["withheld_purchases"] == 0
    assert result["imbalance_eur"] == 0.0 and result["revenue_eur"] == pytest.approx(
        -20.0
    )
    stored = battery.initial_soc_mwh + battery.charge_efficiency * 1.0
    assert result["soc_end_mwh"] == pytest.approx(stored)
    surplus = (stored - battery.initial_soc_mwh) * battery.discharge_efficiency
    assert result["soc_value_eur"] == pytest.approx(
        surplus * (60.0 - battery.degradation_eur_per_mwh)
    )
    assert result["cycles"] == 0.0


def test_the_page_states_the_verdict_from_the_intervals() -> None:
    def interval(mean: float, low: float, high: float) -> dict[str, float]:
        return {
            "mean": mean,
            "low": low,
            "high": high,
            "days": 730,
            "total": 730 * mean,
        }

    arm = {
        "pnl_eur": 149498.0,
        "capture": 0.901,
        "withheld_sales": 0,
        "withheld_purchases": 0,
        "withheld_on_spike_days": 0,
        "undelivered_mwh": 0.0,
        "imbalance_eur": 0.0,
        "soc_value_eur": 0.0,
        "cycles_per_day": 1.73,
    }
    diff = {
        k: interval(0.0, -1.0, 1.0)
        for k in ("all", "summer", "winter", "spike_days", "other_days", "no_costs")
    }
    summary: dict[str, Any] = {
        "days": 730,
        "first_day": "2024-06-01",
        "last_day": "2026-05-31",
        "perfect_foresight_pnl_eur": 165922.0,
        "arms": {name: dict(arm) for name in bc.ARMS},
        "difference": {
            name: {k: dict(v) for k, v in diff.items()} for name in bc.LEVELS
        },
    }
    rejected = bc.results_markdown(summary)
    summary["difference"]["limits_q25"]["all"] = interval(2.0, 0.5, 3.5)
    adopted = bc.results_markdown(summary)

    assert "Neither arm is adopted" in rejected
    assert "Adopted: limits_q25" in adopted and "—" not in adopted
