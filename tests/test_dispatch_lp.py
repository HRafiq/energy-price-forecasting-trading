"""The HiGHS linear program agrees with the production optimiser where it should."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.trading.dispatch_lp import DayProgram, solve_day
from src.trading.optimizer import optimize_dispatch, product_blocks


def _day(hourly: bool = False) -> tuple[pd.Series, pd.Series]:
    index = pd.date_range("2025-03-10 23:00", periods=96, freq="15min", tz="UTC")
    hour = np.arange(96) // 4
    # Small noise breaks price ties, so the optimum schedule is unique: one price
    # per hourly product, or one per quarter-hour.
    drawn = np.random.default_rng(4).normal(0.0, 0.5, 96)
    noise = drawn[::4][hour] if hourly else drawn
    prices = 60.0 + 40.0 * np.sin(2 * np.pi * (hour - 5) / 24) + (hour == 19) * 90.0
    prices = prices + noise
    minutes = pd.Series(60 if hourly else 15, index=index)
    return pd.Series(prices, index=index), minutes


@pytest.mark.parametrize("hourly", [False, True])
def test_the_relaxation_reproduces_the_integer_optimum(
    settings: Settings, hourly: bool
) -> None:
    prices, minutes = _day(hourly)
    products = product_blocks(pd.DatetimeIndex(prices.index), minutes)
    program = DayProgram.build(96, 0.25, settings.battery, products)

    net, value = solve_day(program, prices.to_numpy())
    milp = optimize_dispatch(prices, settings.battery, products=products)

    assert value == pytest.approx(milp.objective_eur, abs=1e-6)
    assert net == pytest.approx(milp.schedule["net_mw"].to_numpy(), abs=1e-6)
    if hourly:
        # One schedule per hourly product.
        assert np.abs(np.diff(net.reshape(24, 4), axis=1)).max() < 1e-9


def test_the_day_ends_where_it_started_and_respects_the_cycle_cap(
    settings: Settings,
) -> None:
    prices, minutes = _day()
    battery = settings.battery
    program = DayProgram.build(96, 0.25, battery, None)

    net, _ = solve_day(program, prices.to_numpy())

    charge, discharge = np.maximum(-net, 0.0), np.maximum(net, 0.0)
    stored = (
        battery.charge_efficiency * 0.25 * charge
        - 0.25 / battery.discharge_efficiency * discharge
    ).cumsum() + battery.initial_soc_mwh
    assert stored[-1] == pytest.approx(battery.initial_soc_mwh, abs=1e-9)
    assert stored.min() >= -1e-9 and stored.max() <= battery.capacity_mwh + 1e-9
    assert battery.max_cycles_per_day is not None
    assert (0.25 / battery.discharge_efficiency * discharge).sum() <= (
        battery.max_cycles_per_day * battery.capacity_mwh + 1e-9
    )
    assert net.max() <= battery.power_mw + 1e-9


def test_a_flat_day_trades_nothing_and_the_price_count_is_checked(
    settings: Settings,
) -> None:
    program = DayProgram.build(96, 0.25, settings.battery, None)

    net, value = solve_day(program, np.full(96, 50.0))

    assert value == pytest.approx(0.0, abs=1e-9) and np.abs(net).max() < 1e-9
    with pytest.raises(ValueError, match="expected 96 prices"):
        solve_day(program, np.full(95, 50.0))
