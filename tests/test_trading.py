"""Battery optimizer, settlement and strategies.

The toy cases are small enough to check by hand; the arithmetic is in each
docstring. A dynamic program over state of charge checks optimality on random
lossless days, the same formulation built as matrices and solved by SciPy's HiGHS
checks days with losses, hourly products and a cycle cap, and perfect foresight
must bound every forecast strategy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
import yaml
from numpy.typing import NDArray
from pydantic import ValidationError

from src.config import DEFAULT_SETTINGS_PATH, Settings
from src.forecasting.base import quantile_column
from src.forecasting.walkforward import HoldoutAccessError
from src.trading import run_strategies as runner_module
from src.trading.battery import Battery
from src.trading.optimizer import (
    DispatchError,
    optimize_dispatch,
    period_hours,
    product_blocks,
)
from src.trading.run_strategies import (
    check_ceiling,
    run_strategies,
    select_days,
    summarise,
)
from src.trading.settlement import settle
from src.trading.strategies import (
    MEDIAN_FORECAST,
    PERFECT_FORESIGHT,
    DayResult,
    Strategy,
    build_strategies,
    dispatch_day,
    quantile_aware,
)

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
Z_SCORES = {
    0.05: -1.645,
    0.10: -1.2816,
    0.25: -0.6745,
    0.50: 0.0,
    0.75: 0.6745,
    0.90: 1.2816,
    0.95: 1.645,
}


def prices(values: Sequence[float], start: str = "2025-11-03 23:00") -> pd.Series:
    index = pd.date_range(start, periods=len(values), freq="15min", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


def day_frame(
    settings: Settings,
    day: date,
    seed: int = 0,
    product_minutes: int = 15,
    band_scale: float = 25.0,
) -> pd.DataFrame:
    """A forecast fan and realised prices shaped like a German day."""
    start = settings.market.local_midnight_utc(day)
    end = settings.market.local_midnight_utc(day + timedelta(days=1))
    index = pd.date_range(
        start, end, freq="15min", inclusive="left", name="timestamp_utc"
    )
    hours = np.arange(len(index)) / 4
    median = (
        70
        + 60 * np.sin((hours - 8) / 24 * 2 * np.pi)
        - 50 * np.exp(-(((hours - 13) / 2) ** 2))
    )
    frame = pd.DataFrame(index=index)
    for q in QUANTILES:
        frame[quantile_column(q)] = median + band_scale * Z_SCORES[q]
    rng = np.random.default_rng(seed)
    frame["actual"] = median + rng.normal(0, 25, len(index))
    frame["target_day"] = day
    frame["price_product_minutes"] = product_minutes
    return frame


def no_simultaneous(schedule: pd.DataFrame) -> bool:
    both = (schedule["charge_mw"] > 1e-9) & (schedule["discharge_mw"] > 1e-9)
    return not bool(both.any())


# --- battery ----------------------------------------------------------------


def test_battery_derives_efficiency_split_and_duration() -> None:
    battery = Battery(
        power_mw=1,
        capacity_mwh=2,
        round_trip_efficiency=0.81,
        degradation_eur_per_mwh=8,
    )
    assert battery.charge_efficiency == pytest.approx(0.9)
    assert battery.discharge_efficiency == pytest.approx(0.9)
    assert battery.duration_hours == 2
    assert battery.initial_soc_mwh == 1.0
    assert battery.max_cycles_per_day is None


@pytest.mark.parametrize(
    "override",
    [
        {"round_trip_efficiency": 1.2},
        {"initial_soc_fraction": 1.5},
        {"max_cycles_per_day": 0},
        {"power_mw": 0},
        {"unknown": 1},
    ],
)
def test_battery_rejects_invalid_parameters(override: dict[str, float]) -> None:
    params = {
        "power_mw": 1.0,
        "capacity_mwh": 2.0,
        "round_trip_efficiency": 0.9,
        "degradation_eur_per_mwh": 8.0,
    }
    with pytest.raises(ValidationError):
        Battery.model_validate(params | override)


def test_settings_battery_matches_the_documented_defaults(settings: Settings) -> None:
    battery = settings.battery
    assert (battery.power_mw, battery.capacity_mwh) == (1.0, 2.0)
    assert battery.round_trip_efficiency == 0.9
    assert battery.degradation_eur_per_mwh == 8.0
    assert battery.initial_soc_mwh == 1.0
    assert battery.max_cycles_per_day == 2.0
    assert settings.trading.dispatch_quantiles == (0.25, 0.10)


# --- optimizer: hand-checked cases ------------------------------------------


def test_toy_case_matches_hand_calculation() -> None:
    """1 MW / 0.45 MWh, 81% round trip (90% each way), €8 wear, starts empty.

    Prices 10, 10, 100, 101 over four quarter-hours. Charging 1 MW for 15 minutes
    stores 0.25 x 0.9 = 0.225 MWh, so two periods fill the 0.45 MWh. Each MWh
    bought at 10 sells 0.81 MWh for over 90 net of wear, so both periods charge.
    Selling starts with the dearer last period at full power, which drains
    0.25 / 0.9 = 0.2778 MWh; the rest, 0.1722 MWh, leaves at 0.62 MW in the
    period before. Value: 0.25 x (92 x 0.62 + 93 x 1 - 10 - 10) = 32.51.
    """
    battery = Battery(
        power_mw=1,
        capacity_mwh=0.45,
        round_trip_efficiency=0.81,
        degradation_eur_per_mwh=8,
        initial_soc_fraction=0,
    )
    result = optimize_dispatch(prices([10, 10, 100, 101]), battery)
    s = result.schedule
    np.testing.assert_allclose(s["charge_mw"], [1, 1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(s["discharge_mw"], [0, 0, 0.62, 1], atol=1e-6)
    np.testing.assert_allclose(
        s["soc_mwh"], [0.225, 0.45, 0.45 - 0.62 / 3.6, 0], atol=1e-6
    )
    np.testing.assert_allclose(s["net_mw"], s["discharge_mw"] - s["charge_mw"])
    assert result.objective_eur == pytest.approx(32.51, abs=1e-6)


@pytest.mark.parametrize(("offset", "trades"), [(-0.5, False), (0.5, True)])
def test_trades_only_beyond_the_breakeven(offset: float, trades: bool) -> None:
    """Selling pays only above buy / 0.90 + wear = 50 / 0.9 + 8.

    One hour at €50 then one hour at the sell price. Four full charging quarters
    store 0.9487 MWh and return 0.9 MWh, so the value is 0.9 x (sell - 63.56).
    """
    battery = Battery(
        power_mw=1,
        capacity_mwh=1,
        round_trip_efficiency=0.9,
        degradation_eur_per_mwh=8,
        initial_soc_fraction=0,
    )
    breakeven = 50 / 0.9 + 8
    result = optimize_dispatch(prices([50] * 4 + [breakeven + offset] * 4), battery)
    if trades:
        assert result.schedule["discharge_mw"].sum() * 0.25 == pytest.approx(0.9)
        assert result.objective_eur == pytest.approx(0.9 * offset, abs=1e-6)
    else:
        assert (result.schedule[["charge_mw", "discharge_mw"]] == 0).all().all()
        assert result.objective_eur == 0


def test_charges_when_paid_to_at_negative_prices() -> None:
    """M4: at -€30 the battery is paid to charge.

    Buying 1 MWh at -30 earns 30; it returns 0.9 MWh sold at €0 with €8 wear
    per MWh, so the day is worth 30 - 0.9 x 8 = 22.8.
    """
    battery = Battery(
        power_mw=1,
        capacity_mwh=1,
        round_trip_efficiency=0.9,
        degradation_eur_per_mwh=8,
        initial_soc_fraction=0,
    )
    day = prices([-30] * 4 + [0] * 4)
    result = optimize_dispatch(day, battery)
    np.testing.assert_allclose(result.schedule["charge_mw"].iloc[:4], 1, atol=1e-6)
    assert result.objective_eur == pytest.approx(22.8, abs=1e-6)
    settled = settle(result.schedule, day, battery)
    assert settled.revenue_eur == pytest.approx(30)
    assert settled.pnl_eur == pytest.approx(22.8, abs=1e-6)


def test_binary_blocks_the_energy_burning_loop() -> None:
    """At -€30 an LP charges 1 MW and discharges 0.9 MW at once.

    The state of charge does not move and the net purchase of 0.1 MW earns
    0.25 x 30 x 0.1 = 0.75 per quarter-hour, 3.0 over four. The MILP must
    alternate instead: two charging quarters store 0.4743 MWh and two
    discharging quarters return 0.9 MW each, worth 0.25 x 30 x (2 - 1.8) = 1.5.
    """
    battery = Battery(
        power_mw=1, capacity_mwh=2, round_trip_efficiency=0.9, degradation_eur_per_mwh=0
    )
    day = prices([-30] * 4)
    relaxed = optimize_dispatch(day, battery, integer=False)
    exact = optimize_dispatch(day, battery)
    assert relaxed.objective_eur == pytest.approx(3.0, abs=1e-6)
    assert not no_simultaneous(relaxed.schedule)
    assert exact.objective_eur == pytest.approx(1.5, abs=1e-6)
    assert no_simultaneous(exact.schedule)


# --- optimizer: structure ----------------------------------------------------


def test_hourly_products_share_one_schedule_per_hour(settings: Settings) -> None:
    frame = day_frame(settings, date(2025, 3, 12), product_minutes=60)
    # The model forecasts quarter-hours; a wobble inside each hour must not split it.
    frame["q50"] += np.tile([3.0, -2.0, 1.0, -2.0], len(frame) // 4)
    result = dispatch_day(frame, MEDIAN_FORECAST, settings.battery)
    hours = result.schedule.groupby(pd.DatetimeIndex(result.schedule.index).floor("h"))
    assert (hours["charge_mw"].nunique() == 1).all()
    assert (hours["discharge_mw"].nunique() == 1).all()
    assert result.settlement.discharged_mwh > 0


def test_product_blocks_follow_utc_hours_on_the_long_dst_day(
    settings: Settings,
) -> None:
    day = date(2024, 10, 27)
    index = pd.date_range(
        settings.market.local_midnight_utc(day),
        settings.market.local_midnight_utc(day + timedelta(days=1)),
        freq="15min",
        inclusive="left",
    )
    labels = product_blocks(index, np.full(len(index), 60))
    assert len(index) == 100
    assert len(np.unique(labels)) == 25
    assert (pd.Series(labels).value_counts() == 4).all()
    assert len(np.unique(product_blocks(index, np.full(len(index), 15)))) == 100


@pytest.mark.parametrize(
    ("day", "periods", "product_minutes"),
    [
        (date(2024, 10, 27), 100, 60),
        (date(2025, 3, 30), 92, 60),
        (date(2025, 10, 1), 96, 15),
        (date(2025, 10, 26), 100, 15),
        (date(2026, 3, 29), 92, 15),
    ],
)
def test_dst_and_switch_days_respect_every_physical_limit(
    settings: Settings, day: date, periods: int, product_minutes: int
) -> None:
    battery = settings.battery
    frame = day_frame(settings, day, seed=3, product_minutes=product_minutes)
    result = dispatch_day(frame, PERFECT_FORESIGHT, battery)
    s = result.schedule
    assert len(s) == periods
    if product_minutes == 60:
        hours = s.groupby(pd.DatetimeIndex(s.index).floor("h"))
        assert len(hours) == periods // 4
        assert (hours[["charge_mw", "discharge_mw"]].nunique() == 1).all().all()
    assert battery.max_cycles_per_day is not None
    assert result.settlement.cycles <= battery.max_cycles_per_day + 1e-6
    assert s["soc_mwh"].iloc[-1] == pytest.approx(battery.initial_soc_mwh, abs=1e-6)
    assert s["soc_mwh"].between(-1e-9, battery.capacity_mwh + 1e-9).all()
    assert s[["charge_mw", "discharge_mw"]].le(battery.power_mw + 1e-9).all().all()
    assert no_simultaneous(s)
    # State of charge follows the dynamics from the configured start.
    dt = 0.25
    expected = battery.initial_soc_mwh + np.cumsum(
        battery.charge_efficiency * dt * s["charge_mw"]
        - dt / battery.discharge_efficiency * s["discharge_mw"]
    )
    np.testing.assert_allclose(s["soc_mwh"], expected, atol=1e-6)


def test_cycle_cap_limits_throughput() -> None:
    swings = ([0.0] * 4 + [120.0] * 4) * 4
    base = {
        "power_mw": 1.0,
        "capacity_mwh": 1.0,
        "round_trip_efficiency": 0.9,
        "degradation_eur_per_mwh": 0.0,
        "initial_soc_fraction": 0.0,
    }
    free = Battery.model_validate(base)
    capped = Battery.model_validate(base | {"max_cycles_per_day": 1.0})
    day = prices(swings)
    free_cycles = settle(optimize_dispatch(day, free).schedule, day, free).cycles
    capped_cycles = settle(optimize_dispatch(day, capped).schedule, day, capped).cycles
    assert free_cycles > 1.5
    assert capped_cycles == pytest.approx(1.0, abs=1e-6)


def _dynamic_program(values: NDArray[np.float64], degradation: float) -> float:
    """Exact optimum for a lossless 1 MW / 1 MWh battery starting half full.

    Each quarter-hour moves the state of charge by at most 0.25 MWh, so four
    steps span the capacity. With no losses the linear program's vertices sit on
    whole steps, so this search over steps finds the true optimum.
    """
    best = {2: 0.0}
    for price in values:
        following: dict[int, float] = {}
        for level, value in best.items():
            for move, reward in (
                (1, -price * 0.25),
                (0, 0.0),
                (-1, (price - degradation) * 0.25),
            ):
                new = level + move
                if 0 <= new <= 4:
                    following[new] = max(following.get(new, -math.inf), value + reward)
        best = following
    return best[2]


@pytest.mark.parametrize("seed", range(12))
def test_optimizer_matches_an_exact_dynamic_program(seed: int) -> None:
    rng = np.random.default_rng(seed)
    values = rng.normal(40, 60, 16).round(2)
    degradation = float(rng.choice([0.0, 5.0]))
    battery = Battery(
        power_mw=1,
        capacity_mwh=1,
        round_trip_efficiency=1.0,
        degradation_eur_per_mwh=degradation,
    )
    result = optimize_dispatch(prices(list(values)), battery)
    assert result.objective_eur == pytest.approx(
        _dynamic_program(values, degradation), abs=1e-6
    )


def _scipy_optimum(
    sell: NDArray[np.float64], battery: Battery, products: NDArray[np.int64]
) -> float:
    """The same problem written as matrices and solved by HiGHS through SciPy.

    Columns: charge, discharge and state of charge per period, then the binary.
    """
    from scipy.optimize import Bounds, LinearConstraint, milp

    n, dt = len(sell), 0.25
    power, energy = battery.power_mw, battery.capacity_mwh
    eta_c, eta_d = battery.charge_efficiency, battery.discharge_efficiency
    charge, discharge, soc, mode = 0, n, 2 * n, 3 * n
    cost = np.zeros(4 * n)
    cost[charge:discharge] = dt * sell
    cost[discharge:soc] = -dt * (sell - battery.degradation_eur_per_mwh)
    rows: list[NDArray[np.float64]] = []
    lower: list[float] = []
    upper: list[float] = []

    def add(coefficients: dict[int, float], low: float, high: float) -> None:
        row = np.zeros(4 * n)
        for column, value in coefficients.items():
            row[column] = value
        rows.append(row)
        lower.append(low)
        upper.append(high)

    start = battery.initial_soc_mwh
    for t in range(n):
        add({charge + t: 1.0, mode + t: -power}, -np.inf, 0.0)
        add({discharge + t: 1.0, mode + t: power}, -np.inf, power)
        balance = {soc + t: 1.0, charge + t: -eta_c * dt, discharge + t: dt / eta_d}
        if t > 0:
            balance[soc + t - 1] = -1.0
        add(balance, start if t == 0 else 0.0, start if t == 0 else 0.0)
        if t > 0 and products[t] == products[t - 1]:
            add({charge + t: 1.0, charge + t - 1: -1.0}, 0.0, 0.0)
            add({discharge + t: 1.0, discharge + t - 1: -1.0}, 0.0, 0.0)
    add({soc + n - 1: 1.0}, start, start)
    if battery.max_cycles_per_day is not None:
        add(
            {discharge + t: dt / eta_d for t in range(n)},
            -np.inf,
            battery.max_cycles_per_day * energy,
        )
    result = milp(
        cost,
        constraints=LinearConstraint(np.vstack(rows), lower, upper),
        integrality=np.concatenate([np.zeros(3 * n), np.ones(n)]),
        bounds=Bounds(
            np.zeros(4 * n),
            np.concatenate([np.full(2 * n, power), np.full(n, energy), np.ones(n)]),
        ),
        options={"mip_rel_gap": 0.0},
    )
    assert result.success
    return float(-result.fun)


@pytest.mark.parametrize("seed", range(8))
def test_optimizer_matches_an_independent_scipy_formulation(seed: int) -> None:
    """Losses, wear, a cycle cap and, on odd seeds, hourly products."""
    rng = np.random.default_rng(100 + seed)
    values = rng.normal(50, 80, 32).round(2)
    battery = Battery(
        power_mw=1.0,
        capacity_mwh=float(rng.choice([0.5, 1.0, 2.0])),
        round_trip_efficiency=0.88,
        degradation_eur_per_mwh=8.0,
        initial_soc_fraction=float(rng.choice([0.0, 0.5])),
        max_cycles_per_day=float(rng.choice([1.0, 2.0])),
    )
    step = 4 if seed % 2 else 1
    products = (np.arange(32) // step).astype(np.int64)
    ours = optimize_dispatch(prices(list(values)), battery, products=products)
    expected = _scipy_optimum(values, battery, products)
    assert ours.objective_eur == pytest.approx(expected, abs=1e-3)


def test_optimizer_rejects_bad_inputs() -> None:
    battery = Battery(
        power_mw=1, capacity_mwh=1, round_trip_efficiency=0.9, degradation_eur_per_mwh=0
    )
    with pytest.raises(ValueError, match="finite"):
        optimize_dispatch(prices([1.0, float("nan")]), battery)
    with pytest.raises(ValueError, match="same index"):
        optimize_dispatch(
            prices([1.0, 2.0]), battery, buy_prices=prices([1.0, 2.0], "2025-01-01")
        )
    with pytest.raises(ValueError, match="contiguous"):
        optimize_dispatch(
            prices([1.0, 2.0, 3.0]), battery, products=np.array([1, 2, 1])
        )
    gap = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(
            ["2025-01-01 00:00", "2025-01-01 00:15", "2025-01-01 00:45"], tz="UTC"
        ),
    )
    with pytest.raises(ValueError, match="evenly spaced"):
        optimize_dispatch(gap, battery)
    assert period_hours(pd.DatetimeIndex(prices([1.0, 2.0]).index)) == 0.25


# --- settlement ---------------------------------------------------------------


def test_settlement_arithmetic() -> None:
    """Charge 1 MW at €20, discharge 0.5 MW at €80, one quarter-hour each.

    Revenue 0.25 x (-20 + 40) = 5; 0.125 MWh discharged costs 1.25 in wear;
    cycles 0.125 / 0.9 / 2 MWh.
    """
    battery = Battery(
        power_mw=1,
        capacity_mwh=2,
        round_trip_efficiency=0.81,
        degradation_eur_per_mwh=10,
    )
    day = prices([20, 80])
    schedule = pd.DataFrame(
        {"charge_mw": [1.0, 0.0], "discharge_mw": [0.0, 0.5]}, index=day.index
    )
    result = settle(schedule, day, battery)
    assert result.revenue_eur == pytest.approx(5.0)
    assert result.degradation_eur == pytest.approx(1.25)
    assert result.pnl_eur == pytest.approx(3.75)
    assert result.charged_mwh == pytest.approx(0.25)
    assert result.discharged_mwh == pytest.approx(0.125)
    assert result.cycles == pytest.approx(0.125 / 0.9 / 2)


# --- strategies -------------------------------------------------------------


def test_strategy_definitions(settings: Settings) -> None:
    assert quantile_aware(0.25) == Strategy("quantile_q25", "q25", "q75")
    assert quantile_aware(0.10) == Strategy("quantile_q10", "q10", "q90")
    for level in (0.0, 0.5, 0.7):
        with pytest.raises(ValueError):
            quantile_aware(level)
    names = [s.name for s in build_strategies(settings)]
    assert names == [
        "perfect_foresight",
        "median_forecast",
        "quantile_q25",
        "quantile_q10",
    ]
    assert PERFECT_FORESIGHT.uses_realised_prices
    assert not any(s.uses_realised_prices for s in build_strategies(settings)[1:])


def test_forecast_strategies_never_read_realised_prices(settings: Settings) -> None:
    frame = day_frame(settings, date(2025, 11, 20), seed=5)
    corrupted = frame.assign(actual=frame["actual"] * -3 + 500)
    columns = ["charge_mw", "discharge_mw", "soc_mwh"]
    for strategy in build_strategies(settings)[1:]:
        honest = dispatch_day(frame, strategy, settings.battery)
        leaked = dispatch_day(corrupted, strategy, settings.battery)
        pd.testing.assert_frame_equal(
            honest.schedule[columns], leaked.schedule[columns]
        )
        assert honest.planned_value_eur == pytest.approx(leaked.planned_value_eur)
    honest_pf = dispatch_day(frame, PERFECT_FORESIGHT, settings.battery)
    leaked_pf = dispatch_day(corrupted, PERFECT_FORESIGHT, settings.battery)
    assert not honest_pf.schedule[columns].equals(leaked_pf.schedule[columns])


def test_quantile_aware_holds_back_when_the_fan_is_wide(settings: Settings) -> None:
    """With a wide fan, q25 selling never clears q75 buying plus losses."""
    frame = day_frame(settings, date(2025, 11, 20), band_scale=100.0)
    median = dispatch_day(frame, MEDIAN_FORECAST, settings.battery)
    cautious = dispatch_day(frame, quantile_aware(0.25), settings.battery)
    assert median.settlement.discharged_mwh > 0.5
    assert cautious.settlement.discharged_mwh == 0
    assert cautious.settlement.pnl_eur == 0


@pytest.mark.parametrize("seed", range(4))
def test_perfect_foresight_bounds_every_strategy(settings: Settings, seed: int) -> None:
    frame = day_frame(settings, date(2025, 12, 1) + timedelta(days=seed), seed=seed)
    results = [
        dispatch_day(frame, s, settings.battery) for s in build_strategies(settings)
    ]
    ceiling = results[0].settlement.pnl_eur
    assert ceiling == pytest.approx(results[0].planned_value_eur, abs=1e-6)
    for result in results[1:]:
        assert result.settlement.pnl_eur <= ceiling + 1e-6


def test_dispatch_day_rejects_mixed_frames(settings: Settings) -> None:
    one = day_frame(settings, date(2025, 11, 20))
    two = day_frame(settings, date(2025, 11, 21))
    with pytest.raises(ValueError, match="exactly one target day"):
        dispatch_day(pd.concat([one, two]), MEDIAN_FORECAST, settings.battery)
    with pytest.raises(ValueError, match="lacks columns"):
        dispatch_day(one.drop(columns="q50"), MEDIAN_FORECAST, settings.battery)


# --- runner -----------------------------------------------------------------


def _forecasts(settings: Settings, days: list[date]) -> pd.DataFrame:
    return pd.concat([day_frame(settings, day, seed=i) for i, day in enumerate(days)])


def test_select_days_refuses_the_holdout_and_explains_skips(settings: Settings) -> None:
    columns = {"actual", "q50", "q25", "q75", "q10", "q90", "price_product_minutes"}
    first = date(2025, 11, 1)
    day = [first + timedelta(days=i) for i in range(5)]
    frames = _forecasts(settings, [day[0], day[1], day[2], day[4]])
    frames = frames.drop(frames[frames["target_day"] == day[1]].index[5])
    frames.loc[frames["target_day"] == day[2], "actual"] = np.nan
    mixed = frames[frames["target_day"] == day[4]].index[:8]
    frames.loc[mixed, "price_product_minutes"] = 60
    days, skipped = select_days(frames, day[0], day[4], settings, columns)
    assert days == [day[0]]
    assert skipped == {
        day[1]: "incomplete periods",
        day[2]: "missing price or forecast",
        day[3]: "no forecast",
        day[4]: "mixed product lengths",
    }
    holdout = settings.evaluation.holdout_start
    with pytest.raises(HoldoutAccessError):
        select_days(frames, first, holdout, settings, columns)
    warm_up = settings.evaluation.validation_start - timedelta(days=1)
    with pytest.raises(ValueError, match="validation window"):
        select_days(frames, warm_up, first, settings, columns)


def test_run_strategies_end_to_end(settings: Settings) -> None:
    days = [date(2025, 10, 26), date(2025, 11, 20)]
    strategies = build_strategies(settings)
    holdout = settings.evaluation.holdout_start
    forecasts = _forecasts(settings, days)
    dispatch, pnl, failed = run_strategies(
        forecasts, days, strategies, settings.battery, holdout_start=holdout
    )
    assert failed == {}
    assert len(pnl) == len(days) * len(strategies)
    assert len(dispatch) == (100 + 96) * len(strategies)
    assert {"target_day", "strategy", "net_mw", "soc_mwh", "realised_price"} <= set(
        dispatch.columns
    )
    names = [strategy.name for strategy in strategies]
    summary = summarise(pnl, names)
    assert list(summary.index) == names
    assert summary.loc["perfect_foresight", "capture_ratio"] == pytest.approx(1.0)
    assert summary.loc["median_forecast", "days"] == 2
    with pytest.raises(HoldoutAccessError):
        run_strategies(
            forecasts, [holdout], strategies, settings.battery, holdout_start=holdout
        )


def test_run_strategies_records_a_failed_solve(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = [date(2025, 11, 20), date(2025, 11, 21)]
    real = dispatch_day

    def flaky(
        frame: pd.DataFrame,
        strategy: Strategy,
        battery: Battery,
        *,
        time_limit_s: float,
    ) -> DayResult:
        if frame["target_day"].iloc[0] == days[1]:
            raise DispatchError("no proven optimum")
        return real(frame, strategy, battery, time_limit_s=time_limit_s)

    monkeypatch.setattr(runner_module, "dispatch_day", flaky)
    _, pnl, failed = run_strategies(
        _forecasts(settings, days),
        days,
        build_strategies(settings),
        settings.battery,
        holdout_start=settings.evaluation.holdout_start,
    )
    assert set(pnl["target_day"]) == {days[0]}
    assert failed == {days[1]: "solver failed: no proven optimum"}


def test_check_ceiling_catches_a_strategy_above_perfect_foresight() -> None:
    pnl = pd.DataFrame(
        {
            "target_day": [date(2025, 1, 1)] * 2,
            "strategy": ["perfect_foresight", "median_forecast"],
            "pnl_eur": [100.0, 101.0],
        }
    )
    with pytest.raises(RuntimeError, match="beat perfect foresight"):
        check_ceiling(pnl)


# --- config -----------------------------------------------------------------


@pytest.mark.parametrize("levels", [[0.3], [0.6], [0.25, 0.25], []])
def test_dispatch_quantiles_must_mirror_forecast_quantiles(levels: list[float]) -> None:
    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["trading"]["dispatch_quantiles"] = levels
    with pytest.raises(ValidationError):
        Settings.model_validate(raw)
