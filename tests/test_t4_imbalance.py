"""T4 imbalance experiment: outage physics and reBAP settlement on synthetic days."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.health.experiments.t4_imbalance import (
    evaluate_day,
    outage_windows,
    random_start,
    run_experiment,
    simulate_delivery,
    value_soc_gap,
    worst_start,
)
from src.trading.battery import Battery


def _battery(initial_soc_fraction: float = 0.5, capacity_mwh: float = 2.0) -> Battery:
    # Round trip 0.81 keeps the numbers exact: 0.9 each way.
    return Battery(
        power_mw=1,
        capacity_mwh=capacity_mwh,
        round_trip_efficiency=0.81,
        degradation_eur_per_mwh=8,
        initial_soc_fraction=initial_soc_fraction,
    )


# Hourly periods (dt = 1 h) so the hand calculations stay readable.
NET = np.array([-1.0, 0.0, 0.81, 0.0])
PRICE = np.array([20.0, 30.0, 120.0, 50.0])
REBAP = np.array([30.0, 40.0, 200.0, 60.0])


def test_no_outage_gives_zero_deviation_and_cost() -> None:
    battery = _battery()
    outcome = evaluate_day(
        NET, PRICE, REBAP, np.zeros((1, 4), dtype=bool), battery, 1.0, 70.0
    )
    assert np.abs(outcome.deviation_mw).max() == pytest.approx(0.0, abs=1e-12)
    assert outcome.cost_eur[0] == pytest.approx(0.0, abs=1e-9)
    assert outcome.cost_zero_soc_eur[0] == pytest.approx(0.0, abs=1e-9)
    assert outcome.soc_end_mwh[0] == pytest.approx(1.0)
    assert outcome.undelivered_mwh[0] == pytest.approx(0.0, abs=1e-12)


def test_outage_on_discharge_block_pays_rebap_on_the_shortfall() -> None:
    """Hand-computed day, 1 MW / 2 MWh, 0.9 each way, wear 8, start 1 MWh.

    Committed: charge 1 MW at 20 (cells 1.0 -> 1.9), idle, discharge 0.81 MW at
    120 (cells lose 0.81 / 0.9 = 0.9 -> 1.0), idle.
    Committed P&L = -1 * 20 + 0.81 * 120 - 8 * 0.81 = -20 + 97.2 - 6.48 = 70.72.

    Outage in period 2 only: the discharge is missed, deviation -0.81 MW.
    Imbalance cash = 1 h * (-0.81) * 200 = -162. Wear on actual discharge = 0.
    Day-ahead revenue stays 77.2, so P&L with the gap at zero = -84.8 and
    cost at zero = 70.72 + 84.8 = 155.52.
    The cells end at 1.9 MWh, 0.9 above plan. Sold at the next reBAP of 70:
    0.9 * 0.9 * (70 - 8) = 0.81 * 62 = 50.22. P&L = -34.58, cost = 105.30.
    """
    battery = _battery()
    mask = outage_windows(4, 1, [2])
    outcome = evaluate_day(NET, PRICE, REBAP, mask, battery, 1.0, 70.0)
    assert outcome.committed_pnl_eur == pytest.approx(70.72)
    assert outcome.day_ahead_revenue_eur == pytest.approx(77.2)
    np.testing.assert_allclose(outcome.deviation_mw[0], [0.0, 0.0, -0.81, 0.0])
    assert outcome.imbalance_cash_eur[0] == pytest.approx(-162.0)
    assert outcome.wear_eur[0] == pytest.approx(0.0)
    assert outcome.pnl_zero_soc_eur[0] == pytest.approx(-84.8)
    assert outcome.cost_zero_soc_eur[0] == pytest.approx(155.52)
    assert outcome.soc_end_mwh[0] == pytest.approx(1.9)
    assert outcome.soc_value_eur[0] == pytest.approx(50.22)
    assert outcome.pnl_eur[0] == pytest.approx(-34.58)
    assert outcome.cost_eur[0] == pytest.approx(105.30)
    assert outcome.undelivered_mwh[0] == pytest.approx(0.81)


def test_missed_charge_clips_the_later_discharge() -> None:
    """Start 0.45 MWh. Committed: charge 1 MW (cells -> 1.35), discharge 0.9 MW.

    The discharge needs 0.9 / 0.9 = 1.0 MWh from the cells. With the charge lost
    to an outage only 0.45 MWh is stored, which delivers 0.45 * 0.9 = 0.405 MW.
    Deviations: +1.0 MW (the charge not bought) and 0.405 - 0.9 = -0.495 MW.
    With reBAP 10 then 100: cash = 1.0 * 10 - 0.495 * 100 = -39.5.
    The cells end empty, 0.45 below plan (the plan ends at 0.35, the start is
    0.45, so the gap to the start is -0.45 MWh).
    """
    battery = _battery(initial_soc_fraction=0.225)
    net = np.array([-1.0, 0.9])
    delivery = simulate_delivery(net, outage_windows(2, 1, [0]), battery, 1.0)
    np.testing.assert_allclose(delivery.actual_net_mw[0], [0.0, 0.405])
    np.testing.assert_allclose(delivery.discharge_mw[0], [0.0, 0.405])
    assert delivery.soc_mwh[0, -1] == pytest.approx(0.0)

    outcome = evaluate_day(
        net,
        np.array([5.0, 90.0]),
        np.array([10.0, 100.0]),
        outage_windows(2, 1, [0]),
        battery,
        1.0,
        0.0,
    )
    np.testing.assert_allclose(outcome.deviation_mw[0], [1.0, -0.495])
    assert outcome.imbalance_cash_eur[0] == pytest.approx(-39.5)
    assert outcome.wear_eur[0] == pytest.approx(8 * 0.405)


def test_committed_charge_that_would_overfill_is_clipped() -> None:
    # Full battery after a missed discharge: the later charge only fills the room.
    battery = _battery(initial_soc_fraction=1.0)
    net = np.array([0.9, -1.0])
    delivery = simulate_delivery(net, outage_windows(2, 1, [0]), battery, 1.0)
    np.testing.assert_allclose(delivery.actual_net_mw[0], [0.0, 0.0])
    assert delivery.soc_mwh[0, -1] == pytest.approx(2.0)


def test_negative_rebap_pays_the_shortfall() -> None:
    battery = _battery()
    rebap = REBAP.copy()
    rebap[2] = -100.0
    outcome = evaluate_day(
        NET, PRICE, rebap, outage_windows(4, 1, [2]), battery, 1.0, -100.0
    )
    # Shortfall of 0.81 MWh at -100: the battery is paid 81.
    assert outcome.imbalance_cash_eur[0] == pytest.approx(81.0)
    # The 0.9 MWh surplus is kept rather than dumped at -100: worth zero.
    assert outcome.soc_value_eur[0] == pytest.approx(0.0)


def test_soc_gap_value_signs() -> None:
    battery = _battery()
    # 0.9 MWh surplus sold at 70: 0.9 * 0.9 * (70 - 8) = 50.22.
    assert value_soc_gap([1.9], battery, 70.0)[0] == pytest.approx(50.22)
    # 0.45 MWh missing bought at 100: 0.45 / 0.9 * 100 = 50 paid.
    assert value_soc_gap([0.55], battery, 100.0)[0] == pytest.approx(-50.0)
    # The same energy bought at -100 earns 50.
    assert value_soc_gap([0.55], battery, -100.0)[0] == pytest.approx(50.0)


def test_worst_window_search_finds_the_known_worst_start() -> None:
    # Discharge 0.45 MW in periods 5 and 6 at a reBAP of 1000; nothing else.
    battery = _battery()
    net = np.zeros(8)
    net[[5, 6]] = 0.45
    price = np.full(8, 50.0)
    rebap = np.zeros(8)
    rebap[[5, 6]] = 1000.0
    assert worst_start(net, price, rebap, battery, 1.0, 0.0, 2) == 5


def test_random_start_is_reproducible_and_inside_the_day() -> None:
    day = date(2025, 3, 4)
    starts = [random_start(day, 96, 8, 7) for _ in range(3)]
    assert len(set(starts)) == 1
    assert 0 <= starts[0] <= 88
    other_days = {random_start(date(2025, 3, d), 96, 8, 7) for d in range(1, 29)}
    assert len(other_days) > 1
    with pytest.raises(ValueError):
        outage_windows(96, 8, [89])


def _dispatch(days: int) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.date_range("2025-03-01 23:00", periods=96 * days, freq="15min", tz="UTC")
    rng = np.random.default_rng(3)
    day_net = np.zeros(96)
    day_net[8:12] = -1.0
    day_net[70:74] = 0.81
    frames = []
    for strategy in ("median_forecast", "perfect_foresight"):
        frames.append(
            pd.DataFrame(
                {
                    "target_day": [
                        date(2025, 3, 2) + timedelta(days=i // 96)
                        for i in range(96 * days)
                    ],
                    "strategy": strategy,
                    "net_mw": np.tile(day_net, days),
                    "realised_price": rng.normal(80, 30, 96 * days),
                },
                index=index,
            )
        )
    frame = pd.concat(frames)
    rebap = pd.Series(rng.normal(90, 60, 96 * days), index=index)
    return frame, rebap


def test_run_experiment_is_deterministic_and_complete() -> None:
    dispatch, rebap = _dispatch(3)
    battery = _battery()
    first, checks = run_experiment(dispatch, rebap, battery, seed=11)
    second, _ = run_experiment(dispatch, rebap, battery, seed=11)
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 3 * 2 * 3
    assert checks["no_outage_max_cost_eur"] < 1e-9
    random_rows = first[first["scenario"] == "random_2h"]
    # Every strategy meets the same outage on the same day.
    assert random_rows.groupby("target_day")["outage_start_utc"].nunique().max() == 1
    worst = first[first["scenario"] == "worst_2h"].set_index(["target_day", "strategy"])
    rand = random_rows.set_index(["target_day", "strategy"])
    assert (worst["cost_eur"] >= rand["cost_eur"] - 1e-9).all()
