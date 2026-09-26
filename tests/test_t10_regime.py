"""The regime test: the exact solve, the windows, and the verdicts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.health.experiments import t10_regime as rg
from src.trading.dispatch_lp import DayProgram
from src.trading.optimizer import product_blocks


def _day(settings: Settings, prices: np.ndarray) -> tuple[Any, ...]:
    index = pd.date_range("2026-01-01", periods=len(prices), freq="h", tz="UTC")
    minutes = pd.Series(np.full(len(prices), 60), index=index)
    products = product_blocks(index, minutes)
    program = DayProgram.build(len(prices), 1.0, settings.battery, products)
    return program, index, products


def test_an_ordinary_day_never_needs_the_milp(settings: Settings) -> None:
    """With positive prices the relaxation is tight, so the fast path is taken."""
    prices = np.array([20.0] * 6 + [40.0] * 6 + [30.0] * 6 + [150.0] * 6)
    program, index, products = _day(settings, prices)

    net, resolved = rg.solve_exactly(program, prices, index, products, settings.battery)

    assert resolved is False
    assert net.sum() < 0  # round-trip losses: more bought than sold
    assert np.isfinite(net).all()


def test_a_deeply_negative_price_forces_the_exact_solve(settings: Settings) -> None:
    """Being paid to charge makes the relaxation run both flows at once."""
    prices = np.array([-500.0] * 8 + [50.0] * 8 + [200.0] * 8)
    program, index, products = _day(settings, prices)

    net, resolved = rg.solve_exactly(program, prices, index, products, settings.battery)

    assert resolved is True
    # The production MILP never runs both flows, so every period is one or other.
    assert np.isfinite(net).all() and len(net) == 24


def test_both_solvers_reach_the_same_value_where_the_relaxation_is_tight(
    settings: Settings,
) -> None:
    """Not the same schedule: a degenerate day has several equally good ones."""
    from src.trading.dispatch_lp import solve_day
    from src.trading.optimizer import optimize_dispatch

    prices = np.array([10.0] * 8 + [60.0] * 8 + [180.0] * 8)
    program, index, products = _day(settings, prices)

    _, relaxed_value = solve_day(program, prices)
    exact = optimize_dispatch(
        pd.Series(prices, index=index), settings.battery, products=products
    )

    assert relaxed_value == pytest.approx(exact.objective_eur, abs=1e-6)


def test_the_windows_are_two_full_years_that_do_not_overlap() -> None:
    assert rg.WINDOWS["crisis"][1] < rg.WINDOWS["recent"][0]
    for first, last in (rg.WINDOWS["crisis"], rg.WINDOWS["recent"]):
        assert (last - first).days + 1 == 730


def test_the_page_reports_each_verdict_and_says_why_the_product_split_is_missing() -> (
    None
):
    def arms(capture: float) -> dict[str, Any]:
        return {
            arm: {
                "pnl_eur": 100000.0 * capture,
                "capture": capture,
                "losing_days": 10,
                "cycles_per_day": 1.6,
            }
            for arm in rg.ARMS
        }

    def profile(capture: float, ordinary: float) -> dict[str, Any]:
        return {
            "days": 730,
            "first_day": "2021-06-01",
            "last_day": "2023-05-31",
            "mean_price_spread_eur": 143.0,
            "ordinary_day_share": ordinary,
            "mean_shape_agreement": 0.79,
            "days_resolved_with_the_milp": dict.fromkeys(rg.ARMS, 2),
            "arms": arms(capture),
        }

    summary: dict[str, Any] = {
        "windows": {"crisis": profile(0.845, 0.315), "recent": profile(0.849, 0.410)},
        "seasons": {
            name: {"Jun-Sep": profile(0.87, 0.5), "Oct-May": profile(0.83, 0.3)}
            for name in ("crisis", "recent")
        },
        "product_overlap": {"quarter_hourly_days": 243, "of_which_summer": 0},
        "ceiling_drift_eur": 0.0,
        "capture_drop_points": 0.004,
        "ordinary_threshold": rg.ORDINARY,
        "yesterday_loses_in_recent": True,
        "verdicts": {
            "R1_capture_falls_5_points": False,
            "R2_yesterday_overtakes_the_rule": False,
            "R3_fewer_ordinary_days": True,
        },
    }
    yearly = pd.DataFrame(
        {"days": [365], "mean_agreement": [0.796], "ordinary_share": [0.312]},
        index=pd.Index([2021], name="year"),
    )

    page = rg.results_markdown(summary, yearly)

    assert page.count("HOLDS") == 1 and page.count("fails") == 2
    assert "+0.4 points" in page
    assert "0 fall in June to September" in page
    assert "not available" in page


def test_a_window_profile_divides_every_arm_by_that_windows_own_ceiling() -> None:
    days = pd.date_range("2024-06-01", periods=4, freq="D").date
    frame = pd.DataFrame(
        {
            "shape_agreement": [0.95, 0.80, 0.99, 0.50],
            "spread": [100.0] * 4,
            "perfect_foresight": [100.0] * 4,
            "fixed_shape_seasonal": [80.0, 90.0, 70.0, 60.0],
            "fixed_shape": [50.0] * 4,
            "yesterday": [-10.0, 40.0, 40.0, 40.0],
            **{f"{arm}_cycles": [1.5] * 4 for arm in rg.ARMS},
            **{f"{arm}_milp": [False] * 4 for arm in rg.ARMS},
        },
        index=pd.Index(days, name="target_day"),
    )

    profile = rg._profile(frame)

    assert profile["arms"]["fixed_shape_seasonal"]["capture"] == pytest.approx(0.75)
    assert profile["arms"]["yesterday"]["losing_days"] == 1
    # Two of the four days clear T9's 0.90 threshold.
    assert profile["ordinary_day_share"] == pytest.approx(0.5)
