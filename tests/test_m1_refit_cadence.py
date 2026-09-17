"""Weekly against 28-day refits: the pieces that decide the comparison."""

from __future__ import annotations

import math
from datetime import date, timedelta

import pandas as pd
import pytest

from src.health.experiments import m1_refit_cadence as rc
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT

START = date(2024, 6, 1)


def _days(n: int) -> list[date]:
    return [START + timedelta(days=i) for i in range(n)]


def test_the_control_arm_is_the_phase_2_cadence() -> None:
    assert rc.CADENCES == {rc.CONTROL: 28, rc.CANDIDATE: 7}


def test_refit_days_follow_the_walk_forward_rule() -> None:
    days = _days(30)

    assert rc.refit_days(days, 7) == [days[0], days[7], days[14], days[21], days[28]]
    assert rc.refit_days(days, 28) == [days[0], days[28]]


def test_a_gap_in_the_days_counts_as_elapsed_time() -> None:
    days = [START, START + timedelta(days=1), START + timedelta(days=19)]

    assert rc.refit_days(days, 7) == [START, START + timedelta(days=19)]


def test_pairs_are_only_the_days_both_arms_have() -> None:
    days = _days(20)
    control = pd.DataFrame({"target_day": days, "pinball": 5.0})
    # The candidate lacks the last three days and has one the control lacks.
    candidate = pd.DataFrame(
        {"target_day": [*days[:-3], date(2025, 1, 1)], "pinball": 4.5}
    )

    result = rc.paired_difference(control, candidate, "pinball")

    assert result["days"] == 17
    assert result["mean"] == pytest.approx(-0.5)
    assert result["low"] == pytest.approx(-0.5)
    assert result["high"] == pytest.approx(-0.5)


def test_pairs_follow_the_day_not_the_row_order() -> None:
    days = _days(10)
    control = pd.DataFrame(
        {"target_day": days, "pinball": [float(i) for i in range(10)]}
    )
    candidate = control.assign(pinball=control["pinball"] - 1.0).sample(
        frac=1.0, random_state=3
    )

    result = rc.paired_difference(control, candidate, "pinball")

    # Paired by day, every difference is exactly -1, so the interval collapses.
    # Paired by row position after the shuffle, it would not.
    assert result["low"] == pytest.approx(-1.0)
    assert result["high"] == pytest.approx(-1.0)


def test_the_reading_follows_the_interval_and_the_direction() -> None:
    assert rc.verdict({"low": -0.3, "high": -0.1}, better="lower") == "better"
    assert rc.verdict({"low": 0.1, "high": 0.3}, better="lower") == "worse"
    assert rc.verdict({"low": -2.0, "high": 9.0}, better="higher") == "within noise"
    assert rc.verdict({"low": -9.0, "high": -1.0}, better="higher") == "worse"
    with pytest.raises(ValueError, match="better"):
        rc.verdict({"low": 0.0, "high": 1.0}, better="closer")


def test_daily_profit_separates_median_dispatch_from_the_ceiling() -> None:
    pnl = pd.DataFrame(
        {
            "target_day": [
                START,
                START,
                START + timedelta(days=1),
                START + timedelta(days=1),
            ],
            "strategy": [
                MEDIAN_FORECAST.name,
                PERFECT_FORESIGHT.name,
                MEDIAN_FORECAST.name,
                PERFECT_FORESIGHT.name,
            ],
            "pnl_eur": [80.0, 100.0, 90.0, 120.0],
        }
    )

    profit = rc.daily_profit(pnl)

    assert profit["target_day"].tolist() == [START, START + timedelta(days=1)]
    assert profit["pnl_eur"].tolist() == [80.0, 90.0]
    assert profit["perfect_foresight_pnl_eur"].tolist() == [100.0, 120.0]


def test_the_results_page_states_the_reproduction_and_both_readings() -> None:
    table = pd.DataFrame(
        {
            "refit_every_days": [28, 7],
            "fits": [27, 107],
            "mean_pinball": [5.11, 5.05],
            "coverage_90": [0.856, 0.861],
            "coverage_50": [0.49, 0.50],
            "mae_median": [16.03, 15.9],
            "capture": [0.901, 0.905],
            "pnl_eur": [149_498.0, 150_100.0],
            "run_seconds": [600.0, math.nan],
        },
        index=pd.Index([rc.CONTROL, rc.CANDIDATE], name="arm"),
    )
    comparison = {
        "pinball": {"mean": -0.06, "low": -0.09, "high": -0.03, "days": 730.0},
        "coverage_90": {"mean": 0.005, "low": -0.002, "high": 0.012, "days": 730.0},
        "pnl_eur": {"mean": 0.8, "low": -1.5, "high": 3.1, "days": 730.0},
    }

    page = rc.results_markdown(
        table, comparison, days=_days(760), scored=_days(730), traded=730, gap=0.0
    )

    assert "reproduces the saved comparison forecasts exactly" in page
    assert "| every_7_days | 7 | 107 |" in page and "reused |" in page
    assert page.count("| better |") == 1 and "within noise |" in page
    assert "for 4.0 times the fits" in page
    assert chr(0x2014) not in page and chr(0x2013) not in page
