"""Scenario dispatch: the linearity property, the copula, and what a hedge costs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import Settings
from src.health.experiments import t7_scenario_dispatch as t7
from src.trading.dispatch_lp import DayProgram, solve_day
from src.trading.optimizer import period_hours

Floats = NDArray[np.float64]
LEVELS = np.array([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])


def _fan(n: int = 8, centre: float = 100.0, width: float = 40.0) -> Floats:
    """A symmetric quantile fan, the same shape for every period."""
    offsets = (LEVELS - 0.5) * 2 * width
    return np.tile(centre + offsets, (n, 1))


def test_a_price_at_the_median_ranks_in_the_middle() -> None:
    quantiles = _fan()
    actual = quantiles[:, 3].copy()  # q50 for every period

    ranks = t7.day_ranks(quantiles, LEVELS, actual)

    assert np.allclose(ranks, 0.5)


def test_prices_outside_the_fan_pin_at_its_edges() -> None:
    """Flat tails: the copula cannot see how far outside a price landed."""
    quantiles = _fan()
    below = np.full(8, -500.0)
    above = np.full(8, 5000.0)

    assert np.allclose(t7.day_ranks(quantiles, LEVELS, below), LEVELS[0])
    assert np.allclose(t7.day_ranks(quantiles, LEVELS, above), LEVELS[-1])


def test_ranks_read_back_through_their_own_day_return_the_prices() -> None:
    """The round trip that makes the empirical copula meaningful."""
    quantiles = _fan()
    actual = np.array([70.0, 100.0, 130.0, 95.0, 118.0, 88.0, 105.0, 112.0])

    ranks = t7.day_ranks(quantiles, LEVELS, actual)
    back = t7.scenario_prices(quantiles, LEVELS, ranks.reshape(1, -1))

    assert np.allclose(back[0], actual, atol=1e-9)


def _program(settings: Settings, n: int = 8) -> DayProgram:
    return DayProgram.build(n, 0.25, settings.battery)


def test_risk_neutral_scenario_dispatch_is_dispatch_at_the_scenario_mean(
    settings: Settings,
) -> None:
    """The plan's central argument, and the reason the arms are risk-sensitive.

    Profit is linear in price and no constraint depends on price, so the
    expected profit of a fixed schedule is its profit at the mean price. A
    thousand coherent paths cannot move the risk-neutral schedule.
    """
    rng = np.random.default_rng(3)
    program = _program(settings)
    prices = rng.normal(100.0, 45.0, size=(200, 8))

    scenario_net = t7.solve_cvar(program, prices, 0.0)
    mean_net, _ = solve_day(program, prices.mean(axis=0))

    assert np.allclose(scenario_net, mean_net, atol=1e-6)


def test_a_hedge_gives_up_expected_value_to_lift_the_worst_scenarios(
    settings: Settings,
) -> None:
    """Which is why it can only win if the forecast's own mean is wrong."""
    rng = np.random.default_rng(11)
    program = _program(settings)
    prices = rng.normal(100.0, 60.0, size=(200, 8))

    def outcomes(net: Floats) -> Floats:
        return np.array(
            [t7.settle(net, p, settings.battery, program.dt) for p in prices]
        )

    neutral = outcomes(t7.solve_cvar(program, prices, 0.0))
    hedged = outcomes(t7.solve_cvar(program, prices, 0.5))
    tail = int(t7.CVAR_ALPHA * len(prices))

    assert hedged.mean() <= neutral.mean() + 1e-6
    assert np.sort(hedged)[:tail].mean() >= np.sort(neutral)[:tail].mean() - 1e-6


def test_the_page_states_the_verdict_from_the_intervals() -> None:
    def interval(mean: float, low: float, high: float) -> dict[str, float]:
        return {
            "mean": mean,
            "low": low,
            "high": high,
            "days": 526,
            "total": 526 * mean,
        }

    summary: dict[str, Any] = {
        "days": 526,
        "first_day": "2024-12-19",
        "last_day": "2026-05-31",
        "scenarios": 200,
        "cvar_alpha": 0.2,
        "arms": {
            "median": {"pnl_eur": 106662.0},
            **{
                a: {"pnl_eur": 105000.0, "expected_at_scenario_mean_eur": 108000.0}
                for a in t7.ARMS
            },
        },
        "difference": {
            a: {s: interval(-1.8, -2.6, -1.0) for s in ("all", "summer", "winter")}
            for a in t7.ARMS
        },
    }
    rejected = t7.results_markdown(summary)
    summary["difference"]["cvar_25"]["all"] = interval(2.0, 0.5, 3.5)
    adopted = t7.results_markdown(summary)

    assert "Neither risk-sensitive arm is adopted" in rejected
    assert "Adopted: cvar_25" in adopted


def test_period_hours_matches_the_production_optimiser(settings: Settings) -> None:
    """The experiment must settle on the same period length the backtest uses."""
    import pandas as pd

    index = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    assert period_hours(pd.DatetimeIndex(index)) == pytest.approx(0.25)


def test_a_fitted_tail_carries_a_spike_that_a_flat_one_clamps() -> None:
    """14.4% of real periods land outside the fan; a clamp cannot hold them."""
    quantiles = _fan()
    spike = np.full(8, 900.0)
    tails = t7.Tails(lower=0.6, upper=0.6)

    flat_back = t7.scenario_prices(
        quantiles, LEVELS, t7.day_ranks(quantiles, LEVELS, spike).reshape(1, -1)
    )[0]
    tailed_back = t7.scenario_prices(
        quantiles,
        LEVELS,
        t7.day_ranks(quantiles, LEVELS, spike, tails).reshape(1, -1),
        tails,
    )[0]

    assert np.allclose(flat_back, quantiles[:, -1])  # clamped to the fan's top
    assert (tailed_back > quantiles[:, -1] * 2).all()  # well past it


def test_the_tails_are_fitted_from_how_far_prices_actually_ran_past_the_fan() -> None:
    quantiles = _fan()
    top, middle = quantiles[0, -1], quantiles[0, 3]
    overshoot = 44.0
    actual = np.full(8, middle)
    actual[0] = top + overshoot

    tails = t7.fit_tails(quantiles, LEVELS, actual)

    # The scale is the overshoot as a share of the fan's own upper half-width,
    # so a wide-fan day gets a proportionally longer tail.
    assert tails.upper == pytest.approx(overshoot / (top - middle))
    assert tails.lower == 0.0


def test_a_moderate_overshoot_survives_the_round_trip(settings: Settings) -> None:
    """Exactness matters where the data lives, not at implausible extremes."""
    quantiles = _fan()
    tails = t7.Tails(lower=0.6, upper=0.6)
    actual = np.array([150.0, 145.0, 55.0, 100.0, 160.0, 50.0, 120.0, 80.0])

    ranks = t7.day_ranks(quantiles, LEVELS, actual, tails)
    back = t7.scenario_prices(quantiles, LEVELS, ranks.reshape(1, -1), tails)[0]

    assert np.allclose(back, actual, atol=1e-6)
