"""Training on the money: the refit calendar and the direction of the gradient."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pytest
from numpy.typing import NDArray

from src.config import Settings
from src.forecasting.run_comparison import REFIT_EVERY_DAYS as COMPARISON_CADENCE
from src.health.experiments import t2_money_loss as m
from src.trading.dispatch_lp import DayProgram, solve_day


def test_refits_fall_on_the_comparison_runs_days() -> None:
    days = [date(2024, 3, 30) + timedelta(days=i) for i in range(70)]

    fits = m.refit_days(days)

    assert m.REFIT_EVERY_DAYS == COMPARISON_CADENCE[m.PRODUCTION] == 28
    assert fits == [date(2024, 3, 30), date(2024, 4, 27), date(2024, 5, 25)]


def _group(settings: Settings, prices: NDArray[np.float64]) -> m._DayGroup:
    program = DayProgram.build(96, 0.25, settings.battery, None)
    net_real, _ = solve_day(program, prices)
    return m._DayGroup(np.arange(96), program, net_real, prices)


def test_the_gradient_pushes_the_guess_towards_what_the_battery_missed(
    settings: Settings,
) -> None:
    hour = np.arange(96) // 4
    real = 50.0 + (hour == 19) * 200.0 + (hour == 3) * -30.0
    group = _group(settings, real)
    # A guess that puts the peak at 17:00 instead of 19:00.
    guess = 50.0 + (hour == 17) * 200.0 + (hour == 3) * -30.0

    grad, loss = m.spo_plus_gradient(guess, [group])

    # The real schedule sells at 19:00 and the shifted one does not: push up.
    assert grad[hour == 19].mean() < 0
    # The shifted schedule sells at 17:00 where the real one does not: push down.
    assert grad[hour == 17].mean() > 0
    assert loss > 0

    perfect, zero_loss = m.spo_plus_gradient(real, [group])
    assert np.abs(perfect).max() < 1e-9 and zero_loss == pytest.approx(0.0, abs=1e-6)
