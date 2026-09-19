"""Training on the money: the refit calendar and the direction of the gradient."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
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


def test_the_rival_is_q50_plus_the_hour_bias_of_the_days_before_each_refit() -> None:
    days = [date(2024, 3, 30) + timedelta(days=i) for i in range(35)]
    index = pd.DatetimeIndex(
        [
            pd.Timestamp(d, tz="Europe/Berlin") + pd.Timedelta(hours=h)
            for d in days
            for h in range(24)
        ]
    ).tz_convert("UTC")
    hour = index.tz_convert("Europe/Berlin").hour
    saved = pd.DataFrame(
        {
            "target_day": [d for d in days for _ in range(24)],
            "q50": 50.0,
            # The real price runs 10 above q50 at 19:00, 5 below at 03:00.
            "actual": 50.0
            + np.where(hour == 19, 10.0, 0.0)
            - np.where(hour == 3, 5.0, 0.0),
        },
        index=index,
    )

    rival = m._rival_points(saved, "Europe/Berlin")

    first_block = saved["target_day"] < date(2024, 4, 27)
    # No history before the first refit: the rival is q50 itself.
    assert (rival[first_block] == 50.0).all()
    second = ~first_block
    assert rival[second & (hour == 19)].unique() == pytest.approx([60.0])
    assert rival[second & (hour == 3)].unique() == pytest.approx([45.0])
    assert rival[second & (hour == 12)].unique() == pytest.approx([50.0])


def test_robustness_reads_the_gain_off_the_daily_profits() -> None:
    days = [date(2024, 6, 1) + timedelta(days=i) for i in range(400)]
    gain = np.full(400, 1.0)
    gain[150] = 300.0  # 2024-10-29, in the strongest quarter
    gain[10] = -2.0
    pnl = pd.DataFrame(
        {
            "target_day": days * 2,
            "strategy": ["median_forecast"] * 400 + ["money"] * 400,
            "pnl_eur": np.concatenate([np.full(400, 100.0), 100.0 + gain]),
        }
    )
    saved = pd.DataFrame(
        {"target_day": days, "actual": [-1.0 if i < 5 else 20.0 for i in range(400)]}
    )

    out = m.robustness(pnl, saved)

    assert out["total_eur"] == pytest.approx(gain.sum())
    assert out["top_days"][0] == {"day": "2024-10-29", "eur": 300.0}
    assert out["wins"] == 399 and out["losses"] == 1 and out["ties"] == 0
    assert out["without_2024q4"]["days"] == 400 - 92
    assert out["by_year"]["2025"]["days"] == 186
    assert out["negative_price_days"]["days"] == 5
    assert out["without_top10"]["mean"] == pytest.approx(
        (gain.sum() - 300.0 - 9.0) / 390
    )
