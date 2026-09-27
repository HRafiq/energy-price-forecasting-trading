"""The fixed-shape benchmark: what it averages, and what it is not allowed to see."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.health.experiments import t8_fixed_shape as fs

TZ = "Europe/Berlin"


def _hourly(start: str, hours: int, values: Any) -> pd.Series:
    index = pd.date_range(start, periods=hours, freq="h", tz="UTC")
    return pd.Series(np.asarray(values, dtype="float64"), index=index)


def test_the_window_stops_at_the_instant_the_delivery_day_begins() -> None:
    """The leakage guarantee: nothing from the target day reaches the average."""
    prices = _hourly("2026-05-31 00:00", 48, np.arange(48.0))
    start = pd.Timestamp("2026-05-31 22:00", tz="UTC")  # local midnight, 1 June

    window = fs.trailing_window(prices, start, days=10)

    assert window.index.max() < start
    assert len(window) == 22 and window.iloc[-1] == 21.0


def test_the_window_reaches_back_no_further_than_its_span() -> None:
    prices = _hourly("2026-01-01 00:00", 24 * 40, np.zeros(24 * 40))
    start = pd.Timestamp("2026-02-10 00:00", tz="UTC")

    window = fs.trailing_window(prices, start, days=10)

    assert window.index.min() == pd.Timestamp("2026-01-31 00:00", tz="UTC")


def test_the_shape_is_the_mean_of_each_local_time_of_day() -> None:
    """Two days of history, so each local hour averages exactly two prices."""
    prices = _hourly(
        "2026-05-29 22:00", 48, list(range(24)) + [v * 3 for v in range(24)]
    )
    index = pd.date_range("2026-05-31 22:00", periods=24, freq="h", tz="UTC")

    shape = fs.shape_for(prices, index, TZ, None)

    assert shape == pytest.approx([(v + 3 * v) / 2 for v in range(24)])


def test_a_month_filter_averages_only_that_month() -> None:
    june = _hourly("2026-05-31 22:00", 24, np.full(24, 10.0))
    july = _hourly("2026-06-30 22:00", 24, np.full(24, 90.0))
    prices = pd.concat([june, july])
    index = pd.DatetimeIndex(july.index)

    assert fs.shape_for(prices, index, TZ, 7) == pytest.approx(np.full(24, 90.0))
    assert fs.shape_for(prices, index, TZ, 6) == pytest.approx(np.full(24, 10.0))
    assert fs.shape_for(prices, index, TZ, None) == pytest.approx(np.full(24, 50.0))


def test_a_long_day_takes_each_local_clock_hour_from_the_same_clock_hour() -> None:
    """25 local hours in October; the repeated 02:00 gets 02:00's average."""
    history = _hourly("2026-10-23 22:00", 24, np.arange(24.0) * 10)
    index = pd.date_range("2026-10-24 22:00", periods=25, freq="h", tz="UTC")

    shape = fs.shape_for(history, index, TZ, None)

    assert len(shape) == 25
    local = index.tz_convert(TZ)
    repeated = [i for i, t in enumerate(local) if t.hour == 2]
    assert len(repeated) == 2
    assert shape[repeated[0]] == pytest.approx(shape[repeated[1]])


def test_a_clock_time_with_no_history_falls_back_to_the_window_mean() -> None:
    history = _hourly("2026-05-31 22:00", 12, np.full(12, 40.0))
    index = pd.date_range("2026-05-31 22:00", periods=24, freq="h", tz="UTC")

    shape = fs.shape_for(history, index, TZ, None)

    assert np.isfinite(shape).all() and shape == pytest.approx(np.full(24, 40.0))


def test_settling_pays_the_realised_price_and_charges_wear_on_what_is_sold(
    settings: Settings,
) -> None:
    net = np.zeros(96)
    net[8:12] = -1.0  # buy 1 MW for an hour
    net[76:80] = 1.0  # sell 1 MW for an hour
    price = np.full(96, 50.0)
    price[8:12], price[76:80] = 20.0, 150.0

    earned = fs.settle(net, price, settings.battery, 0.25)

    wear = settings.battery.degradation_eur_per_mwh
    assert earned == pytest.approx(150.0 - 20.0 - wear)


def test_the_page_reports_the_decomposition_from_the_captures() -> None:
    def interval(mean: float) -> dict[str, float]:
        return {"mean": mean, "low": mean - 1, "high": mean + 1, "days": 730}

    summary: dict[str, Any] = {
        "days": 730,
        "first_day": "2024-06-01",
        "last_day": "2026-05-31",
        "trailing_days": 730,
        "naive_capture": 0.7765,
        "arms": {
            "perfect_foresight": {
                "pnl_eur": 165922.0,
                "capture": 1.0,
                "losing_days": 0,
            },
            "median_forecast": {
                "pnl_eur": 149498.0,
                "capture": 0.901,
                "losing_days": 15,
            },
            "fixed_shape": {"pnl_eur": 113974.0, "capture": 0.687, "losing_days": 83},
            "fixed_shape_seasonal": {
                "pnl_eur": 140923.0,
                "capture": 0.849,
                "losing_days": 15,
            },
        },
        "cycles_per_day": {"fixed_shape": 1.9, "fixed_shape_seasonal": 1.76},
        "forecast_buys": {
            arm: {s: interval(10.0) for s in ("all", "summer", "winter")}
            for arm in fs.ARMS
        },
    }

    page = fs.results_markdown(summary)

    # The best fixed arm sets the split, not the first one listed.
    assert "no forecast at all: **84.9%**" in page
    assert "adds on top of it: **+5.2%**" in page
    assert "still leaves on the table: **9.9%**" in page
    assert "77.6%" in page


def test_a_quarter_hourly_day_averages_each_quarter_hour_separately() -> None:
    """The 2025-10-01 switch: the shape must carry structure inside the hour.

    Averaging by the hour alone would flatten every quarter-hour to its hourly
    mean. That silently costs the benchmark a third of its profit, so pin it.
    """
    index = pd.date_range("2026-05-31 22:00", periods=96, freq="15min", tz="UTC")
    # A price that moves within every hour, not just between hours.
    within_hour = np.tile([0.0, 10.0, 20.0, 30.0], 24)
    history = pd.Series(within_hour, index=index)

    shape = fs.shape_for(history, index, TZ, None)

    assert shape == pytest.approx(within_hour)
    # The hourly-only mistake would put 15.0 everywhere; make sure it would fail.
    assert not np.allclose(shape, np.full(96, 15.0))


def test_quarter_hours_of_the_same_clock_time_average_across_days() -> None:
    two_days = pd.date_range("2026-05-30 22:00", periods=192, freq="15min", tz="UTC")
    first, second = (
        np.tile([0.0, 10.0, 20.0, 30.0], 24),
        np.tile([4.0, 14.0, 24.0, 34.0], 24),
    )
    history = pd.Series(np.concatenate([first, second]), index=two_days)
    index = pd.date_range("2026-06-01 22:00", periods=96, freq="15min", tz="UTC")

    shape = fs.shape_for(history, index, TZ, None)

    assert shape == pytest.approx(np.tile([2.0, 12.0, 22.0, 32.0], 24))
