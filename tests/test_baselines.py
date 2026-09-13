from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import ForecastError, QuantileForecast, quantile_columns
from src.forecasting.baselines import build_baselines, clock_table, values_on
from src.forecasting.information import build_information_set
from src.timegrid import delivery_periods
from tests.fakes import day_clock_price, synthetic_market

TZ = "Europe/Berlin"
QUARTER = pd.Timedelta(minutes=15)
TARGET = date(2024, 6, 15)
EPOCH = date(2000, 1, 1).toordinal()


def _day_base(day: date) -> float:
    return 1000.0 * (day.toordinal() - EPOCH)


def _forecast(
    settings: Settings, frame: pd.DataFrame, which: int, day: date = TARGET
) -> QuantileForecast:
    model = build_baselines(settings)[which]
    return model.forecast(
        build_information_set(frame, day, settings, model.lookback_days)
    )


def test_naive_is_exact_when_each_day_is_yesterday_plus_a_constant(
    settings: Settings,
) -> None:
    frame = synthetic_market(settings, date(2024, 4, 1), 91, day_clock_price(TZ))
    result = _forecast(settings, frame, which=0)

    actual = frame[PRICE_SERIES].reindex(result.values.index).to_numpy()
    for column in result.values.columns:
        np.testing.assert_allclose(result.values[column].to_numpy(), actual)
    assert result.fallback_periods == 0


def test_seasonal_naive_is_exact_when_prices_repeat_weekly(settings: Settings) -> None:
    def weekly(index: pd.DatetimeIndex) -> NDArray[np.float64]:
        local = index.tz_convert(TZ)
        return np.asarray(
            local.dayofweek * 100 + local.hour * 60 + local.minute, dtype=float
        )

    frame = synthetic_market(settings, date(2024, 4, 1), 91, weekly)
    result = _forecast(settings, frame, which=1)

    actual = frame[PRICE_SERIES].reindex(result.values.index).to_numpy()
    np.testing.assert_allclose(result.values["q50"].to_numpy(), actual)
    np.testing.assert_allclose(result.values["q05"].to_numpy(), actual)


def test_quantile_bands_widen_in_hours_with_larger_errors(settings: Settings) -> None:
    def noisy_evenings(index: pd.DatetimeIndex) -> NDArray[np.float64]:
        local = index.tz_convert(TZ)
        parity = np.asarray([d.toordinal() % 2 for d in local.date])
        swing = np.where(local.hour == 19, np.where(parity == 0, 40.0, -40.0), 0.0)
        return 50.0 + swing

    frame = synthetic_market(settings, date(2024, 4, 1), 91, noisy_evenings)
    values = _forecast(settings, frame, which=0).values
    local_hour = pd.DatetimeIndex(values.index).tz_convert(TZ).hour
    width = values["q95"] - values["q05"]

    assert width[local_hour == 19].min() > 60.0
    assert width[local_hour == 3].max() == 0.0
    assert list(values.columns) == quantile_columns(settings.forecasting.quantiles)


def test_missing_source_day_falls_back_to_the_second_lag(settings: Settings) -> None:
    frame = synthetic_market(settings, date(2024, 4, 1), 91, day_clock_price(TZ))
    previous_day = pd.DatetimeIndex(frame.index).tz_convert(
        TZ
    ).date == TARGET - timedelta(days=1)
    frame.loc[previous_day, PRICE_SERIES] = np.nan

    result = _forecast(settings, frame, which=0)

    actual = frame[PRICE_SERIES].reindex(result.values.index).to_numpy()
    # The point comes from one week back and its range from one-week errors, which
    # are exactly 7000 here, so the forecast lands on the actual price again.
    np.testing.assert_allclose(result.values["q50"].to_numpy(), actual)
    assert result.fallback_periods == 96


def test_with_no_source_day_the_last_published_price_is_used(
    settings: Settings,
) -> None:
    frame = synthetic_market(settings, date(2024, 4, 1), 91, day_clock_price(TZ))
    local_dates = pd.DatetimeIndex(frame.index).tz_convert(TZ).date
    for gone in (TARGET - timedelta(days=1), TARGET - timedelta(days=7)):
        frame.loc[local_dates == gone, PRICE_SERIES] = np.nan

    result = _forecast(settings, frame, which=0)

    last_published = _day_base(TARGET - timedelta(days=2)) + 23 * 60 + 45
    np.testing.assert_allclose(result.values["q50"].to_numpy(), last_published + 7000.0)


def test_autumn_target_day_repeats_the_source_value_in_its_extra_hour(
    settings: Settings,
) -> None:
    frame = synthetic_market(settings, date(2024, 10, 20), 10, day_clock_price(TZ))
    table = clock_table(frame[PRICE_SERIES], TZ, QUARTER)
    target = delivery_periods(date(2024, 10, 27), TZ, QUARTER)

    values = values_on(table, date(2024, 10, 26), target, TZ)

    base = _day_base(date(2024, 10, 26))
    two_oclock = values[np.asarray(target.tz_convert(TZ).hour == 2)]
    assert len(values) == 100
    assert sorted(two_oclock) == sorted([base + m for m in (120, 135, 150, 165)] * 2)


def test_spring_source_day_fills_its_missing_hour_from_0145(settings: Settings) -> None:
    frame = synthetic_market(settings, date(2024, 3, 25), 10, day_clock_price(TZ))
    table = clock_table(frame[PRICE_SERIES], TZ, QUARTER)
    target = delivery_periods(date(2024, 4, 1), TZ, QUARTER)

    values = values_on(table, date(2024, 3, 31), target, TZ)

    base = _day_base(date(2024, 3, 31))
    local = target.tz_convert(TZ)
    assert set(values[np.asarray(local.hour == 2)]) == {base + 105}
    assert values[np.asarray((local.hour == 3) & (local.minute == 0))][0] == base + 180


def test_autumn_source_day_keeps_the_first_of_its_repeated_hour(
    settings: Settings,
) -> None:
    frame = synthetic_market(settings, date(2024, 10, 26), 3)
    table = clock_table(frame[PRICE_SERIES], TZ, QUARTER)
    first_two_oclock = pd.Timestamp("2024-10-27 00:00", tz="UTC")  # 02:00 summer time

    assert table.xs(date(2024, 10, 27))[120] == frame.at[first_two_oclock, PRICE_SERIES]


def test_a_fallback_point_takes_its_range_from_the_fallback_lag(
    settings: Settings,
) -> None:
    rng = np.random.default_rng(7)

    def noisy(index: pd.DatetimeIndex) -> NDArray[np.float64]:
        return np.asarray(50.0 + rng.normal(0.0, 20.0, len(index)), dtype=np.float64)

    frame = synthetic_market(settings, date(2024, 4, 1), 91, noisy)
    local_dates = pd.DatetimeIndex(frame.index).tz_convert(TZ).date
    frame.loc[local_dates == TARGET - timedelta(days=1), PRICE_SERIES] = np.nan
    model = build_baselines(settings)[0]
    info = build_information_set(frame, TARGET, settings, model.lookback_days)

    result = model.forecast(info)

    table = clock_table(info.history[PRICE_SERIES], TZ, QUARTER)
    weekly = model._error_offsets(table, TARGET, model.fallback_lag_days)
    backup = values_on(table, TARGET - timedelta(days=7), info.target_index, TZ)
    hours = np.asarray(info.target_index.tz_convert(TZ).hour)
    np.testing.assert_allclose(
        result.values["q05"].to_numpy(), backup + weekly[0.05][hours]
    )
    np.testing.assert_allclose(
        result.values["q95"].to_numpy(), backup + weekly[0.95][hours]
    )


def test_too_little_error_history_is_refused(settings: Settings) -> None:
    frame = synthetic_market(settings, date(2024, 4, 1), 91, day_clock_price(TZ))
    local_dates = pd.DatetimeIndex(frame.index).tz_convert(TZ).date
    for k in range(2, 22):
        frame.loc[local_dates == TARGET - timedelta(days=k), PRICE_SERIES] = np.nan

    with pytest.raises(ForecastError, match="days of errors"):
        _forecast(settings, frame, which=0)


def test_a_few_missing_periods_never_stop_the_whole_forecast(
    settings: Settings,
) -> None:
    """The reviewer's case: the longer lag lacks error history, day D lacks an hour."""
    frame = synthetic_market(settings, date(2024, 4, 1), 91, day_clock_price(TZ))
    local = pd.DatetimeIndex(frame.index).tz_convert(TZ)
    for k in range(21, 36):
        frame.loc[local.date == TARGET - timedelta(days=k), PRICE_SERIES] = np.nan
    evening = (local.date == TARGET - timedelta(days=1)) & (local.hour == 18)
    frame.loc[evening, PRICE_SERIES] = np.nan

    result = _forecast(settings, frame, which=0)

    assert result.fallback_periods == 4
    actual = frame[PRICE_SERIES].reindex(result.values.index).to_numpy()
    at_six = np.asarray(pd.DatetimeIndex(result.values.index).tz_convert(TZ).hour == 18)
    # Fallback periods: last week's price with the one-day range, 6000 below actual.
    np.testing.assert_allclose(
        result.values["q50"].to_numpy()[at_six], actual[at_six] - 6000.0
    )
    np.testing.assert_allclose(
        result.values["q50"].to_numpy()[~at_six], actual[~at_six]
    )
