"""Features may only use what was published when each day's forecast was issued."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.config import CARBON_COLUMN, GAS_COLUMN, PRICE_SERIES, Settings
from src.features.build import EXTRA_FEATURE_GROUPS, FEATURE_GROUPS, build_features
from src.forecasting.information import build_information_set, issue_time_utc
from tests.fakes import day_clock_price, synthetic_market

TZ = "Europe/Berlin"
QUARTER = pd.Timedelta(minutes=15)


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    from src.config import load_settings

    settings = load_settings()
    frame = synthetic_market(settings, date(2024, 3, 10), 240, day_clock_price(TZ))
    rng = np.random.default_rng(3)
    for column in frame.columns:
        if column != PRICE_SERIES and column != "price_product_minutes":
            frame[column] = rng.normal(100.0, 30.0, len(frame))
    return frame


@pytest.fixture(scope="module")
def full_features(market: pd.DataFrame) -> pd.DataFrame:
    from src.config import load_settings

    features: pd.DataFrame = build_features(market, load_settings())
    return features


def _day_rows(frame: pd.DataFrame, day: date) -> pd.DataFrame:
    local = pd.DatetimeIndex(frame.index).tz_convert(TZ)
    rows: pd.DataFrame = frame.loc[local.date == day]
    return rows


@pytest.mark.parametrize(
    "target",
    [date(2024, 4, 1), date(2024, 6, 15), date(2024, 10, 27), date(2024, 10, 28)],
)
def test_features_equal_those_built_from_the_information_set(
    settings: Settings, market: pd.DataFrame, full_features: pd.DataFrame, target: date
) -> None:
    info = build_information_set(market, target, settings, lookback_days=10)

    from_information = _day_rows(build_features(info.history, settings), target)

    pd.testing.assert_frame_equal(from_information, _day_rows(full_features, target))


def test_every_listed_feature_is_built(full_features: pd.DataFrame) -> None:
    groups = {**FEATURE_GROUPS, **EXTRA_FEATURE_GROUPS}
    listed = [name for group in groups.values() for name in group]
    assert list(full_features.columns) == listed


def test_price_lags_follow_local_clock_time(
    market: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    noon = pd.Timestamp("2024-06-15 12:00", tz=TZ).tz_convert("UTC")
    yesterday_noon = noon - pd.Timedelta(days=1)
    assert (
        full_features.at[noon, "price_lag_1d"]
        == market.at[yesterday_noon, PRICE_SERIES]
    )

    # 02:30 on the day after the spring change takes 01:45 from the short day.
    after_spring = pd.Timestamp("2024-04-01 02:30", tz=TZ).tz_convert("UTC")
    short_day_0145 = pd.Timestamp("2024-03-31 01:45", tz=TZ).tz_convert("UTC")
    assert (
        full_features.at[after_spring, "price_lag_1d"]
        == market.at[short_day_0145, PRICE_SERIES]
    )


def test_fuel_features_come_from_the_day_before(
    market: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    target = date(2024, 6, 15)
    previous = _day_rows(market, target - timedelta(days=1))
    rows = _day_rows(full_features, target)
    assert set(rows[GAS_COLUMN]) == {previous[GAS_COLUMN].iloc[-1]}
    assert set(rows[CARBON_COLUMN]) == {previous[CARBON_COLUMN].iloc[-1]}


def test_measured_values_stop_at_the_publication_cutoff(
    settings: Settings, market: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    target = date(2024, 6, 15)
    cutoff = issue_time_utc(target, settings) - pd.Timedelta(
        minutes=settings.availability.actuals_lag_minutes
    )
    idx = pd.DatetimeIndex(market.index)
    last_end = cutoff.floor("15min")
    window = (idx + QUARTER <= cutoff) & (idx >= last_end - pd.Timedelta(hours=24))
    assert int(window.sum()) == 96
    expected = market.loc[window, "load_actual_mw"].mean()

    values = _day_rows(full_features, target)["load_actual_last_24h_mw"].unique()
    assert len(values) == 1
    assert values[0] == pytest.approx(expected)


def test_national_holidays_are_flagged(full_features: pd.DataFrame) -> None:
    assert set(_day_rows(full_features, date(2024, 10, 3))["is_holiday"]) == {1.0}
    assert set(_day_rows(full_features, date(2024, 10, 2))["is_holiday"]) == {0.0}


def test_weather_means_need_every_point(
    settings: Settings, market: pd.DataFrame
) -> None:
    frame = market.copy()
    day = date(2024, 6, 15)
    rows = pd.DatetimeIndex(frame.index).tz_convert(TZ).date == day
    frame.loc[rows, settings.weather.column("bavaria", "shortwave_radiation")] = np.nan

    features = _day_rows(build_features(frame, settings), day)

    assert features["wx_radiation_mean"].isna().all()
    assert features["wx_radiation_south_mean"].isna().all()
    assert features["wx_wind_speed_onshore_mean"].notna().all()


def test_missing_input_columns_are_refused(
    settings: Settings, market: pd.DataFrame
) -> None:
    column = settings.weather.column("bavaria", "temperature_2m")
    with pytest.raises(ValueError, match=column):
        build_features(market.drop(columns=column), settings)


def test_a_partial_previous_day_leaves_its_statistics_empty(
    settings: Settings, market: pd.DataFrame
) -> None:
    frame = market.copy()
    target = date(2024, 6, 15)
    local = pd.DatetimeIndex(frame.index).tz_convert(TZ)
    afternoon_before = (local.date == target - timedelta(days=1)) & (local.hour >= 12)
    frame.loc[afternoon_before, PRICE_SERIES] = np.nan

    features = _day_rows(build_features(frame, settings), target)
    hours = pd.DatetimeIndex(features.index).tz_convert(TZ).hour

    assert features["price_prev_day_mean"].isna().all()
    assert features["price_prev_day_last"].isna().all()
    assert features.loc[hours == 15, "price_mean_same_clock_7d"].isna().all()
    assert features.loc[hours == 8, "price_mean_same_clock_7d"].notna().all()


def test_a_measured_window_with_a_gap_is_empty(
    settings: Settings, market: pd.DataFrame
) -> None:
    frame = market.copy()
    target = date(2024, 6, 15)
    cutoff = issue_time_utc(target, settings) - pd.Timedelta(
        minutes=settings.availability.actuals_lag_minutes
    )
    inside_window = cutoff.floor("15min") - pd.Timedelta(hours=5)
    assert inside_window in frame.index
    frame.loc[inside_window, "load_actual_mw"] = np.nan

    features = _day_rows(build_features(frame, settings), target)

    assert features["load_actual_last_24h_mw"].isna().all()
    assert features["wind_actual_last_24h_mw"].notna().all()


def test_spike_drivers_are_one_value_per_day_from_the_right_windows(
    settings: Settings, market: pd.DataFrame, full_features: pd.DataFrame
) -> None:
    day = date(2024, 6, 15)
    rows = _day_rows(full_features, day)
    local = pd.DatetimeIndex(rows.index).tz_convert(TZ)
    evening = rows[(local.hour >= 17) & (local.hour <= 20)]
    afternoon = rows[(local.hour >= 12) & (local.hour <= 15)]
    for column in EXTRA_FEATURE_GROUPS["spike_drivers"]:
        assert rows[column].nunique() == 1, column

    assert rows["load_forecast_evening_mean_mw"].iloc[0] == pytest.approx(
        evening["load_forecast_mw"].mean()
    )
    assert rows["load_forecast_evening_ramp_mw"].iloc[0] == pytest.approx(
        evening["load_forecast_mw"].mean() - afternoon["load_forecast_mw"].mean()
    )
    assert rows["wx_radiation_afternoon_mean"].iloc[0] == pytest.approx(
        afternoon["wx_radiation_mean"].mean()
    )
    assert rows["residual_persistence_evening_max_mw"].iloc[0] == pytest.approx(
        evening["residual_load_persistence_mw"].max()
    )
    # Yesterday's evening prices, and the week before, from the market itself.
    prices = market[PRICE_SERIES]
    price_local = pd.DatetimeIndex(prices.index).tz_convert(TZ)
    price_evening = prices[(price_local.hour >= 17) & (price_local.hour <= 20)]
    evening_days = pd.Series(
        price_evening.to_numpy(),
        index=price_local[(price_local.hour >= 17) & (price_local.hour <= 20)].date,
    )
    by_day = evening_days.groupby(level=0).max()
    yesterday = day - timedelta(days=1)
    assert rows["price_prev_day_evening_max"].iloc[0] == pytest.approx(
        by_day[yesterday]
    )
    week = [day - timedelta(days=k) for k in range(1, 8)]
    assert rows["price_evening_max_7d"].iloc[0] == pytest.approx(
        max(by_day[d] for d in week)
    )
    day_max = prices.groupby(price_local.date).max()
    spikes = sum(
        day_max[d] >= settings.evaluation.spike_threshold_eur_mwh for d in week
    )
    assert rows["spike_days_last_7d"].iloc[0] == spikes
