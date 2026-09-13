from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import CARBON_COLUMN, DERIVED_COLUMNS, GAS_COLUMN, Settings
from src.ingest.build_inputs import join_inputs
from tests.fakes import synthetic_market


def _market(settings: Settings) -> pd.DataFrame:
    full = synthetic_market(settings, date(2024, 10, 25), 4)
    keep = list(settings.smard.series) + list(DERIVED_COLUMNS)
    return full[keep]


def _weather(settings: Settings, market: pd.DataFrame) -> pd.DataFrame:
    index = pd.DatetimeIndex(market.index)[8:]  # starts two hours in
    values = np.arange(len(index), dtype=float)
    return pd.DataFrame({c: values for c in settings.weather.columns}, index=index)


def _fuels() -> pd.DataFrame:
    days = [
        date(2024, 10, 25),
        date(2024, 10, 26),
        date(2024, 10, 27),
        date(2024, 10, 28),
    ]
    return pd.DataFrame(
        {
            GAS_COLUMN: [30.0, 31.0, 32.0, 33.0],
            CARBON_COLUMN: [60.0, 61.0, 62.0, np.nan],
        },
        index=days,
    )


def test_inputs_follow_the_availability_column_order(settings: Settings) -> None:
    market = _market(settings)
    frame = join_inputs(market, _weather(settings, market), _fuels(), settings)

    assert list(frame.columns) == list(settings.availability.columns)
    assert frame.index.equals(market.index)


def test_weather_aligns_by_timestamp_and_is_missing_before_it_starts(
    settings: Settings,
) -> None:
    market = _market(settings)
    weather = _weather(settings, market)
    frame = join_inputs(market, weather, _fuels(), settings)

    column = settings.weather.columns[0]
    assert frame[column].iloc[:8].isna().all()
    pd.testing.assert_series_equal(
        frame[column].iloc[8:], weather[column], check_names=False, check_freq=False
    )


def test_fuels_follow_the_local_delivery_day_including_dst(settings: Settings) -> None:
    market = _market(settings)
    frame = join_inputs(market, _weather(settings, market), _fuels(), settings)

    local = pd.DatetimeIndex(frame.index).tz_convert(settings.market.timezone)
    dst_day = local.date == date(2024, 10, 27)
    assert int(dst_day.sum()) == 100
    assert set(frame.loc[dst_day, GAS_COLUMN]) == {32.0}
    assert frame.loc[local.date == date(2024, 10, 28), CARBON_COLUMN].isna().all()


def test_missing_weather_columns_are_refused(settings: Settings) -> None:
    market = _market(settings)
    weather = _weather(settings, market).drop(columns=settings.weather.columns[-1])
    with pytest.raises(ValueError, match="weather data lacks"):
        join_inputs(market, weather, _fuels(), settings)
