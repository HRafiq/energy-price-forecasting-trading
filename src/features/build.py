"""Feature matrix for the price models, built only from data known at issue time.

Each row is one delivery period of local day X. Every feature in that row uses
only information published by 11:40 on day X-1, the issue time for day X,
following the column rules in ``config/settings.yaml``:

* calendar: clock time, weekday, national holiday, season, DST day length,
  product type, all fixed in advance;
* price history: the same local clock time one, two and seven days earlier, the
  seven-day mean at that clock time, and statistics of day X-1;
* load: the grid operators' load forecast for day X and its change from day X-1;
* grid-operator wind and solar forecasts for day X-1, published at 18:00 on
  X-2, and a residual-load estimate from them and the load forecast;
* measured wind, solar and load over the 24 hours before the publication cutoff;
* weather forecasts for day X issued two days ahead, averaged over onshore,
  offshore and southern points, with a simple wind-power curve;
* fuels: the last gas and carbon prices published before day X-1, and the
  short-run marginal cost of a gas plant they imply.

``tests/test_features.py`` proves the rule: features built from a target day's
information set equal features built from the full dataset for that day.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta

import holidays
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import (
    CARBON_COLUMN,
    GAS_COLUMN,
    PRICE_SERIES,
    PRODUCT_COLUMN,
    RESOLUTION_STEP,
    Settings,
)
from src.features.clock import clock_minutes, clock_table, lag_by_clock, local_dates
from src.forecasting.information import issue_time_utc
from src.timegrid import ensure_utc_index, expected_periods

__all__ = ["FEATURE_GROUPS", "build_features"]

#: Dataset columns the features read, besides the weather columns.
BASE_INPUTS = (
    PRICE_SERIES,
    PRODUCT_COLUMN,
    "load_forecast_mw",
    "load_actual_mw",
    "wind_onshore_actual_mw",
    "wind_offshore_actual_mw",
    "solar_actual_mw",
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "solar_forecast_mw",
    GAS_COLUMN,
    CARBON_COLUMN,
)

OFFSHORE_POINTS = ("north_sea_offshore", "baltic_offshore")
SOUTH_POINTS = ("baden_wuerttemberg", "bavaria")
#: Open-Meteo reports wind speed in km/h by default.
KMH_PER_MS = 3.6
#: Generic turbine curve in m/s: cut-in, rated, cut-out.
CUT_IN_MS, RATED_MS, CUT_OUT_MS = 3.0, 12.0, 25.0
#: Typical combined-cycle gas plant: electrical efficiency and gas emissions.
CCGT_EFFICIENCY = 0.55
GAS_T_CO2_PER_MWH_THERMAL = 0.202

FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "calendar": (
        "clock_minute",
        "weekday",
        "is_weekend",
        "is_holiday",
        "day_of_year_sin",
        "day_of_year_cos",
        "periods_in_day",
        "quarter_hour_product",
    ),
    "price_history": (
        "price_lag_1d",
        "price_lag_2d",
        "price_lag_7d",
        "price_mean_same_clock_7d",
        "price_prev_day_mean",
        "price_prev_day_min",
        "price_prev_day_max",
        "price_prev_day_std",
        "price_prev_day_last",
    ),
    "load": (
        "load_forecast_mw",
        "load_forecast_day_mean_mw",
        "load_forecast_change_1d_mw",
    ),
    "grid_operator_forecasts": (
        "wind_onshore_forecast_mw_lag_1d",
        "wind_offshore_forecast_mw_lag_1d",
        "solar_forecast_mw_lag_1d",
        "residual_load_persistence_mw",
    ),
    "measured": (
        "wind_actual_last_24h_mw",
        "solar_actual_last_24h_mw",
        "load_actual_last_24h_mw",
    ),
    "weather": (
        "wx_wind_speed_onshore_mean",
        "wx_wind_speed_offshore_mean",
        "wx_wind_power_onshore",
        "wx_wind_power_offshore",
        "wx_radiation_mean",
        "wx_radiation_south_mean",
        "wx_temperature_mean",
    ),
    "fuels": (GAS_COLUMN, CARBON_COLUMN, "ccgt_marginal_cost_eur_mwh"),
}


def _by_day(
    values: pd.Series,
    dates: NDArray[np.object_],
    days_back: int,
    how: str,
    periods: dict[date, int] | None = None,
) -> NDArray[np.float64]:
    """A daily statistic of ``values`` for the local day ``days_back`` before a row.

    With ``periods`` given, a day only counts when every one of its periods has a
    value; a partly published day gives NaN instead of a misleading statistic.
    """
    grouped = values.groupby(dates)
    daily = grouped.agg(how)
    if periods is not None:
        expected = pd.Series(
            [periods.get(day, -1) for day in daily.index], index=daily.index
        )
        daily = daily.where(grouped.count() == expected)
    shift = timedelta(days=days_back)
    return daily.reindex([day - shift for day in dates]).to_numpy(dtype="float64")


def _row_mean(
    stack: NDArray[np.float64], require_all: bool = False
) -> NDArray[np.float64]:
    """Mean over the first axis; NaN if all values are NaN, or any with require_all."""
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    valid = finite.all(axis=0) if require_all else count > 0
    return np.where(valid, total / np.maximum(count, 1), np.nan)


def _wind_power(speed_kmh: NDArray[np.float64]) -> NDArray[np.float64]:
    """Share of rated output on a generic turbine curve; NaN stays NaN."""
    speed = speed_kmh / KMH_PER_MS
    ramp = np.clip((speed - CUT_IN_MS) / (RATED_MS - CUT_IN_MS), 0.0, 1.0) ** 3
    power = np.where(speed >= CUT_OUT_MS, 0.0, ramp)
    return np.where(np.isfinite(speed), power, np.nan)


def _window_mean_before(
    series: pd.Series,
    step: pd.Timedelta,
    cutoffs: Sequence[pd.Timestamp],
    hours: int,
) -> NDArray[np.float64]:
    """Mean over the ``hours`` before each cutoff; NaN unless every period has a value.

    The window holds the periods in the ``hours`` before the last period end at or
    before the cutoff. A window with any missing period
    gives NaN, so a later-than-assumed publication cannot silently change the
    feature between backtest and live use.
    """
    idx = ensure_utc_index(series.index)
    starts = idx.to_numpy(dtype="datetime64[ns]").astype("int64")
    ends = starts + step.value
    values = series.to_numpy(dtype="float64")
    ok = np.isfinite(values)
    total = np.concatenate([[0.0], np.cumsum(np.where(ok, values, 0.0))])
    count = np.concatenate([[0], np.cumsum(ok)])
    window = pd.Timedelta(hours=hours)
    expected = int(window / step)
    result = np.full(len(cutoffs), np.nan)
    for i, cutoff in enumerate(cutoffs):
        hi = int(np.searchsorted(ends, cutoff.value, side="right"))
        # The window is the whole periods before the last period end by the cutoff.
        last_end = cutoff.value - cutoff.value % step.value
        lo = int(np.searchsorted(starts, last_end - window.value, side="left"))
        if hi > lo and int(count[hi] - count[lo]) == expected:
            result[i] = (total[hi] - total[lo]) / expected
    return result


def build_features(frame: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Features for every period of ``frame``, each from its own issue time."""
    idx = ensure_utc_index(frame.index)
    required = [*BASE_INPUTS, *settings.weather.columns]
    absent = [column for column in required if column not in frame.columns]
    if absent:
        raise ValueError(f"cannot build features without input columns {absent}")
    tz = settings.market.timezone
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    local = idx.tz_convert(tz)
    dates = local_dates(idx, tz)
    codes, uniques = pd.factorize(dates)
    days: list[date] = list(uniques)
    out: dict[str, NDArray[np.float64]] = {}

    # Calendar: fixed in advance.
    weekday = np.asarray(local.dayofweek, dtype="float64")
    german_holidays = holidays.country_holidays(
        "DE", years=sorted({d.year for d in days})
    )
    day_of_year = np.asarray(local.dayofyear, dtype="float64")
    out["clock_minute"] = clock_minutes(idx, tz).astype("float64")
    out["weekday"] = weekday
    out["is_weekend"] = (weekday >= 5).astype("float64")
    out["is_holiday"] = np.asarray(
        [day in german_holidays for day in days], dtype="float64"
    )[codes]
    out["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    periods = {day: expected_periods(day, tz, step) for day in days}
    out["periods_in_day"] = np.asarray([periods[day] for day in days], dtype="float64")[
        codes
    ]
    out["quarter_hour_product"] = np.where(
        frame[PRODUCT_COLUMN].to_numpy(dtype="float64") == 15, 1.0, 0.0
    )

    # Price history: day X-1 and earlier.
    price = frame[PRICE_SERIES]
    price_table = clock_table(price, tz, step)
    lags = {
        k: lag_by_clock(price, k, tz, step, table=price_table).to_numpy()
        for k in range(1, 8)
    }
    out["price_lag_1d"] = lags[1]
    out["price_lag_2d"] = lags[2]
    out["price_lag_7d"] = lags[7]
    out["price_mean_same_clock_7d"] = _row_mean(
        np.vstack([lags[k] for k in range(1, 8)]), require_all=True
    )
    for stat in ("mean", "min", "max", "std", "last"):
        out[f"price_prev_day_{stat}"] = _by_day(price, dates, 1, stat, periods)

    # Load forecast for day X, published before the gate.
    load_forecast = frame["load_forecast_mw"]
    out["load_forecast_mw"] = load_forecast.to_numpy(dtype="float64")
    out["load_forecast_day_mean_mw"] = _by_day(load_forecast, dates, 0, "mean", periods)
    out["load_forecast_change_1d_mw"] = (
        out["load_forecast_mw"] - lag_by_clock(load_forecast, 1, tz, step).to_numpy()
    )

    # Grid-operator wind and solar forecasts for day X-1.
    renewables = np.zeros(len(idx))
    for column in (
        "wind_onshore_forecast_mw",
        "wind_offshore_forecast_mw",
        "solar_forecast_mw",
    ):
        lagged = lag_by_clock(frame[column], 1, tz, step).to_numpy()
        out[f"{column}_lag_1d"] = lagged
        renewables = renewables + lagged
    out["residual_load_persistence_mw"] = out["load_forecast_mw"] - renewables

    # Measured values up to the publication cutoff for day X.
    lag = pd.Timedelta(minutes=settings.availability.actuals_lag_minutes)
    cutoffs = [issue_time_utc(day, settings) - lag for day in days]
    wind_actual = frame["wind_onshore_actual_mw"] + frame["wind_offshore_actual_mw"]
    for name, series in (
        ("wind_actual_last_24h_mw", wind_actual),
        ("solar_actual_last_24h_mw", frame["solar_actual_mw"]),
        ("load_actual_last_24h_mw", frame["load_actual_mw"]),
    ):
        out[name] = _window_mean_before(series, step, cutoffs, hours=24)[codes]

    # Weather forecasts for day X, issued two days ahead.
    weather = settings.weather

    def stack(points: Sequence[str], variable: str) -> NDArray[np.float64]:
        columns = [weather.column(p, variable) for p in points if p in weather.points]
        return np.vstack(
            [
                frame[c].to_numpy(dtype="float64")
                if c in frame
                else np.full(len(idx), np.nan)
                for c in columns
            ]
        )

    onshore = [p for p in weather.points if p not in OFFSHORE_POINTS]
    offshore = [p for p in weather.points if p in OFFSHORE_POINTS]
    wind_on = stack(onshore, "wind_speed_100m")
    wind_off = stack(offshore, "wind_speed_100m")
    out["wx_wind_speed_onshore_mean"] = _row_mean(wind_on, require_all=True)
    out["wx_wind_speed_offshore_mean"] = _row_mean(wind_off, require_all=True)
    out["wx_wind_power_onshore"] = _row_mean(_wind_power(wind_on), require_all=True)
    out["wx_wind_power_offshore"] = _row_mean(_wind_power(wind_off), require_all=True)
    out["wx_radiation_mean"] = _row_mean(
        stack(list(weather.points), "shortwave_radiation"), require_all=True
    )
    out["wx_radiation_south_mean"] = _row_mean(
        stack(SOUTH_POINTS, "shortwave_radiation"), require_all=True
    )
    out["wx_temperature_mean"] = _row_mean(
        stack(list(weather.points), "temperature_2m"), require_all=True
    )

    # Fuels: last prices published before day X-1, stored on day X-1's rows.
    gas = _by_day(frame[GAS_COLUMN], dates, 1, "last")
    carbon = _by_day(frame[CARBON_COLUMN], dates, 1, "last")
    out[GAS_COLUMN] = gas
    out[CARBON_COLUMN] = carbon
    out["ccgt_marginal_cost_eur_mwh"] = (
        gas + carbon * GAS_T_CO2_PER_MWH_THERMAL
    ) / CCGT_EFFICIENCY

    ordered = [name for group in FEATURE_GROUPS.values() for name in group]
    missing = [name for name in ordered if name not in out]
    if missing:
        raise RuntimeError(
            f"feature groups list features that were not built: {missing}"
        )
    return pd.DataFrame({name: out[name] for name in ordered}, index=idx)
