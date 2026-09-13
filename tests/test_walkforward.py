from __future__ import annotations

import dataclasses
from datetime import date, timedelta

import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import (
    ForecastError,
    QuantileForecast,
    make_forecast,
    quantile_column,
)
from src.forecasting.information import InformationSet
from src.forecasting.walkforward import HoldoutAccessError, day_range, run_walk_forward
from tests.fakes import day_clock_price, synthetic_market


class SpyForecaster:
    name = "spy"
    lookback_days: int | None = 3

    def __init__(self, settings: Settings) -> None:
        self.quantiles = settings.forecasting.quantiles
        self.fit_days: list[date] = []
        self.last_price_seen: list[tuple[date, pd.Timestamp]] = []

    def fit(self, info: InformationSet) -> None:
        self.fit_days.append(info.target_day)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        last = info.history[PRICE_SERIES].last_valid_index()
        assert isinstance(last, pd.Timestamp)
        self.last_price_seen.append((info.target_day, last))
        raw = pd.DataFrame(
            {quantile_column(q): q for q in self.quantiles}, index=info.target_index
        )
        return make_forecast(self.name, info, raw, self.quantiles)


class WrongDayForecaster(SpyForecaster):
    def forecast(self, info: InformationSet) -> QuantileForecast:
        good = super().forecast(info)
        return dataclasses.replace(good, target_day=info.target_day + timedelta(days=1))


def _market(settings: Settings) -> pd.DataFrame:
    return synthetic_market(
        settings, date(2024, 6, 1), 20, day_clock_price(settings.market.timezone)
    )


def _with_holdout(settings: Settings, holdout: date) -> Settings:
    evaluation = settings.evaluation.model_copy(update={"holdout_start": holdout})
    return settings.model_copy(update={"evaluation": evaluation})


def test_each_forecast_sees_only_prices_before_its_target_day(
    settings: Settings,
) -> None:
    frame = _market(settings)
    spy = SpyForecaster(settings)
    days = day_range(date(2024, 6, 5), date(2024, 6, 11))

    out = run_walk_forward(frame, spy, days, settings)

    for day, last_seen in spy.last_price_seen:
        assert last_seen < settings.market.local_midnight_utc(day)
    assert len(out) == 7 * 96
    assert list(out["model"].unique()) == ["spy"]
    pd.testing.assert_series_equal(
        out["actual"], frame[PRICE_SERIES].reindex(out.index), check_names=False
    )


def test_models_refit_on_their_schedule(settings: Settings) -> None:
    spy = SpyForecaster(settings)
    days = day_range(date(2024, 6, 5), date(2024, 6, 11))

    run_walk_forward(_market(settings), spy, days, settings, refit_every_days=3)

    assert spy.fit_days == [date(2024, 6, 5), date(2024, 6, 8), date(2024, 6, 11)]


def test_hold_out_days_are_refused_unless_explicitly_allowed(
    settings: Settings,
) -> None:
    guarded = _with_holdout(settings, date(2024, 6, 10))
    days = day_range(date(2024, 6, 8), date(2024, 6, 11))

    with pytest.raises(HoldoutAccessError, match="2 target days"):
        run_walk_forward(_market(settings), SpyForecaster(settings), days, guarded)

    out = run_walk_forward(
        _market(settings), SpyForecaster(settings), days, guarded, allow_holdout=True
    )
    assert out["target_day"].nunique() == 4


def test_target_days_must_be_strictly_increasing(settings: Settings) -> None:
    days = [date(2024, 6, 6), date(2024, 6, 5)]
    with pytest.raises(ValueError, match="strictly increasing"):
        run_walk_forward(_market(settings), SpyForecaster(settings), days, settings)


def test_a_forecast_for_the_wrong_day_is_rejected(settings: Settings) -> None:
    with pytest.raises(ForecastError, match="breaks the contract"):
        run_walk_forward(
            _market(settings),
            WrongDayForecaster(settings),
            [date(2024, 6, 6)],
            settings,
        )


class CrossingForecaster(SpyForecaster):
    def forecast(self, info: InformationSet) -> QuantileForecast:
        good = super().forecast(info)
        values = good.values.copy()
        values.iloc[:, 0] = values.iloc[:, -1] + 1.0
        return dataclasses.replace(good, values=values)


class MissingValueForecaster(SpyForecaster):
    def forecast(self, info: InformationSet) -> QuantileForecast:
        good = super().forecast(info)
        values = good.values.copy()
        values.iloc[0, 3] = float("nan")
        return dataclasses.replace(good, values=values)


def test_forecasts_that_skip_validation_are_still_checked(settings: Settings) -> None:
    day = [date(2024, 6, 6)]
    with pytest.raises(ForecastError, match="crossing quantiles"):
        run_walk_forward(_market(settings), CrossingForecaster(settings), day, settings)
    with pytest.raises(ForecastError, match="missing or infinite"):
        run_walk_forward(
            _market(settings), MissingValueForecaster(settings), day, settings
        )
