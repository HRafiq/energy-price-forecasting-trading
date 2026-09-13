"""M3: a model can only ever see what was published at the issue time."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.forecasting.base import (
    Forecaster,
    QuantileForecast,
    make_forecast,
    quantile_column,
)
from src.forecasting.baselines import build_baselines
from src.forecasting.information import (
    InformationSet,
    build_information_set,
    issue_time_utc,
)
from tests.fakes import day_clock_price, synthetic_market

TARGET = date(2024, 6, 15)
QUARTER = pd.Timedelta(minutes=15)
TARGET_START = pd.Timestamp("2024-06-14 22:00", tz="UTC")
TARGET_END = pd.Timestamp("2024-06-15 22:00", tz="UTC")


class GreedyForecaster:
    """Uses every value it is given, so only the information set can protect it."""

    name = "greedy"
    lookback_days: int | None = None
    fit_lookback_days: int | None = None

    def __init__(self, quantiles: tuple[float, ...]) -> None:
        self.quantiles = quantiles

    def fit(self, info: InformationSet) -> None:
        return None

    def forecast(self, info: InformationSet) -> QuantileForecast:
        total = float(np.nansum(info.history.to_numpy(dtype="float64")))
        raw = pd.DataFrame(
            {quantile_column(q): total + q for q in self.quantiles},
            index=info.target_index,
        )
        return make_forecast(self.name, info, raw, self.quantiles)


def _market(settings: Settings) -> pd.DataFrame:
    return synthetic_market(
        settings, date(2024, 4, 1), 91, day_clock_price(settings.market.timezone)
    )


def _corrupt_everything_unpublished(
    frame: pd.DataFrame, settings: Settings
) -> pd.DataFrame:
    """Overwrite all data unknown at issue time, independently of the code."""
    out = frame.astype("float64")
    idx = pd.DatetimeIndex(out.index)
    issue = (
        (pd.Timestamp(TARGET - timedelta(days=1)) + pd.Timedelta(hours=11, minutes=40))
        .tz_localize(settings.market.timezone)
        .tz_convert("UTC")
    )
    actuals_known_until = issue - pd.Timedelta(
        minutes=settings.availability.actuals_lag_minutes
    )
    garbage = -9.0e9
    out.loc[idx >= TARGET_END, :] = garbage
    for column, rule in settings.availability.columns.items():
        if rule == "before_target_day":
            hidden = idx >= TARGET_START
        elif rule == "through_target_day":
            hidden = idx >= TARGET_END
        else:
            hidden = idx + QUARTER > actuals_known_until
        out.loc[hidden, column] = garbage
    return out


def test_issue_time_is_1140_local_on_the_day_before(settings: Settings) -> None:
    assert issue_time_utc(TARGET, settings) == pd.Timestamp(
        "2024-06-14 09:40", tz="UTC"
    )


def test_each_column_ends_where_its_publication_rule_says(settings: Settings) -> None:
    info = build_information_set(_market(settings), TARGET, settings)
    history = info.history

    assert settings.availability.actuals_lag_minutes == 180
    assert history.index.max() == TARGET_END - QUARTER
    for column in (
        "price_eur_mwh",
        "wind_onshore_forecast_mw",
        "solar_forecast_mw",
        "residual_load_forecast_mw",
    ):
        assert history[column].last_valid_index() == TARGET_START - QUARTER, column
    for column in ("load_forecast_mw", "price_product_minutes"):
        assert history[column].last_valid_index() == TARGET_END - QUARTER, column
    # 11:40 local less 180 minutes is 08:40 local, 06:40 UTC. The last period that
    # has ended by then is 06:15 to 06:30 UTC.
    for column in ("load_actual_mw", "solar_actual_mw", "residual_load_actual_mw"):
        assert history[column].last_valid_index() == pd.Timestamp(
            "2024-06-14 06:15", tz="UTC"
        ), column
    assert info.target_index[0] == TARGET_START
    assert len(info.target_index) == 96


def test_columns_without_a_rule_are_refused(settings: Settings) -> None:
    frame = _market(settings)
    frame["secret"] = 1.0
    with pytest.raises(ValueError, match="secret"):
        build_information_set(frame, TARGET, settings)


def test_lookback_limits_the_history(settings: Settings) -> None:
    info = build_information_set(_market(settings), TARGET, settings, lookback_days=2)
    assert info.history.index.min() == TARGET_START - pd.Timedelta(days=2)


ModelFactory = Callable[[Settings], Forecaster]
FACTORIES: dict[str, ModelFactory] = {
    "greedy": lambda s: GreedyForecaster(s.forecasting.quantiles),
    "naive": lambda s: build_baselines(s)[0],
    "seasonal naive": lambda s: build_baselines(s)[1],
}


@pytest.mark.parametrize("model_name", list(FACTORIES))
def test_forecasts_ignore_everything_unpublished_at_issue_time(
    settings: Settings, model_name: str
) -> None:
    model = FACTORIES[model_name](settings)
    clean = _market(settings)
    dirty = _corrupt_everything_unpublished(clean, settings)

    before = model.forecast(
        build_information_set(clean, TARGET, settings, model.lookback_days)
    )
    after = model.forecast(
        build_information_set(dirty, TARGET, settings, model.lookback_days)
    )

    pd.testing.assert_frame_equal(before.values, after.values)


def test_the_leakage_test_would_notice_a_change_in_published_data(
    settings: Settings,
) -> None:
    model = GreedyForecaster(settings.forecasting.quantiles)
    clean = _market(settings)
    changed = clean.copy()
    moment = TARGET_START - pd.Timedelta(hours=12)
    price = changed["price_eur_mwh"]
    changed["price_eur_mwh"] = price.where(changed.index != moment, price + 1.0)

    before = model.forecast(build_information_set(clean, TARGET, settings))
    after = model.forecast(build_information_set(changed, TARGET, settings))

    assert not before.values.equals(after.values)
