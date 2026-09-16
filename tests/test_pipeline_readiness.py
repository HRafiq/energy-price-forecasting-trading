"""The pipeline's readiness check: which feeds arrived for a delivery day."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CARBON_COLUMN, GAS_COLUMN, PRICE_SERIES, Settings
from src.pipeline import readiness as rd
from src.timegrid import local_day_bounds_utc
from tests.fakes import synthetic_market

TARGET = date(2024, 5, 20)


@pytest.fixture
def market(settings: Settings) -> pd.DataFrame:
    """Three complete days around the target day."""
    return synthetic_market(settings, TARGET - timedelta(days=1), 3)


def _blank(
    frame: pd.DataFrame, columns: list[str], day: date, settings: Settings
) -> pd.DataFrame:
    start, end = local_day_bounds_utc(day, settings.market.timezone)
    out = frame.copy()
    on_day = (out.index >= start) & (out.index < end)
    out.loc[on_day, columns] = np.nan
    return out


def test_every_feed_present_is_ready(settings: Settings, market: pd.DataFrame) -> None:
    result = rd.check_readiness(market, TARGET, settings)

    assert result.ready and result.missing == ()
    assert [feed.name for feed in result.feeds] == list(rd.FEED_NAMES)
    assert all(feed.expected == feed.present for feed in result.feeds[:3])
    assert result.as_dict()["target_day"] == str(TARGET)
    assert result.checked_utc.utcoffset() == timedelta(0)


def test_missing_prices_for_the_issue_day(
    settings: Settings, market: pd.DataFrame
) -> None:
    frame = _blank(market, [PRICE_SERIES], TARGET - timedelta(days=1), settings)

    result = rd.check_readiness(frame, TARGET, settings)

    assert not result.ready and result.missing == ("prices",)
    prices = result.feeds[0]
    assert (prices.present, prices.expected) == (0, 96)
    assert "96 missing" in prices.detail


def test_a_partly_published_load_forecast_is_not_ready(
    settings: Settings, market: pd.DataFrame
) -> None:
    start, _ = local_day_bounds_utc(TARGET, settings.market.timezone)
    frame = market.copy()
    late = frame.index >= start + pd.Timedelta(hours=18)
    frame.loc[late, "load_forecast_mw"] = np.nan

    result = rd.check_readiness(frame, TARGET, settings)

    load = next(f for f in result.feeds if f.name == "load_forecast")
    assert not result.ready and load.present == 72 and load.expected == 96


def test_missing_weather_is_reported_on_its_own(
    settings: Settings, market: pd.DataFrame
) -> None:
    frame = _blank(market, rd.weather_columns(market), TARGET, settings)

    result = rd.check_readiness(frame, TARGET, settings)

    assert result.missing == ("weather",)
    assert next(f for f in result.feeds if f.name == "prices").ready


def test_fuels_report_the_last_settlement_before_delivery(
    settings: Settings, market: pd.DataFrame
) -> None:
    ready = rd.check_readiness(market, TARGET, settings)
    fuels = next(f for f in ready.feeds if f.name == "fuels")
    assert fuels.ready and "days before delivery" in fuels.detail

    empty = market.copy()
    empty[[GAS_COLUMN, CARBON_COLUMN]] = np.nan
    missing = rd.check_readiness(empty, TARGET, settings)
    assert missing.missing == ("fuels",)
    assert "no gas or carbon price yet" in missing.feeds[-1].detail


def test_command_exits_one_until_every_feed_arrives(
    settings: Settings,
    market: pd.DataFrame,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "inputs.parquet"
    market.to_parquet(path)
    data = settings.data.model_copy(update={"inputs_file": path.name})
    monkeypatch.setattr(
        rd,
        "load_settings",
        lambda config=None: settings.model_copy(update={"data": data}),
    )
    monkeypatch.setattr(type(data), "inputs_path", property(lambda self: path))

    assert rd.main(["--day", str(TARGET)]) == 0

    gap = _blank(market, [PRICE_SERIES], TARGET - timedelta(days=1), settings)
    gap.to_parquet(path)
    assert rd.main(["--day", str(TARGET), "--json"]) == 1
