"""The fallback chain: which rung runs, what is skipped, and what is recorded."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import ForecastError, QuantileForecast, make_forecast
from src.forecasting.information import InformationSet, build_information_set
from src.pipeline import chain
from src.pipeline.readiness import FeedStatus, Readiness, check_readiness
from tests.fakes import synthetic_market

TARGET = date(2024, 5, 20)


def _why(result: chain.ChainResult) -> str:
    """Readable assertion message: every rung and what it did."""
    return "; ".join(f"{a.step}: {a.detail}" for a in result.attempts)


def _readiness(*missing: str) -> Readiness:
    feeds = tuple(
        FeedStatus(name, name not in missing, 96, 0 if name in missing else 96, name)
        for name in ("prices", "load_forecast", "weather", "fuels")
    )
    return Readiness(TARGET, datetime.now(UTC), feeds)


@dataclass
class FakeModel:
    """A forecaster that records its calls and answers with a flat fan."""

    settings: Settings
    name: str = "fake"
    fitted: list[date] = field(default_factory=list)

    @property
    def lookback_days(self) -> int | None:
        return 3

    @property
    def fit_lookback_days(self) -> int | None:
        return 3

    def fit(self, info: InformationSet) -> None:
        self.fitted.append(info.target_day)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        quantiles = self.settings.forecasting.quantiles
        values = pd.DataFrame(
            {
                f"q{int(q * 100):02d}": [50.0 + 10 * q] * len(info.target_index)
                for q in quantiles
            },
            index=info.target_index,
        )
        return make_forecast(self.name, info, values, quantiles)


@pytest.fixture
def market(settings: Settings) -> pd.DataFrame:
    return synthetic_market(settings, TARGET - timedelta(days=60), 62)


def test_steps_are_ordered_and_filtered_by_what_arrived() -> None:
    assert [s.name for s in chain.STEPS] == [
        "production",
        "no_weather",
        "seasonal_naive",
        "naive",
    ]
    assert [s.name for s in chain.steps_for(_readiness())] == [
        s.name for s in chain.STEPS
    ]
    assert [s.name for s in chain.steps_for(_readiness("weather"))] == [
        "no_weather",
        "seasonal_naive",
        "naive",
    ]
    # Yesterday's prices missing leaves only the step that needs no feed today.
    assert [s.name for s in chain.steps_for(_readiness("prices"))] == ["seasonal_naive"]


def test_the_supplied_model_takes_the_production_rung(
    settings: Settings, market: pd.DataFrame
) -> None:
    model = FakeModel(settings)

    result = chain.run_chain(market, TARGET, settings, _readiness(), model=model)

    assert result.step == "production" and not result.degraded
    assert result.label == "production model"
    # A registry model arrives fitted, so the chain must not train it again.
    assert model.fitted == []
    assert len(result.forecast.values) == 96
    assert [a.used for a in result.attempts] == [True]
    assert result.seconds >= 0.0


def test_a_missing_weather_feed_skips_the_production_rung(
    settings: Settings, market: pd.DataFrame
) -> None:
    model = FakeModel(settings)

    result = chain.run_chain(
        market, TARGET, settings, _readiness("weather", "load_forecast"), model=model
    )

    assert result.step == "seasonal_naive", _why(result)
    assert result.degraded and model.fitted == []
    skipped = {a.step: a.detail for a in result.attempts if not a.used}
    assert "waiting on load_forecast, weather" in skipped["production"]
    assert "waiting on load_forecast" in skipped["no_weather"]


def test_a_failing_rung_is_recorded_and_the_chain_moves_on(
    settings: Settings, market: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Broken(FakeModel):
        def fit(self, info: InformationSet) -> None:
            raise ForecastError("no training history")

    monkeypatch.setattr(
        chain,
        "STEPS",
        (
            chain.ChainStep("production", "production model", frozenset(), Broken),
            *chain.STEPS[1:],
        ),
    )

    result = chain.run_chain(market, TARGET, settings, _readiness())

    # The next rung down takes over, and the failure is kept in the record.
    assert result.step == "no_weather", _why(result)
    failed = next(a for a in result.attempts if a.step == "production")
    assert not failed.used and "no training history" in failed.detail


def test_the_chain_raises_when_no_rung_can_run(
    settings: Settings, market: pd.DataFrame
) -> None:
    empty = market.copy()
    empty[PRICE_SERIES] = float("nan")

    with pytest.raises(ForecastError, match="no chain step could forecast"):
        chain.run_chain(empty, TARGET, settings, _readiness("prices"))


def test_readiness_and_chain_agree_on_a_real_frame(
    settings: Settings, market: pd.DataFrame
) -> None:
    ready = check_readiness(market, TARGET, settings)
    info = build_information_set(market, TARGET, settings, 3)

    assert ready.ready
    assert [s.name for s in chain.steps_for(ready)] == [s.name for s in chain.STEPS]
    assert info.target_day == TARGET


def test_a_rung_that_fails_unexpectedly_is_stepped_past_not_fatal(
    settings: Settings, market: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desk must still bid when a rung breaks in a way nobody anticipated.

    A model that will not deserialise, a missing solver binary, a library that
    differs on the machine of the day: none of these raise ForecastError, and
    before this they ended the run with no forecast, no incident and no record.
    """

    class Unpickleable:
        """Stands in for a rung that breaks for a reason of its own."""

        name = "unpickleable"
        lookback_days = 10
        fit_lookback_days = 10

        def fit(self, info: InformationSet) -> None:
            raise AttributeError("numpy.ndarray has no attribute _reconstruct")

        def forecast(self, info: InformationSet) -> QuantileForecast:
            raise AssertionError("never reached")

    broken_rung = chain.STEPS[0]
    monkeypatch.setattr(
        chain,
        "STEPS",
        (
            chain.ChainStep(
                broken_rung.name,
                broken_rung.label,
                broken_rung.needs,
                lambda s: Unpickleable(),
            ),
            *chain.STEPS[1:],
        ),
    )

    result = chain.run_chain(market, TARGET, settings, _readiness())

    assert result.step != "production" and result.degraded
    broken = next(a for a in result.attempts if a.step == "production")
    assert not broken.used and "AttributeError" in broken.detail
