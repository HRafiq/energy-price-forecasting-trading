"""D1 experiment: blanking the weather feed, the outage model and the comparison."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import (
    QuantileForecast,
    make_forecast,
    quantile_column,
    quantile_columns,
)
from src.forecasting.information import InformationSet, build_information_set
from src.forecasting.models.gradient_boosting import (
    DEFAULT_LGBM_PARAMS,
    LightGBMConformalModel,
)
from src.forecasting.walkforward import (
    HoldoutAccessError,
    day_range,
    run_walk_forward,
)
from src.health.experiments import d1_missing_weather as d1
from src.trading.strategies import MEDIAN_FORECAST, PERFECT_FORESIGHT
from tests.fakes import synthetic_market
from tests.test_trading import day_frame

TZ = "Europe/Berlin"
TARGET = date(2024, 5, 20)
SMALL_LGBM = {
    **DEFAULT_LGBM_PARAMS,
    "n_estimators": 30,
    "num_leaves": 15,
    "min_child_samples": 20,
}


def _price(index: pd.DatetimeIndex) -> NDArray[np.float64]:
    local = index.tz_convert(TZ)
    clock = np.asarray(local.hour * 60 + local.minute, dtype=float)
    noise = np.random.default_rng(11).normal(0.0, 5.0, len(index))
    return 60.0 + 30.0 * np.sin(2 * np.pi * clock / 1440) + noise


@pytest.fixture(scope="module")
def market() -> pd.DataFrame:
    from src.config import load_settings

    settings = load_settings()
    frame = synthetic_market(settings, date(2024, 2, 1), 110, _price)
    rng = np.random.default_rng(5)
    for column in frame.columns:
        if column not in (PRICE_SERIES, "price_product_minutes"):
            frame[column] = rng.normal(100.0, 20.0, len(frame))
    return frame


def _model(settings: Settings) -> LightGBMConformalModel:
    return LightGBMConformalModel(
        settings, training_days=60, calibration_days=14, params=dict(SMALL_LGBM)
    )


def test_blanking_removes_only_the_delivery_days_weather(
    settings: Settings, market: pd.DataFrame
) -> None:
    info = build_information_set(market, TARGET, settings, lookback_days=5)
    blanked = d1.blank_weather(info)
    weather = d1.weather_columns(market)
    others = [c for c in market.columns if c not in weather]
    on_day = blanked.history.index >= info.target_index[0]

    assert weather and all(c.startswith("wx_") for c in weather)
    assert blanked.history.loc[on_day, weather].isna().all().all()
    pd.testing.assert_frame_equal(
        blanked.history.loc[~on_day, weather], info.history.loc[~on_day, weather]
    )
    pd.testing.assert_frame_equal(blanked.history[others], info.history[others])
    assert info.history.loc[on_day, weather].notna().any().any()


def test_weather_columns_are_required() -> None:
    with pytest.raises(ValueError, match="no weather columns"):
        d1.weather_columns(pd.DataFrame({"price": [1.0]}))


def test_outage_model_keeps_the_published_forecast_and_returns_the_outage(
    settings: Settings, market: pd.DataFrame
) -> None:
    days = day_range(TARGET, TARGET + timedelta(days=2))
    outage_model = d1.WeatherOutageModel(_model(settings))
    outage = run_walk_forward(market, outage_model, days, settings, refit_every_days=7)
    plain = run_walk_forward(market, _model(settings), days, settings, 7)

    published = pd.concat(outage_model.published)
    published.index.name = "timestamp_utc"
    assert d1.reproduction_gap(published, plain, settings.forecasting.quantiles) < 1e-9
    assert outage["target_day"].nunique() == 3
    assert outage["model"].iloc[0] == "lightgbm_conformal_weather_missing"
    quantiles = outage[[c for c in outage.columns if c.startswith("q")]].to_numpy()
    assert np.isfinite(quantiles).all()


class RecordingModel:
    """Keeps every information set it forecasts from.

    Its quantiles sit near 1000 when the delivery day's weather is blank and near
    0 otherwise, so the two forecasts cannot be confused.
    """

    name = "recording"
    lookback_days: int | None = 5
    fit_lookback_days: int | None = 5

    def __init__(self, settings: Settings) -> None:
        self.quantiles = settings.forecasting.quantiles
        self.histories: list[pd.DataFrame] = []

    def fit(self, info: InformationSet) -> None:
        return None

    def forecast(self, info: InformationSet) -> QuantileForecast:
        self.histories.append(info.history)
        weather = d1.weather_columns(info.history)
        on_day = info.history.index >= info.target_index[0]
        blank = bool(info.history.loc[on_day, weather].isna().all().all())
        level = 1000.0 if blank else 0.0
        raw = pd.DataFrame(
            {quantile_column(q): level + q for q in self.quantiles},
            index=info.target_index,
        )
        return make_forecast(self.name, info, raw, self.quantiles)


def test_outage_model_returns_the_forecast_made_on_the_blanked_information(
    settings: Settings, market: pd.DataFrame
) -> None:
    info = build_information_set(market, TARGET, settings, lookback_days=5)
    spy = RecordingModel(settings)
    outage = d1.WeatherOutageModel(spy)

    returned = outage.forecast(info)

    published_history, blanked_history = spy.histories
    weather = d1.weather_columns(market)
    on_day = info.history.index >= info.target_index[0]
    pd.testing.assert_frame_equal(published_history, info.history)
    assert blanked_history.loc[on_day, weather].isna().all().all()
    assert (returned.values.to_numpy(dtype="float64") >= 1000.0).all()
    (kept,) = outage.published
    columns = quantile_columns(settings.forecasting.quantiles)
    assert (kept[columns].to_numpy(dtype="float64") < 1000.0).all()
    assert (kept["target_day"] == TARGET).all()


def test_holdout_days_are_refused(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdout = settings.evaluation.holdout_start
    days = [holdout - timedelta(days=1), holdout]

    with pytest.raises(HoldoutAccessError):
        d1.check_days(days, settings)
    d1.check_days(days[:1], settings)

    data = settings.data.model_copy(update={"processed_dir": tmp_path})
    local = settings.model_copy(update={"data": data})
    comparison = tmp_path / "forecasts" / "comparison"
    comparison.mkdir(parents=True)
    saved = pd.DataFrame({"target_day": days, "q50": [1.0, 2.0]})
    for name in (d1.PRODUCTION, *d1.BASELINES):
        saved.to_parquet(comparison / f"{name}.parquet")

    def no_forecasts(*_: object, **__: object) -> pd.DataFrame:
        raise AssertionError("forecast before the hold-out check")

    monkeypatch.setattr(d1, "load_settings", lambda config=None: local)
    monkeypatch.setattr(d1, "run_walk_forward", no_forecasts)
    with pytest.raises(HoldoutAccessError):
        d1.main(["--workers", "1"])


def _saved(days: list[date], model: str) -> pd.DataFrame:
    return pd.DataFrame(
        {"target_day": days, "model": model, "q50": np.arange(len(days), dtype=float)}
    )


def test_cached_forecasts_are_reused_only_for_the_same_days_and_model(
    tmp_path: Path,
) -> None:
    days = [date(2024, 6, 1), date(2024, 6, 2)]
    path = tmp_path / "d1_forecasts_weather_missing.parquet"
    builds: list[list[date]] = []

    def build() -> pd.DataFrame:
        builds.append(days)
        return _saved(days, d1.OUTAGE_MODEL).assign(q50=[7.0, 8.0])

    _, reused = d1.cached_forecasts(path, build, days, d1.OUTAGE_MODEL)
    assert (reused, len(builds)) == (False, 1)
    frame, reused = d1.cached_forecasts(path, build, days, d1.OUTAGE_MODEL)
    assert (reused, len(builds)) == (True, 1)
    assert frame["q50"].tolist() == [7.0, 8.0]

    # Fewer days than requested: recomputed and saved again.
    _saved(days[:1], d1.OUTAGE_MODEL).to_parquet(path)
    _, reused = d1.cached_forecasts(path, build, days, d1.OUTAGE_MODEL)
    assert (reused, len(builds)) == (False, 2)
    assert pd.read_parquet(path)["q50"].tolist() == [7.0, 8.0]

    # Another model's forecasts under this name: recomputed.
    _saved(days, d1.FALLBACK_MODEL).to_parquet(path)
    _, reused = d1.cached_forecasts(path, build, days, d1.OUTAGE_MODEL)
    assert (reused, len(builds)) == (False, 3)

    # A companion file that no longer matches: recomputed.
    _, reused = d1.cached_forecasts(
        path, build, days, d1.OUTAGE_MODEL, also_valid=lambda: False
    )
    assert (reused, len(builds)) == (False, 4)


def test_published_forecasts_match_on_days_alone() -> None:
    days = [date(2024, 6, 1), date(2024, 6, 2)]
    published = pd.DataFrame({"target_day": days, "q50": [1.0, 2.0]})

    assert d1.matches_days(published, days, None)
    assert not d1.matches_days(published, days, d1.PRODUCTION)
    assert not d1.matches_days(published, days[:1], None)
    assert not d1.matches_days(published, [*days, date(2024, 6, 3)], None)


def test_reused_arms_keep_their_recorded_runtime(tmp_path: Path) -> None:
    path = tmp_path / "d1_missing_weather.json"
    assert d1.previous_run_seconds(path) == {}
    path.write_text("{", encoding="utf-8")
    assert d1.previous_run_seconds(path) == {}

    path.write_text(
        json.dumps(
            {"run_seconds": {"weather_missing": 1088.7, "fallback_no_weather": 495.1}}
        ),
        encoding="utf-8",
    )
    earlier = d1.previous_run_seconds(path)

    assert earlier == {"weather_missing": 1088.7, "fallback_no_weather": 495.1}
    assert d1.reused_run_seconds(
        {"weather_missing": True, "fallback_no_weather": False}, earlier
    ) == {"weather_missing": 1088.7}
    assert d1.reused_run_seconds({"weather_missing": True}, {}) == {}


def test_reproduction_gap_measures_and_refuses_missing_periods(
    settings: Settings,
) -> None:
    saved = day_frame(settings, date(2025, 11, 20))
    shifted = saved.assign(q95=saved["q95"] + 0.5)
    quantiles = settings.forecasting.quantiles
    assert d1.reproduction_gap(saved, saved, quantiles) == 0.0
    assert d1.reproduction_gap(shifted, saved, quantiles) == pytest.approx(0.5)
    with pytest.raises(ValueError, match="do not cover"):
        d1.reproduction_gap(saved, saved.iloc[4:], quantiles)


def test_traded_days_are_the_days_every_arm_can_trade(settings: Settings) -> None:
    days = [date(2025, 11, 20), date(2025, 11, 21), date(2025, 11, 22)]
    complete = pd.concat(day_frame(settings, day, seed=i) for i, day in enumerate(days))
    gappy = complete.copy()
    gap = gappy["q50"].to_numpy(copy=True)
    gap[gappy["target_day"].to_numpy() == days[1]] = np.nan
    gappy["q50"] = gap

    traded = d1.common_traded_days({"a": complete, "b": gappy}, days, settings)
    assert traded == [days[0], days[2]]


def test_summary_prices_each_arm_against_full(settings: Settings) -> None:
    day = date(2025, 11, 20)
    full = day_frame(settings, day, band_scale=10.0)
    wide = day_frame(settings, day, band_scale=40.0)

    def pnl(median: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "target_day": [day, day],
                "strategy": [PERFECT_FORESIGHT.name, MEDIAN_FORECAST.name],
                "pnl_eur": [100.0, median],
            }
        )

    table = d1.summarise(
        {"full": full, "weather_missing": wide},
        {"full": pnl(90.0), "weather_missing": pnl(70.0)},
        settings,
    )
    assert table.loc["full", "pinball_vs_full"] == 0.0
    assert table.loc["weather_missing", "capture"] == pytest.approx(0.7)
    assert table.loc["weather_missing", "capture_points_vs_full"] == pytest.approx(
        -20.0
    )
    assert table.loc["weather_missing", "pnl_eur_vs_full"] == pytest.approx(-20.0)
    coverage = table["coverage_90"].to_numpy(dtype="float64")
    wide, full_coverage = (
        coverage[list(table.index).index("weather_missing")],
        coverage[0],
    )
    assert wide >= full_coverage
