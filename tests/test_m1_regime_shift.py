"""M1 regime shift experiment: refit schedules, scoring, aggregation and outputs."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings
from src.features.build import FEATURE_GROUPS
from src.forecasting.base import QuantileForecast, make_forecast, quantile_column
from src.forecasting.evaluate import pinball
from src.forecasting.information import InformationSet
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.walkforward import HoldoutAccessError, day_range
from src.health.experiments.m1_regime_shift import (
    ARMS,
    Arm,
    build_forecaster,
    check_days,
    daily_metrics,
    dispatch_pnl,
    experiment_features,
    experiment_payload,
    load_checkpoint,
    main,
    min_rolling,
    period_table,
    refit_interval,
    report,
    rolling_coverage,
    run_arm,
    setup_description,
    tradable_days,
    training_range,
    write_atomically,
)
from tests.fakes import day_clock_price, synthetic_market

ARM = {arm.name: arm for arm in ARMS}
#: A fixed fan for hand-checked scores: q05 = 0 up to q95 = 60.
LEVELS = dict(
    zip((0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95), range(0, 70, 10), strict=True)
)


class SpyForecaster:
    name = "spy"
    lookback_days: int | None = 3
    fit_lookback_days: int | None = 5

    def __init__(self, settings: Settings) -> None:
        self.quantiles = settings.forecasting.quantiles
        self.fit_days: list[date] = []

    def fit(self, info: InformationSet) -> None:
        self.fit_days.append(info.target_day)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        raw = pd.DataFrame(
            {quantile_column(q): q for q in self.quantiles}, index=info.target_index
        )
        return make_forecast(self.name, info, raw, self.quantiles)


def _fits(settings: Settings, arm: Arm, days: list[date]) -> list[date]:
    frame = synthetic_market(
        settings,
        days[0] - timedelta(days=6),
        len(days) + 7,
        day_clock_price(settings.market.timezone),
    )
    spy = SpyForecaster(settings)
    out = run_arm(frame, arm, days, settings, forecaster=spy)
    assert sorted(set(out["target_day"])) == days
    assert set(out["arm"]) == {arm.name}
    return spy.fit_days


# --- arms ---------------------------------------------------------------------


def test_frozen_arm_fits_exactly_once_over_the_whole_span(settings: Settings) -> None:
    days = day_range(date(2021, 1, 1), date(2021, 4, 10))

    assert _fits(settings, ARM["frozen"], days) == [date(2021, 1, 1)]
    full_span = day_range(date(2021, 1, 1), date(2023, 12, 31))
    assert refit_interval(ARM["frozen"], full_span) > len(full_span) - 1


def test_quarterly_and_monthly_arms_refit_on_their_schedule(
    settings: Settings,
) -> None:
    days = day_range(date(2021, 1, 1), date(2021, 4, 10))

    quarterly = _fits(settings, ARM["quarterly"], days)
    monthly = _fits(settings, ARM["monthly"], days)

    assert quarterly == [date(2021, 1, 1), date(2021, 4, 2)]
    assert monthly == [date(2021, 1, 1) + timedelta(days=28 * k) for k in range(4)]
    assert refit_interval(ARM["quarterly"], days) == 91
    assert refit_interval(ARM["naive_previous_day"], days) == 1


def test_lightgbm_arms_share_the_frozen_setup_without_weather(
    settings: Settings,
) -> None:
    names = experiment_features()
    assert not set(names) & set(FEATURE_GROUPS["weather"])
    assert set(FEATURE_GROUPS["fuels"]) <= set(names)

    for arm in ("frozen", "quarterly", "monthly"):
        model = build_forecaster(ARM[arm], settings)
        assert isinstance(model, LightGBMConformalModel)
        assert model.training_days == 730
        assert model.calibration_days == 42
        assert model._names == names
    assert build_forecaster(ARM["naive_previous_day"], settings).name == (
        "naive_previous_day"
    )


def test_hold_out_days_are_refused_before_any_fit(settings: Settings) -> None:
    holdout = settings.evaluation.holdout_start
    days = day_range(holdout - timedelta(days=2), holdout)
    spy = SpyForecaster(settings)
    frame = synthetic_market(settings, holdout - timedelta(days=10), 12)

    with pytest.raises(HoldoutAccessError):
        check_days(days, settings)
    with pytest.raises(HoldoutAccessError):
        run_arm(frame, ARM["frozen"], days, settings, forecaster=spy)
    assert spy.fit_days == []
    with pytest.raises(HoldoutAccessError):
        main(
            [
                "--first-day",
                str(holdout - timedelta(days=1)),
                "--last-day",
                str(holdout),
            ]
        )
    with pytest.raises(HoldoutAccessError):
        dispatch_pnl(pd.DataFrame(), days, settings)
    check_days(days[:-1], settings)


def test_checkpoint_is_reused_only_for_the_same_days(tmp_path: Path) -> None:
    days = [date(2021, 1, 1), date(2021, 1, 2)]
    frame = pd.DataFrame({"target_day": days, "q50": [1.0, 2.0]})
    path = tmp_path / "m1_regime_forecasts_frozen.parquet"

    assert load_checkpoint(path, days) is None
    write_atomically(path, frame.to_parquet)

    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    saved = load_checkpoint(path, days)
    assert saved is not None
    assert list(saved["q50"]) == [1.0, 2.0]
    assert load_checkpoint(path, [*days, date(2021, 1, 3)]) is None


# --- daily scores and trading ----------------------------------------------------


def _day(settings: Settings, day: date, actual: np.ndarray[Any, Any]) -> pd.DataFrame:
    start = settings.market.local_midnight_utc(day)
    index = pd.date_range(start, periods=len(actual), freq="15min", name="t")
    frame = pd.DataFrame(
        {quantile_column(q): v for q, v in LEVELS.items()}, index=index
    )
    frame["actual"] = actual
    frame["target_day"] = day
    frame["price_product_minutes"] = 15
    return frame


def test_daily_metrics_score_each_day(settings: Settings) -> None:
    calm = np.array([30.0, 55.0, 65.0, -5.0])
    spike = np.array([30.0, 30.0, 30.0, 250.0])
    forecasts = pd.concat(
        [
            _day(settings, date(2021, 1, 1), calm),
            _day(settings, date(2021, 1, 2), spike),
        ]
    )

    daily = daily_metrics(forecasts, settings).set_index("target_day")

    first = daily.loc[date(2021, 1, 1)]
    assert first["coverage_90"] == pytest.approx(0.5)
    assert first["coverage_50"] == pytest.approx(0.25)
    assert first["mae_q50"] == pytest.approx((0 + 25 + 35 + 35) / 4)
    expected = np.mean(
        [pinball(calm, np.full(4, level), q).mean() for q, level in LEVELS.items()]
    )
    assert first["pinball"] == pytest.approx(expected)
    assert not first["spike_day"]
    assert bool(daily.at[date(2021, 1, 2), "spike_day"])
    assert float(daily.at[date(2021, 1, 2), "coverage_90"]) == pytest.approx(0.75)


def _full_day(settings: Settings, day: date, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    hours = np.arange(96) / 4
    shape = 60 + 40 * np.sin((hours - 8) / 24 * 2 * np.pi)
    frame = _day(settings, day, shape + rng.normal(0, 15, 96))
    for q, z in zip(
        sorted(LEVELS), (-1.6, -1.3, -0.7, 0.0, 0.7, 1.3, 1.6), strict=True
    ):
        frame[quantile_column(q)] = shape + 20 * z
    return frame


def test_tradable_days_skip_incomplete_days(settings: Settings) -> None:
    whole = _full_day(settings, date(2021, 3, 1), 1)
    partial = _full_day(settings, date(2021, 3, 2), 2).iloc[:90]

    days, skipped = tradable_days(pd.concat([whole, partial]), settings)

    assert days == [date(2021, 3, 1)]
    assert skipped == {date(2021, 3, 2): "incomplete periods"}


def test_dispatch_settles_median_and_perfect_foresight(settings: Settings) -> None:
    days = [date(2021, 3, 1), date(2021, 3, 2)]
    forecasts = pd.concat([_full_day(settings, d, i) for i, d in enumerate(days)])

    pnl = dispatch_pnl(forecasts, days, settings, workers=1)

    assert list(pnl["target_day"]) == days
    assert (pnl["perfect_foresight_pnl_eur"] > 0).all()
    assert (pnl["pnl_eur"] <= pnl["perfect_foresight_pnl_eur"] + 1e-2).all()


def test_dispatch_retries_failed_days_once_one_at_a_time(settings: Settings) -> None:
    days = [date(2021, 3, 1), date(2021, 3, 2)]
    calls: list[tuple[list[date], int]] = []

    def runner(
        forecasts: pd.DataFrame,
        run_days: list[date],
        *args: Any,
        workers: int,
        **kw: Any,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[date, str]]:
        calls.append((list(run_days), workers))
        failed = {days[1]: "solver failed"} if workers > 1 else {}
        done = [d for d in run_days if d not in failed]
        pnl = pd.DataFrame(
            [
                {"target_day": d, "strategy": s, "pnl_eur": v}
                for d in done
                for s, v in (("perfect_foresight", 10.0), ("median_forecast", 6.0))
            ]
        )
        return pd.DataFrame(), pnl, failed

    pnl = dispatch_pnl(pd.DataFrame(), days, settings, workers=4, runner=runner)

    assert calls == [(days, 4), ([days[1]], 1)]
    assert list(pnl["target_day"]) == days
    assert list(pnl["pnl_eur"]) == [6.0, 6.0]


# --- aggregation and payload ------------------------------------------------------


def _daily(first: date, n: int, arms: tuple[str, ...]) -> pd.DataFrame:
    rows = []
    for arm in arms:
        for i, day in enumerate(day_range(first, first + timedelta(days=n - 1))):
            rows.append(
                {
                    "target_day": day,
                    "arm": arm,
                    "coverage_90": 0.0 if arm == "frozen" and 30 <= i <= 33 else 1.0,
                    "coverage_50": 0.5,
                    "pinball": float(i),
                    "mae_q50": 2.0,
                    "spike_day": i % 10 == 0,
                    "pnl_eur": 1.0,
                    "perfect_foresight_pnl_eur": 4.0,
                }
            )
    return pd.DataFrame(rows)


def test_rolling_coverage_and_its_minimum() -> None:
    first = date(2021, 1, 1)
    daily = _daily(first, 120, ("frozen", "monthly"))

    rolling = rolling_coverage(daily, window=28)

    assert rolling["frozen"].iloc[:27].isna().all()
    assert rolling["frozen"].iloc[27] == pytest.approx(1.0)
    lowest = min_rolling(rolling)
    value, window_end = lowest["frozen"]
    assert value == pytest.approx(24 / 28)
    assert window_end == first + timedelta(days=33)
    assert lowest["monthly"][0] == pytest.approx(1.0)

    gap = daily[daily["target_day"] != first + timedelta(days=50)]
    with_gap = rolling_coverage(gap, window=28)["monthly"]
    # Every window that contains the missing day 50 ends on days 50 to 77.
    assert len(with_gap) == 120
    assert with_gap.iloc[27:50].notna().all()
    assert with_gap.iloc[50:78].isna().all()
    assert with_gap.iloc[78:].notna().all()


def test_period_tables_average_scores_and_sum_pnl() -> None:
    daily = _daily(date(2021, 3, 30), 4, ("frozen",))
    daily.loc[3, "pnl_eur"] = np.nan

    quarters = period_table(daily, "quarter").set_index("period")
    total = period_table(daily, "total").set_index("period")

    assert list(quarters.index) == ["2021Q1", "2021Q2"]
    assert quarters.at["2021Q1", "days"] == 2
    assert quarters.at["2021Q1", "pinball"] == pytest.approx(0.5)
    assert quarters.at["2021Q1", "spike_days"] == 1
    assert quarters.at["2021Q1", "capture_ratio"] == pytest.approx(0.25)
    assert quarters.at["2021Q2", "traded_days"] == 1
    assert quarters.at["2021Q2", "pnl_eur"] == pytest.approx(1.0)
    assert total.at["total", "days"] == 4
    assert total.at["total", "pnl_eur"] == pytest.approx(3.0)
    assert total.at["total", "perfect_foresight_pnl_eur"] == pytest.approx(12.0)
    assert list(period_table(daily, "year")["period"]) == ["2021"]
    with pytest.raises(ValueError):
        period_table(daily, "month")


def test_payload_has_the_dashboard_shape(settings: Settings) -> None:
    first = date(2021, 1, 1)
    arms = tuple(arm.name for arm in ARMS)
    daily = _daily(first, 120, arms)
    setup = setup_description(settings, first, first + timedelta(days=119), ARMS)

    payload = experiment_payload(daily, settings, setup)
    text = json.dumps(payload, allow_nan=False)
    decoded = json.loads(text)

    assert decoded["coverage_target"] == 0.9
    assert decoded["arms"] == list(arms)
    rolling = decoded["rolling_coverage_90"]
    assert rolling["window_days"] == 28
    assert len(rolling["dates"]) == 120 - 27
    assert rolling["dates"][0] == (first + timedelta(days=27)).isoformat()
    assert set(rolling["arms"]) == set(arms)
    assert all(len(v) == len(rolling["dates"]) for v in rolling["arms"].values())
    assert (
        decoded["min_rolling_coverage_90"]["frozen"]["window_end"]
        == (first + timedelta(days=33)).isoformat()
    )
    assert {row["period"] for row in decoded["quarterly"]} == {"2021Q1", "2021Q2"}
    assert len(decoded["total"]) == len(arms)
    assert decoded["setup"]["feature_groups"] == [
        g for g in FEATURE_GROUPS if g != "weather"
    ]
    assert [a["refit_every_days"] for a in decoded["setup"]["arms"]] == [
        None,
        91,
        28,
        1,
    ]
    assert len(text) < 200_000
    assert PRICE_SERIES not in text


def test_training_range_reads_only_the_frozen_training_window(
    settings: Settings,
) -> None:
    first = date(2021, 1, 1)
    start = settings.market.local_midnight_utc(first - timedelta(days=731))
    end = settings.market.local_midnight_utc(first + timedelta(days=1))
    index = pd.date_range(start, end, freq="15min", inclusive="left")
    frame = pd.DataFrame({PRICE_SERIES: 10.0}, index=index)
    frame.iloc[:96, 0] = 999.0  # the day before the window
    frame.loc[index >= settings.market.local_midnight_utc(first), PRICE_SERIES] = 500.0
    forecasts = pd.DataFrame(
        {"q50": [20.0, 30.0], "q95": [40.0, 50.0], "actual": [500.0, 450.0]}
    )

    limits = training_range(forecasts, frame, first, settings)

    assert limits["training_max"] == pytest.approx(10.0)
    assert limits["training_mean"] == pytest.approx(10.0)
    assert limits["q50_max"] == pytest.approx(30.0)
    assert limits["q95_max"] == pytest.approx(50.0)
    assert limits["actual_max"] == pytest.approx(500.0)


def test_report_states_measured_limits_and_the_prior_expectation(
    settings: Settings,
) -> None:
    first = date(2021, 1, 1)
    daily = _daily(first, 120, tuple(arm.name for arm in ARMS))
    context = pd.DataFrame(
        {
            "mean_price": [50.0, 60.0],
            "max_price": [137.0, 140.0],
            "spike_days": [0, 0],
            "mean_gas": [18.0, 25.0],
        },
        index=["2021Q1", "2021Q2"],
    )
    limits = {
        "training_mean": 34.2,
        "training_p99": 73.1,
        "training_max": 200.04,
        "q50_max": 100.349,
        "q95_max": 138.2,
        "actual_max": 871.0,
    }

    text = report(
        daily, context, settings, first, first + timedelta(days=119), {}, {}, limits
    )

    assert "its median forecast peaked at €100.35" in text
    assert "highest €200.04" in text
    assert "never exceeded" not in text
    assert "## Against the expectation recorded before running" in text
    assert "frozen coverage near 58%" in text
    assert "86% to 90%" in text
    assert not {"\u2013", "\u2014"} & set(text)
