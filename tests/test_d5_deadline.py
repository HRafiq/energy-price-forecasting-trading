"""D5 experiment: failure draws, the fallback chain, timing and incidents."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.config import Settings
from src.forecasting.walkforward import HoldoutAccessError
from src.health.experiments import d5_deadline as d5
from src.health.incidents import load_incidents, upsert_incidents

RUNTIMES = {
    "full": 1.2,
    "fallback_no_weather": 1.0,
    "naive_previous_day": 0.3,
    "seasonal_naive_previous_week": 0.3,
}


def _days(n: int) -> list[date]:
    return [date(2025, 1, 1) + timedelta(days=i) for i in range(n)]


def _daily(days: list[date]) -> pd.DataFrame:
    earnings = {
        "full": 100.0,
        "fallback_no_weather": 90.0,
        "naive_previous_day": 60.0,
        "seasonal_naive_previous_week": 70.0,
    }
    return pd.DataFrame(
        [
            {
                "target_day": day,
                "arm": arm,
                "pinball": 10.0 - pnl / 20,
                "pnl_eur": pnl,
                "perfect_foresight_pnl_eur": 120.0,
            }
            for day in days
            for arm, pnl in earnings.items()
        ]
    )


def test_failure_draws_are_reproducible_and_near_their_rates() -> None:
    days = _days(4000)
    first, second = d5.draw_failures(days), d5.draw_failures(days)
    pd.testing.assert_frame_equal(first, second)
    for column, rate in d5.FAILURE_RATES.items():
        assert first[column].mean() == pytest.approx(rate, abs=0.012)
    assert not d5.draw_failures(days, seed=7).equals(first)


@pytest.mark.parametrize(
    ("failures", "step"),
    [
        (d5.Failures(), "full"),
        (d5.Failures(weather_late=True), "fallback_no_weather"),
        (d5.Failures(model_fails=True), "naive_previous_day"),
        (d5.Failures(weather_late=True, model_fails=True), "naive_previous_day"),
        (d5.Failures(prices_late=True), "seasonal_naive_previous_week"),
        (
            d5.Failures(weather_late=True, model_fails=True, prices_late=True),
            "seasonal_naive_previous_week",
        ),
    ],
)
def test_chain_takes_the_first_step_that_can_run(
    failures: d5.Failures, step: str
) -> None:
    assert d5.choose_step(failures) == step


def test_submission_time_counts_waits_and_attempted_steps() -> None:
    assert d5.submission_minutes(d5.Failures(), RUNTIMES) == pytest.approx(1.2 / 60)
    # Weather late: wait 5 minutes, skip the full model, run the no-weather model.
    assert d5.submission_minutes(
        d5.Failures(weather_late=True), RUNTIMES
    ) == pytest.approx(5 + 1.0 / 60)
    # Model step fails: both models ran and failed, then naive previous day.
    assert d5.submission_minutes(
        d5.Failures(model_fails=True), RUNTIMES
    ) == pytest.approx((1.2 + 1.0 + 0.3) / 60)
    # Prices late: wait 5 minutes, everything but seasonal naive is skipped.
    assert d5.submission_minutes(
        d5.Failures(prices_late=True), RUNTIMES
    ) == pytest.approx(5 + 0.3 / 60)


def test_submission_time_for_combined_failures() -> None:
    # Weather and prices late: two waits, and only seasonal naive runs.
    assert d5.submission_minutes(
        d5.Failures(weather_late=True, prices_late=True), RUNTIMES
    ) == pytest.approx(10 + 0.3 / 60)
    # Weather late and the model failing: one wait, the full model skipped, the
    # model without weather ran and failed, then naive previous day.
    assert d5.submission_minutes(
        d5.Failures(weather_late=True, model_fails=True), RUNTIMES
    ) == pytest.approx(5 + (1.0 + 0.3) / 60)


def test_a_submission_after_the_gate_is_not_on_time(settings: Settings) -> None:
    slow = {**RUNTIMES, "fallback_no_weather": 16 * 60.0}
    days = _days(2)
    failures = pd.DataFrame(
        {
            "target_day": days,
            "weather_late": [False, True],
            "model_fails": [False, False],
            "prices_late": [False, False],
        }
    )

    chain = d5.run_chain(failures, _daily(days), slow)

    assert chain["on_time"].tolist() == [True, False]
    assert chain.loc[1, "submitted_minutes_after_issue"] == pytest.approx(21.0)
    assert d5.summarise(chain)["on_time_share_with_chain"] == 0.5
    (incident,) = d5.step_incidents(chain, settings)
    assert "after the 12:00 gate" in incident.detail
    assert incident.action.endswith("forecast issued 12:01")


def test_prices_late_with_a_failing_model_is_late_data_in_whole_seconds(
    settings: Settings,
) -> None:
    runtimes = {
        **RUNTIMES,
        "fallback_no_weather": 1.2345,
        "seasonal_naive_previous_week": 0.2345,
    }
    days = _days(2)
    failures = pd.DataFrame(
        {
            "target_day": days,
            "weather_late": [False, True],
            "model_fails": [True, False],
            "prices_late": [True, False],
        }
    )

    chain = d5.run_chain(failures, _daily(days), runtimes)
    prices, weather = d5.step_incidents(chain, settings)

    assert chain["step"].tolist() == [
        "seasonal_naive_previous_week",
        "fallback_no_weather",
    ]
    assert prices.type == "late_data"
    assert "model step failed" in prices.detail
    assert "yesterday's prices late" in prices.detail
    for incident in (prices, weather):
        assert incident.detected_utc.microsecond == 0
        assert incident.action.startswith("Simulated fallback: ")
    # 11:40 plus a 5 minute wait plus 1.2345 s, cut to the second.
    assert weather.action.endswith("forecast issued 11:45")
    assert weather.detected_utc.second == 1


def test_realised_failure_rates_are_stated_beside_the_nominal_ones() -> None:
    days = _days(4)
    failures = pd.DataFrame(
        {
            "target_day": days,
            "weather_late": [True, False, False, False],
            "model_fails": [False, False, False, False],
            "prices_late": [True, True, False, False],
        }
    )

    realised = d5.realised_failure_rates(failures)
    chain = d5.run_chain(failures, _daily(days), RUNTIMES)
    text = d5._markdown(d5.summarise(chain), RUNTIMES, realised)

    assert realised == {"weather_late": 0.25, "model_fails": 0.0, "prices_late": 0.5}
    assert "the seed drew 25.0%, 0.0% and 50.0%" in text
    assert "measured on the machine that ran this experiment" in text


def test_holdout_days_are_refused(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdout = settings.evaluation.holdout_start
    days = [holdout - timedelta(days=1), holdout]

    with pytest.raises(HoldoutAccessError):
        d5.check_days(days, settings)
    d5.check_days(days[:1], settings)

    def no_models(_: Settings) -> dict[str, object]:
        raise AssertionError("models built before the hold-out check")

    monkeypatch.setattr(d5, "_chain_models", no_models)
    with pytest.raises(HoldoutAccessError):
        d5.measure_runtimes(pd.DataFrame(), days, settings)

    data = settings.data.model_copy(update={"processed_dir": tmp_path})
    local = settings.model_copy(update={"data": data})
    (tmp_path / "experiments").mkdir()
    _daily(days).to_parquet(
        tmp_path / "experiments" / "d1_missing_weather_daily.parquet"
    )

    def no_runtimes(*_: object) -> dict[str, float]:
        raise AssertionError("runtimes measured before the hold-out check")

    monkeypatch.setattr(d5, "load_settings", lambda config=None: local)
    monkeypatch.setattr(d5, "measure_runtimes", no_runtimes)
    with pytest.raises(HoldoutAccessError):
        d5.main([])


class _FakeStep:
    lookback_days: int | None = None
    fit_lookback_days: int | None = None

    def fit(self, info: object) -> None:
        return None

    def forecast(self, info: object) -> None:
        return None


def test_runtimes_are_measured_on_inputs_cut_before_the_holdout(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    holdout = settings.evaluation.holdout_start
    cut = pd.Timestamp(settings.market.local_midnight_utc(holdout))
    index = pd.date_range(
        cut - pd.Timedelta(days=3), cut + pd.Timedelta(days=3), freq="1h"
    )
    inputs = pd.DataFrame({"x": 1.0}, index=index)
    latest: list[pd.Timestamp] = []

    def fake_information_set(
        frame: pd.DataFrame, day: date, _: Settings, lookback: int | None
    ) -> object:
        latest.append(pd.Timestamp(frame.index.max()))
        return object()

    monkeypatch.setattr(
        d5, "_chain_models", lambda _: {name: _FakeStep() for name in d5.CHAIN}
    )
    monkeypatch.setattr(d5, "build_information_set", fake_information_set)

    runtimes = d5.measure_runtimes(
        inputs, [holdout - timedelta(days=2), holdout - timedelta(days=1)], settings
    )

    assert set(runtimes) == set(d5.CHAIN)
    assert latest
    assert max(latest) < cut
    assert d5.before_holdout(inputs, settings).index.max() < cut


def test_run_chain_prices_each_day_and_the_counterfactual() -> None:
    days = _days(3)
    failures = pd.DataFrame(
        {
            "target_day": days,
            "weather_late": [False, True, False],
            "model_fails": [False, False, True],
            "prices_late": [False, False, False],
        }
    )
    chain = d5.run_chain(failures, _daily(days), RUNTIMES)

    assert chain["step"].tolist() == [
        "full",
        "fallback_no_weather",
        "naive_previous_day",
    ]
    assert chain["on_time"].all()
    assert chain["pnl_eur"].tolist() == [100.0, 90.0, 60.0]
    assert chain["pnl_without_chain_eur"].tolist() == [100.0, 0.0, 0.0]
    summary = d5.summarise(chain)
    assert summary["on_time_share_with_chain"] == 1.0
    assert summary["on_time_share_without_chain"] == pytest.approx(1 / 3)
    assert summary["fallback_by_step"] == {
        "fallback_no_weather": 1,
        "naive_previous_day": 1,
        "seasonal_naive_previous_week": 0,
    }
    assert summary["capture_chain"] == pytest.approx(250 / 360)


def test_fallback_days_write_idempotent_incidents(
    settings: Settings, tmp_path: Path
) -> None:
    days = _days(3)
    failures = pd.DataFrame(
        {
            "target_day": days,
            "weather_late": [False, True, False],
            "model_fails": [False, False, True],
            "prices_late": [False, False, False],
        }
    )
    chain = d5.run_chain(failures, _daily(days), RUNTIMES)
    incidents = d5.step_incidents(chain, settings)

    assert [i.type for i in incidents] == ["late_data", "pipeline"]
    assert all(i.source == "d5_deadline" and i.status == "resolved" for i in incidents)
    weather = incidents[0]
    assert weather.delivery_day == days[1]
    assert (
        weather.action == "Simulated fallback: model without weather used; "
        "forecast issued 11:45"
    )
    assert weather.metrics["pnl_lost_eur"] == pytest.approx(10.0)

    path = tmp_path / "incidents.jsonl"
    upsert_incidents(incidents, path)
    first = path.read_bytes()
    upsert_incidents(d5.step_incidents(chain, settings), path)
    assert path.read_bytes() == first
    assert len(load_incidents(path)) == 2
