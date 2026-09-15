"""M2 drift monitor: rolling scores, the frozen threshold rule and episodes."""

from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.health.drift import (
    SOURCE,
    AlertEpisode,
    Thresholds,
    alert_episodes,
    apply_thresholds,
    daily_scores,
    drift_incidents,
    fit_thresholds,
    longest_run,
    rolling_scores,
    select_coverage_threshold,
    select_pinball_threshold,
)
from src.health.experiments import m2_drift as m2
from src.health.incidents import IN_SAMPLE_NOTE

QUANTILE_COLUMNS = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


def _day_rows(day: date, center: float, actual: list[float]) -> pd.DataFrame:
    index = pd.date_range(pd.Timestamp(day, tz="UTC"), periods=len(actual), freq="6h")
    offsets = np.array([-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0])
    frame = pd.DataFrame(
        {col: center + offsets[i] for i, col in enumerate(QUANTILE_COLUMNS)},
        index=index,
    )
    frame["actual"] = actual
    frame["target_day"] = day
    return frame


def _random_forecasts(
    first: date, days: int, seed: int, spread: float = 8.0
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    parts = [
        _day_rows(
            first + timedelta(days=i),
            50.0,
            list(50.0 + rng.normal(0.0, spread * (1.0 + (i % 17) / 10.0), 4)),
        )
        for i in range(days)
    ]
    return pd.concat(parts)


def _daily(
    coverage: list[float], pinball: list[float], periods: list[int]
) -> pd.DataFrame:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(len(coverage))]
    return pd.DataFrame(
        {"periods": periods, "coverage_90": coverage, "pinball": pinball},
        index=pd.Index(days, name="target_day"),
    )


# --- daily and rolling -----------------------------------------------------


def test_daily_scores_coverage_pinball_and_untraded_days(settings: Settings) -> None:
    quantiles = settings.forecasting.quantiles
    constant = _day_rows(date(2025, 1, 1), 10.0, [12.0] * 4)
    for column in QUANTILE_COLUMNS:
        constant[column] = 10.0
    inside_half = _day_rows(date(2025, 1, 2), 50.0, [50.0, 55.0, 70.0, 20.0])
    missing = _day_rows(date(2025, 1, 3), 50.0, [50.0, np.nan, 50.0, 50.0])

    daily = daily_scores(pd.concat([constant, inside_half, missing]), quantiles)

    assert list(daily.index) == [date(2025, 1, 1), date(2025, 1, 2)]
    assert daily["periods"].tolist() == [4, 4]
    assert daily["coverage_90"].tolist() == [0.0, 0.5]
    # Every quantile at 10 with the price at 12: loss 2q, mean of q is 0.5.
    assert daily["pinball"].iloc[0] == pytest.approx(1.0)


def test_rolling_scores_pool_periods_and_wait_for_a_full_window() -> None:
    daily = _daily([1.0, 0.0, 0.5], [2.0, 5.0, 1.0], [4, 2, 2])

    rolled = rolling_scores(daily, window=2)

    assert np.isnan(rolled["rolling_coverage_90"].iloc[0])
    assert rolled["rolling_coverage_90"].iloc[1] == pytest.approx(4.0 / 6.0)
    assert rolled["rolling_pinball"].iloc[1] == pytest.approx((8.0 + 10.0) / 6.0)
    assert rolled["rolling_coverage_90"].iloc[2] == pytest.approx(0.25)
    assert rolled["rolling_pinball"].iloc[2] == pytest.approx(3.0)


# --- threshold rule --------------------------------------------------------


def test_coverage_threshold_is_the_highest_grid_value_within_five_percent() -> None:
    # 20 days: 18 at 90%, one at 70% and one at 72%. One day in 20 is the most
    # allowed, so the threshold can rise to 72% (strictly below) but not 72.5%.
    values = pd.Series([0.9] * 18 + [0.70, 0.72])

    assert select_coverage_threshold(values) == pytest.approx(0.72)


def test_coverage_threshold_rises_to_the_bulk_when_one_day_is_low() -> None:
    values = pd.Series([0.9] * 19 + [0.5])

    assert select_coverage_threshold(values) == pytest.approx(0.9)


def test_pinball_threshold_is_the_lowest_grid_value_within_five_percent() -> None:
    # 18 days at ratio 1.0, one at 1.33, one at 1.9: 1.30 leaves two above, 1.35 one.
    values = pd.Series([1.0] * 18 + [1.33, 1.9])

    assert select_pinball_threshold(values) == pytest.approx(1.35)


def _window_settings(settings: Settings) -> Settings:
    evaluation = settings.evaluation.model_copy(
        update={
            "validation_start": date(2024, 3, 1),
            "holdout_start": date(2024, 6, 1),
        }
    )
    return settings.model_copy(update={"evaluation": evaluation})


def test_threshold_selection_cannot_see_holdout_rows(settings: Settings) -> None:
    local = _window_settings(settings)
    history = _random_forecasts(date(2024, 1, 20), 133, seed=3)
    assert max(history["target_day"]) == date(2024, 5, 31)
    # Hold-out rows with prices far outside every range and huge losses.
    holdout = _random_forecasts(date(2024, 6, 1), 60, seed=4, spread=400.0)

    alone, _ = fit_thresholds(history, local)
    with_holdout, validation = fit_thresholds(pd.concat([history, holdout]), local)

    assert with_holdout == alone
    assert max(validation.index) == date(2024, 5, 31)
    assert min(validation.index) == date(2024, 3, 1)
    assert alone.validation_days == 92
    assert alone.coverage_alert_share <= 0.05
    assert alone.pinball_alert_share <= 0.05

    # The same rows labelled as validation would move the thresholds: the test
    # above is sensitive to them.
    later = local.model_copy(
        update={
            "evaluation": local.evaluation.model_copy(
                update={"holdout_start": date(2024, 8, 1)}
            )
        }
    )
    widened, _ = fit_thresholds(pd.concat([history, holdout]), later)
    assert (widened.coverage, widened.pinball_ratio) != (
        alone.coverage,
        alone.pinball_ratio,
    )


def test_compute_fixes_thresholds_before_it_reads_the_holdout_file(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = settings.data.model_copy(update={"processed_dir": tmp_path})
    local = _window_settings(settings.model_copy(update={"data": data}))
    frames = {
        "comparison": _random_forecasts(date(2024, 1, 20), 133, seed=3),
        "calm": _random_forecasts(date(2024, 6, 1), 60, seed=4),
        "wild": _random_forecasts(date(2024, 6, 1), 60, seed=4, spread=400.0),
    }
    paths = {name: tmp_path / f"{name}.parquet" for name in frames}
    for name, frame in frames.items():
        frame.assign(fallback_periods=0).to_parquet(paths[name])

    reads: list[Path] = []
    read_at_fit: list[list[Path]] = []
    read_parquet = pd.read_parquet
    fit = fit_thresholds

    def spy_read(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        reads.append(Path(path))
        return read_parquet(path, *args, **kwargs)

    def spy_fit(
        forecasts: pd.DataFrame, fit_settings: Settings
    ) -> tuple[Thresholds, pd.DataFrame]:
        read_at_fit.append(list(reads))
        return fit(forecasts, fit_settings)

    monkeypatch.setattr(pd, "read_parquet", spy_read)
    monkeypatch.setattr(m2, "fit_thresholds", spy_fit)

    results = {}
    for holdout in ("calm", "wild"):
        reads.clear()
        results[holdout] = m2.compute(local, paths["comparison"], paths[holdout])
        assert reads == [paths["comparison"], paths[holdout]]
    assert read_at_fit == [[paths["comparison"]], [paths["comparison"]]]

    # A wildly different hold-out file changes the hold-out alerts, not the
    # thresholds.
    assert results["calm"].thresholds == results["wild"].thresholds
    assert (
        results["wild"].summaries["holdout"].coverage_alert_share
        > results["calm"].summaries["holdout"].coverage_alert_share
    )


# --- alerts, episodes, incidents -------------------------------------------


def _thresholds() -> Thresholds:
    return Thresholds(
        coverage=0.8,
        pinball_ratio=1.5,
        pinball_median=4.0,
        validation_days=5,
        coverage_alert_share=0.0,
        pinball_alert_share=0.0,
    )


def test_apply_thresholds_flags_strictly_and_never_without_a_window() -> None:
    rolled = pd.DataFrame(
        {
            "rolling_coverage_90": [np.nan, 0.8, 0.79, 0.9],
            "rolling_pinball": [np.nan, 6.0, 6.4, 4.0],
        }
    )

    flagged = apply_thresholds(rolled, _thresholds())

    assert flagged["coverage_alert"].tolist() == [False, False, True, False]
    assert flagged["pinball_alert"].tolist() == [False, False, True, False]
    assert flagged["rolling_pinball_ratio"].iloc[1] == pytest.approx(1.5)


def test_episodes_group_consecutive_alert_days_within_each_window() -> None:
    days = [date(2026, 5, 27) + timedelta(days=i) for i in range(7)]
    series = pd.DataFrame(
        {
            "coverage_alert": [False, True, True, False, True, True, True],
            "rolling_coverage_90": [0.85, 0.78, 0.75, 0.82, 0.79, 0.7, 0.72],
            "pinball_alert": [False, True, True, False, False, False, False],
            "rolling_pinball_ratio": [1.0, 1.6, 1.7, 1.2, 1.0, 1.0, 1.0],
        },
        index=pd.Index(days, name="target_day"),
    )
    windows = pd.Series(["validation"] * 5 + ["holdout"] * 2, index=series.index)

    coverage = alert_episodes(series, "coverage", windows)
    pinball = alert_episodes(series, "pinball", windows)

    assert [(e.window, e.start, e.end, e.days) for e in coverage] == [
        ("validation", days[1], days[2], 2),
        ("validation", days[4], days[4], 1),
        ("holdout", days[5], days[6], 2),
    ]
    assert coverage[0].first_value == 0.78
    assert (coverage[0].extreme_value, coverage[0].extreme_day) == (0.75, days[2])
    assert [e.open_at_end for e in coverage] == [False, True, True]
    assert (pinball[0].extreme_value, pinball[0].extreme_day) == (1.7, days[2])

    records = drift_incidents(coverage + pinball, _thresholds(), "Europe/Berlin")

    assert len(records) == 4
    assert len({r.incident_id for r in records}) == 4
    for record in records:
        assert (record.type, record.severity, record.status, record.source) == (
            "drift",
            "warning",
            "review",
            SOURCE,
        )
        assert record.action == "Flag for retraining review"
    first = records[0]
    assert first.delivery_day == days[1]
    assert "78.0%" in first.detail
    assert "80.0%" in first.detail
    assert first.metrics["alert_days"] == 2.0
    pinball_record = records[3]
    assert pinball_record.delivery_day == days[1]
    assert "1.600 times" in pinball_record.detail
    # Validation episodes started on days that helped fit the thresholds.
    assert [r.metrics["in_sample"] for r in records] == [1.0, 1.0, 0.0, 1.0]
    assert records[0].detail.endswith(IN_SAMPLE_NOTE)
    assert IN_SAMPLE_NOTE not in records[2].detail


def test_pinball_incident_shows_the_ratio_to_three_decimals() -> None:
    day = date(2026, 6, 10)
    episode = AlertEpisode(
        signal="pinball",
        window="holdout",
        start=day,
        end=day,
        days=1,
        first_value=1.5024,
        extreme_value=1.5024,
        extreme_day=day,
        open_at_end=True,
    )

    (record,) = drift_incidents([episode], _thresholds(), "Europe/Berlin")

    assert "was 1.502 times its validation median" in record.detail
    assert "above the alert threshold of 1.50." in record.detail
    assert record.metrics["in_sample"] == 0.0


def test_longest_run() -> None:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(6)]
    flags = pd.Series([True, False, True, True, True, False], index=days)

    assert longest_run(flags) == (3, days[2], days[4])
    assert longest_run(pd.Series([False], index=days[:1])) == (0, None, None)


def test_thresholds_dataclass_is_frozen() -> None:
    th = _thresholds()
    assert replace(th, coverage=0.7).coverage == 0.7
    with pytest.raises(AttributeError):
        th.coverage = 0.5  # type: ignore[misc]
