"""The live drift check: what it scores, when it may alert, and what it writes."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings
from src.health.drift import Thresholds
from src.health.incidents import default_path, load_incidents
from src.pipeline import drift_check as dc

LIVE_FROM = date(2026, 9, 15)
THRESHOLDS: dict[str, float] = {
    "coverage": 0.74,
    "pinball_ratio": 1.5,
    "pinball_median": 4.9,
    "validation_days": 730,
    "coverage_alert_share": 0.048,
    "pinball_alert_share": 0.042,
}


def _local(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    evaluation = settings.evaluation.model_copy(update={"live_from": LIVE_FROM})
    return settings.model_copy(update={"data": data, "evaluation": evaluation})


def _save_thresholds(settings: Settings, **changes: object) -> None:
    path = settings.data.processed_path / "experiments" / "m2_drift.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": settings.forecasting.production_model,
        "window_days": 28,
        "thresholds": THRESHOLDS,
    } | changes
    path.write_text(json.dumps(payload), encoding="utf-8")


def _index(settings: Settings, day: date) -> pd.DatetimeIndex:
    start = pd.Timestamp(day, tz=settings.market.timezone)
    return pd.date_range(
        start, start + pd.Timedelta(days=1), freq="15min", inclusive="left"
    ).tz_convert("UTC")


def _save_forecast(
    settings: Settings,
    day: date,
    *,
    step: str = "production",
    width: float = 50.0,
    kind: str = "live",
) -> pd.DatetimeIndex:
    """A flat fan around 100 EUR/MWh, ``width`` either side for q05 and q95."""
    index = _index(settings, day)
    spread = {
        "q05": -width,
        "q10": -0.8 * width,
        "q25": -0.4 * width,
        "q50": 0.0,
        "q75": 0.4 * width,
        "q90": 0.8 * width,
        "q95": width,
    }
    frame = pd.DataFrame(
        {name: 100.0 + offset for name, offset in spread.items()}, index=index
    )
    frame.insert(0, "model", "lightgbm_conformal")
    frame["target_day"] = day
    frame["issue_time_utc"] = datetime(2026, 1, 1, tzinfo=UTC)
    frame["chain_step"] = step
    frame["kind"] = kind
    path = settings.data.processed_path / "forecasts" / "production" / f"{day}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path)
    return index


def _prices(index: pd.DatetimeIndex, value: float) -> pd.DataFrame:
    return pd.DataFrame({PRICE_SERIES: np.full(len(index), value)}, index=index)


def test_the_thresholds_are_the_ones_m2_fixed(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    with pytest.raises(FileNotFoundError, match="m2_drift"):
        dc.load_thresholds(local)

    _save_thresholds(local)
    assert dc.load_thresholds(local) == Thresholds(
        coverage=0.74,
        pinball_ratio=1.5,
        pinball_median=4.9,
        validation_days=730,
        coverage_alert_share=0.048,
        pinball_alert_share=0.042,
    )

    _save_thresholds(local, window_days=14)
    with pytest.raises(ValueError, match="14-day window"):
        dc.load_thresholds(local)
    _save_thresholds(local, model="lightgbm_quantile")
    with pytest.raises(ValueError, match="fixed thresholds for lightgbm_quantile"):
        dc.load_thresholds(local)


def test_fallback_and_unpriced_days_are_skipped_and_nothing_alerts_while_warming_up(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    _save_thresholds(local)
    days = [LIVE_FROM + timedelta(days=offset) for offset in range(4)]
    parts = [_prices(_save_forecast(local, days[0]), 300.0)]  # far outside the fan
    parts.append(_prices(_save_forecast(local, days[1], step="seasonal_naive"), 100.0))
    parts.append(_prices(_save_forecast(local, days[2]), 100.0))
    unpriced = _prices(_save_forecast(local, days[3]), 100.0)
    unpriced.iloc[-4:] = np.nan
    parts.append(unpriced)
    # A day before live_from and one after --through are never read.
    _save_forecast(local, LIVE_FROM - timedelta(days=1))
    _save_forecast(local, days[3] + timedelta(days=1))

    summary = dc.run_check(local, days[3], frame=pd.concat(parts))

    assert summary["status"] == "warming up: 2 of 28 scored days"
    assert summary["warming_up"] is True and summary["latest"] is None
    assert [s["day"] for s in summary["series"]] == [str(days[0]), str(days[2])]
    assert summary["series"][0]["coverage_90"] == 0.0
    assert summary["skipped"] == [
        {"day": str(days[1]), "reason": "forecast by seasonal_naive"},
        {"day": str(days[3]), "reason": "prices not all published"},
    ]
    assert summary["incidents"] == []
    saved = json.loads(dc.summary_path(local).read_text(encoding="utf-8"))
    assert saved["status"] == summary["status"]


def _month(local: Settings, misses_from: int) -> tuple[list[date], pd.DataFrame]:
    """30 production days; prices fall outside the fan from day ``misses_from`` on."""
    days = [LIVE_FROM + timedelta(days=offset) for offset in range(30)]
    parts = []
    for number, day in enumerate(days):
        index = _save_forecast(local, day)
        parts.append(_prices(index, 400.0 if number >= misses_from else 100.0))
    return days, pd.concat(parts)


def test_a_coverage_collapse_alerts_once_the_window_is_full_and_the_log_follows_it(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    _save_thresholds(local)
    days, prices = _month(local, misses_from=20)

    summary = dc.run_check(local, days[-1], frame=prices)

    assert summary["scored_days"] == 30 and summary["warming_up"] is False
    assert summary["status"] == "in alert"
    latest = summary["latest"]
    assert latest["day"] == str(days[-1]) and latest["coverage_alert"] is True
    assert latest["rolling_coverage_90"] == pytest.approx(18 / 28)
    live = [i for i in load_incidents(default_path(local)) if i.source == "live_drift"]
    coverage = [i for i in live if "coverage" in i.detail]
    # 28-day windows ending on day 28, 29 and 30 hold 8, 9 and 10 missed days.
    assert len(coverage) == 1 and coverage[0].delivery_day == days[27]
    assert coverage[0].metrics["in_sample"] == 0.0 and coverage[0].status == "review"
    assert coverage[0].incident_id in summary["incidents"]

    # Rebuilt, not appended: a rerun with the same record changes nothing.
    again = dc.run_check(local, days[-1], frame=prices)
    assert again["incidents"] == summary["incidents"]
    assert len(
        [i for i in load_incidents(default_path(local)) if i.source == "live_drift"]
    ) == len(live)


def test_a_steady_month_reports_no_alert(settings: Settings, tmp_path: Path) -> None:
    local = _local(settings, tmp_path)
    _save_thresholds(local)
    days, prices = _month(local, misses_from=99)

    summary = dc.run_check(local, days[-1], frame=prices)

    assert summary["status"] == "no alert" and summary["incidents"] == []
    assert summary["latest"]["rolling_coverage_90"] == 1.0


def test_a_day_before_live_from_is_refused(settings: Settings, tmp_path: Path) -> None:
    local = _local(settings, tmp_path)
    with pytest.raises(ValueError, match="before live_from"):
        dc.run_check(local, LIVE_FROM - timedelta(days=1), frame=pd.DataFrame())


def test_an_alert_keeps_the_time_its_first_run_found_it(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    _save_thresholds(local)
    days, prices = _month(local, misses_from=20)
    first_run = datetime(2026, 10, 13, 8, 45, tzinfo=UTC)
    later_run = datetime(2026, 10, 14, 8, 45, tzinfo=UTC)

    dc.run_check(local, days[-2], frame=prices, now_utc=first_run)
    dc.run_check(local, days[-1], frame=prices, now_utc=later_run)

    (coverage,) = [
        i
        for i in load_incidents(default_path(local))
        if i.source == "live_drift" and "coverage" in i.detail
    ]
    assert coverage.detected_utc == first_run
    assert "The alert ran 3 scored days" in coverage.detail
    assert "still in alert on the last scored day" in coverage.detail


def test_twenty_seven_missed_days_still_cannot_alert(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    _save_thresholds(local)
    days, prices = _month(local, misses_from=0)

    summary = dc.run_check(local, days[26], frame=prices)

    assert summary["status"] == "warming up: 27 of 28 scored days"
    assert summary["incidents"] == []
    assert dc.run_check(local, days[27], frame=prices)["status"] == "in alert"


def test_a_reconstructed_day_is_not_scored(settings: Settings, tmp_path: Path) -> None:
    """It was forecast after the fact, so it says nothing about the live model."""
    local = _local(settings, tmp_path)
    day = local.evaluation.live_from
    index = _save_forecast(local, day, kind="backfill")

    frame, skipped = dc.live_forecasts(local, day, _prices(index, 100.0)[PRICE_SERIES])

    assert frame.empty
    assert skipped == [{"day": str(day), "reason": "reconstructed, not a live bid"}]


def test_a_forecast_saved_before_the_kind_column_is_still_scored(
    settings: Settings, tmp_path: Path
) -> None:
    """Every forecast written before the column existed was a live bid."""
    local = _local(settings, tmp_path)
    day = local.evaluation.live_from
    index = _save_forecast(local, day)
    path = local.data.processed_path / "forecasts" / "production" / f"{day}.parquet"
    pd.read_parquet(path).drop(columns=["kind"]).to_parquet(path)

    frame, skipped = dc.live_forecasts(local, day, _prices(index, 100.0)[PRICE_SERIES])

    assert not frame.empty and skipped == []
