"""Reconstructing a day the desk never bid on, without it passing for a bid."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import QuantileForecast, make_forecast
from src.forecasting.information import InformationSet
from src.health.incidents import default_path, load_incidents
from src.pipeline import daily_run as dr
from src.pipeline.model_source import RefitOutcome
from src.pipeline.plan import load_plan
from tests.fakes import synthetic_market
from tests.test_pipeline_daily_run import LIVE_DAY, FakeModel, _local

#: Well after the delivery day, which is what makes it a reconstruction.
AFTER = datetime(2026, 9, 25, 12, 0, tzinfo=UTC)


def _before_gate(settings: Settings) -> datetime:
    """An hour before LIVE_DAY's gate, when a live run would still be in time."""
    gate = settings.market.local_midnight_utc(LIVE_DAY - timedelta(days=1))
    hours, minutes = (int(p) for p in settings.market.gate_closure_local.split(":"))
    return gate + timedelta(hours=hours, minutes=minutes) - timedelta(hours=1)


#: The registered version was already serving before LIVE_DAY.
SERVING_SINCE = (LIVE_DAY - timedelta(days=2), "3")


@pytest.fixture(autouse=True)
def _no_real_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dr,
        "refresh_production_model",
        lambda settings, day, frame, **kwargs: RefitOutcome(
            "not_due", "fake registry", "3", "3"
        ),
    )
    day, version = SERVING_SINCE
    monkeypatch.setattr(dr, "served_version", lambda settings, **k: (version, day))


@pytest.fixture
def market(settings: Settings) -> pd.DataFrame:
    return synthetic_market(settings, LIVE_DAY - timedelta(days=40), 42)


def _backfill(
    settings: Settings,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    *,
    now: datetime = AFTER,
    day: date = LIVE_DAY,
) -> dr.RunRecord:
    monkeypatch.setattr(dr, "load_model", lambda s, **k: (FakeModel(settings), "3"))
    return dr.run_day(
        settings,
        day,
        frame=market,
        use_registry=True,
        now_utc=now,
        kind=dr.KIND_BACKFILL,
    )


def test_a_reconstruction_has_no_issue_time_and_no_gate_verdict(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)

    record = _backfill(local, market, monkeypatch)

    assert record.kind == dr.KIND_BACKFILL
    assert record.issued_utc is None and record.on_time is None
    saved = json.loads(dr.record_path(local, LIVE_DAY).read_text())
    assert saved["kind"] == "backfill"
    assert saved["issued_utc"] is None and saved["on_time"] is None
    assert saved["minutes_before_gate"] is None
    # It is still a real schedule with a real value, which is the point of it.
    assert record.planned_value_eur > 0
    assert load_plan(local, LIVE_DAY).schedule["discharge_mw"].sum() > 0


def test_a_reconstruction_writes_no_incident(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Long after the gate, so a live run at this moment would be a late bid."""
    local = _local(settings, tmp_path)

    record = _backfill(local, market, monkeypatch)

    assert record.incidents == ()
    assert load_incidents(default_path(local)) == []


def test_the_saved_forecast_says_it_was_reconstructed(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The drift monitor reads this folder and must leave the day out."""
    local = _local(settings, tmp_path)

    _backfill(local, market, monkeypatch)

    saved = pd.read_parquet(
        local.data.processed_path / "forecasts" / "production" / f"{LIVE_DAY}.parquet"
    )
    assert saved["kind"].iloc[0] == "backfill"


def test_a_day_whose_gate_is_still_open_is_refused(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Before the gate the day can still be bid, so it must be."""
    local = _local(settings, tmp_path)
    before_gate = _before_gate(local)

    with pytest.raises(ValueError, match="has not yet passed"):
        _backfill(local, market, monkeypatch, now=before_gate)


def test_a_model_newer_than_the_day_is_refused(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """It was fitted on data the desk did not have, so it is not a forecast."""
    local = _local(settings, tmp_path)
    monkeypatch.setattr(
        dr, "served_version", lambda settings, **k: ("4", LIVE_DAY + timedelta(days=1))
    )

    with pytest.raises(ValueError, match="cannot reconstruct the day"):
        _backfill(local, market, monkeypatch)


def test_a_reconstruction_does_not_refit_the_served_model(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A refit would train on the days after the one being reconstructed."""
    local = _local(settings, tmp_path)
    called: list[date] = []

    def _refit(
        settings: Settings, day: date, frame: object, **kwargs: object
    ) -> RefitOutcome:
        called.append(day)
        return RefitOutcome("refitted", "should not happen", "3", "4")

    monkeypatch.setattr(dr, "refresh_production_model", _refit)

    record = _backfill(local, market, monkeypatch)

    assert called == []
    saved = json.loads(dr.record_path(local, LIVE_DAY).read_text())
    assert saved["refit"] is None and record.kind == dr.KIND_BACKFILL


def test_the_deadline_check_reports_no_gate_for_a_reconstruction(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    _backfill(local, market, monkeypatch)

    assert dr.check_deadline(local, LIVE_DAY) is None
    # A late bid writes a critical incident; a reconstruction must write none.
    assert load_incidents(default_path(local)) == []


@dataclass
class TellTaleModel:
    """A model whose forecast is the mean of every price it was allowed to see.

    ``FakeModel`` answers from the clock alone, so it would agree with itself
    whether or not the delivery day leaked into its information set. This one
    cannot: if the target day's realised prices were visible, its answer would
    move.
    """

    settings: Settings
    name: str = "tell_tale"
    seen: list[float] = field(default_factory=list)

    @property
    def lookback_days(self) -> int | None:
        return None

    @property
    def fit_lookback_days(self) -> int | None:
        return None

    def fit(self, info: InformationSet) -> None:
        return None

    def forecast(self, info: InformationSet) -> QuantileForecast:
        quantiles = self.settings.forecasting.quantiles
        level = float(info.history[PRICE_SERIES].mean())
        self.seen.append(level)
        hours = np.asarray(info.target_index.tz_convert(info.tz).hour, dtype="float64")
        # A shape the battery can trade, anchored on what the model could see.
        curve = level + np.where((hours >= 17) & (hours < 21), 120.0, -40.0)
        values = pd.DataFrame(
            {f"q{int(q * 100):02d}": curve + 2 * q for q in quantiles},
            index=info.target_index,
        )
        return make_forecast(self.name, info, values, quantiles)


def test_a_reconstruction_cannot_see_the_day_it_forecasts(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the whole thing: no look-ahead, so the numbers are real.

    A reconstruction runs on today's dataset, which already holds the delivery
    day's realised prices; a live run on the day before could not have had
    them. So the same day is reconstructed twice: once from the full dataset,
    and once from one with that day's prices blanked out, which is what the
    desk actually had before the gate. A chain that is point-in-time gives the
    same forecast either way. One that peeks does not.
    """
    full_side = _local(settings, tmp_path / "full")
    blind_side = _local(settings, tmp_path / "blind")
    watcher = TellTaleModel(full_side)
    monkeypatch.setattr(dr, "load_model", lambda s, **k: (watcher, "3"))

    blinded = market.copy()
    start = settings.market.local_midnight_utc(LIVE_DAY)
    end = settings.market.local_midnight_utc(LIVE_DAY + timedelta(days=1))
    on_the_day = (blinded.index >= start) & (blinded.index < end)
    assert blinded.loc[on_the_day, PRICE_SERIES].notna().all()
    blinded.loc[on_the_day, PRICE_SERIES] = np.nan

    for side, frame in ((full_side, market), (blind_side, blinded)):
        dr.run_day(
            side,
            LIVE_DAY,
            frame=frame,
            use_registry=True,
            now_utc=AFTER,
            kind=dr.KIND_BACKFILL,
        )

    folder = "forecasts", "production", f"{LIVE_DAY}.parquet"
    knowing = pd.read_parquet(full_side.data.processed_path.joinpath(*folder))
    blind = pd.read_parquet(blind_side.data.processed_path.joinpath(*folder))
    quantiles = [c for c in knowing.columns if c.startswith("q")]
    pd.testing.assert_frame_equal(knowing[quantiles], blind[quantiles])
    # The model answered from the data rather than the clock, so the comparison
    # above could have failed.
    with_day, without_day = watcher.seen
    assert np.isfinite(with_day) and with_day == pytest.approx(without_day)


def test_a_rerun_may_not_turn_a_reconstruction_into_a_bid(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """--replace repairs a run; it must not add a day the desk never traded."""
    local = _local(settings, tmp_path)
    _backfill(local, market, monkeypatch)

    with pytest.raises(ValueError, match="may not change what a day was"):
        dr.run_day(
            local,
            LIVE_DAY,
            frame=market,
            use_registry=True,
            now_utc=AFTER,
            replace=True,
        )
    saved = json.loads(dr.record_path(local, LIVE_DAY).read_text())
    assert saved["kind"] == "backfill"
