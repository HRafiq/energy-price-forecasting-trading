"""Delivery days the desk did not bid on: which days count, and what is recorded."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from src.config import Settings
from src.health.incidents import (
    Incident,
    default_path,
    load_incidents,
    make_incident_id,
    upsert_incidents,
)
from src.pipeline import gaps

DETECTED = datetime(2026, 9, 21, 10, 0, tzinfo=UTC)


def _local(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    return settings.model_copy(update={"data": data})


def _record(settings: Settings, day: date, kind: str = "live") -> None:
    folder = settings.data.processed_path / "pipeline" / "runs"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{day}.json").write_text(
        json.dumps({"target_day": str(day), "kind": kind}), encoding="utf-8"
    )


def _first(settings: Settings) -> date:
    return settings.evaluation.live_from


def test_days_before_the_first_bid_are_not_gaps(
    settings: Settings, tmp_path: Path
) -> None:
    """The desk did not exist yet, so its absence is not an outage."""
    local = _local(settings, tmp_path)
    started = _first(local) + timedelta(days=5)
    _record(local, started)

    assert gaps.gap_days(local, started) == []


def test_a_day_between_bids_with_no_record_is_a_gap(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    first = _first(local)
    _record(local, first)
    _record(local, first + timedelta(days=3))

    found = gaps.gap_days(local, first + timedelta(days=3))

    assert found == [first + timedelta(days=1), first + timedelta(days=2)]


def test_a_reconstruction_does_not_close_the_gap(
    settings: Settings, tmp_path: Path
) -> None:
    """The battery still traded nothing that day; only a bid closes a gap."""
    local = _local(settings, tmp_path)
    first = _first(local)
    dark = first + timedelta(days=1)
    _record(local, first)
    _record(local, dark, kind="backfill")

    assert gaps.gap_days(local, dark) == [dark]


def test_the_incident_names_the_day_and_says_nothing_was_traded(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    first = _first(local)
    dark = first + timedelta(days=1)
    _record(local, first)
    _record(local, first + timedelta(days=2))

    written = gaps.record_gaps(local, first + timedelta(days=2), now_utc=DETECTED)

    assert [i.delivery_day for i in written] == [dark]
    incident = written[0]
    assert incident.source == "desk_offline" and incident.severity == "critical"
    assert incident.status == "review" and incident.type == "pipeline"
    assert str(dark) in incident.detail and "traded nothing" in incident.detail
    logged = load_incidents(default_path(local))
    assert [i.incident_id for i in logged] == [incident.incident_id]


def test_a_day_the_pipeline_ran_and_failed_on_is_not_called_offline(
    settings: Settings, tmp_path: Path
) -> None:
    """That run wrote its own incident; the machine was not the problem."""
    local = _local(settings, tmp_path)
    first = _first(local)
    failed_on = first + timedelta(days=1)
    _record(local, first)
    _record(local, first + timedelta(days=2))
    no_forecast = Incident(
        incident_id=make_incident_id("pipeline", "pipeline", failed_on),
        delivery_day=failed_on,
        detected_utc=DETECTED,
        type="pipeline",
        severity="critical",
        detail="Live run produced no forecast: no chain step could forecast.",
        action="No schedule was committed; the gate closed without a bid",
        status="review",
        source="pipeline",
    )
    upsert_incidents([no_forecast], default_path(local))

    assert gaps.gap_days(local, first + timedelta(days=2)) == []
    assert gaps.record_gaps(local, first + timedelta(days=2), now_utc=DETECTED) == []
    # The run's own incident is untouched, and it is the only one.
    assert [i.source for i in load_incidents(default_path(local))] == ["pipeline"]


def test_a_gap_keeps_the_time_it_was_first_found(
    settings: Settings, tmp_path: Path
) -> None:
    """A rerun must not redate an outage to whenever the scan last ran."""
    local = _local(settings, tmp_path)
    first = _first(local)
    _record(local, first)
    _record(local, first + timedelta(days=2))
    through = first + timedelta(days=2)
    found = gaps.record_gaps(local, through, now_utc=DETECTED)

    later = gaps.record_gaps(local, through, now_utc=DETECTED + timedelta(days=3))

    assert [i.detected_utc for i in later] == [i.detected_utc for i in found]
    assert later[0].detected_utc == DETECTED


def test_rewriting_the_gaps_leaves_other_sources_alone(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    first = _first(local)
    dark = first + timedelta(days=1)
    _record(local, first)
    _record(local, first + timedelta(days=2))
    other = Incident(
        incident_id=make_incident_id("observed", "tail_miss", dark),
        delivery_day=dark,
        detected_utc=DETECTED,
        type="tail_miss",
        severity="warning",
        detail="A tail miss on the same delivery day, from a different source.",
        action="None",
        status="resolved",
        source="observed",
    )
    upsert_incidents([other], default_path(local))

    gaps.record_gaps(local, first + timedelta(days=2), now_utc=DETECTED)
    _record(local, dark)  # the gap closes
    gaps.record_gaps(local, first + timedelta(days=2), now_utc=DETECTED)

    assert [i.incident_id for i in load_incidents(default_path(local))] == [
        other.incident_id
    ]


def test_a_rerun_that_finds_fewer_gaps_drops_the_stale_ones(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    first = _first(local)
    dark = first + timedelta(days=1)
    _record(local, first)
    _record(local, first + timedelta(days=2))
    gaps.record_gaps(local, first + timedelta(days=2), now_utc=DETECTED)

    _record(local, dark)  # the day was repaired with a real bid
    written = gaps.record_gaps(local, first + timedelta(days=2), now_utc=DETECTED)

    assert written == []
    assert load_incidents(default_path(local)) == []


def test_an_unreadable_run_record_is_not_mistaken_for_a_bid(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    first = _first(local)
    _record(local, first)
    folder = local.data.processed_path / "pipeline" / "runs"
    (folder / f"{first + timedelta(days=1)}.json").write_text("{", encoding="utf-8")

    assert gaps.gap_days(local, first + timedelta(days=1)) == [
        first + timedelta(days=1)
    ]


def test_the_cli_reports_the_days_it_found(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = _local(settings, tmp_path)
    first = _first(local)
    _record(local, first)
    _record(local, first + timedelta(days=2))
    monkeypatch.setattr(gaps, "load_settings", lambda path: local)

    assert gaps.main(["--through", str(first + timedelta(days=2))]) == 0
    assert str(first + timedelta(days=1)) in capsys.readouterr().out
