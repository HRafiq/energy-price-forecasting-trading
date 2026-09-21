"""The live record: what the export joins, and what the API serves from it."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api import service
from api.main import app
from src.config import Settings
from src.export import live
from src.health.incidents import (
    Incident,
    default_path,
    make_incident_id,
    upsert_incidents,
)


def _local(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    return settings.model_copy(update={"data": data})


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(
    day: str, issued: str, on_time: bool, step: str, **more: object
) -> dict[str, object]:
    return {
        "target_day": day,
        "issued_utc": issued,
        "gate_utc": f"{day}T10:00:00+00:00",
        "on_time": on_time,
        "minutes_before_gate": 25.0 if on_time else -80.0,
        "step": step,
        "model": "lightgbm_conformal" if step == "production" else "seasonal_naive",
        "model_version": "1" if step == "production" else None,
        "readiness": {"missing": [] if step == "production" else ["weather"]},
        "planned_value_eur": 300.0,
        **more,
    }


@pytest.fixture
def sources(settings: Settings, tmp_path: Path) -> Settings:
    local = _local(settings, tmp_path)
    processed = local.data.processed_path
    live_from = local.evaluation.live_from
    first = live_from.isoformat()
    second = (live_from + timedelta(days=1)).isoformat()
    before = (live_from - timedelta(days=1)).isoformat()
    _write(
        processed / "pipeline" / "runs" / f"{first}.json",
        _record(
            first,
            f"{first}T09:34:00+00:00",
            True,
            "production",
            refit={"status": "not_due"},
        ),
    )
    _write(
        processed / "pipeline" / "runs" / f"{second}.json",
        _record(second, f"{second}T08:00:00+00:00", False, "seasonal_naive"),
    )
    # A record from before live_from is a rehearsal and stays out.
    _write(
        processed / "pipeline" / "runs" / f"{before}.json",
        _record(before, f"{before}T09:00:00+00:00", True, "production"),
    )
    _write(
        processed / "pipeline" / "settlements" / f"{first}.json",
        {
            "target_day": first,
            "pnl_eur": 308.41,
            "cycles": 2.0,
            "settled_utc": f"{second}T11:13:45+00:00",
        },
    )
    _write(
        processed / "health" / "live_drift.json",
        {
            "status": "warming up: 1 of 28 scored days",
            "warming_up": True,
            "scored_days": 1,
            "window_days": 28,
            "through": first,
            "latest": None,
            "skipped": [{"day": second, "reason": "forecast by seasonal_naive"}],
        },
    )
    late = Incident(
        incident_id=make_incident_id(
            "pipeline", "late_data", date.fromisoformat(second)
        ),
        delivery_day=date.fromisoformat(second),
        detected_utc=datetime(2026, 9, 16, 11, 21, tzinfo=UTC),
        type="late_data",
        severity="critical",
        detail="Live run: feeds missing at issue time: weather.",
        action="Fallback: seasonal naive used",
        status="review",
        source="pipeline",
    )
    other = late.model_copy(
        update={
            "incident_id": make_incident_id(
                "d5_deadline", "pipeline", date.fromisoformat(second)
            ),
            "source": "d5_deadline",
        }
    )
    upsert_incidents([late, other], default_path(local))
    return local


def test_the_live_record_joins_runs_settlements_incidents_and_drift(
    sources: Settings,
) -> None:
    record = live.build_live_record(sources)

    first, second = (d["target_day"] for d in record["days"])
    assert first == sources.evaluation.live_from.isoformat()
    assert record["timezone"] == "Europe/Berlin" and record["gate_local"] == "12:00"
    day_one, day_two = record["days"]
    assert day_one["issued_local"] == f"{first} 11:34" and day_one["on_time"] is True
    assert day_one["settled"] is True and day_one["pnl_eur"] == 308.41
    assert (
        day_one["settled_local"] == f"{second} 13:13" and day_one["refit"] == "not_due"
    )
    assert day_two["settled"] is False and day_two["pnl_eur"] is None
    # Copied from the record, never recomputed from the timestamps.
    assert day_two["on_time"] is False and day_two["refit"] is None
    assert day_two["step"] == "seasonal_naive" and day_two["feeds_missing"] == [
        "weather"
    ]
    # Only the live pipeline's own incidents are attached; the D5 one is not.
    assert [i["source"] for i in day_two["incidents"]] == ["pipeline"]
    assert day_one["kind"] == "live" and day_two["kind"] == "live"
    totals = record["totals"]
    assert totals == {
        "days": 2,
        "not_bid_days": 0,
        "dark_days": 0,
        "reconstructed_days": 0,
        "settled_days": 1,
        "on_time_days": 1,
        "production_days": 1,
        "steps": {"production": 1, "seasonal_naive": 1},
        "settled_pnl_eur": 308.41,
        "planned_value_settled_days_eur": 300.0,
        "open_incidents": 1,
    }
    assert (
        record["drift"]["scored_days"] == 1
        and record["drift"]["skipped"][0]["day"] == second
    )


def test_an_empty_pipeline_gives_an_empty_record(
    settings: Settings, tmp_path: Path
) -> None:
    record = live.build_live_record(_local(settings, tmp_path))

    assert record["days"] == [] and record["drift"] is None
    assert record["totals"]["days"] == 0 and record["totals"]["settled_pnl_eur"] == 0.0


def test_the_export_writes_the_file_the_api_serves(
    sources: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "dashboard"
    monkeypatch.setenv("DASHBOARD_ROOT", str(root))
    service.clear_cache()
    client = TestClient(app)

    missing = client.get("/api/live")
    assert missing.status_code == 503 and "not exported" in missing.json()["detail"]

    written = live.export_live(sources, root)
    served = client.get("/api/live")

    assert served.status_code == 200 and served.json() == written
    assert not list(root.glob(".*.partial"))
    service.clear_cache()


def test_a_run_that_produced_no_forecast_still_gets_a_row(
    sources: Settings,
) -> None:
    day = sources.evaluation.live_from + timedelta(days=5)
    failure = Incident(
        incident_id=make_incident_id("pipeline", "pipeline", day),
        delivery_day=day,
        detected_utc=datetime(2026, 9, 21, 9, 45, tzinfo=UTC),
        type="pipeline",
        severity="critical",
        detail="Live run produced no forecast: no chain step could forecast.",
        action="No schedule was committed; the gate closed without a bid",
        status="review",
        source="pipeline",
    )
    upsert_incidents([failure], default_path(sources))
    broken = sources.data.processed_path / "pipeline" / "runs" / "2026-09-30.json"
    broken.write_text("{not json", encoding="utf-8")

    record = live.build_live_record(sources)

    row = next(d for d in record["days"] if d["target_day"] == day.isoformat())
    assert row["step"] is None and row["settled"] is False
    assert row["on_time"] is None and row["planned_value_eur"] is None
    assert [i["incident_id"] for i in row["incidents"]] == [failure.incident_id]
    assert record["totals"]["open_incidents"] == 2
    assert "2026-09-30" not in [d["target_day"] for d in record["days"]]


def test_a_reconstruction_appears_but_counts_in_no_live_total(
    sources: Settings,
) -> None:
    """It was never bid, so it is not a live day, an on-time day or live profit."""
    day = (sources.evaluation.live_from + timedelta(days=2)).isoformat()
    _write(
        sources.data.processed_path / "pipeline" / "runs" / f"{day}.json",
        _record(day, "", False, "production", kind="backfill")
        | {"issued_utc": None, "on_time": None, "minutes_before_gate": None},
    )
    _write(
        sources.data.processed_path / "pipeline" / "settlements" / f"{day}.json",
        {"pnl_eur": 999.0, "cycles": 1.5, "settled_utc": f"{day}T12:00:00+00:00"},
    )

    record = live.build_live_record(sources)

    row = next(d for d in record["days"] if d["target_day"] == day)
    assert row["kind"] == "backfill" and row["issued_local"] is None
    # Not False: it had no gate to miss, and False reads as a missed gate.
    assert row["on_time"] is None and row["minutes_before_gate"] is None
    assert row["planned_value_eur"] == 300.0 and row["pnl_eur"] == 999.0
    totals = record["totals"]
    assert totals["reconstructed_days"] == 1 and totals["not_bid_days"] == 1
    # Unchanged by the reconstruction: two live days, one settled, 308.41 earned.
    assert totals["days"] == 2 and totals["settled_days"] == 1
    assert totals["on_time_days"] == 1 and totals["production_days"] == 1
    assert totals["settled_pnl_eur"] == 308.41
    assert totals["steps"] == {"production": 1, "seasonal_naive": 1}


def test_a_day_the_desk_never_ran_gets_a_row_of_its_own(
    sources: Settings,
) -> None:
    day = sources.evaluation.live_from + timedelta(days=4)
    dark = Incident(
        incident_id=make_incident_id("desk_offline", "pipeline", day),
        delivery_day=day,
        detected_utc=datetime(2026, 9, 21, 10, 0, tzinfo=UTC),
        type="pipeline",
        severity="critical",
        detail=f"No schedule was committed for {day}: the battery traded nothing.",
        action="A scheduled runner would close this gap",
        status="review",
        source="desk_offline",
    )
    upsert_incidents([dark], default_path(sources))

    record = live.build_live_record(sources)

    row = next(d for d in record["days"] if d["target_day"] == day.isoformat())
    assert row["kind"] is None and row["step"] is None and row["settled"] is False
    assert row["on_time"] is None
    assert [i["source"] for i in row["incidents"]] == ["desk_offline"]
    assert record["totals"]["dark_days"] == 1 and record["totals"]["days"] == 2
    assert record["totals"]["not_bid_days"] == 1
