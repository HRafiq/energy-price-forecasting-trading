"""The incident log store and the observed-incident rules."""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config import Settings
from src.health.incidents import (
    IN_SAMPLE_NOTE,
    Incident,
    daily_outside_share,
    fallback_incidents,
    filter_incidents,
    load_incidents,
    make_incident_id,
    observed_incidents,
    replace_incidents,
    tail_miss_incidents,
    tail_miss_threshold,
    untraded_day_incidents,
    upsert_incidents,
)

QUANTILES = ("q05", "q10", "q25", "q50", "q75", "q90", "q95")


def _incident(
    day: date,
    kind: str = "data_gap",
    source: str = "observed",
    detail: str = "Prices missing for 96 periods.",
) -> Incident:
    return Incident.model_validate(
        {
            "incident_id": make_incident_id(source, kind, day),
            "delivery_day": day,
            "detected_utc": datetime(2026, 9, 14, 10, 0, tzinfo=UTC),
            "type": kind,
            "severity": "warning",
            "detail": detail,
            "action": "Day skipped.",
            "status": "resolved",
            "source": source,
            "metrics": {"periods": 96.0},
        }
    )


def _frame(
    actuals: dict[date, list[float]], low: float = 10.0, high: float = 20.0
) -> pd.DataFrame:
    """Four periods per day; q05 = ``low``, q95 = ``high``, median between."""
    parts = []
    for day, prices in actuals.items():
        index = pd.date_range(
            pd.Timestamp(day, tz="UTC"), periods=len(prices), freq="6h"
        )
        part = pd.DataFrame(
            {
                col: np.linspace(low, high, len(QUANTILES))[i]
                for i, col in enumerate(QUANTILES)
            },
            index=index,
        )
        part["actual"] = prices
        part["target_day"] = day
        part["fallback_periods"] = 0
        parts.append(part)
    return pd.concat(parts)


# --- ids and model ---------------------------------------------------------


def test_ids_are_deterministic_and_distinguish_their_inputs() -> None:
    day = date(2026, 9, 13)
    first = make_incident_id("observed", "data_gap", day)

    assert first == make_incident_id("observed", "data_gap", day)
    assert len(first) == 16
    others = {
        make_incident_id("m2_drift", "data_gap", day),
        make_incident_id("observed", "tail_miss", day),
        make_incident_id("observed", "data_gap", day + timedelta(days=1)),
        make_incident_id("observed", "data_gap", day, key="coverage"),
    }
    assert first not in others
    assert len(others) == 4


def test_detected_time_must_be_utc() -> None:
    record = _incident(date(2026, 9, 13)).model_dump()
    record["detected_utc"] = datetime(2026, 9, 14, 10, 0)
    with pytest.raises(ValidationError):
        Incident.model_validate(record)


# --- store -----------------------------------------------------------------


def test_upsert_is_idempotent_and_replaces_by_id(tmp_path: Path) -> None:
    path = tmp_path / "health" / "incidents.jsonl"
    batch = [_incident(date(2026, 9, 13)), _incident(date(2026, 9, 12), "tail_miss")]

    upsert_incidents(batch, path)
    first_bytes = path.read_bytes()
    upsert_incidents(batch, path)

    assert path.read_bytes() == first_bytes
    assert len(load_incidents(path)) == 2

    changed = _incident(date(2026, 9, 13), detail="Prices missing, now published.")
    records = upsert_incidents([changed], path)
    assert len(records) == 2
    by_id = {record.incident_id: record for record in load_incidents(path)}
    assert by_id[changed.incident_id].detail == "Prices missing, now published."


def test_replace_drops_stale_records_of_its_sources_only(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    stale = _incident(date(2026, 9, 2), "tail_miss")
    drift = _incident(date(2026, 9, 3), "drift", source="m2_drift")
    deadline = _incident(date(2026, 9, 4), "pipeline", source="d5_deadline")
    upsert_incidents([_incident(date(2026, 9, 1)), stale, drift, deadline], path)

    fresh = _incident(date(2026, 9, 1), detail="Prices missing, now published.")
    records = replace_incidents([fresh], ["observed"], path)

    assert records == load_incidents(path)
    assert {(r.source, r.delivery_day) for r in records} == {
        ("observed", date(2026, 9, 1)),
        ("m2_drift", date(2026, 9, 3)),
        ("d5_deadline", date(2026, 9, 4)),
    }
    assert stale not in records
    assert drift in records and deadline in records
    assert fresh in records

    first_bytes = path.read_bytes()
    replace_incidents([fresh], ["observed"], path)
    assert path.read_bytes() == first_bytes

    # A batch without records clears the source.
    replace_incidents([], ["observed"], path)
    assert load_incidents(path) == [drift, deadline]


def test_replace_refuses_records_of_sources_it_does_not_replace(
    tmp_path: Path,
) -> None:
    path = tmp_path / "incidents.jsonl"
    other = _incident(date(2026, 9, 1), "drift", source="m2_drift")

    with pytest.raises(ValueError, match="not being replaced"):
        replace_incidents([other], ["observed"], path)
    assert not path.exists()


def test_round_trip_keeps_every_field(tmp_path: Path) -> None:
    path = tmp_path / "incidents.jsonl"
    original = _incident(date(2026, 9, 13))

    upsert_incidents([original], path)
    (loaded,) = load_incidents(path)

    assert loaded == original
    assert loaded.detected_utc.utcoffset() == timedelta(0)
    assert json.loads(path.read_text(encoding="utf-8"))["metrics"] == {"periods": 96.0}


def test_missing_file_is_an_empty_log(tmp_path: Path) -> None:
    assert load_incidents(tmp_path / "absent.jsonl") == []


def test_atomic_write_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "incidents.jsonl"
    upsert_incidents([_incident(date(2026, 9, 13))], path)
    assert [p.name for p in tmp_path.iterdir()] == ["incidents.jsonl"]
    before = path.read_bytes()

    def fail(*_: object) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="disk full"):
        upsert_incidents([_incident(date(2026, 9, 1))], path)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["incidents.jsonl"]


def test_filter_by_date_range_type_and_source() -> None:
    records = [
        _incident(date(2026, 9, 1)),
        _incident(date(2026, 9, 5), "tail_miss"),
        _incident(date(2026, 9, 9), "drift", source="m2_drift"),
    ]

    assert len(filter_incidents(records)) == 3
    in_range = filter_incidents(records, start=date(2026, 9, 2), end=date(2026, 9, 9))
    assert [r.delivery_day.day for r in in_range] == [5, 9]
    assert [r.type for r in filter_incidents(records, types=["drift"])] == ["drift"]
    observed = filter_incidents(records, sources=["observed"], types=["tail_miss"])
    assert [r.delivery_day.day for r in observed] == [5]


# --- observed rules --------------------------------------------------------


def test_untraded_manifest_days_become_critical_data_gaps() -> None:
    manifest = {
        "run_id": "backtest-test",
        "created_utc": "2026-09-14T16:21:50+00:00",
        "days": [
            {"date": "2026-09-12", "window": "holdout", "traded": True},
            {
                "date": "2026-09-13",
                "window": "holdout",
                "traded": False,
                "skip_reason": "missing price or forecast",
            },
        ],
    }

    (record,) = untraded_day_incidents(manifest)

    assert record.delivery_day == date(2026, 9, 13)
    assert (record.type, record.severity, record.status) == (
        "data_gap",
        "critical",
        "resolved",
    )
    assert "missing price or forecast" in record.detail
    assert record.detected_utc == datetime(2026, 9, 14, 16, 21, 50, tzinfo=UTC)


def test_fallback_periods_become_warning_data_gaps(settings: Settings) -> None:
    frame = _frame({date(2025, 3, 1): [15.0] * 4, date(2025, 3, 2): [15.0] * 4})
    frame.loc[frame["target_day"] == date(2025, 3, 2), "fallback_periods"] = 3

    (record,) = fallback_incidents(frame, settings)

    assert record.delivery_day == date(2025, 3, 2)
    assert record.severity == "warning"
    assert "3 of 4 periods" in record.detail
    # Issued 11:40 Berlin (CET) the day before delivery.
    assert record.detected_utc == datetime(2025, 3, 1, 10, 40, tzinfo=UTC)


def test_tail_miss_cutoff_is_the_validation_99th_percentile() -> None:
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(4)]
    # Outside-range shares per day: 0, 0.25, 0.5, 1.0.
    frame = _frame(
        {
            days[0]: [15.0, 15.0, 15.0, 15.0],
            days[1]: [25.0, 15.0, 15.0, 15.0],
            days[2]: [25.0, 5.0, 15.0, 15.0],
            days[3]: [25.0, 5.0, 30.0, 0.0],
        }
    )

    shares = daily_outside_share(frame)

    assert shares.tolist() == [0.0, 0.25, 0.5, 1.0]
    assert tail_miss_threshold(frame) == pytest.approx(
        float(np.percentile([0.0, 0.25, 0.5, 1.0], 99))
    )


def test_tail_miss_flags_days_at_the_cutoff_and_names_the_worst_miss() -> None:
    days = [date(2025, 1, 6), date(2025, 1, 7)]
    frame = _frame({days[0]: [15.0, 25.0, 15.0, 15.0], days[1]: [15.0, 4.0, 60.0, 2.0]})

    records = tail_miss_incidents(
        frame, threshold=0.5, timezone="UTC", fit_window=(days[0], days[0])
    )

    (record,) = records
    assert record.delivery_day == days[1]
    assert (record.type, record.status) == ("tail_miss", "review")
    assert record.metrics["outside_share"] == pytest.approx(0.75)
    # Worst miss: 60 against q95 = 20, at 12:00.
    assert record.metrics["worst_actual_eur_mwh"] == 60.0
    assert record.metrics["worst_band_edge_eur_mwh"] == 20.0
    assert "3 of 4 periods" in record.detail
    assert "12:00 local: 60.00 €/MWh against q95 20.00 €/MWh" in record.detail
    # The flagged day lies after the fit window: a hold-out style detection.
    assert record.metrics["in_sample"] == 0.0
    assert IN_SAMPLE_NOTE not in record.detail


def test_tail_miss_includes_a_day_exactly_at_the_cutoff() -> None:
    days = [date(2025, 1, 9), date(2025, 1, 10)]
    # Outside-range shares: exactly 0.5, then 0.25.
    frame = _frame(
        {days[0]: [25.0, 5.0, 15.0, 15.0], days[1]: [25.0, 15.0, 15.0, 15.0]}
    )

    records = tail_miss_incidents(
        frame, threshold=0.5, timezone="UTC", fit_window=(days[0], days[1])
    )

    assert [r.delivery_day for r in records] == [days[0]]
    assert records[0].metrics["outside_share"] == 0.5


def test_tail_miss_marks_days_inside_the_fit_window_as_in_sample() -> None:
    days = [date(2026, 5, 31), date(2026, 6, 1)]
    frame = _frame({day: [25.0, 5.0, 15.0, 15.0] for day in days})

    inside, outside = tail_miss_incidents(
        frame, threshold=0.5, timezone="UTC", fit_window=(date(2024, 6, 1), days[0])
    )

    assert inside.delivery_day == days[0]
    assert inside.metrics["in_sample"] == 1.0
    assert inside.detail.endswith(IN_SAMPLE_NOTE)
    assert outside.metrics["in_sample"] == 0.0
    assert IN_SAMPLE_NOTE not in outside.detail


def test_tail_miss_skips_periods_without_a_price() -> None:
    day = date(2025, 1, 8)
    frame = _frame({day: [25.0, float("nan"), 15.0, 15.0]})

    (record,) = tail_miss_incidents(
        frame, threshold=0.3, timezone="UTC", fit_window=(day, day)
    )

    assert "1 of 3 periods" in record.detail


def test_observed_incidents_merge_rules_and_cutoff_comes_from_validation(
    settings: Settings, tmp_path: Path
) -> None:
    data = settings.data.model_copy(update={"processed_dir": tmp_path})
    local = settings.model_copy(update={"data": data})
    run = tmp_path / "dashboard" / "run-a"
    run.mkdir(parents=True)
    (tmp_path / "dashboard" / "latest.json").write_text('{"run_id": "run-a"}')
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": "run-a",
                "created_utc": "2026-09-14T16:00:00+00:00",
                "days": [
                    {
                        "date": "2026-06-02",
                        "window": "holdout",
                        "traded": False,
                        "skip_reason": "missing price or forecast",
                    }
                ],
            }
        )
    )
    validation = _frame(
        {date(2026, 5, 1) + timedelta(days=i): [15.0] * 4 for i in range(10)}
        | {date(2026, 5, 11): [25.0, 15.0, 15.0, 15.0]}
    )
    holdout = _frame(
        {
            date(2026, 6, 1): [25.0, 25.0, 15.0, 15.0],
            date(2026, 6, 2): [15.0] * 4,
            date(2026, 6, 3): [15.0] * 4,
        }
    )
    holdout.loc[holdout["target_day"] == date(2026, 6, 2), "fallback_periods"] = 1
    holdout.loc[holdout["target_day"] == date(2026, 6, 3), "fallback_periods"] = 2

    records = observed_incidents(local, validation, holdout)

    summary = [(r.delivery_day, r.type, r.severity) for r in records]
    assert summary == [
        (date(2026, 5, 11), "tail_miss", "warning"),
        (date(2026, 6, 1), "tail_miss", "warning"),
        (date(2026, 6, 2), "data_gap", "critical"),
        (date(2026, 6, 3), "data_gap", "warning"),
    ]
    # Validation ends the day before the hold-out start, 2026-06-01.
    in_sample = {r.delivery_day: r.metrics["in_sample"] for r in records[:2]}
    assert in_sample == {date(2026, 5, 11): 1.0, date(2026, 6, 1): 0.0}
