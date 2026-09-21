"""Delivery days the desk did not bid on, recorded so the gap is visible.

The live pipeline runs only while the machine hosting it is on. A day it missed
leaves no run record and no plan, so without a record of its own it is simply
absent from the Live tab: the track record looks unbroken when it is not. This
module names those days and writes one incident for each, which is the honest
way for an operations record to carry an outage.

A gap is a delivery day that has no **live** run record, between the first day
the desk actually bid and the day asked for. Days before the first bid are not
gaps, because there was no desk yet. A reconstruction (``kind: backfill``) is
not a bid, so backfilling a day does not close its gap: the day stays a day the
battery traded nothing.

A day the pipeline ran and failed on is not a gap either. That run writes its
own critical incident and no run record, and reporting it a second time as a
dark day would both double-count it and say something false: the pipeline was
running, and it is the failure, not the machine, that needs looking at.

The incidents have their own source, ``desk_offline``, so a rerun that finds
fewer gaps drops the ones it no longer finds instead of leaving them behind. A
gap that is still a gap keeps the detection time its first scan recorded: the
day the outage was found does not move every time the scan runs again.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from src.config import Settings, load_settings
from src.health.incidents import (
    Incident,
    default_path,
    load_incidents,
    make_incident_id,
    replace_incidents,
)
from src.pipeline.daily_run import KIND_LIVE
from src.pipeline.daily_run import SOURCE as PIPELINE_SOURCE

__all__ = [
    "SOURCE",
    "desk_offline_incident",
    "gap_days",
    "main",
    "record_gaps",
    "run_kinds",
]

#: Incident source for a day the desk did not bid on.
SOURCE = "desk_offline"


def run_kinds(settings: Settings) -> dict[date, str]:
    """Every delivery day with a run record, and whether it was bid or reconstructed."""
    folder = settings.data.processed_path / "pipeline" / "runs"
    kinds: dict[date, str] = {}
    for path in sorted(folder.glob("*.json")) if folder.exists() else []:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(record, dict) and "target_day" in record:
            day = date.fromisoformat(str(record["target_day"]))
            kinds[day] = str(record.get("kind", KIND_LIVE))
    return kinds


def gap_days(settings: Settings, through: date) -> list[date]:
    """Delivery days up to ``through`` that the desk ran through but did not bid.

    Counted from the first day it did bid, so the days before the pipeline
    existed are not reported as outages, and skipping days the pipeline ran and
    failed on, which report themselves.
    """
    kinds = run_kinds(settings)
    bid = sorted(day for day, kind in kinds.items() if kind == KIND_LIVE)
    if not bid:
        return []
    # A run that produced no forecast wrote this incident instead of a record.
    failed = {
        incident.delivery_day
        for incident in load_incidents(default_path(settings))
        if incident.source == PIPELINE_SOURCE
    }
    day = max(bid[0], settings.evaluation.live_from)
    days: list[date] = []
    while day <= through:
        if kinds.get(day) != KIND_LIVE and day not in failed:
            days.append(day)
        day += timedelta(days=1)
    return days


def desk_offline_incident(day: date, detected_utc: datetime) -> Incident:
    """One dark delivery day, as the health log carries it."""
    return Incident(
        incident_id=make_incident_id(SOURCE, "pipeline", day),
        delivery_day=day,
        detected_utc=detected_utc.replace(microsecond=0),
        type="pipeline",
        severity="critical",
        detail=(
            f"No schedule was committed for {day}: the pipeline did not run on "
            f"{day - timedelta(days=1)}, so the gate closed without a bid and the "
            "battery traded nothing that day."
        ),
        action=(
            "The pipeline runs only while its host machine is on; a scheduled "
            "runner independent of that machine would close this gap"
        ),
        status="review",
        source=SOURCE,
    )


def record_gaps(
    settings: Settings, through: date, *, now_utc: datetime | None = None
) -> list[Incident]:
    """Rewrite the dark-day incidents through ``through`` and return them.

    The whole source is rewritten, so a day that is no longer a gap drops out.
    A day that is still one keeps the time the scan that first found it
    recorded, rather than being redated on every run.
    """
    path = default_path(settings)
    now = (now_utc or datetime.now(UTC)).replace(microsecond=0)
    found_before = {
        incident.incident_id: incident.detected_utc
        for incident in load_incidents(path)
        if incident.source == SOURCE
    }
    gaps = []
    for day in gap_days(settings, through):
        incident = desk_offline_incident(day, now)
        first_seen = found_before.get(incident.incident_id)
        if first_seen is not None:
            incident = incident.model_copy(update={"detected_utc": first_seen})
        gaps.append(incident)
    replace_incidents(gaps, [SOURCE], path)
    return gaps


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record the delivery days the desk did not bid on."
    )
    parser.add_argument("--through", type=date.fromisoformat, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    gaps = record_gaps(settings, args.through)
    if not gaps:
        print(f"no day without a bid through {args.through}")
        return 0
    days = ", ".join(str(incident.delivery_day) for incident in gaps)
    print(f"{len(gaps)} day(s) without a bid through {args.through}: {days}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
