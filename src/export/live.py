"""The live record, exported for the dashboard's Live tab.

The daily pipeline leaves one run record per delivery day (``pipeline/runs``),
one settlement once the day's prices are published (``pipeline/settlements``),
incidents in the health log with source ``pipeline``, ``live_drift`` and
``desk_offline``, and the drift check's summary
(``health/live_drift.json``). This module joins them into
one file, ``<dashboard root>/live.json``, one entry per delivery day from
``evaluation.live_from`` on, with the totals a reader wants first: how many days
are live, how many bids made the gate, what settled, and how the plan compared.

A day the desk never bid on may carry a reconstruction: the same chain, run
after the fact on the data that was available before its gate, to show what the
model would have bid. It is recorded with ``kind`` ``backfill`` and it is kept
out of every total that describes live trading, because it was not a bid. The
totals count live days only; reconstructions are counted separately.

Nothing here recomputes anything: every figure is copied from a record the
pipeline wrote, so the tab shows what happened, as it was recorded.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings
from src.health.incidents import default_path, load_incidents

__all__ = ["LIVE_FILE", "LIVE_SOURCES", "build_live_record", "export_live"]

LIVE_FILE = "live.json"
#: Incident sources the live pipeline writes.
LIVE_SOURCES = frozenset({"pipeline", "live_drift", "desk_offline"})
KIND_LIVE = "live"


def _read(path: Path) -> dict[str, Any] | None:
    """A JSON object from ``path``, or None when absent, malformed or not an object."""
    if not path.exists():
        return None
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def _gate_verdict(record: dict[str, Any]) -> bool:
    """Whether a live bid made its gate, as the run record put it."""
    return bool(record.get("on_time", False))


def _number(value: object) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _local_clock(stamp: str | None, timezone: str) -> str | None:
    if not stamp:
        return None
    return pd.Timestamp(stamp).tz_convert(timezone).strftime("%Y-%m-%d %H:%M")


def build_live_record(settings: Settings) -> dict[str, Any]:
    """Every live delivery day, its settlement if any, its incidents, and totals."""
    processed = settings.data.processed_path
    tz = settings.market.timezone
    live_from = settings.evaluation.live_from
    runs_dir = processed / "pipeline" / "runs"
    settlements_dir = processed / "pipeline" / "settlements"
    incidents = [
        incident
        for incident in load_incidents(default_path(settings))
        if incident.source in LIVE_SOURCES
    ]
    by_day: dict[str, list[dict[str, Any]]] = {}
    for incident in incidents:
        by_day.setdefault(str(incident.delivery_day), []).append(
            {
                "incident_id": incident.incident_id,
                "type": incident.type,
                "severity": incident.severity,
                "status": incident.status,
                "source": incident.source,
                "detail": incident.detail,
            }
        )

    records: dict[str, dict[str, Any]] = {}
    run_paths = sorted(runs_dir.glob("*.json")) if runs_dir.exists() else []
    for path in run_paths:
        record = _read(path)
        if record is not None and "target_day" in record:
            records[str(record["target_day"])] = record
    # A run that produced no forecast, and a day the desk never ran at all,
    # wrote an incident but no run record; the day still gets a row, with no
    # step, so the missed bid is not hidden.
    incident_days = {
        day
        for day, items in by_day.items()
        if any(i["source"] in ("pipeline", "desk_offline") for i in items)
    }
    days: list[dict[str, Any]] = []
    for day in sorted(set(records) | incident_days):
        if day < str(live_from):
            continue
        record = records.get(day, {})
        settlement = _read(settlements_dir / f"{day}.json")
        readiness = record.get("readiness")
        refit = record.get("refit")
        # A run record written before the kind field existed was a live bid:
        # nothing else could write one at the time.
        kind = str(record.get("kind", KIND_LIVE)) if record else None
        days.append(
            {
                "target_day": day,
                # None for a day with no run record at all: the desk was dark.
                "kind": kind,
                "issued_local": _local_clock(record.get("issued_utc"), tz),
                # None on a day that was not bid: it had no gate to make or
                # miss, and False there would read as a missed gate.
                "on_time": _gate_verdict(record) if kind == KIND_LIVE else None,
                "minutes_before_gate": _number(record.get("minutes_before_gate")),
                "step": record.get("step"),
                "model": record.get("model"),
                "model_version": record.get("model_version"),
                "feeds_missing": list(readiness.get("missing", []))
                if isinstance(readiness, dict)
                else [],
                "refit": refit.get("status") if isinstance(refit, dict) else None,
                "planned_value_eur": _number(record.get("planned_value_eur")),
                "settled": settlement is not None,
                "pnl_eur": _number(settlement.get("pnl_eur")) if settlement else None,
                "cycles": _number(settlement.get("cycles")) if settlement else None,
                "settled_local": _local_clock(
                    settlement.get("settled_utc") if settlement else None, tz
                ),
                "incidents": by_day.get(day, []),
            }
        )

    # Every total below describes live trading, so it is computed over the days
    # the desk actually bid. A reconstruction is reported beside them, never
    # inside them: counting one as a bid would overstate the record.
    live = [d for d in days if d["kind"] == KIND_LIVE]
    reconstructed = [d for d in days if d["kind"] not in (None, KIND_LIVE)]
    settled = [d for d in live if d["settled"]]
    steps = Counter(str(d["step"]) for d in live)
    drift = _read(processed / "health" / "live_drift.json")
    totals = {
        "days": len(live),
        "dark_days": sum(1 for d in days if d["kind"] is None),
        "reconstructed_days": len(reconstructed),
        "settled_days": len(settled),
        "on_time_days": sum(1 for d in live if d["on_time"]),
        "not_bid_days": sum(1 for d in days if d["kind"] != KIND_LIVE),
        "production_days": steps.get("production", 0),
        "steps": dict(sorted(steps.items())),
        "settled_pnl_eur": float(sum(d["pnl_eur"] or 0.0 for d in settled)),
        "planned_value_settled_days_eur": float(
            sum(d["planned_value_eur"] or 0.0 for d in settled)
        ),
        "open_incidents": sum(
            1 for d in days for i in d["incidents"] if i["status"] == "review"
        ),
    }
    return {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "live_from": str(live_from),
        "timezone": tz,
        "gate_local": settings.market.gate_closure_local,
        "totals": totals,
        "drift": (
            {
                "status": drift.get("status"),
                "warming_up": drift.get("warming_up"),
                "scored_days": drift.get("scored_days"),
                "window_days": drift.get("window_days"),
                "through": drift.get("through"),
                "latest": drift.get("latest"),
                "skipped": drift.get("skipped", []),
            }
            if drift
            else None
        ),
        "days": days,
    }


def export_live(settings: Settings, root: Path) -> dict[str, Any]:
    """Write ``live.json`` at the dashboard root and return its contents."""
    payload = build_live_record(settings)
    root.mkdir(parents=True, exist_ok=True)
    target = root / LIVE_FILE
    partial = target.with_name(f".{target.name}.partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    partial.replace(target)
    return payload
