"""Queries behind the Model health tab, over a run's exported ``health`` folder.

``python -m src.export.artifacts --steps health`` copies the Phase 6 experiment
results and the incident log into ``<run>/health/``. These functions read that
folder and return plain dictionaries, like ``api.service``:

* ``regime``: the M1 regime shift backtest, 2021 to 2023;
* ``drift``: the M2 drift monitor over the validation window and the hold-out;
* ``incidents``: the incident log, filtered, faceted and paged;
* ``ops``: the four operations tiles, from D5, M2 and the incident log.

Nothing is recomputed from forecasts and nothing is rounded: values are served as
the experiments wrote them and the dashboard formats them once. Every
endpoint serves ``generated_utc``, when its source was written, from ``index.json``.
The M1 file's ``coverage_alert_threshold`` is not served: the drift threshold comes
from M2.

Each incident carries a provenance category: ``observed`` (fixed rules over saved
backtest outputs, and the live pipeline's own runs), ``measured`` (the drift monitor
over real saved forecasts, in the M2 experiment and in the live daily check) or
``simulated`` (failure injection, D5), and ``in_sample`` when the record says whether
its day lies in the window its threshold was fitted on.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, get_args

from api.service import ArtifactsMissingError, RequestError
from src.health.incidents import OBSERVED, Incident, IncidentType

__all__ = [
    "DRIFT_WINDOWS",
    "HEALTH_FILES",
    "INCIDENT_TYPES",
    "MAX_LIMIT",
    "MEASURED_SOURCES",
    "OBSERVED_SOURCES",
    "SIMULATED_SOURCES",
    "Health",
    "check_incident_query",
    "clear_cache",
    "drift",
    "incidents",
    "load_health",
    "ops",
    "provenance",
    "regime",
]

HEALTH_DIR = "health"
HEALTH_FILES = {
    "regime": "m1_regime_experiment.json",
    "drift": "m2_drift.json",
    "deadline": "d5_deadline.json",
    "incidents": "incidents.json",
    "index": "index.json",
}
DRIFT_WINDOWS = ("validation", "holdout", "all")
INCIDENT_TYPES: tuple[str, ...] = get_args(IncidentType)
MAX_LIMIT = 500
#: Sources whose incidents record real events: the fixed rules over saved backtest
#: outputs, the live pipeline's own runs, and the days it did not run at all.
OBSERVED_SOURCES = frozenset({OBSERVED, "pipeline", "desk_offline"})
#: Sources whose incidents are measured on real saved forecasts: the M2 experiment
#: and the live pipeline's daily drift check.
MEASURED_SOURCES = frozenset({"m2_drift", "live_drift"})
#: Sources whose incidents come from failure injection, not from real operations.
SIMULATED_SOURCES = frozenset({"d5_deadline"})
SOURCE_PATTERN = re.compile(r"[a-z0-9_]+")
EXPORT_HINT = "then python -m src.export.artifacts --steps health"
MISSING = {
    "regime": (
        "the M1 regime shift experiment is not in this run's health export; run "
        f"python -m src.health.experiments.m1_regime_shift, {EXPORT_HINT}"
    ),
    "drift": (
        "the M2 drift monitor is not in this run's health export; run "
        f"python -m src.health.experiments.m2_drift, {EXPORT_HINT}"
    ),
    "incidents": (
        "the incident log is not in this run's health export; write it with the "
        f"Phase 6 experiments, {EXPORT_HINT}"
    ),
}
PERIOD_KEYS = (
    "arm",
    "period",
    "days",
    "coverage_90",
    "coverage_50",
    "pinball",
    "mae_q50",
    "capture_ratio",
    "pnl_eur",
    "perfect_foresight_pnl_eur",
)
SERIES_KEYS = (
    "target_day",
    "window",
    "coverage_90",
    "pinball",
    "rolling_coverage_90",
    "rolling_pinball_ratio",
    "coverage_alert",
    "pinball_alert",
)
DEADLINE_KEYS = (
    "days",
    "on_time_share_with_chain",
    "on_time_share_without_chain",
    "fallback_days",
    "fallback_by_step",
    "latest_submission_minutes_after_issue",
    "capture_full",
    "capture_chain",
    "capture_without_chain",
)


@dataclass(frozen=True)
class Health:
    """One run's health export, loaded once; a file not exported is ``None``."""

    run_id: str
    manifest: dict[str, Any]
    regime: dict[str, Any] | None
    drift: dict[str, Any] | None
    deadline: dict[str, Any] | None
    incidents: tuple[Incident, ...] | None
    index: dict[str, Any] | None

    def generated_utc(self, key: str) -> str | None:
        """When the source of a health file was written, from ``index.json``."""
        if self.index is None:
            return None
        name = HEALTH_FILES[key]
        for entry in self.index.get("files", []):
            if entry.get("file") == name:
                stamp = entry.get("generated_utc")
                return None if stamp is None else str(stamp)
        return None


# --- loading -------------------------------------------------------------------


def _signature(folder: Path) -> tuple[tuple[int, int], ...]:
    """Modification time in nanoseconds and size of the manifest and health files.

    The size catches a rewrite within the same timestamp tick, and ``index.json``
    carries the export time, so a re-export always changes the key.
    """
    paths = [folder / "manifest.json"]
    paths += [folder / HEALTH_DIR / name for name in HEALTH_FILES.values()]
    signature = []
    for path in paths:
        try:
            stat = path.stat()
        except OSError:
            signature.append((-1, -1))
        else:
            signature.append((stat.st_mtime_ns, stat.st_size))
    return tuple(signature)


def load_health(root: Path, run_id: str) -> Health:
    """A run's health files, re-read whenever an export has rewritten one."""
    folder = root / run_id
    if not (folder / "manifest.json").exists():
        raise ArtifactsMissingError(f"run {run_id!r} not found")
    return _read_health(root, run_id, _signature(folder))


def clear_cache() -> None:
    _read_health.cache_clear()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


@lru_cache(maxsize=4)
def _read_health(
    root: Path, run_id: str, signature: tuple[tuple[int, int], ...]
) -> Health:
    del signature  # part of the cache key only
    folder = root / run_id / HEALTH_DIR
    try:
        records = _read_json(folder / HEALTH_FILES["incidents"])
        return Health(
            run_id=run_id,
            manifest=json.loads(
                (root / run_id / "manifest.json").read_text(encoding="utf-8")
            ),
            regime=_read_json(folder / HEALTH_FILES["regime"]),
            drift=_read_json(folder / HEALTH_FILES["drift"]),
            deadline=_read_json(folder / HEALTH_FILES["deadline"]),
            incidents=None
            if records is None
            else tuple(Incident.model_validate(record) for record in records),
            index=_read_json(folder / HEALTH_FILES["index"]),
        )
    except (OSError, ValueError) as exc:
        raise ArtifactsMissingError(
            f"the health files of run {run_id!r} could not be read, probably while "
            "an export rewrites them; retry in a moment"
        ) from exc


# --- helpers ---------------------------------------------------------------------


def _require(data: dict[str, Any] | None, key: str) -> dict[str, Any]:
    if data is None:
        raise ArtifactsMissingError(MISSING[key])
    return data


def _run_match(health: Health, data: dict[str, Any]) -> dict[str, Any]:
    """Whether the M2 series ends on the run's last day, so it belongs to the run."""
    series = data.get("series") or []
    last = series[-1]["target_day"] if series else None
    run_last = health.manifest.get("last_day")
    return {
        "last_target_day": last,
        "run_last_day": run_last,
        "matches_run": last is not None and last == run_last,
    }


def provenance(source: str) -> str:
    """``observed``, ``measured`` (drift monitor) or ``simulated`` (injection).

    A source outside the explicit sets is treated as simulated: it did not come
    from the fixed rules over saved outputs, and nothing vouches that it measured
    real operations.
    """
    if source in OBSERVED_SOURCES:
        return "observed"
    if source in MEASURED_SOURCES:
        return "measured"
    return "simulated"


def _incident_record(incident: Incident) -> dict[str, Any]:
    in_sample = incident.metrics.get("in_sample")
    return incident.model_dump(mode="json") | {
        "provenance": provenance(incident.source),
        "in_sample": None if in_sample is None else bool(in_sample),
    }


# --- endpoints -----------------------------------------------------------------------


def regime(health: Health) -> dict[str, Any]:
    """Rolling 90% coverage per arm and the per-year table of the M1 backtest."""
    data = _require(health.regime, "regime")
    setup = data["setup"]
    groups = [str(group) for group in setup.get("feature_groups", [])]
    rolling = data["rolling_coverage_90"]
    keys = (
        "summary",
        "model",
        "training_days",
        "calibration_days",
        "first_target_day",
        "last_target_day",
        "spike_threshold_eur_mwh",
        "battery",
        "arms",
    )
    return {
        "experiment": data.get("experiment", "m1_regime_shift"),
        "generated_utc": health.generated_utc("regime"),
        "historical_backtest": True,
        "setup": {key: setup.get(key) for key in keys}
        | {"feature_groups": groups, "weather_features": "weather" in groups},
        "coverage_target": data["coverage_target"],
        "rolling_coverage_90": {
            "window_days": rolling["window_days"],
            "dates": rolling["dates"],
            "arms": {arm: list(values) for arm, values in rolling["arms"].items()},
        },
        "min_rolling_coverage_90": data.get("min_rolling_coverage_90", {}),
        "periods": [
            {key: row.get(key) for key in PERIOD_KEYS}
            for row in [*data.get("yearly", []), *data.get("total", [])]
        ],
    }


def drift(health: Health, window: str = "all") -> dict[str, Any]:
    """The M2 rolling signals, thresholds, alert episodes and window summaries."""
    if window not in DRIFT_WINDOWS:
        raise RequestError(
            f"unknown window {window!r}; choose one of {', '.join(DRIFT_WINDOWS)}"
        )
    data = _require(health.drift, "drift")

    def kept(name: str) -> bool:
        return window == "all" or name == window

    return {
        "experiment": "m2_drift",
        "generated_utc": health.generated_utc("drift"),
        "model": data.get("model"),
        "window": window,
        "window_days": data["window_days"],
        "holdout_start": data["holdout_start"],
        "rule": data.get("rule"),
        **_run_match(health, data),
        "thresholds": data["thresholds"],
        "windows": {
            name: summary
            for name, summary in data.get("windows", {}).items()
            if kept(name)
        },
        "episodes": [e for e in data.get("episodes", []) if kept(e["window"])],
        "series": [
            {key: row.get(key) for key in SERIES_KEYS}
            for row in data["series"]
            if kept(row["window"])
        ],
    }


def check_incident_query(
    start: date | None,
    end: date | None,
    types: Sequence[str] | None,
    sources: Sequence[str] | None,
    limit: int,
    offset: int,
) -> None:
    """Refuse a malformed incident query before any file is read."""
    unknown = [t for t in types or () if t not in INCIDENT_TYPES]
    if unknown:
        raise RequestError(
            f"unknown incident type {unknown[0]!r}; choose from "
            f"{', '.join(INCIDENT_TYPES)}"
        )
    for source in sources or ():
        if not SOURCE_PATTERN.fullmatch(source):
            raise RequestError(
                f"source must be lower-case letters, digits and underscores, "
                f"not {source!r}"
            )
    if start is not None and end is not None and start > end:
        raise RequestError(f"start {start} is after end {end}")
    if not 1 <= limit <= MAX_LIMIT:
        raise RequestError(f"limit must be 1 to {MAX_LIMIT}")
    if offset < 0:
        raise RequestError("offset must be 0 or more")


def _count(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def incidents(
    health: Health,
    *,
    start: date | None = None,
    end: date | None = None,
    types: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Incidents newest first, with faceted counts per type and per source.

    ``total`` counts every record matching all filters. The type counts apply every
    filter except the type filter, and the source counts every filter except the
    source filter, so a filter menu can show what each choice would return.
    """
    check_incident_query(start, end, types, sources, limit, offset)
    if health.incidents is None:
        raise ArtifactsMissingError(MISSING["incidents"])
    in_range = [
        incident
        for incident in health.incidents
        if (start is None or incident.delivery_day >= start)
        and (end is None or incident.delivery_day <= end)
    ]

    def type_ok(incident: Incident) -> bool:
        return not types or incident.type in types

    def source_ok(incident: Incident) -> bool:
        return not sources or incident.source in sources

    matched = sorted(
        (i for i in in_range if type_ok(i) and source_ok(i)),
        key=lambda i: (i.delivery_day, i.detected_utc, i.incident_id),
        reverse=True,
    )
    type_counts = dict.fromkeys(INCIDENT_TYPES, 0) | _count(
        i.type for i in in_range if source_ok(i)
    )
    source_counts = _count(i.source for i in in_range if type_ok(i))
    return {
        "generated_utc": health.generated_utc("incidents"),
        "total": len(matched),
        "limit": limit,
        "offset": offset,
        "counts": {
            "type": type_counts,
            "source": source_counts,
            "provenance": dict.fromkeys(("observed", "measured", "simulated"), 0)
            | _count(provenance(i.source) for i in matched),
        },
        "source_provenance": {source: provenance(source) for source in source_counts},
        "incidents": [
            _incident_record(incident) for incident in matched[offset : offset + limit]
        ],
    }


def _deadline(health: Health) -> dict[str, Any]:
    data = health.deadline
    if data is None:
        return {"available": False, "detail": "D5 not exported yet"}
    setup = data.get("setup", {})
    summary = data["summary"]
    first, last = setup.get("days", [None, None])
    return {
        "available": True,
        "source": "d5_deadline",
        "simulation": True,
        "generated_utc": health.generated_utc("deadline"),
        "period": {"first_day": first, "last_day": last},
        "issue_local": setup.get("issue_local"),
        "gate_local": setup.get("gate_local"),
        "failure_rates": setup.get("failure_rates"),
        "seed": setup.get("seed"),
        "realised_failure_rates": data.get(
            "realised_failure_rates", summary.get("realised_failure_rates")
        ),
    } | {key: summary.get(key) for key in DEADLINE_KEYS}


def _fallbacks(health: Health, deadline: dict[str, Any]) -> dict[str, Any]:
    observed = None
    if health.incidents is not None:
        observed = sum(
            1
            for incident in health.incidents
            if incident.source == OBSERVED and "fallback_periods" in incident.metrics
        )
    manifest = health.manifest
    # Observed and simulated fallbacks cover different periods and different kinds
    # of evidence, so they are served side by side and never added.
    common = {
        "observed": observed,
        "observed_rule": "observed data_gap incidents with fallback forecast periods",
        "observed_period": {
            "first_day": manifest.get("first_day"),
            "last_day": manifest.get("last_day"),
        },
        "observed_generated_utc": health.generated_utc("incidents"),
    }
    if not deadline["available"]:
        return {"available": False, "detail": deadline["detail"]} | common
    return {
        "available": True,
        "simulated_days": int(deadline["fallback_days"]),
        "simulated_source": "d5_deadline",
        "simulated_period": deadline["period"],
        "simulated_generated_utc": deadline["generated_utc"],
    } | common


def _drift_now(health: Health) -> dict[str, Any]:
    data = health.drift
    if data is None:
        return {"available": False, "detail": "M2 not exported yet"}
    series = data["series"]
    holdout = [row for row in series if row["window"] == "holdout"]
    last = (holdout or series)[-1]
    thresholds = data["thresholds"]
    return {
        "available": True,
        "source": "m2_drift",
        "generated_utc": health.generated_utc("drift"),
        "as_of": last["target_day"],
        "window": last["window"],
        "window_days": data["window_days"],
        "holdout_start": data["holdout_start"],
        "holdout_days": len(holdout),
        **_run_match(health, data),
        "coverage": {
            "value": last["rolling_coverage_90"],
            "threshold": thresholds["coverage"],
            "alert": bool(last["coverage_alert"]),
        },
        "pinball_ratio": {
            "value": last["rolling_pinball_ratio"],
            "threshold": thresholds["pinball_ratio"],
            "alert": bool(last["pinball_alert"]),
        },
    }


def ops(health: Health) -> dict[str, Any]:
    """The operations tiles; a source not exported yet is marked unavailable."""
    if all(
        part is None
        for part in (health.regime, health.drift, health.deadline, health.incidents)
    ):
        raise ArtifactsMissingError(
            f"run {health.run_id!r} has no health export; run "
            "python -m src.export.artifacts --steps health"
        )
    deadline = _deadline(health)
    return {
        "run_id": health.run_id,
        "deadline": deadline,
        "fallbacks": _fallbacks(health, deadline),
        "drift": _drift_now(health),
    }
