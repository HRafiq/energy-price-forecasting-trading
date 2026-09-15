"""The incident log: dated records of what went wrong and what was done about it.

The Model health tab lists these records. Injected experiments and genuinely
observed events both land here, and later pipeline stages write to the same store,
so the interface is deliberately small:

* ``Incident``: one record, validated by pydantic;
* ``make_incident_id``: a deterministic id, so a rerun replaces a record instead
  of duplicating it;
* ``upsert_incidents``, ``replace_incidents``, ``load_incidents`` and
  ``filter_incidents`` over a JSON Lines file, by default
  ``data/processed/health/incidents.jsonl``. ``upsert_incidents`` adds to the
  log; ``replace_incidents`` swaps out every record of the given sources, so a
  rerun that no longer finds an incident removes it. Every write
  goes to a temporary file in the same folder and is moved into place with
  ``os.replace``, so a reader never sees a half-written log.

Observed incidents are derived from saved Phase 4 and Phase 5 outputs by fixed
rules, decided before counting and never adjusted to change the counts:

* **data_gap, day not traded.** Every delivery day the latest dashboard export
  (``data/processed/dashboard/<run>/manifest.json``, the run named in
  ``latest.json``) lists with ``traded == false``. Severity critical, because no
  schedule was traded; status resolved, because the backtest skipped the day and
  listed the reason. ``detected_utc`` is the export's ``created_utc``.
* **data_gap, fallback forecast.** Every delivery day on which the production
  model's saved forecasts (validation comparison and hold-out files) needed
  fallback periods, ``fallback_periods > 0``. Severity warning, status resolved.
  ``detected_utc`` is the forecast issue time on the day before delivery.
* **tail_miss.** For each delivery day with published prices, the share of
  periods whose realised price lies outside the production model's q05 to q95
  range. The cut-off is the 99th percentile (numpy's default linear
  interpolation) of that daily share over the validation window only
  (``evaluation.validation_start`` to the day before ``evaluation.holdout_start``),
  and it is applied unchanged to validation and hold-out days. A day whose share
  is at or above the cut-off is a tail miss. The detail names the day's worst
  miss: the period whose realised price is furthest outside the range, with the
  price and the band edge it crossed. A validation day that is flagged helped
  set the cut-off, so its detail says it is in-sample and its
  ``metrics["in_sample"]`` is 1.0; hold-out days get 0.0. Severity warning,
  status review.
  ``detected_utc`` is the start of the local day after delivery, when a daily
  monitoring batch would run.

The two data_gap rules share one id per delivery day and source, so a day that is
both untraded and fallback-forecast is a single record, the untraded one.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from src.config import Settings

__all__ = [
    "IN_SAMPLE_NOTE",
    "OBSERVED",
    "TAIL_MISS_PERCENTILE",
    "Incident",
    "IncidentType",
    "Severity",
    "Status",
    "daily_outside_share",
    "default_path",
    "fallback_incidents",
    "filter_incidents",
    "load_incidents",
    "make_incident_id",
    "observed_incidents",
    "replace_incidents",
    "tail_miss_incidents",
    "tail_miss_threshold",
    "untraded_day_incidents",
    "upsert_incidents",
]

IncidentType = Literal[
    "data_gap", "late_data", "tail_miss", "drift", "pipeline", "drawdown"
]
Severity = Literal["info", "warning", "critical"]
Status = Literal["resolved", "review"]

OBSERVED = "observed"
#: Appended to the detail of an incident whose day helped fit its own threshold.
IN_SAMPLE_NOTE = (
    "In-sample: its threshold was fitted on the validation window that contains "
    "this day."
)
#: Percentile of the validation daily outside-range share that marks a tail miss.
TAIL_MISS_PERCENTILE = 99.0


class Incident(BaseModel):
    """One dated incident as the Model health tab shows it."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    incident_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    delivery_day: date
    detected_utc: datetime
    type: IncidentType
    severity: Severity
    detail: str = Field(min_length=1)
    action: str = Field(min_length=1)
    status: Status
    source: str = Field(pattern=r"^[a-z0-9_]+$")
    metrics: dict[str, float] = Field(default_factory=dict)

    @field_validator("detected_utc")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("detected_utc must be timezone-aware UTC")
        return value


def make_incident_id(
    source: str, incident_type: str, delivery_day: date, key: str = ""
) -> str:
    """Stable id from source, type and delivery day, plus an optional key.

    ``key`` separates incidents that share all three, for example two drift
    signals whose alerts start on the same day.
    """
    text = f"{source}|{incident_type}|{delivery_day.isoformat()}|{key}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def default_path(settings: Settings) -> Path:
    return settings.data.processed_path / "health" / "incidents.jsonl"


# --- store -----------------------------------------------------------------


def load_incidents(path: Path) -> list[Incident]:
    """Every record in the log, in file order; an absent file is an empty log."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [Incident.model_validate_json(line) for line in fh if line.strip()]


def _sort_key(incident: Incident) -> tuple[date, str, str]:
    return incident.delivery_day, incident.type, incident.incident_id


def upsert_incidents(incidents: Iterable[Incident], path: Path) -> list[Incident]:
    """Insert or replace records by id, write atomically, return the full log.

    The file is kept sorted by delivery day, type and id, so the same records
    always produce the same bytes and a repeated upsert changes nothing.
    """
    merged = {incident.incident_id: incident for incident in load_incidents(path)}
    for incident in incidents:
        merged[incident.incident_id] = incident
    return _write_records(merged.values(), path)


def replace_incidents(
    incidents: Iterable[Incident], sources: Iterable[str], path: Path
) -> list[Incident]:
    """Replace every record of ``sources`` with ``incidents``, return the full log.

    Existing records whose source is in ``sources`` are removed first, so a rerun
    that no longer finds an incident also drops it from the log. Records of other
    sources are kept as they are. The write is atomic and sorted like
    ``upsert_incidents``; a repeated replace changes nothing.
    """
    replaced = set(sources)
    batch = list(incidents)
    outside = sorted({i.source for i in batch} - replaced)
    if outside:
        raise ValueError(f"incidents from sources not being replaced: {outside}")
    merged = {
        incident.incident_id: incident
        for incident in load_incidents(path)
        if incident.source not in replaced
    }
    for incident in batch:
        merged[incident.incident_id] = incident
    return _write_records(merged.values(), path)


def _write_records(incidents: Iterable[Incident], path: Path) -> list[Incident]:
    records = sorted(incidents, key=_sort_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            for incident in records:
                fh.write(incident.model_dump_json() + "\n")
        # mkstemp creates the file readable by its owner only; a log is shared.
        os.chmod(temp_name, 0o644)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise
    return records


def filter_incidents(
    incidents: Iterable[Incident],
    *,
    start: date | None = None,
    end: date | None = None,
    types: Sequence[str] | None = None,
    sources: Sequence[str] | None = None,
) -> list[Incident]:
    """Records with ``start <= delivery_day <= end`` and a listed type and source."""
    return [
        incident
        for incident in incidents
        if (start is None or incident.delivery_day >= start)
        and (end is None or incident.delivery_day <= end)
        and (types is None or incident.type in types)
        and (sources is None or incident.source in sources)
    ]


# --- observed rules --------------------------------------------------------


def _utc(timestamp: pd.Timestamp) -> datetime:
    return timestamp.tz_convert("UTC").to_pydatetime()


def _day_after_local_start(day: date, timezone: str) -> datetime:
    start = pd.Timestamp(datetime.combine(day + timedelta(days=1), time()))
    return _utc(start.tz_localize(timezone))


def untraded_day_incidents(manifest: Mapping[str, Any]) -> list[Incident]:
    """A critical data_gap for every manifest day marked as not traded."""
    detected = _utc(pd.Timestamp(manifest["created_utc"]))
    run_id = str(manifest.get("run_id", "the dashboard export"))
    incidents = []
    for entry in manifest["days"]:
        if entry.get("traded", True):
            continue
        day = date.fromisoformat(entry["date"])
        reason = entry.get("skip_reason") or "no reason recorded"
        incidents.append(
            Incident(
                incident_id=make_incident_id(OBSERVED, "data_gap", day),
                delivery_day=day,
                detected_utc=detected,
                type="data_gap",
                severity="critical",
                detail=(
                    f"Delivery day {day} was not traded in {entry.get('window')} "
                    f"run {run_id}: {reason}."
                ),
                action=(
                    "Backtest skipped the day and listed it with the reason in the "
                    "run notes; no schedule was traded."
                ),
                status="resolved",
                source=OBSERVED,
            )
        )
    return incidents


def fallback_incidents(forecasts: pd.DataFrame, settings: Settings) -> list[Incident]:
    """A warning data_gap for every day whose forecast needed fallback periods."""
    per_day = forecasts.groupby("target_day")["fallback_periods"].max()
    periods = forecasts.groupby("target_day").size()
    issue_hours, issue_minutes = settings.market.forecast_issue_local.split(":")
    incidents = []
    for day, fallback in per_day[per_day > 0].items():
        assert isinstance(day, date)
        issue = pd.Timestamp(
            datetime.combine(
                day - timedelta(days=1), time(int(issue_hours), int(issue_minutes))
            )
        ).tz_localize(settings.market.timezone)
        incidents.append(
            Incident(
                incident_id=make_incident_id(OBSERVED, "data_gap", day),
                delivery_day=day,
                detected_utc=_utc(issue),
                type="data_gap",
                severity="warning",
                detail=(
                    f"The production forecast for {day} needed a fallback source "
                    f"in {int(fallback)} of {int(periods[day])} periods."
                ),
                action=(
                    "Fallback inputs used for those periods; forecast issued at "
                    f"{settings.market.forecast_issue_local}."
                ),
                status="resolved",
                source=OBSERVED,
                metrics={"fallback_periods": float(fallback)},
            )
        )
    return incidents


def daily_outside_share(forecasts: pd.DataFrame) -> pd.Series:
    """Share of priced periods per day with the realised price outside q05 to q95."""
    valid = forecasts[forecasts["actual"].notna()]
    outside = (valid["actual"] < valid["q05"]) | (valid["actual"] > valid["q95"])
    return outside.groupby(valid["target_day"]).mean().astype("float64")


def tail_miss_threshold(validation_forecasts: pd.DataFrame) -> float:
    """The 99th percentile of the daily outside-range share over the given days.

    Pass validation-window forecasts only; the caller does the filtering.
    """
    shares = daily_outside_share(validation_forecasts).to_numpy(dtype="float64")
    if shares.size == 0:
        raise ValueError("no priced days to set the tail-miss threshold from")
    return float(np.percentile(shares, TAIL_MISS_PERCENTILE))


def tail_miss_incidents(
    forecasts: pd.DataFrame,
    threshold: float,
    timezone: str,
    *,
    fit_window: tuple[date, date],
) -> list[Incident]:
    """A tail_miss for every day whose outside-range share reaches ``threshold``.

    ``fit_window`` is the first and last day, inclusive, of the window the
    threshold was fitted on. A flagged day inside it is marked in-sample, in the
    detail and with ``metrics["in_sample"] = 1.0``; any other day gets 0.0.
    """
    fit_first, fit_last = fit_window
    valid = forecasts[forecasts["actual"].notna()]
    shares = daily_outside_share(valid)
    below = (valid["q05"] - valid["actual"]).clip(lower=0.0)
    above = (valid["actual"] - valid["q95"]).clip(lower=0.0)
    distance = pd.concat([below, above], axis=1).max(axis=1)
    incidents = []
    for day, share in shares[shares >= threshold].items():
        assert isinstance(day, date)
        in_day = np.asarray(valid["target_day"] == day, dtype=bool)
        rows = valid[in_day]
        day_distance = distance[in_day]
        position = int(np.argmax(day_distance.to_numpy(dtype="float64")))
        worst = rows.index[position]
        row = rows.iloc[position]
        high = float(above[in_day].iloc[position]) > 0.0
        edge_name = "q95" if high else "q05"
        edge = float(row[edge_name])
        actual = float(row["actual"])
        stamp = pd.Timestamp(worst).tz_convert(timezone)
        outside = int(np.count_nonzero(day_distance.to_numpy(dtype="float64") > 0.0))
        in_sample = fit_first <= day <= fit_last
        incidents.append(
            Incident(
                incident_id=make_incident_id(OBSERVED, "tail_miss", day),
                delivery_day=day,
                detected_utc=_day_after_local_start(day, timezone),
                type="tail_miss",
                severity="warning",
                detail=(
                    f"Realised price left the 90% range in {outside} of {len(rows)} "
                    f"periods ({share:.1%}, rule cut-off {threshold:.1%}). Worst "
                    f"miss at {stamp:%H:%M} local: {actual:.2f} €/MWh against "
                    f"{edge_name} {edge:.2f} €/MWh."
                    + (f" {IN_SAMPLE_NOTE}" if in_sample else "")
                ),
                action="Logged for forecast review; no automatic action.",
                status="review",
                source=OBSERVED,
                metrics={
                    "outside_share": float(share),
                    "threshold": threshold,
                    "worst_actual_eur_mwh": actual,
                    "worst_band_edge_eur_mwh": edge,
                    "worst_miss_eur_mwh": float(day_distance.max()),
                    "in_sample": 1.0 if in_sample else 0.0,
                },
            )
        )
    return incidents


def _latest_manifest(settings: Settings) -> dict[str, Any] | None:
    root = settings.data.processed_path / "dashboard"
    latest = root / "latest.json"
    if not latest.exists():
        return None
    run_id = json.loads(latest.read_text(encoding="utf-8"))["run_id"]
    manifest = root / run_id / "manifest.json"
    if not manifest.exists():
        return None
    loaded: dict[str, Any] = json.loads(manifest.read_text(encoding="utf-8"))
    return loaded


def observed_incidents(
    settings: Settings, validation: pd.DataFrame, holdout: pd.DataFrame
) -> list[Incident]:
    """Observed incidents from the saved manifest and production forecasts.

    ``validation`` must hold only validation-window forecasts; it alone sets the
    tail-miss cut-off, which is then applied to both frames.
    """
    both = pd.concat([validation, holdout])
    incidents: dict[str, Incident] = {}
    for incident in fallback_incidents(both, settings):
        incidents[incident.incident_id] = incident
    manifest = _latest_manifest(settings)
    if manifest is not None:
        # An untraded day overrides a fallback record with the same id.
        for incident in untraded_day_incidents(manifest):
            incidents[incident.incident_id] = incident
    threshold = tail_miss_threshold(validation)
    ev = settings.evaluation
    fit_window = (ev.validation_start, ev.holdout_start - timedelta(days=1))
    for incident in tail_miss_incidents(
        both, threshold, settings.market.timezone, fit_window=fit_window
    ):
        incidents[incident.incident_id] = incident
    return sorted(incidents.values(), key=_sort_key)
