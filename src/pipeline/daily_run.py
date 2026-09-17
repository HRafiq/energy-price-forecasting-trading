"""One day of the live pipeline, from readiness to a committed plan.

    uv run python -m src.pipeline.daily_run --day 2026-09-17
    uv run python -m src.pipeline.daily_run --day 2026-09-16 --settle

The daily job forecasts delivery day D+1 at 11:40 on day D and commits a schedule
before the gate closes at 12:00. This module is what each Airflow task calls, so
the pipeline can also be run by hand, without Airflow:

1. refuse any day before ``evaluation.live_from``, so a live run can never
   re-score the frozen hold-out;
2. check the feeds (:mod:`src.pipeline.readiness`);
3. take the model from the MLflow registry when one is registered, otherwise fit
   the configured production model on the spot;
4. walk the fallback chain (:mod:`src.pipeline.chain`) and keep the first forecast
   it can make;
5. save the forecast, commit the plan (:mod:`src.pipeline.plan`), and write the
   run record;
6. write an incident whenever the chain had to step down, or the forecast missed
   the gate. The command still exits 0: a schedule was committed either way, and
   the lateness lives in the run record and the incident.

Settling is a separate call the next day, once the auction has published the
prices the committed schedule will be paid.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import (
    PRICE_SERIES,
    PRODUCT_COLUMN,
    RESOLUTION_STEP,
    Settings,
    load_settings,
)
from src.forecasting.base import Forecaster, ForecastError, QuantileForecast
from src.health.incidents import (
    Incident,
    IncidentType,
    default_path,
    make_incident_id,
    upsert_incidents,
)
from src.pipeline.chain import STEPS, ChainResult, run_chain
from src.pipeline.model_source import ModelSourceError, load_model
from src.pipeline.plan import (
    Plan,
    load_plan,
    plan_day,
    plan_path,
    save_plan,
    settle_plan,
)
from src.pipeline.readiness import Readiness, check_readiness

__all__ = [
    "SOURCE",
    "IncompleteRunError",
    "PlanExistsError",
    "RunRecord",
    "check_deadline",
    "deadline_missed",
    "main",
    "record_path",
    "run_day",
    "settle_day",
]

SOURCE = "pipeline"
DATA_FEEDS = frozenset({"prices", "load_forecast", "weather", "fuels"})


@dataclass(frozen=True)
class RunRecord:
    """What one live day did, and whether it made the gate."""

    target_day: date
    issued_utc: datetime
    gate_utc: datetime
    on_time: bool
    step: str
    model: str
    model_version: str | None
    readiness: Readiness
    attempts: tuple[dict[str, object], ...]
    planned_value_eur: float
    solve_seconds: float
    incidents: tuple[str, ...] = field(default_factory=tuple)

    @property
    def degraded(self) -> bool:
        return self.step != "production"

    def as_dict(self) -> dict[str, Any]:
        return {
            "target_day": str(self.target_day),
            "issued_utc": self.issued_utc.isoformat(timespec="seconds"),
            "gate_utc": self.gate_utc.isoformat(timespec="seconds"),
            "on_time": self.on_time,
            "minutes_before_gate": round(
                (self.gate_utc - self.issued_utc).total_seconds() / 60, 2
            ),
            "step": self.step,
            "model": self.model,
            "model_version": self.model_version,
            "readiness": self.readiness.as_dict(),
            "attempts": list(self.attempts),
            "planned_value_eur": round(self.planned_value_eur, 2),
            "solve_seconds": round(self.solve_seconds, 3),
            "incidents": list(self.incidents),
        }


def _clock_utc(day: date, clock: str, settings: Settings) -> datetime:
    """A local clock time on ``day``, as a UTC instant."""
    hours, minutes = (int(part) for part in clock.split(":"))
    local = pd.Timestamp(day) + pd.Timedelta(hours=hours, minutes=minutes)
    return local.tz_localize(settings.market.timezone).tz_convert("UTC").to_pydatetime()


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, path)


def record_path(settings: Settings, day: date) -> Path:
    return settings.data.processed_path / "pipeline" / "runs" / f"{day}.json"


def _forecast_path(settings: Settings, day: date) -> Path:
    return settings.data.processed_path / "forecasts" / "production" / f"{day}.parquet"


def _save_forecast(forecast: QuantileForecast, step: str, settings: Settings) -> Path:
    table = forecast.values.copy()
    table.insert(0, "model", forecast.model)
    table["target_day"] = forecast.target_day
    table["issue_time_utc"] = forecast.issue_time_utc
    table["chain_step"] = step
    path = _forecast_path(settings, forecast.target_day)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    table.to_parquet(partial)
    os.replace(partial, path)
    return path


def _model_from_registry(
    settings: Settings,
) -> tuple[Forecaster | None, str | None, str]:
    """The registered model, or nothing and the reason the chain must fit one."""
    try:
        model, version = load_model(settings)
    except ModelSourceError as exc:
        return None, None, str(exc)
    return model, version, ""


def _chain_incident(
    record_day: date,
    result: ChainResult,
    readiness: Readiness,
    issued_utc: datetime,
    on_time: bool,
    settings: Settings,
) -> Incident:
    missing = set(readiness.missing)
    kind: IncidentType = "late_data" if missing & DATA_FEEDS else "pipeline"
    local = pd.Timestamp(issued_utc).tz_convert(settings.market.timezone)
    clock = local.strftime("%H:%M")
    if missing:
        cause = f"feeds missing at issue time: {', '.join(sorted(missing))}"
    elif result.degraded:
        cause = "the production model could not run"
    else:
        cause = "every feed had arrived"
    return Incident(
        incident_id=make_incident_id(SOURCE, kind, record_day),
        delivery_day=record_day,
        detected_utc=issued_utc.replace(microsecond=0),
        type=kind,
        severity="warning" if on_time else "critical",
        detail=(
            f"Live run for {record_day}: {cause}. The {result.label} forecast was "
            f"submitted at {clock} local, "
            f"{'before' if on_time else 'after'} the "
            f"{settings.market.gate_closure_local} gate."
        ),
        action=(
            f"Fallback: {result.label} used; forecast issued {clock}"
            if result.degraded
            else f"Forecast issued {clock} by the {result.label}"
        ),
        status="resolved" if on_time else "review",
        source=SOURCE,
        metrics={"minutes_before_gate": 0.0},
    )


class PlanExistsError(FileExistsError):
    """A schedule is already committed for the day, and a rerun must not replace it."""


class IncompleteRunError(RuntimeError):
    """A plan was saved but the run that saved it never wrote its run record."""


def run_day(
    settings: Settings,
    target_day: date,
    *,
    frame: pd.DataFrame | None = None,
    use_registry: bool = True,
    now_utc: datetime | None = None,
    replace: bool = False,
) -> RunRecord:
    """Forecast and commit one delivery day, and record what happened.

    A day that already has a committed schedule is refused unless ``replace`` is set.
    """
    if target_day < settings.evaluation.live_from:
        raise ValueError(
            f"{target_day} is before live_from {settings.evaluation.live_from}; "
            "the live pipeline does not re-run evaluated days"
        )
    committed = plan_path(settings, target_day)
    recorded = record_path(settings, target_day)
    if not replace and recorded.exists():
        # A bid submitted at the gate is final. Airflow starts a run for the latest
        # slot it missed, and without this a rerun would commit a second, later
        # schedule over the one on record and rewrite its run record as late. The
        # run record is written last, so its presence means that run finished.
        raise PlanExistsError(
            f"a schedule for {target_day} is already committed at {committed}"
        )
    if not replace and committed.exists():
        # A plan without its run record: a run crashed after committing. Skipping
        # would leave the deadline check without a record, and re-forecasting would
        # replace the bid with a later one, so a person decides.
        raise IncompleteRunError(
            f"a schedule for {target_day} exists at {committed} but its run record "
            f"{recorded} does not; inspect it, then rerun with --replace to redo it"
        )
    issued_utc = now_utc or datetime.now(UTC)
    issue_day = target_day - timedelta(days=1)
    gate_utc = _clock_utc(issue_day, settings.market.gate_closure_local, settings)
    on_time = issued_utc < gate_utc

    inputs = frame if frame is not None else pd.read_parquet(settings.data.inputs_path)
    readiness = check_readiness(inputs, target_day, settings)

    model: Forecaster | None = None
    version: str | None = None
    registry_note = "registry not used"
    if use_registry:
        model, version, registry_note = _model_from_registry(settings)

    try:
        result = run_chain(inputs, target_day, settings, readiness, model=model)
    except ForecastError as exc:
        failure = Incident(
            incident_id=make_incident_id(SOURCE, "pipeline", target_day),
            delivery_day=target_day,
            detected_utc=issued_utc.replace(microsecond=0),
            type="pipeline",
            severity="critical",
            detail=f"Live run for {target_day} produced no forecast: {exc}",
            action="No schedule was committed; the gate closed without a bid",
            status="review",
            source=SOURCE,
        )
        upsert_incidents([failure], default_path(settings))
        raise

    _save_forecast(result.forecast, result.step, settings)
    step_minutes = int(
        RESOLUTION_STEP[settings.data.modeling_resolution] / pd.Timedelta(minutes=1)
    )
    products = (
        inputs[PRODUCT_COLUMN]
        .reindex(result.forecast.values.index)
        .fillna(step_minutes)
        .astype("int64")
    )
    plan = plan_day(
        result.forecast.values,
        products,
        settings,
        target_day=target_day,
        model=result.forecast.model,
        step=result.step,
        issued_utc=issued_utc,
    )
    save_plan(plan, settings)

    written: list[str] = []
    if result.degraded or not on_time:
        incident = _chain_incident(
            target_day, result, readiness, issued_utc, on_time, settings
        )
        minutes = round((gate_utc - issued_utc).total_seconds() / 60, 2)
        incident = incident.model_copy(
            update={"metrics": {"minutes_before_gate": minutes}}
        )
        upsert_incidents([incident], default_path(settings))
        written.append(incident.incident_id)

    record = RunRecord(
        target_day=target_day,
        issued_utc=issued_utc,
        gate_utc=gate_utc,
        on_time=on_time,
        step=result.step,
        model=result.forecast.model,
        # The registered version belongs to the production rung. A fallback step
        # forecast with a baseline the registry never served, so it has no version.
        model_version=version if result.step == STEPS[0].name else None,
        readiness=readiness,
        attempts=tuple(attempt.as_dict() for attempt in result.attempts),
        planned_value_eur=plan.planned_value_eur,
        solve_seconds=plan.solve_seconds,
        incidents=tuple(written),
    )
    payload = record.as_dict()
    payload["registry"] = registry_note or f"model version {version}"
    _write_json(payload, record_path(settings, target_day))
    return record


def settle_day(
    settings: Settings, day: date, *, frame: pd.DataFrame | None = None
) -> dict[str, Any]:
    """Value the committed plan for ``day`` at the prices the auction cleared."""
    plan: Plan = load_plan(settings, day)
    inputs = frame if frame is not None else pd.read_parquet(settings.data.inputs_path)
    settlement = settle_plan(plan, inputs[PRICE_SERIES], settings.battery)
    payload = {
        "target_day": str(day),
        "step": plan.step,
        "model": plan.model,
        "strategy": plan.strategy,
        "planned_value_eur": round(plan.planned_value_eur, 2),
        "revenue_eur": round(settlement.revenue_eur, 2),
        "degradation_eur": round(settlement.degradation_eur, 2),
        "pnl_eur": round(settlement.pnl_eur, 2),
        "charged_mwh": round(settlement.charged_mwh, 4),
        "discharged_mwh": round(settlement.discharged_mwh, 4),
        "cycles": round(settlement.cycles, 4),
        "settled_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_json(
        payload,
        settings.data.processed_path / "pipeline" / "settlements" / f"{day}.json",
    )
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one day of the live pipeline.")
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--settle", action="store_true", help="settle the committed plan for that day"
    )
    parser.add_argument(
        "--check-deadline",
        action="store_true",
        help="report whether the schedule for that day made the gate",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="fit the production model instead of loading a registered one",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="replace a schedule already committed for that day; a real bid is "
        "final at the gate, so use this only to repair a broken run",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    if args.check_deadline:
        on_time = check_deadline(settings, args.day)
        state = "before the gate" if on_time else "after the gate, incident written"
        print(f"{args.day}: schedule committed {state}")
        return 0
    if args.settle:
        print(json.dumps(settle_day(settings, args.day), indent=2))
        return 0

    try:
        record = run_day(
            settings, args.day, use_registry=not args.no_registry, replace=args.replace
        )
    except PlanExistsError as exc:
        # Not a failure: the day is already traded, so the DAG carries on to the
        # export, the deadline check and settlement with the schedule on record.
        print(f"{args.day}: not replaced, {exc}; pass --replace to overwrite it")
        return 0
    print(json.dumps(record.as_dict(), indent=2))
    # A committed schedule is a success even when it was late: lateness is carried
    # by the run record and its incident. Failing here would make the DAG run the
    # fallback branch a second time for a day that already has a schedule.
    return 0


def check_deadline(settings: Settings, day: date) -> bool:
    """True when the schedule for ``day`` was committed before the gate.

    Reads the run record the forecast wrote. A late run gets one critical
    incident, so a missed gate is reported in the health log rather than only in
    the scheduler's history.
    """
    path = record_path(settings, day)
    if not path.exists():
        raise FileNotFoundError(f"no run record for {day} at {path}")
    record = json.loads(path.read_text(encoding="utf-8"))
    if bool(record["on_time"]):
        return True
    deadline_missed(str(day))
    return False


def deadline_missed(target_day: str | None = None) -> str:
    """Record that a run did not finish before its deadline.

    Airflow calls this from the DAG's deadline alert. It writes a pipeline
    incident so a missed deadline appears in the Model health log beside the
    fallbacks, rather than only in the scheduler's own history.
    """
    settings = load_settings()
    day = date.fromisoformat(target_day) if target_day else date.today()
    detected = datetime.now(UTC).replace(microsecond=0)
    incident = Incident(
        incident_id=make_incident_id(SOURCE, "pipeline", day, "deadline"),
        delivery_day=day,
        detected_utc=detected,
        type="pipeline",
        severity="critical",
        detail=(
            f"The daily run for {day} passed its deadline before the dashboard "
            "export finished."
        ),
        action="Check the scheduler: the schedule may have been committed late",
        status="review",
        source=SOURCE,
    )
    upsert_incidents([incident], default_path(settings))
    return incident.incident_id


if __name__ == "__main__":
    raise SystemExit(main())
