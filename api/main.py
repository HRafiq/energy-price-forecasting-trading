"""FastAPI app serving the dashboard: thin read-only endpoints over ``api.service``.

    uv run uvicorn api.main:app --port 8000

Every endpoint lives under ``/api``; the contract is in ``docs/dashboard_api.md``.
The Model health tab reads ``/api/model-health/*``, over ``api.health``;
``/api/health`` stays the service status.
When ``frontend/dist`` exists, the built React app is served at ``/``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from api import health as model_health
from api import service
from src.config import REPO_ROOT, load_settings
from src.narration import grounding
from src.narration import payload as narration_payload
from src.narration import provider as narration_provider

__all__ = ["app", "dashboard_root"]

T = TypeVar("T")
WindowKey = Literal["last30", "last90", "validation", "holdout", "all"]
StrategyKey = Literal["median", "q25", "q10"]
FRONTEND = REPO_ROOT / "frontend" / "dist"


@lru_cache(maxsize=1)
def _default_root() -> Path:
    return load_settings().data.processed_path / "dashboard"


def dashboard_root() -> Path:
    """Where the runs live; ``DASHBOARD_ROOT`` overrides the configured folder."""
    override = os.environ.get("DASHBOARD_ROOT")
    return Path(override) if override else _default_root()


app = FastAPI(
    title="Battery trading dashboard API",
    description="Read-only access to the DE-LU battery backtest artifacts.",
    version="1.0.0",
)

Root = Annotated[Path, Depends(dashboard_root)]
Window = Annotated[WindowKey, Query()]
Duration = Annotated[int, Query(ge=1, le=4)]
Degradation = Annotated[int, Query(ge=0, le=25)]
StrategyQuery = Annotated[StrategyKey, Query()]
Power = Annotated[float, Query(ge=0.5, le=5.0)]
DriftWindow = Literal["validation", "holdout", "all"]


def _call(function: Callable[[], T]) -> T:
    """Turn service errors into HTTP answers with a readable detail."""
    try:
        return function()
    except (service.ArtifactsMissingError, service.SolverError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except service.DayNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except service.RequestError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run(root: Path, run_id: str | None) -> service.Run:
    def load() -> service.Run:
        chosen = run_id or service.latest_run_id(root)
        if run_id is not None and run_id not in service.list_run_ids(root):
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return service.load_run(root, chosen)

    return _call(load)


def _day(text: str) -> date:
    return _call(lambda: service.parse_day(text))


@app.get("/api/health")
def health(root: Root) -> dict[str, Any]:
    run = _run(root, None)
    info = service.run_info(run)
    return {
        "status": "ok",
        "run": run.run_id,
        "run_kind": info["run_kind"],
        "traded_days": info["traded_days"],
        "grid_available": info["grid_available"],
        "mode": info["mode"],
    }


@app.get("/api/runs")
def runs(root: Root) -> list[dict[str, Any]]:
    return [
        service.run_info(_run(root, run_id)) for run_id in service.list_run_ids(root)
    ]


@app.get("/api/days")
def days(root: Root, run: str | None = None) -> dict[str, Any]:
    return {"days": _run(root, run).manifest["days"]}


@app.get("/api/runs/{run_id}/summary")
def run_summary(
    run_id: str,
    root: Root,
    window: Window = "last30",
    duration: Duration = 2,
    degradation: Degradation = 8,
    strategy: StrategyQuery = "median",
    power: Power = 1.0,
) -> dict[str, Any]:
    loaded = _run(root, run_id)
    return _call(
        lambda: service.summary(loaded, window, duration, degradation, strategy, power)
    )


@app.get("/api/forecast")
def forecast(root: Root, date: str, run: str | None = None) -> dict[str, Any]:
    loaded, day = _run(root, run), _day(date)
    return _call(lambda: service.forecast_day(loaded, day))


@app.get("/api/dispatch")
def dispatch(
    root: Root,
    date: str,
    run: str | None = None,
    duration: Duration = 2,
    degradation: Degradation = 8,
    strategy: StrategyQuery = "median",
    power: Power = 1.0,
) -> dict[str, Any]:
    loaded, day = _run(root, run), _day(date)
    return _call(
        lambda: service.dispatch(loaded, day, duration, degradation, strategy, power)
    )


@app.get("/api/pnl")
def pnl(
    root: Root,
    run: str | None = None,
    window: Window = "last30",
    duration: Duration = 2,
    degradation: Degradation = 8,
    strategy: StrategyQuery = "median",
    power: Power = 1.0,
) -> dict[str, Any]:
    loaded = _run(root, run)
    return _call(
        lambda: service.pnl_series(
            loaded, window, duration, degradation, strategy, power
        )
    )


@app.get("/api/calibration")
def calibration(
    root: Root, run: str | None = None, window: Window = "last30"
) -> dict[str, Any]:
    loaded = _run(root, run)
    return _call(lambda: service.calibration(loaded, window))


@app.get("/api/error-analysis")
def error_analysis(
    root: Root, run: str | None = None, window: Window = "last30"
) -> dict[str, Any]:
    loaded = _run(root, run)
    return _call(lambda: service.error_analysis(loaded, window))


@app.get("/api/feature-importance")
def feature_importance(root: Root, run: str | None = None) -> dict[str, Any]:
    return _run(root, run).feature_importance


def _health(root: Path, run_id: str | None) -> model_health.Health:
    def load() -> model_health.Health:
        if run_id is not None and run_id not in service.list_run_ids(root):
            raise HTTPException(status_code=404, detail=f"run {run_id!r} not found")
        return model_health.load_health(root, run_id or service.latest_run_id(root))

    return _call(load)


def _optional_day(text: str | None) -> date | None:
    return None if text is None else _day(text)


@app.get("/api/model-health/regime")
def model_health_regime(root: Root, run: str | None = None) -> dict[str, Any]:
    loaded = _health(root, run)
    return _call(lambda: model_health.regime(loaded))


@app.get("/api/model-health/drift")
def model_health_drift(
    root: Root,
    run: str | None = None,
    window: Annotated[DriftWindow, Query()] = "all",
) -> dict[str, Any]:
    loaded = _health(root, run)
    return _call(lambda: model_health.drift(loaded, window))


@app.get("/api/model-health/incidents")
def model_health_incidents(
    root: Root,
    run: str | None = None,
    start: str | None = None,
    end: str | None = None,
    incident_type: Annotated[list[str] | None, Query(alias="type")] = None,
    source: Annotated[list[str] | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=model_health.MAX_LIMIT)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> dict[str, Any]:
    # A malformed query is refused (422) before the run's files are looked up (503).
    first, last = _optional_day(start), _optional_day(end)
    _call(
        lambda: model_health.check_incident_query(
            first, last, incident_type, source, limit, offset
        )
    )
    loaded = _health(root, run)
    return _call(
        lambda: model_health.incidents(
            loaded,
            start=first,
            end=last,
            types=incident_type,
            sources=source,
            limit=limit,
            offset=offset,
        )
    )


@app.get("/api/live")
def live_record(root: Root) -> dict[str, Any]:
    """The live pipeline's record: every delivery day, settlements, incidents.

    Written by the export's ``live`` step at the dashboard root, beside the runs,
    so it is the same whichever run the tab is showing.
    """
    path = root / "live.json"
    if not path.exists():
        raise HTTPException(
            status_code=503,
            detail="live record not exported yet; run python -m src.export.artifacts "
            "--steps live",
        )
    loaded: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return loaded


@app.get("/api/model-health/ops")
def model_health_ops(root: Root, run: str | None = None) -> dict[str, Any]:
    loaded = _health(root, run)
    return _call(lambda: model_health.ops(loaded))


class NarrateRequest(BaseModel):
    """What to write a briefing about: one tab, one day, one battery."""

    model_config = ConfigDict(extra="forbid")

    tab: Literal["overview", "forecast", "trading", "model_health"] = "overview"
    #: The delivery day. Only the overview tab is about one day; the others show a
    #: window, so they may leave it out and take the run's last day.
    date: str | None = None
    window: WindowKey = "last30"
    run: str | None = None
    power: float = Field(default=1.0, ge=0.5, le=5.0)
    duration: int = Field(default=2, ge=1, le=4)
    degradation: int = Field(default=8, ge=0, le=25)
    strategy: StrategyKey = "median"
    question: str | None = Field(default=None, max_length=200)


def _plain_question(question: str | None) -> str | None:
    """A follow-up with its figures removed.

    The question is echoed back inside the answer, so a figure in the question
    would otherwise appear in the briefing as though the data held it. The three
    canned follow-ups carry no numbers; anything typed loses its, whether written
    in digits or in words.
    """
    if not question:
        return None
    return grounding.strip_numerals(question) or None


@app.post("/api/narrate")
def narrate(request: NarrateRequest, root: Root) -> dict[str, Any]:
    """A briefing for one tab, carrying only numbers the payload already holds.

    The deterministic writer answers unless a model is switched on. A model's
    draft is judged by the grounding check: one that states a figure the payload
    does not contain is sent back once, and if the rewrite fails too the
    deterministic writer answers, with the rejected figures in the response. What
    is returned is always grounded, so the dashboard can show it unread.
    """
    loaded = _run(root, request.run)
    health = _health(root, request.run) if request.tab == "model_health" else None
    day = _day(request.date or str(loaded.manifest["last_day"]))
    battery = {
        "power_mw": request.power,
        "duration_h": request.duration,
        "degradation_eur_per_mwh": request.degradation,
        "strategy": request.strategy,
    }

    def build() -> dict[str, Any]:
        try:
            return narration_payload.build_payload(
                loaded,
                health,
                tab=request.tab,
                day=day,
                window=request.window,
                battery=battery,
            )
        except narration_payload.PayloadError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    payload = _call(build)
    question = _plain_question(request.question)
    provider = narration_provider.build_provider()
    rejected: list[str] = []
    reason: str | None = None
    briefing: narration_provider.Briefing | None = None
    report: grounding.GroundingReport | None = None
    retry: narration_provider.Retry | None = None
    attempts = 0
    try:
        # A refused draft is sent back once with the figures that failed; a second
        # refusal hands the briefing to the template. There is no third draft.
        while attempts < narration_provider.MAX_ATTEMPTS:
            attempts += 1
            written = provider.write(payload, question, retry)
            checked = grounding.check_grounding(written.text, payload)
            if checked.grounded:
                briefing, report = written, checked
                break
            figures = tuple(dict.fromkeys(c.text for c in checked.unsupported))
            rejected += [figure for figure in figures if figure not in rejected]
            retry = narration_provider.Retry(written.text, figures)
        if briefing is None:
            reason = (
                f"the briefing stated figures this page does not show, in {attempts} "
                f"draft{'s' if attempts > 1 else ''}"
            )
    except narration_provider.NarrationError as exc:
        reason = str(exc)
    if briefing is None or report is None:
        briefing = narration_provider.TemplateProvider().write(payload, question)
        report = grounding.check_grounding(briefing.text, payload)
    if not report.grounded:  # pragma: no cover - the template quotes the payload
        briefing = narration_provider.TemplateProvider().write(payload, None)
        report = grounding.check_grounding(briefing.text, payload)
        reason = reason or "the briefing could not be grounded"
    return {
        "tab": request.tab,
        "day": str(day),
        "window": request.window,
        "text": briefing.text,
        "provider": briefing.provider,
        "model": briefing.model,
        "grounded": report.grounded,
        "unsupported": [claim.text for claim in report.unsupported],
        "fell_back": reason is not None,
        "attempts": attempts,
        "rejected": rejected,
        "fallback_reason": reason,
        "follow_ups": narration_provider.FOLLOW_UPS,
    }


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
