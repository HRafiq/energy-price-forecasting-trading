"""FastAPI app serving the dashboard: thin read-only endpoints over ``api.service``.

    uv run uvicorn api.main:app --port 8000

Every endpoint lives under ``/api``; the contract is in ``docs/dashboard_api.md``.
When ``frontend/dist`` exists, the built React app is served at ``/``.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from api import service
from src.config import REPO_ROOT, load_settings

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


if FRONTEND.exists():
    app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
