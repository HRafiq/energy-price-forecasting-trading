"""Run the candidate models walk-forward and log every run to MLflow.

    uv run python -m src.forecasting.run_comparison
    uv run python -m src.forecasting.run_comparison --models lightgbm_quantile mstl
    uv run python -m src.forecasting.run_comparison --last-days 7   # quick check

Candidates forecast every day from ``evaluation.validation_start`` minus a
warm-up to the day before the hold-out. The warm-up gives QRA member forecasts
to calibrate on; scores cover the validation window only, or only the requested
days in a quick run. Each model's forecasts are written to
``data/processed/forecasts/comparison/`` and each run is logged to the local
MLflow store ``mlflow.db`` with its parameters, scores and run time. Scores are
saved after every model. QRA runs last, on the saved forecasts of its members;
re-running a member without QRA deletes QRA's results, which would be stale.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fcntl
import json
import re
import time
from collections.abc import Callable, Iterator, Sequence
from datetime import date, timedelta
from pathlib import Path

import mlflow
import pandas as pd

from src.config import REPO_ROOT, Settings, load_settings
from src.forecasting.base import Forecaster
from src.forecasting.baselines import build_baselines
from src.forecasting.evaluate import segment_scores
from src.forecasting.models.gradient_boosting import (
    LightGBMConformalModel,
    LightGBMQuantileModel,
)
from src.forecasting.models.lear import LearModel
from src.forecasting.models.mstl import MSTLModel
from src.forecasting.models.qra import QuantileRegressionAveraging
from src.forecasting.models.quantile_forest import QuantileForestModel
from src.forecasting.walkforward import day_range, run_walk_forward

__all__ = [
    "CANDIDATES",
    "QRA_MEMBERS",
    "REFIT_EVERY_DAYS",
    "first_scored_day",
    "main",
    "qra_target_days",
    "remove_stale_qra",
]

EXPERIMENT = "price-forecast-comparison"
#: Local MLflow store: metadata in SQLite, artifacts in mlruns/, both ignored by git.
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
ARTIFACTS = REPO_ROOT / "mlruns"
SCORES_FILE = "validation_scores.json"

CANDIDATES: dict[str, Callable[[Settings], Forecaster]] = {
    "naive_previous_day": lambda s: build_baselines(s)[0],
    "seasonal_naive_previous_week": lambda s: build_baselines(s)[1],
    "lear": LearModel,
    "lightgbm_quantile": LightGBMQuantileModel,
    "lightgbm_conformal": LightGBMConformalModel,
    "quantile_forest": QuantileForestModel,
    "mstl": MSTLModel,
}
#: Days between refits. Feature models refit every four weeks to bound run time.
REFIT_EVERY_DAYS = {
    "naive_previous_day": 1,
    "seasonal_naive_previous_week": 1,
    "lear": 28,
    "lightgbm_quantile": 28,
    "lightgbm_conformal": 28,
    "quantile_forest": 28,
    "mstl": 1,
    "qra": 7,
}
QRA_MEMBERS = (
    "lear",
    "lightgbm_quantile",
    "lightgbm_conformal",
    "quantile_forest",
    "mstl",
)


def qra_target_days(days: list[date], calibration_days: int) -> list[date]:
    """Days QRA can forecast: members need ``calibration_days`` of forecasts first."""
    first_allowed = days[0] + timedelta(days=calibration_days)
    return [day for day in days if day >= first_allowed]


def first_scored_day(
    days: Sequence[date], settings: Settings, last_days: int | None
) -> date:
    """The validation start, or the first requested day of a quick run."""
    if last_days is None:
        return settings.evaluation.validation_start
    return days[-last_days]


@contextlib.contextmanager
def _scores_lock(out_dir: Path) -> Iterator[None]:
    """Exclusive lock for reading and rewriting the scores file across processes."""
    with (out_dir / f"{SCORES_FILE}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def _write_scores(path: Path, scores: dict[str, object]) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    tmp.replace(path)


def remove_stale_qra(out_dir: Path, models: Sequence[str]) -> bool:
    """Delete QRA results when a member is re-run without QRA; True if removed."""
    if "qra" in models or not any(m in QRA_MEMBERS for m in models):
        return False
    removed = False
    forecasts = out_dir / "qra.parquet"
    if forecasts.exists():
        forecasts.unlink()
        removed = True
    scores_path = out_dir / SCORES_FILE
    with _scores_lock(out_dir):
        if scores_path.exists():
            scores = json.loads(scores_path.read_text(encoding="utf-8"))
            if scores.pop("qra", None) is not None:
                _write_scores(scores_path, scores)
                removed = True
    return removed


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a whole number of at least 1")
    return number


def _metric_key(segment: str, metric: str) -> str:
    raw = f"{segment}.{metric}".replace("%", "pct").replace("€", "eur")
    return re.sub(r"[^A-Za-z0-9_.\-/ ]", "", raw).replace(" ", "_")


def _params(model: object) -> dict[str, str]:
    if not dataclasses.is_dataclass(model):
        return {}
    return {
        f.name: str(getattr(model, f.name))
        for f in dataclasses.fields(model)
        if not f.name.startswith("_") and f.name not in ("settings", "members")
    }


def _save_scores(out_dir: Path, name: str, scores: dict[str, dict[str, float]]) -> None:
    """Merge one model's scores into the file, locked so parallel runs keep theirs."""
    path = out_dir / SCORES_FILE
    with _scores_lock(out_dir):
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        existing[name] = scores
        _write_scores(path, existing)


def _log_run(
    name: str,
    model: object,
    forecasts: pd.DataFrame,
    settings: Settings,
    seconds: float,
    refit_every_days: int,
    first_scored: date,
) -> dict[str, dict[str, float]]:
    scored = forecasts[[day >= first_scored for day in forecasts["target_day"]]]
    scores = segment_scores(scored, settings)
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"model": name, "window": "validation"})
        mlflow.log_params(
            {
                **_params(model),
                "refit_every_days": str(refit_every_days),
                "first_scored_day": str(first_scored),
                "last_day": str(max(forecasts["target_day"])),
            }
        )
        for segment, row in scores.iterrows():
            for metric, value in row.items():
                if pd.notna(value):
                    mlflow.log_metric(
                        _metric_key(str(segment), str(metric)), float(value)
                    )
        mlflow.log_metric("run_seconds", seconds)
    return {
        str(k): {str(m): float(v) for m, v in r.items()} for k, r in scores.iterrows()
    }


def main(argv: list[str] | None = None) -> int:
    names = [*CANDIDATES, "qra"]
    parser = argparse.ArgumentParser(
        description="Compare forecasting models walk-forward."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--models", nargs="+", choices=names, default=names)
    parser.add_argument(
        "--last-days", type=_positive_int, default=None, help="quick run: last N days"
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    warmup_days = QuantileRegressionAveraging(settings, {}).calibration_days + 7
    first = settings.evaluation.validation_start - timedelta(days=warmup_days)
    last = settings.evaluation.holdout_start - timedelta(days=1)
    days = day_range(first, last)
    quick = args.last_days is not None
    if quick:
        available = len(days) - warmup_days
        if args.last_days > available:
            parser.error(f"--last-days can be at most {available}")
        days = days[-(args.last_days + warmup_days) :]
    scored_from = first_scored_day(days, settings, args.last_days)
    frame = pd.read_parquet(settings.data.inputs_path)
    out_dir = (
        settings.data.processed_path
        / "forecasts"
        / ("comparison_quick" if quick else "comparison")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if remove_stale_qra(out_dir, args.models):
        print(
            "removed QRA results built from the previous member forecasts", flush=True
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    experiment = EXPERIMENT + ("-quick" if quick else "")
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=ARTIFACTS.as_uri())
    mlflow.set_experiment(experiment)

    for name in [n for n in args.models if n != "qra"]:
        model = CANDIDATES[name](settings)
        refit = REFIT_EVERY_DAYS[name]
        started = time.perf_counter()
        forecasts = run_walk_forward(
            frame, model, days, settings, refit_every_days=refit
        )
        seconds = time.perf_counter() - started
        forecasts.to_parquet(out_dir / f"{name}.parquet")
        scores = _log_run(name, model, forecasts, settings, seconds, refit, scored_from)
        _save_scores(out_dir, name, scores)
        print(f"{name}: {len(days)} days in {seconds:.0f} s", flush=True)

    if "qra" in args.models:
        members = {m: pd.read_parquet(out_dir / f"{m}.parquet") for m in QRA_MEMBERS}
        qra = QuantileRegressionAveraging(settings, members)
        qra_days = qra_target_days(days, qra.calibration_days)
        refit = REFIT_EVERY_DAYS["qra"]
        started = time.perf_counter()
        forecasts = run_walk_forward(
            frame, qra, qra_days, settings, refit_every_days=refit
        )
        seconds = time.perf_counter() - started
        forecasts.to_parquet(out_dir / "qra.parquet")
        scores = _log_run("qra", qra, forecasts, settings, seconds, refit, scored_from)
        _save_scores(out_dir, "qra", scores)
        print(f"qra: {len(qra_days)} days in {seconds:.0f} s", flush=True)

    print(f"scores in {out_dir / SCORES_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
