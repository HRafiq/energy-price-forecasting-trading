"""M2 experiment: would a drift monitor have caught the hold-out degradation?

    uv run python -m src.health.experiments.m2_drift

Reads saved production forecasts only; nothing is refit and the hold-out is not
rerun. The steps run in this order, and the order is the point:

1. Read the validation comparison forecasts
   (``data/processed/forecasts/comparison/<production model>.parquet``) and fix
   both alert thresholds on validation days, 2024-06-01 to 2026-05-31, with the
   rule in ``src.health.drift``.
2. Only then read the Phase 4 hold-out forecasts
   (``data/processed/forecasts/holdout/<production model>.parquet``) and apply the
   frozen thresholds: first alert date per signal, share of hold-out days in
   alert, longest alert run.

Rolling windows. The comparison file starts on 2024-03-30, 63 days before the
validation window, so the first validation day already has a full 28-day window
from those earlier comparison days. The first hold-out days take their window
from the last validation days. Days before the validation window fill windows
only; they get no row in the outputs and no vote in the thresholds.

Outputs: ``data/processed/experiments/m2_drift_daily.parquet``,
``data/processed/experiments/m2_drift.json`` for the dashboard,
``docs/results/m2_drift.md``, drift incidents plus the observed incidents from
``src.health.incidents`` in the incident log, and an MLflow run in the
``m2-drift-monitor`` experiment.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, Settings, load_settings
from src.health.drift import (
    COVERAGE_STEP,
    MAX_ALERT_SHARE,
    PINBALL_STEP,
    SOURCE,
    WINDOW_DAYS,
    AlertEpisode,
    Thresholds,
    alert_episodes,
    apply_thresholds,
    daily_scores,
    drift_incidents,
    fit_thresholds,
    longest_run,
    rolling_scores,
    window_labels,
)
from src.health.incidents import (
    OBSERVED,
    Incident,
    default_path,
    observed_incidents,
    replace_incidents,
)

__all__ = ["DAILY_COLUMNS", "M2Result", "compute", "main"]

RESULTS_PATH = REPO_ROOT / "docs" / "results" / "m2_drift.md"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
EXPERIMENT = "m2-drift-monitor"
DAILY_COLUMNS = [
    "target_day",
    "window",
    "coverage_90",
    "pinball",
    "rolling_coverage_90",
    "rolling_pinball_ratio",
    "coverage_alert",
    "pinball_alert",
]
RULE = (
    f"Rolling {WINDOW_DAYS} traded days, pooled over periods. Coverage threshold: "
    f"the highest value on a {COVERAGE_STEP * 100:.1f} percentage point grid such "
    f"that at most {MAX_ALERT_SHARE:.0%} of validation days have rolling 90% "
    "interval coverage below it. Pinball threshold: rolling mean pinball divided "
    "by its validation median; the lowest ratio on a "
    f"{PINBALL_STEP:.2f} grid such that at most {MAX_ALERT_SHARE:.0%} of "
    "validation days have a ratio above it. Thresholds use validation days only "
    "(2024-06-01 to 2026-05-31) and are applied unchanged to the hold-out."
)


@dataclass(frozen=True)
class WindowSummary:
    days: int
    coverage_alert_share: float
    pinball_alert_share: float
    first_coverage_alert: date | None
    first_pinball_alert: date | None
    longest_coverage_run: tuple[int, date | None, date | None]
    longest_pinball_run: tuple[int, date | None, date | None]


@dataclass(frozen=True)
class M2Result:
    thresholds: Thresholds
    daily: pd.DataFrame
    episodes: list[AlertEpisode]
    summaries: dict[str, WindowSummary]
    incidents: list[Incident]
    holdout_start: date
    model: str


def _first(flags: pd.Series) -> date | None:
    days = flags.index[flags.to_numpy(dtype=bool)]
    return None if len(days) == 0 else days[0]


def _summary(frame: pd.DataFrame) -> WindowSummary:
    return WindowSummary(
        days=len(frame),
        coverage_alert_share=float(frame["coverage_alert"].mean()),
        pinball_alert_share=float(frame["pinball_alert"].mean()),
        first_coverage_alert=_first(frame["coverage_alert"]),
        first_pinball_alert=_first(frame["pinball_alert"]),
        longest_coverage_run=longest_run(frame["coverage_alert"]),
        longest_pinball_run=longest_run(frame["pinball_alert"]),
    )


def compute(settings: Settings, comparison_path: Path, holdout_path: Path) -> M2Result:
    ev = settings.evaluation
    comparison = pd.read_parquet(comparison_path)
    before_holdout = comparison[
        np.asarray([day < ev.holdout_start for day in comparison["target_day"]])
    ]
    # Step 1: thresholds from validation days. The hold-out file is not open yet.
    thresholds, _ = fit_thresholds(before_holdout, settings)

    # Step 2: the frozen thresholds meet the hold-out.
    holdout = pd.read_parquet(holdout_path)
    holdout = holdout[
        np.asarray([day >= ev.holdout_start for day in holdout["target_day"]])
    ]
    both = pd.concat([before_holdout, holdout]).sort_index()
    rolled = rolling_scores(daily_scores(both, settings.forecasting.quantiles))
    flagged = apply_thresholds(rolled, thresholds)
    windows = window_labels(flagged.index, settings)
    keep = windows != "before"
    flagged = flagged[keep]
    windows = windows[keep]
    if flagged["rolling_coverage_90"].isna().any():
        raise ValueError("a reported day has no full rolling window")

    episodes = sorted(
        alert_episodes(flagged, "coverage", windows)
        + alert_episodes(flagged, "pinball", windows),
        key=lambda e: (e.start, e.signal),
    )
    summaries = {
        name: _summary(flagged[windows == name]) for name in ("validation", "holdout")
    }
    validation_forecasts = before_holdout[
        np.asarray([day >= ev.validation_start for day in before_holdout["target_day"]])
    ]
    incidents = drift_incidents(
        episodes, thresholds, settings.market.timezone
    ) + observed_incidents(settings, validation_forecasts, holdout)

    daily = flagged.assign(window=windows).reset_index()[DAILY_COLUMNS]
    return M2Result(
        thresholds=thresholds,
        daily=daily,
        episodes=episodes,
        summaries=summaries,
        incidents=incidents,
        holdout_start=ev.holdout_start,
        model=settings.forecasting.production_model,
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    return value


def _counts(incidents: list[Incident]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for incident in incidents:
        by_type = counts.setdefault(incident.source, {})
        by_type[incident.type] = by_type.get(incident.type, 0) + 1
    return counts


def _dashboard_json(result: M2Result) -> dict[str, Any]:
    series: list[dict[str, Any]] = [
        {
            str(key): _jsonable(
                value.item() if isinstance(value, np.generic) else value
            )
            for key, value in record.items()
        }
        for record in result.daily.astype(
            {"coverage_alert": bool, "pinball_alert": bool}
        ).to_dict(orient="records")
    ]
    return {
        "generated_by": "python -m src.health.experiments.m2_drift",
        "model": result.model,
        "window_days": WINDOW_DAYS,
        "holdout_start": result.holdout_start.isoformat(),
        "rule": RULE,
        "thresholds": _jsonable(asdict(result.thresholds)),
        "windows": _jsonable(
            {name: asdict(summary) for name, summary in result.summaries.items()}
        ),
        "episodes": _jsonable([asdict(episode) for episode in result.episodes]),
        "incident_counts": _counts(result.incidents),
        "series": series,
    }


def _pct(value: float) -> str:
    return f"{value:.1%}"


def _run_text(run: tuple[int, date | None, date | None]) -> str:
    length, start, end = run
    return "none" if length == 0 else f"{length} traded days, {start} to {end}"


def _range_row(daily: pd.DataFrame, window: str) -> str:
    part = daily[daily["window"] == window].set_index("target_day")
    coverage = part["rolling_coverage_90"]
    ratio = part["rolling_pinball_ratio"]
    return (
        f"| {window} | {len(part)} | "
        f"{_pct(float(coverage.min()))} on {coverage.idxmin()} | "
        f"{float(ratio.max()):.2f} on {ratio.idxmax()} |"
    )


def _markdown(result: M2Result) -> str:
    th = result.thresholds
    val = result.summaries["validation"]
    hold = result.summaries["holdout"]

    def after_start(day: date | None) -> str:
        if day is None:
            return "no alert"
        return f"{day} ({(day - result.holdout_start).days} days after the start)"

    counts = _counts(result.incidents)
    count_rows = [
        f"| {source} | {kind} | {n} |"
        for source, by_type in sorted(counts.items())
        for kind, n in sorted(by_type.items())
    ]
    val_episodes = sum(1 for e in result.episodes if e.window == "validation")
    hold_episodes = sum(1 for e in result.episodes if e.window == "holdout")
    lines = [
        "# M2 drift monitor",
        "",
        "Generated by `python -m src.health.experiments.m2_drift` from saved",
        f"`{result.model}` forecasts. Nothing refit, the hold-out not rerun.",
        "",
        "## Rule",
        "",
        RULE,
        "",
        "## Thresholds, validation days only",
        "",
        "| signal | threshold | validation days in alert |",
        "|---|---|---|",
        f"| rolling 90% coverage | below {_pct(th.coverage)} | "
        f"{_pct(th.coverage_alert_share)} of {th.validation_days} |",
        f"| rolling pinball ratio | above {th.pinball_ratio:.2f} | "
        f"{_pct(th.pinball_alert_share)} of {th.validation_days} |",
        "",
        "Validation median of the rolling mean pinball: "
        f"{th.pinball_median:.3f} €/MWh.",
        f"Longest validation alert run: coverage {_run_text(val.longest_coverage_run)};"
        f" pinball {_run_text(val.longest_pinball_run)}.",
        "",
        "## Hold-out, thresholds frozen",
        "",
        "| signal | first alert | hold-out days in alert | longest run |",
        "|---|---|---|---|",
        f"| rolling 90% coverage | {after_start(hold.first_coverage_alert)} | "
        f"{_pct(hold.coverage_alert_share)} of {hold.days} | "
        f"{_run_text(hold.longest_coverage_run)} |",
        f"| rolling pinball ratio | {after_start(hold.first_pinball_alert)} | "
        f"{_pct(hold.pinball_alert_share)} of {hold.days} | "
        f"{_run_text(hold.longest_pinball_run)} |",
        "",
        f"Alert episodes: {val_episodes} in validation, {hold_episodes} in the "
        "hold-out. Each is one drift incident. Days are traded days: a delivery "
        "day without complete prices is left out of the series.",
        "",
        "## Range of the rolling signals",
        "",
        "| window | traded days | lowest rolling coverage | highest pinball ratio |",
        "|---|---|---|---|",
        *[_range_row(result.daily, name) for name in ("validation", "holdout")],
        "",
        "## Incidents written",
        "",
        "| source | type | count |",
        "|---|---|---|",
        *count_rows,
        "",
    ]
    return "\n".join(lines)


def _log_mlflow(result: M2Result) -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(REPO_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    hold = result.summaries["holdout"]
    metrics = {
        "coverage_threshold": result.thresholds.coverage,
        "pinball_ratio_threshold": result.thresholds.pinball_ratio,
        "pinball_validation_median": result.thresholds.pinball_median,
        "validation_coverage_alert_share": result.thresholds.coverage_alert_share,
        "validation_pinball_alert_share": result.thresholds.pinball_alert_share,
        "holdout_coverage_alert_share": hold.coverage_alert_share,
        "holdout_pinball_alert_share": hold.pinball_alert_share,
        "holdout_longest_coverage_run": float(hold.longest_coverage_run[0]),
        "holdout_longest_pinball_run": float(hold.longest_pinball_run[0]),
        "incidents": float(len(result.incidents)),
    }
    for name, first in (
        ("coverage", hold.first_coverage_alert),
        ("pinball", hold.first_pinball_alert),
    ):
        if first is not None:
            metrics[f"holdout_first_{name}_alert_days_after_start"] = float(
                (first - result.holdout_start).days
            )
    with mlflow.start_run(run_name="m2-drift"):
        mlflow.log_params(
            {
                "model": result.model,
                "window_days": WINDOW_DAYS,
                "max_alert_share": MAX_ALERT_SHARE,
                "coverage_step": COVERAGE_STEP,
                "pinball_step": PINBALL_STEP,
            }
        )
        mlflow.log_metrics(metrics)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fix drift thresholds on validation, then report the hold-out."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--no-mlflow", action="store_true")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    forecasts_dir = settings.data.processed_path / "forecasts"
    model_file = f"{settings.forecasting.production_model}.parquet"
    result = compute(
        settings,
        forecasts_dir / "comparison" / model_file,
        forecasts_dir / "holdout" / model_file,
    )

    out_dir = settings.data.processed_path / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)
    result.daily.to_parquet(out_dir / "m2_drift_daily.parquet")
    (out_dir / "m2_drift.json").write_text(
        json.dumps(_dashboard_json(result), indent=2) + "\n", encoding="utf-8"
    )
    log_path = default_path(settings)
    # Both sources are rebuilt in full here, so stale records of either go.
    records = replace_incidents(result.incidents, (SOURCE, OBSERVED), log_path)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(_markdown(result), encoding="utf-8")
    if not args.no_mlflow:
        _log_mlflow(result)

    th = result.thresholds
    print(
        f"coverage threshold {th.coverage:.3f}, pinball ratio {th.pinball_ratio:.2f}; "
        f"{len(result.incidents)} incidents from {SOURCE} and observed rules, "
        f"log now holds {len(records)}"
    )
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
