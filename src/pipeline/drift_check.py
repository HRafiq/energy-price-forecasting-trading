"""The live drift check: the M2 monitor, run every day on what the pipeline served.

    uv run python -m src.pipeline.drift_check --through 2026-09-18

The M2 experiment fixed two alert thresholds on the validation window: rolling
28-day 90% interval coverage below 74%, and rolling mean pinball loss above 1.5
times its validation median. This check applies those same thresholds, unchanged, to
the live record, after each day is settled:

* **What is scored:** the saved live forecasts (``forecasts/production/<day>.parquet``)
  from ``evaluation.live_from`` through ``--through``, but only days the production
  model forecast. A day a lower rung of the fallback chain covered is skipped: the
  thresholds describe the production model's ranges, and another forecaster's would
  raise alerts about the wrong model. A day without every realised price is skipped
  too, as M2 skips it.
* **Warm-up:** the rolling window needs 28 scored days, so until then the check
  records how many it has and raises nothing. The window is not filled with hold-out
  forecasts: those come from the one-off hold-out run, and the hold-out stays frozen.
* **Alerts:** each run of consecutive scored days in alert on one signal becomes a
  drift incident with source ``live_drift``. The live incidents are rebuilt on every
  run, so an alert that is still open carries its latest length, and each keeps the
  time of the run that first found it.

An alert asks for a retraining review. It does not refit the model early: that rule
was never backtested. Writes ``health/live_drift.json`` beside the incident log.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PRICE_SERIES, Settings, load_settings
from src.health.drift import (
    WINDOW_DAYS,
    Thresholds,
    alert_episodes,
    apply_thresholds,
    daily_scores,
    drift_incidents,
    rolling_scores,
)
from src.health.incidents import default_path, load_incidents, replace_incidents

__all__ = [
    "PRODUCTION_STEP",
    "SOURCE",
    "live_forecasts",
    "load_thresholds",
    "main",
    "run_check",
    "summary_path",
]

SOURCE = "live_drift"
PRODUCTION_STEP = "production"


def summary_path(settings: Settings) -> Path:
    return settings.data.processed_path / "health" / "live_drift.json"


def load_thresholds(settings: Settings) -> Thresholds:
    """The thresholds the M2 experiment fixed on validation, read back unchanged."""
    path = settings.data.processed_path / "experiments" / "m2_drift.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `python -m src.health.experiments.m2_drift` first"
        )
    saved = json.loads(path.read_text(encoding="utf-8"))
    if int(saved["window_days"]) != WINDOW_DAYS:
        raise ValueError(
            f"m2_drift.json used a {saved['window_days']}-day window, the check "
            f"uses {WINDOW_DAYS}"
        )
    if saved["model"] != settings.forecasting.production_model:
        raise ValueError(
            f"m2_drift.json fixed thresholds for {saved['model']}, but the "
            f"production model is {settings.forecasting.production_model}"
        )
    return Thresholds(**saved["thresholds"])


def live_forecasts(
    settings: Settings, through: date, prices: pd.Series
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    """Production-model live forecasts through ``through``, with realised prices.

    Returns the frame in the walk-forward format and the days left out, each with
    the reason.
    """
    folder = settings.data.processed_path / "forecasts" / "production"
    parts: list[pd.DataFrame] = []
    skipped: list[dict[str, str]] = []
    for path in sorted(folder.glob("*.parquet")) if folder.exists() else []:
        day = date.fromisoformat(path.stem)
        if not settings.evaluation.live_from <= day <= through:
            continue
        frame = pd.read_parquet(path)
        # A reconstruction was made after the day it forecasts, from a saved
        # model, with no gate to meet. Scoring it would mix it into the live
        # model's record, which is exactly what the monitor is watching.
        kind = str(frame["kind"].iloc[0]) if "kind" in frame.columns else "live"
        if kind != "live":
            skipped.append({"day": str(day), "reason": "reconstructed, not a live bid"})
            continue
        step = str(frame["chain_step"].iloc[0])
        if step != PRODUCTION_STEP:
            skipped.append({"day": str(day), "reason": f"forecast by {step}"})
            continue
        parts.append(frame)
    if not parts:
        return pd.DataFrame(), skipped
    frame = pd.concat(parts).sort_index()
    frame["actual"] = prices.reindex(frame.index)
    complete = frame["actual"].notna().groupby(frame["target_day"]).all()
    unpriced = [str(day) for day, full in complete.items() if not full]
    skipped += [{"day": day, "reason": "prices not all published"} for day in unpriced]
    return frame, sorted(skipped, key=lambda item: item["day"])


def run_check(
    settings: Settings,
    through: date,
    *,
    frame: pd.DataFrame | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Score the live record through ``through``, rewrite live incidents and summary."""
    if through < settings.evaluation.live_from:
        raise ValueError(
            f"{through} is before live_from {settings.evaluation.live_from}"
        )
    thresholds = load_thresholds(settings)
    inputs = frame if frame is not None else pd.read_parquet(settings.data.inputs_path)
    forecasts, skipped = live_forecasts(settings, through, inputs[PRICE_SERIES])
    quantiles = settings.forecasting.quantiles
    if forecasts.empty:
        daily = daily_scores(pd.DataFrame(columns=["target_day", "actual"]), quantiles)
    else:
        daily = daily_scores(forecasts, quantiles)
    flagged = apply_thresholds(rolling_scores(daily, WINDOW_DAYS), thresholds)
    windows = pd.Series("live", index=flagged.index, dtype="object")
    episodes = [
        episode
        for signal in ("coverage", "pinball")
        for episode in alert_episodes(flagged, signal, windows)
    ]
    found = drift_incidents(
        episodes,
        thresholds,
        settings.market.timezone,
        WINDOW_DAYS,
        source=SOURCE,
        days_word="scored days",
    )
    log = default_path(settings)
    # An alert is detected when a run first scores it, not at a batch's midnight;
    # a rebuilt alert keeps the time its first run recorded.
    earlier = {
        incident.incident_id: incident.detected_utc
        for incident in load_incidents(log)
        if incident.source == SOURCE
    }
    run_utc = (now_utc or datetime.now(UTC)).replace(microsecond=0)
    incidents = [
        incident.model_copy(
            update={"detected_utc": earlier.get(incident.incident_id, run_utc)}
        )
        for incident in found
    ]
    replace_incidents(incidents, [SOURCE], log)

    scored = len(flagged)
    latest: dict[str, Any] | None = None
    status = f"warming up: {scored} of {WINDOW_DAYS} scored days"
    if scored and pd.notna(flagged["rolling_coverage_90"].iloc[-1]):
        row = flagged.iloc[-1]
        latest = {
            "day": str(flagged.index[-1]),
            "rolling_coverage_90": float(row["rolling_coverage_90"]),
            "rolling_pinball_ratio": float(row["rolling_pinball_ratio"]),
            "coverage_alert": bool(row["coverage_alert"]),
            "pinball_alert": bool(row["pinball_alert"]),
        }
        in_alert = latest["coverage_alert"] or latest["pinball_alert"]
        status = "in alert" if in_alert else "no alert"
    summary: dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "through": str(through),
        "model": settings.forecasting.production_model,
        "window_days": WINDOW_DAYS,
        "thresholds": asdict(thresholds),
        "scored_days": scored,
        "warming_up": latest is None,
        "status": status,
        "latest": latest,
        "skipped": skipped,
        "incidents": [incident.incident_id for incident in incidents],
        "series": [
            {
                "day": str(day),
                "coverage_90": float(row["coverage_90"]),
                "pinball": float(row["pinball"]),
                "rolling_coverage_90": None
                if pd.isna(row["rolling_coverage_90"])
                else float(row["rolling_coverage_90"]),
                "rolling_pinball_ratio": None
                if pd.isna(row["rolling_pinball_ratio"])
                else float(row["rolling_pinball_ratio"]),
                "coverage_alert": bool(row["coverage_alert"]),
                "pinball_alert": bool(row["pinball_alert"]),
            }
            for day, row in flagged.iterrows()
        ],
    }
    path = summary_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(partial, path)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the drift monitor on the live record."
    )
    parser.add_argument(
        "--through",
        type=date.fromisoformat,
        required=True,
        help="last settled delivery day to score",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    summary = run_check(load_settings(args.config), args.through)
    print(
        f"live drift through {summary['through']}: {summary['status']}; "
        f"{len(summary['skipped'])} days skipped, "
        f"{len(summary['incidents'])} live drift incidents"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
