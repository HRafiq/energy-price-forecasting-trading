"""D5 experiment: the 12:00 gate closes whether or not the pipeline worked.

    uv run python -m src.health.experiments.d5_deadline

The daily forecast is issued at 11:40 for a gate that closes at 12:00. This
experiment injects pipeline failures on the validation days and runs the
fallback chain, in order:

1. ``full``: the production model;
2. ``fallback_no_weather``: the production model without weather features;
3. ``naive_previous_day``: yesterday's prices with their recent error ranges;
4. ``seasonal_naive_previous_week``: last week's prices.

Naive previous day comes before seasonal naive because it scored better on the
validation window, 9.61 against 11.49 pinball loss.

Failures are drawn once per day with a fixed seed, as scenario assumptions rather
than estimates of how often feeds fail:

* weather feed late on 10% of days: the full model cannot run;
* model step failing on 3% of days: neither LightGBM model can run;
* yesterday's prices late on 1% of days: both LightGBM models and naive previous
  day lose their main input, leaving seasonal naive.

Late data is retried once after a 5-minute wait and, in this scenario, has still
not arrived. Step runtimes are measured here, on the machine that runs the
experiment: building the information set and forecasting with models already
fitted, since models refit ahead of the issue time. The chain's profit comes from
the D1 run, which traded every step on the same days, so nothing is solved again;
run D1 first. Without the chain a failed day has no forecast before the gate, no
position and no profit.

No day on or after ``evaluation.holdout_start`` is used: such days are refused
and the inputs are cut before the hold-out before any runtime is measured.

Every day that needed a fallback writes an incident; a rerun replaces every
earlier ``d5_deadline`` record. Results go to
``docs/results/d5_deadline.md``, ``data/processed/experiments/``, the incident log
and MLflow.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, Settings, load_settings
from src.features.build import FEATURE_GROUPS
from src.forecasting.base import Forecaster
from src.forecasting.baselines import build_baselines
from src.forecasting.information import build_information_set
from src.forecasting.models.common import feature_names
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.walkforward import HoldoutAccessError
from src.health.incidents import (
    Incident,
    IncidentType,
    default_path,
    make_incident_id,
    replace_incidents,
)

__all__ = [
    "CHAIN",
    "FAILURE_RATES",
    "Failures",
    "before_holdout",
    "check_days",
    "choose_step",
    "draw_failures",
    "main",
    "measure_runtimes",
    "realised_failure_rates",
    "run_chain",
    "step_incidents",
    "submission_minutes",
]

SOURCE = "d5_deadline"
CHAIN = (
    "full",
    "fallback_no_weather",
    "naive_previous_day",
    "seasonal_naive_previous_week",
)
STEP_LABELS = {
    "full": "full model",
    "fallback_no_weather": "model without weather",
    "naive_previous_day": "naive previous day",
    "seasonal_naive_previous_week": "seasonal naive previous week",
}
FAILURE_RATES = {"weather_late": 0.10, "model_fails": 0.03, "prices_late": 0.01}
SEED = 6
RETRY_WAIT_MINUTES = 5.0
ISSUE_LOCAL = "11:40"
GATE_LOCAL = "12:00"
RUNTIME_SAMPLE_DAYS = 10
EXPERIMENT = "d5-deadline"
TRACKING_URI = f"sqlite:///{REPO_ROOT / 'mlflow.db'}"
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "d5_deadline.md"


@dataclass(frozen=True)
class Failures:
    """What went wrong on one delivery day."""

    weather_late: bool = False
    model_fails: bool = False
    prices_late: bool = False


def check_days(days: Sequence[date], settings: Settings) -> None:
    """Refuse any delivery day in the hold-out."""
    holdout = settings.evaluation.holdout_start
    inside = [day for day in days if day >= holdout]
    if inside:
        raise HoldoutAccessError(
            f"{len(inside)} delivery days fall in the hold-out starting {holdout}; "
            "D5 never reads it"
        )


def before_holdout(inputs: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """The inputs frame without any row on or after the hold-out start."""
    cut = settings.market.local_midnight_utc(settings.evaluation.holdout_start)
    return inputs.loc[inputs.index < cut]


def draw_failures(days: list[date], seed: int = SEED) -> pd.DataFrame:
    """Independent failure draws per day, reproducible from the seed."""
    rng = np.random.default_rng(seed)
    draws = rng.random((len(days), len(FAILURE_RATES)))
    frame = pd.DataFrame(
        draws < np.asarray(list(FAILURE_RATES.values())),
        columns=list(FAILURE_RATES),
    )
    frame.insert(0, "target_day", days)
    return frame


def realised_failure_rates(failures: pd.DataFrame) -> dict[str, float]:
    """Share of days on which each failure type was actually drawn."""
    return {name: float(failures[name].mean()) for name in FAILURE_RATES}


def choose_step(failures: Failures) -> str:
    """The first chain step that can still run."""
    if failures.prices_late:
        return "seasonal_naive_previous_week"
    if failures.model_fails:
        return "naive_previous_day"
    if failures.weather_late:
        return "fallback_no_weather"
    return "full"


def submission_minutes(failures: Failures, runtimes_s: dict[str, float]) -> float:
    """Minutes after the issue time at which the chosen forecast is submitted.

    Every attempted step costs its runtime; each late input costs one retry wait
    before the chain moves on; a failing model step is noticed after it ran.
    """
    step = choose_step(failures)
    attempted = CHAIN[: CHAIN.index(step) + 1]
    waits = RETRY_WAIT_MINUTES * (
        int(failures.weather_late) + int(failures.prices_late)
    )
    seconds = 0.0
    for name in attempted:
        skipped_for_prices = failures.prices_late and name != step
        skipped_for_weather = failures.weather_late and name == "full"
        if skipped_for_prices or skipped_for_weather:
            continue
        seconds += runtimes_s[name]
    return waits + seconds / 60.0


def _minutes(clock: str) -> int:
    hours, minutes = (int(part) for part in clock.split(":"))
    return 60 * hours + minutes


def run_chain(
    failures: pd.DataFrame, daily: pd.DataFrame, runtimes_s: dict[str, float]
) -> pd.DataFrame:
    """Per day: the step used, when it was submitted and what it earned.

    ``daily`` holds the D1 run's rows: ``target_day``, ``arm``, ``pinball``,
    ``pnl_eur`` and ``perfect_foresight_pnl_eur``.
    """
    slack = _minutes(GATE_LOCAL) - _minutes(ISSUE_LOCAL)
    pnl = daily.pivot(index="target_day", columns="arm", values="pnl_eur").to_dict(
        orient="index"
    )
    pinball = daily.pivot(index="target_day", columns="arm", values="pinball").to_dict(
        orient="index"
    )
    ceiling = daily.groupby("target_day")["perfect_foresight_pnl_eur"].first().to_dict()
    rows = []
    for record in failures.to_dict(orient="records"):
        day = record["target_day"]
        flags = Failures(
            weather_late=bool(record["weather_late"]),
            model_fails=bool(record["model_fails"]),
            prices_late=bool(record["prices_late"]),
        )
        step = choose_step(flags)
        minutes = submission_minutes(flags, runtimes_s)
        chain_pnl = float(pnl[day][step])
        rows.append(
            {
                "target_day": day,
                "weather_late": flags.weather_late,
                "model_fails": flags.model_fails,
                "prices_late": flags.prices_late,
                "step": step,
                "submitted_minutes_after_issue": minutes,
                "on_time": minutes < slack,
                "pinball": float(pinball[day][step]),
                "full_pinball": float(pinball[day]["full"]),
                "pnl_eur": chain_pnl,
                "full_pnl_eur": float(pnl[day]["full"]),
                "pnl_without_chain_eur": chain_pnl if step == "full" else 0.0,
                "perfect_foresight_pnl_eur": float(ceiling[day]),
            }
        )
    return pd.DataFrame(rows)


def step_incidents(chain: pd.DataFrame, settings: Settings) -> list[Incident]:
    """One incident for every day the chain had to fall back."""
    tz = settings.market.timezone
    issue = _minutes(ISSUE_LOCAL)
    incidents = []
    for record in chain[chain["step"] != "full"].to_dict(orient="records"):
        day: date = record["target_day"]
        minutes = float(record["submitted_minutes_after_issue"])
        submitted = (
            pd.Timestamp(day - timedelta(days=1))
            + pd.Timedelta(minutes=issue + minutes)
        ).tz_localize(tz)
        causes = [
            label
            for flag, label in (
                (record["weather_late"], "weather feed late"),
                (record["model_fails"], "model step failed"),
                (record["prices_late"], "yesterday's prices late"),
            )
            if flag
        ]
        kind: IncidentType = (
            "pipeline"
            if record["model_fails"] and not record["prices_late"]
            else "late_data"
        )
        step = STEP_LABELS[str(record["step"])]
        clock = submitted.strftime("%H:%M")
        earned, full = float(record["pnl_eur"]), float(record["full_pnl_eur"])
        timing = "before" if record["on_time"] else "after"
        incidents.append(
            Incident(
                incident_id=make_incident_id(SOURCE, kind, day),
                delivery_day=day,
                detected_utc=submitted.tz_convert("UTC").floor("s").to_pydatetime(),
                type=kind,
                severity="warning",
                detail=(
                    f"Injected failure: {', '.join(causes)}. The {step} forecast was "
                    f"submitted at {clock}, {timing} the {GATE_LOCAL} gate; it "
                    f"earned €{earned:,.0f} against €{full:,.0f} for the full model."
                ),
                action=f"Simulated fallback: {step} used; forecast issued {clock}",
                status="resolved",
                source=SOURCE,
                metrics={
                    "submitted_minutes_after_issue": round(minutes, 3),
                    "pnl_eur": earned,
                    "full_pnl_eur": full,
                    "pnl_lost_eur": full - earned,
                },
            )
        )
    return incidents


def _chain_models(settings: Settings) -> dict[str, Forecaster]:
    no_weather = [group for group in FEATURE_GROUPS if group != "weather"]
    naive, seasonal = build_baselines(settings)
    return {
        "full": LightGBMConformalModel(settings),
        "fallback_no_weather": LightGBMConformalModel(
            settings,
            name="lightgbm_conformal_no_weather",
            _names=feature_names(no_weather),
        ),
        "naive_previous_day": naive,
        "seasonal_naive_previous_week": seasonal,
    }


def measure_runtimes(
    inputs: pd.DataFrame, days: list[date], settings: Settings
) -> dict[str, float]:
    """Median seconds to build the information set and forecast, per chain step.

    Timed on the machine that runs it. Hold-out days are refused and the inputs
    are cut before the hold-out first.
    """
    check_days(days, settings)
    inputs = before_holdout(inputs, settings)
    models = _chain_models(settings)
    runtimes = {}
    for name, model in models.items():
        model.fit(
            build_information_set(inputs, days[0], settings, model.fit_lookback_days)
        )
        seconds = []
        for day in days:
            started = time.perf_counter()
            model.forecast(
                build_information_set(inputs, day, settings, model.lookback_days)
            )
            seconds.append(time.perf_counter() - started)
        runtimes[name] = float(np.median(seconds))
        print(f"{name}: {runtimes[name]:.2f} s per forecast", flush=True)
    return runtimes


def summarise(chain: pd.DataFrame) -> dict[str, Any]:
    days = len(chain)
    ceiling = float(chain["perfect_foresight_pnl_eur"].sum())
    fallback = chain[chain["step"] != "full"]
    return {
        "days": days,
        "on_time_share_with_chain": float(chain["on_time"].mean()),
        "on_time_share_without_chain": float((chain["step"] == "full").mean()),
        "fallback_days": len(fallback),
        "fallback_by_step": {
            step: int((chain["step"] == step).sum()) for step in CHAIN[1:]
        },
        "latest_submission_minutes_after_issue": float(
            chain["submitted_minutes_after_issue"].max()
        ),
        "pnl_full_eur": float(chain["full_pnl_eur"].sum()),
        "pnl_chain_eur": float(chain["pnl_eur"].sum()),
        "pnl_without_chain_eur": float(chain["pnl_without_chain_eur"].sum()),
        "perfect_foresight_pnl_eur": ceiling,
        "capture_full": float(chain["full_pnl_eur"].sum()) / ceiling,
        "capture_chain": float(chain["pnl_eur"].sum()) / ceiling,
        "capture_without_chain": float(chain["pnl_without_chain_eur"].sum()) / ceiling,
        "mean_daily_pinball_full": float(chain["full_pinball"].mean()),
        "mean_daily_pinball_chain": float(chain["pinball"].mean()),
    }


def _markdown(
    summary: dict[str, Any], runtimes: dict[str, float], realised: dict[str, float]
) -> str:
    s = summary
    drawn = (
        f"{realised['weather_late']:.1%}, {realised['model_fails']:.1%} and "
        f"{realised['prices_late']:.1%}"
    )
    lines = [
        "# D5 experiment: the 12:00 deadline and the fallback chain",
        "",
        "Generated by `python -m src.health.experiments.d5_deadline`. Failures are",
        f"injected on {s['days']} validation days with seed {SEED}: weather feed",
        f"late on {FAILURE_RATES['weather_late']:.0%} of days, model step failing on",
        f"{FAILURE_RATES['model_fails']:.0%} and yesterday's prices late on "
        f"{FAILURE_RATES['prices_late']:.0%}; the seed drew {drawn}. These are",
        "scenario assumptions, not estimates. Late data is retried once after 5",
        "minutes and has not arrived.",
        "Profit is the D1 run's median dispatch P&L for each step, 1 MW / 2 MWh.",
        "",
        "## Step runtimes",
        "",
        "Median seconds to build the information set and forecast, models fitted,",
        "measured on the machine that ran this experiment; another machine will",
        "differ.",
        "",
        "| step | seconds |",
        "|---|---|",
        *[f"| {STEP_LABELS[name]} | {runtimes[name]:.2f} |" for name in CHAIN],
        "",
        "## Results",
        "",
        "| | full model every day | with the chain | without the chain |",
        "|---|---|---|---|",
        f"| forecast before the gate | 100.0% | {s['on_time_share_with_chain']:.1%} | "
        f"{s['on_time_share_without_chain']:.1%} |",
        f"| median dispatch P&L (€) | {s['pnl_full_eur']:,.0f} | "
        f"{s['pnl_chain_eur']:,.0f} | {s['pnl_without_chain_eur']:,.0f} |",
        f"| capture vs perfect foresight | {s['capture_full']:.1%} | "
        f"{s['capture_chain']:.1%} | {s['capture_without_chain']:.1%} |",
        "",
        f"Fallback days: {s['fallback_days']}, of which "
        + ", ".join(
            f"{count} {STEP_LABELS[step]}"
            for step, count in s["fallback_by_step"].items()
        )
        + ".",
        f"The latest submission came {s['latest_submission_minutes_after_issue']:.1f} "
        "minutes after the issue time.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Simulate pipeline failures against the 12:00 gate."
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    out = settings.data.processed_path / "experiments"
    daily_path = out / "d1_missing_weather_daily.parquet"
    if not daily_path.exists():
        parser.error("run python -m src.health.experiments.d1_missing_weather first")
    daily = pd.read_parquet(daily_path)
    daily = daily[daily["arm"].isin(CHAIN)]
    complete = daily.dropna(subset=["pnl_eur"]).groupby("target_day")["arm"].nunique()
    days = sorted(complete[complete == len(CHAIN)].index)
    check_days(days, settings)

    inputs = before_holdout(pd.read_parquet(settings.data.inputs_path), settings)
    runtimes = measure_runtimes(inputs, days[:RUNTIME_SAMPLE_DAYS], settings)
    failures = draw_failures(days)
    realised = realised_failure_rates(failures)
    chain = run_chain(failures, daily, runtimes)
    summary = summarise(chain)
    print(json.dumps(summary, indent=2), flush=True)

    out.mkdir(parents=True, exist_ok=True)
    chain_path = out / "d5_deadline_daily.parquet"
    partial = chain_path.with_name(f".{chain_path.name}.partial")
    chain.to_parquet(partial)
    os.replace(partial, chain_path)
    payload = {
        "generated_by": "python -m src.health.experiments.d5_deadline",
        "setup": {
            "chain": list(CHAIN),
            "failure_rates": FAILURE_RATES,
            "seed": SEED,
            "retry_wait_minutes": RETRY_WAIT_MINUTES,
            "issue_local": ISSUE_LOCAL,
            "gate_local": GATE_LOCAL,
            "days": [str(days[0]), str(days[-1])],
        },
        "realised_failure_rates": realised,
        "runtimes_s": runtimes,
        "summary": summary,
    }
    json_path = out / "d5_deadline.json"
    partial = json_path.with_name(f".{json_path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, json_path)

    incidents = step_incidents(chain, settings)
    replace_incidents(incidents, (SOURCE,), default_path(settings))
    print(f"{len(incidents)} fallback incidents written", flush=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(_markdown(summary, runtimes, realised), encoding="utf-8")

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(
            EXPERIMENT, artifact_location=(REPO_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name="fallback-chain"):
        mlflow.log_params(
            {"seed": SEED, **{k: str(v) for k, v in FAILURE_RATES.items()}}
        )
        mlflow.log_metrics(
            {
                key: float(value)
                for key, value in summary.items()
                if isinstance(value, int | float)
            }
        )
    print(f"wrote {RESULTS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
