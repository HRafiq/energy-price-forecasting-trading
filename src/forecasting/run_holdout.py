"""Forecast the hold-out walk-forward, the way the validation comparison did.

    uv run python -m src.forecasting.run_holdout --confirm-holdout
    uv run python -m src.forecasting.run_holdout --confirm-holdout --last 2026-09-14
    uv run python -m src.forecasting.run_holdout --confirm-holdout \
        --models lightgbm_conformal naive_previous_day

This is the only command that forecasts days in the hold-out, and it refuses to
run without ``--confirm-holdout``. The hold-out is forecast once, after every
modelling and trading choice is frozen.

Target days run from ``evaluation.holdout_start`` to ``--last``, or, without it,
to the last local day whose realised prices are complete in the inputs file.
Each model uses the same factory, refit interval, lookbacks and information-set
rules as ``run_comparison``; refits are counted from the hold-out start rather
than from the comparison's first day. Forecasts are written with the same columns
to ``data/processed/forecasts/holdout/<model>.parquet``, and existing files are
not overwritten without ``--overwrite``. Days whose realised prices are incomplete
are still forecast, as a live system would, and are listed so later scoring can
skip them.

The command never scores the forecasts. It prints only models, day and period
counts and run time, and logs each run to the MLflow experiment
``price-forecast-holdout`` with its parameters and run time but no accuracy
metrics, so nobody sees hold-out accuracy before the frozen evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import mlflow
import pandas as pd

from src.config import PRICE_SERIES, RESOLUTION_STEP, Settings, load_settings
from src.forecasting.base import Forecaster
from src.forecasting.run_comparison import (
    ARTIFACTS,
    CANDIDATES,
    REFIT_EVERY_DAYS,
    TRACKING_URI,
)
from src.forecasting.walkforward import day_range, run_walk_forward
from src.timegrid import delivery_periods, ensure_utc_index

__all__ = [
    "EXPERIMENT",
    "HoldoutRun",
    "default_models",
    "holdout_days",
    "incomplete_price_days",
    "last_complete_price_day",
    "main",
    "run_holdout",
]

EXPERIMENT = "price-forecast-holdout"
#: Best validation forecasters after the production model, plus the naive baseline.
COMPARED_WITH_PRODUCTION = (
    "lightgbm_quantile",
    "quantile_forest",
    "naive_previous_day",
)
REFUSAL = (
    "refusing to forecast the hold-out without --confirm-holdout. The hold-out "
    "is forecast once, after every modelling and trading choice is frozen."
)


@dataclass(frozen=True)
class HoldoutRun:
    """What one model's hold-out run produced, without any accuracy figures."""

    model: str
    days: int
    periods: int
    seconds: float
    path: Path


def default_models(settings: Settings) -> list[str]:
    """The production model followed by the models it is compared with."""
    first = settings.forecasting.production_model
    return [first, *(name for name in COMPARED_WITH_PRODUCTION if name != first)]


def last_complete_price_day(frame: pd.DataFrame, settings: Settings) -> date:
    """The last local day on which every delivery period has a realised price.

    Days are local calendar days in ``market.timezone``, so a daylight-saving
    change day needs its 92 or 100 quarter-hours. A final day with any missing
    price, or with rows not yet present, is skipped.
    """
    index = ensure_utc_index(frame.index)
    prices = pd.Series(frame[PRICE_SERIES].to_numpy(), index=index)
    known = prices.dropna().index
    if known.empty:
        raise ValueError("the inputs have no realised prices")
    tz = settings.market.timezone
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    earliest: date = known[0].tz_convert(tz).date()
    day: date = known[-1].tz_convert(tz).date()
    while day >= earliest:
        if prices.reindex(delivery_periods(day, tz, step)).notna().all():
            return day
        day -= timedelta(days=1)
    raise ValueError("the inputs have no local day with complete prices")


def holdout_days(settings: Settings, last: date) -> list[date]:
    """Every target day from the hold-out start to ``last``, inclusive."""
    first = settings.evaluation.holdout_start
    if last < first:
        raise ValueError(f"last day {last} is before the hold-out start {first}")
    return day_range(first, last)


def incomplete_price_days(
    frame: pd.DataFrame, settings: Settings, days: Sequence[date]
) -> list[date]:
    """Target days on which some delivery period has no realised price.

    Such a day cannot be scored or traded, and the days after it lose their lagged
    prices as well.
    """
    index = ensure_utc_index(frame.index)
    prices = pd.Series(frame[PRICE_SERIES].to_numpy(), index=index)
    tz = settings.market.timezone
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    return [
        day
        for day in days
        if not prices.reindex(delivery_periods(day, tz, step)).notna().all()
    ]


def _params(model: object) -> dict[str, str]:
    """Public dataclass fields of a model; mirrors the comparison's run params."""
    if not dataclasses.is_dataclass(model):
        return {}
    return {
        f.name: str(getattr(model, f.name))
        for f in dataclasses.fields(model)
        if not f.name.startswith("_") and f.name not in ("settings", "members")
    }


def _log_run(
    name: str,
    model: Forecaster,
    days: Sequence[date],
    refit_every_days: int,
    seconds: float,
    production: bool,
) -> None:
    """Record parameters and run time only; hold-out accuracy stays unlogged."""
    with mlflow.start_run(run_name=name):
        mlflow.set_tags(
            {
                "model": name,
                "window": "holdout",
                "production_model": str(production).lower(),
            }
        )
        mlflow.log_params(
            {
                **_params(model),
                "refit_every_days": str(refit_every_days),
                "first_day": str(days[0]),
                "last_day": str(days[-1]),
                "target_days": str(len(days)),
                "lookback_days": str(model.lookback_days),
                "fit_lookback_days": str(model.fit_lookback_days),
            }
        )
        mlflow.log_metric("run_seconds", seconds)


def run_holdout(
    frame: pd.DataFrame,
    settings: Settings,
    models: Sequence[str],
    last: date,
    out_dir: Path,
    factories: Mapping[str, Callable[[Settings], Forecaster]] = CANDIDATES,
    log: bool = True,
) -> list[HoldoutRun]:
    """Forecast every hold-out day to ``last`` with each model and save the result.

    Rows after ``last`` are cut first, so no model and no output can contain a
    later day. Each model refits on its comparison schedule, counted from the
    hold-out start. With ``log`` each run is logged to the active MLflow
    experiment.
    """
    days = holdout_days(settings, last)
    frame = frame.loc[
        ensure_utc_index(frame.index)
        < settings.market.local_midnight_utc(last + timedelta(days=1))
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    runs: list[HoldoutRun] = []
    for name in models:
        model = factories[name](settings)
        refit = REFIT_EVERY_DAYS[name]
        started = time.perf_counter()
        forecasts = run_walk_forward(
            frame, model, days, settings, refit_every_days=refit, allow_holdout=True
        )
        seconds = time.perf_counter() - started
        path = out_dir / f"{name}.parquet"
        forecasts.to_parquet(path)
        if log:
            production = name == settings.forecasting.production_model
            _log_run(name, model, days, refit, seconds, production)
        run = HoldoutRun(name, len(days), len(forecasts), seconds, path)
        runs.append(run)
        print(
            f"{name}: {run.days} days, {run.periods} periods in {seconds:.0f} s",
            flush=True,
        )
    return runs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Forecast the hold-out walk-forward, without scoring it."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CANDIDATES),
        default=None,
        help="default: the production model, lightgbm_quantile, quantile_forest "
        "and naive_previous_day",
    )
    parser.add_argument(
        "--last",
        type=date.fromisoformat,
        default=None,
        help="last target day; default: the last day with complete prices",
    )
    parser.add_argument(
        "--confirm-holdout",
        action="store_true",
        help="required: confirms that every choice is frozen",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing hold-out forecasts; the hold-out is forecast once",
    )
    args = parser.parse_args(argv)
    if not args.confirm_holdout:
        print(REFUSAL, file=sys.stderr)
        return 2

    settings = load_settings(args.config)
    models = list(dict.fromkeys(args.models or default_models(settings)))
    out_dir = settings.data.processed_path / "forecasts" / "holdout"
    existing = [name for name in models if (out_dir / f"{name}.parquet").exists()]
    if existing and not args.overwrite:
        parser.error(
            f"hold-out forecasts already exist for {', '.join(existing)}; the "
            "hold-out is forecast once. Pass --overwrite only to repeat a run on "
            "purpose."
        )
    frame = pd.read_parquet(settings.data.inputs_path)
    last = args.last or last_complete_price_day(frame, settings)
    try:
        days = holdout_days(settings, last)
    except ValueError as exc:
        parser.error(str(exc))
    print(
        f"hold-out forecasts for {days[0]} to {days[-1]} ({len(days)} days): "
        f"{', '.join(models)}",
        flush=True,
    )
    incomplete = incomplete_price_days(frame, settings, days)
    if incomplete:
        print(
            "days without complete realised prices, forecast but not scorable: "
            + ", ".join(str(day) for day in incomplete),
            flush=True,
        )

    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACTS.as_uri())
    mlflow.set_experiment(EXPERIMENT)

    started = time.perf_counter()
    run_holdout(frame, settings, models, last, out_dir)
    print(
        f"wrote {len(models)} forecast files to {out_dir} "
        f"in {time.perf_counter() - started:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
