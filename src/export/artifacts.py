"""Dashboard artifacts: everything the Phase 5 dashboard reads, in one run folder.

    uv run python -m src.export.artifacts                      # every step
    uv run python -m src.export.artifacts --steps core         # the quick files only
    uv run python -m src.export.artifacts --steps grid --workers 9
    uv run python -m src.export.artifacts --steps health       # Model health files

A run is one release of the backtest, named after its last delivery day and written
to ``data/processed/dashboard/<run_id>/``; ``latest.json`` beside the runs names the
newest one.

- ``manifest.json``: dates, every day with its window and whether it traded, the
  reference battery, the control grid and how the dashboard uses it.
- ``forecasts.parquet``: walk-forward forecasts of the production model and the
  naive baseline for every day of the validation window and the hold-out.
- ``pnl_grid.parquet``: daily P&L of perfect foresight and the three dashboard
  strategies for every battery duration and wear price in the grid, at 1 MW. With
  duration, wear and the state-of-charge rules fixed, a price taker's optimal
  schedule scales exactly with power, so the dashboard multiplies by its power
  slider instead of solving again.
- ``hour_value.parquet``: cash perfect foresight moves in each local hour of each
  day, for the reference battery; the dashboard shows it as the value at stake
  beside forecast error.
- ``attribution.parquet``: the Shapley split of median dispatch's gap to perfect
  foresight by local hour block and error direction, for the reference battery.
- ``feature_importance.json``: LightGBM gain of the production model trained for
  the last day.
- ``health/``: copies of the Phase 6 experiment results the Model health tab reads
  (``m1_regime_experiment.json``, ``m2_drift.json``, ``d5_deadline.json``), the
  incident log as one JSON list (``incidents.json``) and ``index.json``, which lists
  the files present, where each came from, when its source was generated and last
  modified, and whether M2 ends on the run's last day. A source that does not exist
  is skipped with a note and any earlier copy of it is removed.

The hold-out was evaluated once in Phase 4. The dashboard reports those days; nothing
is chosen from them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, date, datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from src.config import REPO_ROOT, Settings, load_settings
from src.forecasting.information import build_information_set
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.production import build_production_model
from src.health.incidents import default_path, load_incidents
from src.trading.run_strategies import run_strategies, select_days
from src.trading.strategies import (
    MEDIAN_FORECAST,
    PERFECT_FORESIGHT,
    PRODUCT,
    REALISED,
    Strategy,
    quantile_aware,
)

__all__ = [
    "BASELINE_MODEL",
    "GRID_DEGRADATION_EUR_PER_MWH",
    "GRID_DURATIONS_H",
    "HEALTH_EXPERIMENTS",
    "HEALTH_INCIDENTS",
    "POWER_MW",
    "attribution_table",
    "build_manifest",
    "combined_forecasts",
    "dashboard_strategies",
    "export_health",
    "feature_label",
    "grid_fingerprint",
    "hour_value_table",
    "main",
    "pnl_grid",
    "traded_days",
    "write_json",
    "write_parquet",
]

BASELINE_MODEL = "naive_previous_day"
GRID_DURATIONS_H = (1, 2, 3, 4)
GRID_DEGRADATION_EUR_PER_MWH = tuple(range(26))
POWER_MW = {"min": 0.5, "max": 5.0, "step": 0.5}
GRID_COLUMNS = (
    "duration_h",
    "degradation_eur_per_mwh",
    "strategy",
    "target_day",
    "pnl_eur",
    "revenue_eur",
    "degradation_eur",
    "discharged_mwh",
    "cycles",
)
FEATURE_LABELS = {
    "clock_minute": "Time of day",
    "weekday": "Day of week",
    "is_weekend": "Weekend",
    "is_holiday": "Public holiday",
    "price_lag_1d": "Price, same quarter-hour yesterday",
    "price_lag_2d": "Price, same quarter-hour two days ago",
    "price_lag_7d": "Price, same quarter-hour last week",
    "price_mean_same_clock_7d": "Price, same quarter-hour, 7-day mean",
    "price_prev_day_mean": "Price, yesterday's mean",
    "price_prev_day_min": "Price, yesterday's minimum",
    "price_prev_day_max": "Price, yesterday's maximum",
    "price_prev_day_std": "Price, yesterday's volatility",
    "price_prev_day_last": "Price, yesterday's last quarter-hour",
    "day_of_year_sin": "Season, sine",
    "day_of_year_cos": "Season, cosine",
    "periods_in_day": "Quarter-hours in the day",
    "quarter_hour_product": "15-minute products",
    "load_forecast_mw": "Load forecast",
    "load_forecast_day_mean_mw": "Load forecast, day mean",
    "load_forecast_change_1d_mw": "Load forecast, change from yesterday",
    "wind_onshore_forecast_mw_lag_1d": "Grid operator onshore wind forecast, yesterday",
    "wind_offshore_forecast_mw_lag_1d": (
        "Grid operator offshore wind forecast, yesterday"
    ),
    "solar_forecast_mw_lag_1d": "Grid operator solar forecast, yesterday",
    "residual_load_persistence_mw": "Residual load, persistence",
    "wind_actual_last_24h_mw": "Wind generation, last 24 hours",
    "solar_actual_last_24h_mw": "Solar generation, last 24 hours",
    "load_actual_last_24h_mw": "Load, last 24 hours",
    "wx_wind_speed_onshore_mean": "Onshore wind speed, weather forecast",
    "wx_wind_speed_offshore_mean": "Offshore wind speed, weather forecast",
    "wx_wind_power_onshore": "Onshore wind power, weather forecast",
    "wx_wind_power_offshore": "Offshore wind power, weather forecast",
    "wx_radiation_mean": "Solar radiation, weather forecast",
    "wx_radiation_south_mean": "Solar radiation in the south, weather forecast",
    "wx_temperature_mean": "Temperature, weather forecast",
    "gas_ttf_eur_mwh": "Gas price, TTF",
    "carbon_eua_eur_t": "Carbon price, EU allowances",
    "ccgt_marginal_cost_eur_mwh": "Gas plant marginal cost",
}
#: Experiment results the Model health tab reads, from ``<processed>/experiments``.
HEALTH_EXPERIMENTS = ("m1_regime_experiment.json", "m2_drift.json", "d5_deadline.json")
#: The incident log, exported as one JSON list rather than JSON Lines.
HEALTH_INCIDENTS = "incidents.json"
MODE = (
    "Battery results come from a pre-computed grid at 1 MW for every duration and "
    "wear price, scaled by the power setting. Each day's schedule is solved on "
    "request with the same optimizer."
)


def dashboard_strategies(settings: Settings) -> dict[str, Strategy]:
    """The dispatch dial: median plus the configured quantile-aware levels."""
    dial: dict[str, Strategy] = {"median": MEDIAN_FORECAST}
    for level in settings.trading.dispatch_quantiles:
        strategy = quantile_aware(level)
        dial[strategy.sell_column] = strategy
    return dial


def feature_label(name: str) -> str:
    """A readable label for a model feature."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]
    text = name.replace("_eur_mwh", " €/MWh").replace("_mw", " MW")
    for code, words in (("_1d", " yesterday"), ("_7d", " last week"), ("wx_", "")):
        text = text.replace(code, words)
    text = text.replace("_", " ").strip()
    return text[:1].upper() + text[1:]


def combined_forecasts(settings: Settings, model: str) -> pd.DataFrame:
    """Validation-window and hold-out forecasts of one model, with their window."""
    forecasts = settings.data.processed_path / "forecasts"
    comparison = pd.read_parquet(forecasts / "comparison" / f"{model}.parquet")
    validation = comparison[
        (comparison["target_day"] >= settings.evaluation.validation_start)
        & (comparison["target_day"] < settings.evaluation.holdout_start)
    ]
    parts = [validation.assign(window="validation")]
    holdout_path = forecasts / "holdout" / f"{model}.parquet"
    if holdout_path.exists():
        parts.append(pd.read_parquet(holdout_path).assign(window="holdout"))
    frame = pd.concat(parts).sort_index()
    frame.index.name = "timestamp_utc"
    return frame


def traded_days(
    forecasts: pd.DataFrame, settings: Settings
) -> tuple[list[date], dict[date, str]]:
    """Days every dashboard strategy can trade, and why the others cannot."""
    strategies = (PERFECT_FORESIGHT, *dashboard_strategies(settings).values())
    columns = {REALISED, PRODUCT}
    for strategy in strategies:
        columns |= {strategy.sell_column, strategy.buy_column}
    return select_days(
        forecasts,
        settings.evaluation.validation_start,
        max(forecasts["target_day"]),
        settings,
        columns,
        allow_holdout=True,
    )


def build_manifest(
    settings: Settings,
    run_id: str,
    days: list[date],
    skipped: dict[date, str],
    *,
    run_kind: Literal["backtest", "live"] = "backtest",
    issued_utc: str | None = None,
    data_through: date | None = None,
) -> dict[str, Any]:
    """Everything the dashboard needs to know about a run before reading its files.

    ``run_kind`` is ``"backtest"`` for a replay of history and ``"live"`` for a run
    the daily pipeline produced. A live run also carries ``issued_utc``, when it
    issued its forecast, and ``data_through``, the last delivery day with published
    prices, which can be behind the day it forecasts. A backtest leaves
    ``issued_utc`` null and its prices run to its last day.
    """
    holdout = settings.evaluation.holdout_start
    every_day = sorted(set(days) | set(skipped))
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {
        "run_id": run_id,
        "run_kind": run_kind,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "issued_utc": issued_utc,
        "source_commit": commit,
        "model": settings.forecasting.production_model,
        "baseline_model": BASELINE_MODEL,
        "first_day": str(every_day[0]),
        "last_day": str(every_day[-1]),
        "data_through": str(data_through if data_through else every_day[-1]),
        "holdout_start": str(holdout),
        "timezone": settings.market.timezone,
        "forecast_issue_local": settings.market.forecast_issue_local,
        "gate_closure_local": settings.market.gate_closure_local,
        "days": [
            {
                "date": str(day),
                "window": "holdout" if day >= holdout else "validation",
                "traded": day not in skipped,
                "skip_reason": skipped.get(day),
            }
            for day in every_day
        ],
        "reference_battery": settings.battery.model_dump(),
        "grid": {
            "durations_h": list(GRID_DURATIONS_H),
            "degradation_eur_per_mwh": list(GRID_DEGRADATION_EUR_PER_MWH),
            "power_mw": POWER_MW,
            "strategies": {
                key: strategy.name
                for key, strategy in dashboard_strategies(settings).items()
            },
        },
        "mode": MODE,
    }


def _replace_atomically(path: Path, write: Callable[[Path], object]) -> None:
    """Write beside ``path`` and swap it in, so a reader never sees half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    try:
        write(partial)
        os.replace(partial, path)
    finally:
        # After a successful swap the partial is gone; after a failed write it
        # would otherwise be left beside the real file.
        partial.unlink(missing_ok=True)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    _replace_atomically(path, frame.to_parquet)


def write_json(payload: Any, path: Path) -> None:
    text = json.dumps(payload, indent=2)
    _replace_atomically(path, lambda target: target.write_text(text, encoding="utf-8"))


def grid_fingerprint(
    forecasts: pd.DataFrame, days: list[date], settings: Settings
) -> str:
    """A hash of everything a grid cell depends on besides duration and wear."""
    strategies = (PERFECT_FORESIGHT, *dashboard_strategies(settings).values())
    columns = {REALISED, PRODUCT, "target_day"}
    for strategy in strategies:
        columns |= {strategy.sell_column, strategy.buy_column}
    used = forecasts.loc[forecasts["target_day"].isin(days), sorted(columns)]
    digest = hashlib.sha256(
        pd.util.hash_pandas_object(used.sort_index(), index=True).to_numpy().tobytes()
    )
    context = {
        "days": [str(day) for day in days],
        "battery": settings.battery.model_dump(),
        "strategies": [[s.name, s.sell_column, s.buy_column] for s in strategies],
    }
    digest.update(json.dumps(context, sort_keys=True, default=str).encode())
    return digest.hexdigest()


def _prepare_checkpoints(folder: Path, fingerprint: str) -> None:
    """Drop saved cells that were solved for other inputs."""
    stamp = folder / "fingerprint.json"
    saved = None
    if stamp.exists():
        try:
            saved = json.loads(stamp.read_text(encoding="utf-8")).get("fingerprint")
        except ValueError:
            saved = None
    if saved != fingerprint:
        for part in folder.glob("*.parquet"):
            part.unlink()
    write_json({"fingerprint": fingerprint}, stamp)


def pnl_grid(
    forecasts: pd.DataFrame,
    days: list[date],
    settings: Settings,
    *,
    durations: tuple[int, ...] = GRID_DURATIONS_H,
    degradations: tuple[int, ...] = GRID_DEGRADATION_EUR_PER_MWH,
    workers: int = 1,
    checkpoint_dir: Path | None = None,
) -> pd.DataFrame:
    """Daily P&L at 1 MW for every duration, wear price and dashboard strategy.

    Each cell is saved to ``checkpoint_dir`` as soon as it is solved, so an
    interrupted export resumes where it stopped. Saved cells are reused only when
    the forecasts, days, battery and strategies they were solved for are unchanged.
    """
    strategies = (PERFECT_FORESIGHT, *dashboard_strategies(settings).values())
    parts: list[pd.DataFrame] = []
    total = len(durations) * len(degradations)
    started = time.perf_counter()
    if checkpoint_dir is not None:
        _prepare_checkpoints(
            checkpoint_dir, grid_fingerprint(forecasts, days, settings)
        )
    for duration in durations:
        for wear in degradations:
            path = (
                checkpoint_dir / f"duration{duration}_wear{wear}.parquet"
                if checkpoint_dir is not None
                else None
            )
            if path is not None and path.exists():
                parts.append(pd.read_parquet(path))
                continue
            battery = settings.battery.model_copy(
                update={
                    "power_mw": 1.0,
                    "capacity_mwh": float(duration),
                    "degradation_eur_per_mwh": float(wear),
                }
            )
            _, pnl, failed = run_strategies(
                forecasts,
                days,
                strategies,
                battery,
                holdout_start=settings.evaluation.holdout_start,
                time_limit_s=settings.trading.solver_time_limit_s,
                workers=workers,
                allow_holdout=True,
            )
            if failed:
                # A pooled solve now and then fails for no reason in the model: the
                # same days solve on their own. Retry them once, one at a time.
                _, retried, still_failed = run_strategies(
                    forecasts,
                    sorted(failed),
                    strategies,
                    battery,
                    holdout_start=settings.evaluation.holdout_start,
                    time_limit_s=settings.trading.solver_time_limit_s,
                    workers=1,
                    allow_holdout=True,
                )
                if still_failed:
                    raise RuntimeError(
                        f"solver failed for duration {duration} h, wear €{wear}: "
                        f"{still_failed}"
                    )
                pnl = pd.concat([pnl, retried], ignore_index=True).sort_values(
                    ["target_day", "strategy"], ignore_index=True
                )
            cell = pnl.assign(duration_h=duration, degradation_eur_per_mwh=wear)[
                list(GRID_COLUMNS)
            ]
            if path is not None:
                write_parquet(cell, path)
            parts.append(cell)
            elapsed = time.perf_counter() - started
            print(
                f"grid {len(parts)}/{total}: {duration} h, €{wear}/MWh "
                f"({elapsed:.0f} s)",
                flush=True,
            )
    return pd.concat(parts, ignore_index=True)


def hour_value_table(settings: Settings, model: str) -> pd.DataFrame:
    """Cash perfect foresight moves per local hour and day, reference battery."""
    backtest = settings.data.processed_path / "backtest"
    parts = []
    for suite in ("validation", "holdout"):
        path = backtest / suite / "dispatch.parquet"
        if not path.exists():
            continue
        dispatch = pd.read_parquet(
            path,
            columns=["model", "target_day", "strategy", "net_mw", "realised_price"],
        )
        parts.append(
            dispatch[
                (dispatch["strategy"] == PERFECT_FORESIGHT.name)
                & (dispatch["model"] == model)
            ]
        )
    dispatch = pd.concat(parts)
    local = pd.DatetimeIndex(dispatch.index).tz_convert(settings.market.timezone)
    period_hours = 0.25
    frame = pd.DataFrame(
        {
            "target_day": dispatch["target_day"].to_numpy(),
            "hour": local.hour,
            "cash_eur": (period_hours * dispatch["realised_price"] * dispatch["net_mw"])
            .abs()
            .to_numpy(),
        }
    )
    return frame.groupby(["target_day", "hour"], as_index=False).agg(
        cash_eur=("cash_eur", "sum")
    )


def attribution_table(settings: Settings) -> pd.DataFrame:
    """Shapley shares of median dispatch's gap, validation and hold-out days."""
    backtest = settings.data.processed_path / "backtest"
    columns = ["target_day", "block", "direction", "periods", "cost_eur"]
    parts = []
    for path in (
        backtest / "attribution" / "costs.parquet",
        backtest / "holdout" / "attribution_costs.parquet",
    ):
        if path.exists():
            costs = pd.read_parquet(path)
            parts.append(costs.loc[costs["strategy"] == MEDIAN_FORECAST.name, columns])
    return pd.concat(parts, ignore_index=True).sort_values(["target_day", "block"])


def feature_importance(settings: Settings, last_day: date) -> dict[str, Any]:
    """LightGBM gain of the production model trained as for ``last_day``."""
    model = build_production_model(settings)
    if not isinstance(model, LightGBMConformalModel):
        raise TypeError("feature importance needs the LightGBM production model")
    frame = pd.read_parquet(settings.data.inputs_path)
    model.fit(build_information_set(frame, last_day, settings, model.fit_lookback_days))
    gains = model.feature_importance()
    total = float(gains.sum())
    top = gains.sort_values(ascending=False).head(12)
    return {
        "model": model.name,
        "trained_for_day": str(last_day),
        "importance": "gain",
        "features": [
            {
                "feature": str(name),
                "label": feature_label(str(name)),
                "gain_share": float(gain) / total,
            }
            for name, gain in top.items()
        ],
    }


def _utc_stamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat(timespec="seconds")


def _source_entry(name: str, source: Path, label: str) -> dict[str, Any]:
    """Index entry of one copied file: where it came from and when it was written.

    ``generated_utc`` is the source's own ``generated_utc`` field when it carries
    one, otherwise its modification time; ``source_modified_utc`` is always the
    modification time, and ``exported_utc`` when this copy was made.
    """
    modified = _utc_stamp(source.stat().st_mtime)
    generated = modified
    if source.suffix == ".json":
        try:
            stamp = json.loads(source.read_text(encoding="utf-8")).get("generated_utc")
        except (ValueError, AttributeError):
            stamp = None
        if isinstance(stamp, str):
            generated = stamp
    return {
        "file": name,
        "source": label,
        "generated_utc": generated,
        "source_modified_utc": modified,
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }


def _drift_run_match(source: Path, run_dir: Path) -> dict[str, Any]:
    """Whether M2's last target day is the run's last day, from the manifest."""
    try:
        series = json.loads(source.read_text(encoding="utf-8")).get("series") or []
        last = str(series[-1]["target_day"]) if series else None
    except (ValueError, AttributeError, KeyError, TypeError):
        last = None
    manifest = run_dir / "manifest.json"
    run_last = None
    if manifest.exists():
        try:
            run_last = json.loads(manifest.read_text(encoding="utf-8")).get("last_day")
        except (ValueError, AttributeError):
            run_last = None
    return {
        "last_target_day": last,
        "run_last_day": run_last,
        "matches_run": last is not None and last == run_last,
    }


def _remove_stale(folder: Path, name: str, label: str) -> None:
    """Drop a copy whose source is gone, so the tab never shows stale data."""
    print(f"health: {label} not found, skipped", flush=True)
    copy = folder / name
    if copy.exists():
        copy.unlink()
        print(f"health: removed the earlier copy of {name}", flush=True)


#: What a live run adds to the health folder: the day the pipeline actually ran.
HEALTH_LIVE_DAY = "live_day.json"


def _live_day_entry(
    settings: Settings, folder: Path, day: date | None
) -> dict[str, Any] | None:
    """The pipeline's own record for ``day``, copied in beside the experiments.

    A live export otherwise shows backtest artifacts under a live label. This is
    the one file that says what today's run did: which chain step issued the
    forecast, which registered model served it, what it committed and whether it
    made the gate.
    """
    if day is None:
        _remove_stale(folder, HEALTH_LIVE_DAY, f"pipeline/runs/{HEALTH_LIVE_DAY}")
        return None
    source = settings.data.processed_path / "pipeline" / "runs" / f"{day}.json"
    label = f"pipeline/runs/{day}.json"
    if not source.exists():
        _remove_stale(folder, HEALTH_LIVE_DAY, label)
        return None
    record = json.loads(source.read_text(encoding="utf-8"))
    summary = {
        "target_day": record["target_day"],
        "issued_utc": record["issued_utc"],
        "on_time": record["on_time"],
        "minutes_before_gate": record["minutes_before_gate"],
        "step": record["step"],
        "model": record["model"],
        "model_version": record.get("model_version"),
        "planned_value_eur": record["planned_value_eur"],
        "readiness": record["readiness"],
        "attempts": record["attempts"],
        "incidents": record["incidents"],
    }
    write_json(summary, folder / HEALTH_LIVE_DAY)
    return _source_entry(HEALTH_LIVE_DAY, source, label)


def export_health(
    settings: Settings, run_dir: Path, *, live_day: date | None = None
) -> dict[str, Any]:
    """Copy the Model health sources into ``<run_dir>/health`` and index them.

    Each file is swapped in whole and ``index.json`` goes last. A missing source is
    skipped with a printed note and any copy an earlier export left is removed, so
    the tab answers "not exported" rather than serving data that no longer has a
    source. M2's entry records whether its last target day is the run's last day.
    """
    processed = settings.data.processed_path
    folder = run_dir / "health"
    files: dict[str, dict[str, Any]] = {}
    for name in HEALTH_EXPERIMENTS:
        source = processed / "experiments" / name
        label = f"experiments/{name}"
        if not source.exists():
            _remove_stale(folder, name, label)
            continue
        entry = _source_entry(name, source, label)
        _replace_atomically(folder / name, partial(shutil.copyfile, source))
        if name == "m2_drift.json":
            entry |= _drift_run_match(source, run_dir)
            if not entry["matches_run"]:
                print(
                    f"health: m2_drift.json ends on {entry['last_target_day']}, "
                    f"the run on {entry['run_last_day']}; rerun M2",
                    flush=True,
                )
        files[name] = entry
    log = default_path(settings)
    label = f"health/{log.name}"
    if log.exists():
        entry = _source_entry(HEALTH_INCIDENTS, log, label)
        incidents = load_incidents(log)
        write_json(
            [incident.model_dump(mode="json") for incident in incidents],
            folder / HEALTH_INCIDENTS,
        )
        files[HEALTH_INCIDENTS] = entry | {"records": len(incidents)}
    else:
        _remove_stale(folder, HEALTH_INCIDENTS, label)
    live_entry = _live_day_entry(settings, folder, live_day)
    if live_entry is not None:
        files[HEALTH_LIVE_DAY] = live_entry
    index = {
        "exported_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "files": [files[name] for name in sorted(files)],
    }
    write_json(index, folder / "index.json")
    return index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the dashboard artifacts.")
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=("core", "grid", "health", "live"),
        default=["core", "grid", "health", "live"],
    )
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--run-kind",
        choices=("backtest", "live"),
        default="backtest",
        help="a live run is one the daily pipeline produced, not a backtest",
    )
    parser.add_argument(
        "--issued-utc",
        default=None,
        help="when a live run issued its forecast, ISO 8601",
    )
    parser.add_argument(
        "--data-through",
        type=date.fromisoformat,
        default=None,
        help="last delivery day whose prices are published",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be at least 1")

    settings = load_settings(args.config)
    model = settings.forecasting.production_model
    forecasts = combined_forecasts(settings, model)
    # A backtest run is named after the last day it covers. A live run is named
    # after the day the pipeline produced it, which is later than the backtest
    # forecasts it reads, so the two never collide and the name is not misleading.
    last_forecast_day = max(forecasts["target_day"])
    if args.run_kind == "live":
        issued = (
            pd.Timestamp(args.issued_utc)
            if args.issued_utc
            else pd.Timestamp.now(tz="UTC")
        )
        # The flag says UTC, so a value without a zone is read as UTC rather than
        # raising. Airflow passes an aware timestamp; a person often does not.
        if issued.tzinfo is None:
            issued = issued.tz_localize("UTC")
        run_id = f"live-{issued.tz_convert(settings.market.timezone).date()}"
    else:
        run_id = f"backtest-{last_forecast_day}"
    root = settings.data.processed_path / "dashboard"
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    days: list[date] = []
    skipped: dict[date, str] = {}
    if {"core", "grid"} & set(args.steps):
        days, skipped = traded_days(forecasts, settings)
        print(f"{run_id}: {len(days)} traded days, {len(skipped)} skipped", flush=True)

    if "core" in args.steps:
        # Every file is swapped in whole, and the manifest goes last: the API
        # treats a folder with a manifest as a run.
        baseline = combined_forecasts(settings, BASELINE_MODEL)
        write_parquet(pd.concat([forecasts, baseline]), run_dir / "forecasts.parquet")
        write_parquet(hour_value_table(settings, model), run_dir / "hour_value.parquet")
        write_parquet(attribution_table(settings), run_dir / "attribution.parquet")
        write_json(
            feature_importance(settings, days[-1]),
            run_dir / "feature_importance.json",
        )
        write_json(
            build_manifest(
                settings,
                run_id,
                days,
                skipped,
                run_kind=args.run_kind,
                issued_utc=args.issued_utc,
                data_through=args.data_through,
            ),
            run_dir / "manifest.json",
        )
        print("core files written", flush=True)

    if "grid" in args.steps:
        parts_dir = run_dir / "grid_parts"
        grid = pnl_grid(
            forecasts, days, settings, workers=args.workers, checkpoint_dir=parts_dir
        )
        write_parquet(grid, run_dir / "pnl_grid.parquet")
        shutil.rmtree(parts_dir)
        print(f"grid written: {len(grid):,} rows", flush=True)

    if "health" in args.steps:
        # A live run points at the day the pipeline forecast, which is the day
        # after the last published prices.
        live_day = (
            args.data_through + timedelta(days=1)
            if args.run_kind == "live" and args.data_through
            else None
        )
        index = export_health(settings, run_dir, live_day=live_day)
        print(f"health files written: {len(index['files'])}", flush=True)

    write_json({"run_id": run_id}, root / "latest.json")
    # After latest.json, so a fault in the live record never leaves the run
    # exported but unpublished.
    if "live" in args.steps:
        from src.export.live import export_live

        live = export_live(settings, root)
        print(
            f"live record written: {live['totals']['days']} days, "
            f"{live['totals']['settled_days']} settled",
            flush=True,
        )
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
