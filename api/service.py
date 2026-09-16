"""Queries behind the dashboard API, over one run of dashboard artifacts.

A run is the folder ``python -m src.export.artifacts`` writes. Every function here
reads numbers the backtest already produced, filters them to a window and a battery,
and aggregates; the only thing computed fresh is one day's schedule, solved with the
same optimizer the backtest used. The functions take loaded data and return plain
dictionaries, so they are tested without HTTP.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.forecasting.base import quantile_column
from src.forecasting.evaluate import score
from src.trading.backtest import max_drawdown
from src.trading.battery import Battery
from src.trading.optimizer import DispatchError
from src.trading.strategies import (
    MEDIAN_FORECAST,
    PERFECT_FORESIGHT,
    Strategy,
    dispatch_day,
    quantile_aware,
)

__all__ = [
    "INTERVALS",
    "WINDOWS",
    "ArtifactsMissingError",
    "DayNotFoundError",
    "RequestError",
    "Run",
    "SolverError",
    "Window",
    "calibration",
    "clear_cache",
    "dispatch",
    "error_analysis",
    "forecast_day",
    "latest_run_id",
    "list_run_ids",
    "load_run",
    "parse_day",
    "pnl_series",
    "resolve_window",
    "run_info",
    "summary",
]

WINDOWS = ("last30", "last90", "validation", "holdout", "all")
WINDOW_DAYS = {"last30": 30, "last90": 90}
#: Central intervals whose coverage the calibration panel reports.
INTERVALS = ((0.5, 0.25, 0.75), (0.8, 0.10, 0.90), (0.9, 0.05, 0.95))
#: A live solve that takes longer than this is abandoned; backtest days solve in
#: well under a second.
SOLVE_TIME_LIMIT_S = 10.0
DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


class ArtifactsMissingError(RuntimeError):
    """A run, or a file the request needs, has not been exported."""


class RequestError(ValueError):
    """A request the run cannot answer as asked: a bad date, window or battery."""


class DayNotFoundError(LookupError):
    """A well-formed date the run has no forecast for."""


class SolverError(RuntimeError):
    """The live solve did not finish with an optimal schedule."""


@dataclass(frozen=True)
class Run:
    """One run's artifacts, loaded once."""

    run_id: str
    manifest: dict[str, Any]
    forecasts: pd.DataFrame
    grid: pd.DataFrame | None
    hour_value: pd.DataFrame
    attribution: pd.DataFrame
    feature_importance: dict[str, Any]

    @property
    def quantiles(self) -> tuple[float, ...]:
        columns = [
            c for c in self.forecasts.columns if c[:1] == "q" and c[1:].isdigit()
        ]
        return tuple(sorted(int(c[1:]) / 100 for c in columns))


@dataclass(frozen=True)
class Window:
    """The days a request covers."""

    key: str
    first_day: date
    last_day: date
    traded_days: tuple[date, ...]
    skipped_days: tuple[date, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "first_day": str(self.first_day),
            "last_day": str(self.last_day),
            "traded_days": len(self.traded_days),
            "skipped_days": [str(day) for day in self.skipped_days],
        }


# --- loading -------------------------------------------------------------------


def list_run_ids(root: Path) -> list[str]:
    """Run folders that have a manifest, newest first."""
    if not root.exists():
        return []
    runs = [p.name for p in root.iterdir() if (p / "manifest.json").exists()]
    return sorted(runs, reverse=True)


def latest_run_id(root: Path) -> str:
    pointer = root / "latest.json"
    if pointer.exists():
        run_id = str(json.loads(pointer.read_text(encoding="utf-8"))["run_id"])
        if (root / run_id / "manifest.json").exists():
            return run_id
    runs = list_run_ids(root)
    if not runs:
        raise ArtifactsMissingError(
            "no dashboard run found; run python -m src.export.artifacts first"
        )
    return runs[0]


RUN_FILES = (
    "manifest.json",
    "forecasts.parquet",
    "pnl_grid.parquet",
    "hour_value.parquet",
    "attribution.parquet",
    "feature_importance.json",
)


def _signature(folder: Path) -> tuple[float, ...]:
    """Modification times of a run's files; a new export changes it."""
    return tuple(
        (folder / name).stat().st_mtime if (folder / name).exists() else -1.0
        for name in RUN_FILES
    )


def load_run(root: Path, run_id: str) -> Run:
    """A run's artifacts, re-read whenever an export has rewritten its files."""
    folder = root / run_id
    if not (folder / "manifest.json").exists():
        raise ArtifactsMissingError(f"run {run_id!r} not found")
    return _read_run(root, run_id, _signature(folder))


def clear_cache() -> None:
    _read_run.cache_clear()


@lru_cache(maxsize=4)
def _read_run(root: Path, run_id: str, signature: tuple[float, ...]) -> Run:
    """Read a run's files; a run without its grid still serves the other panels."""
    del signature  # part of the cache key only
    folder = root / run_id
    manifest_path = folder / "manifest.json"
    grid_path = folder / "pnl_grid.parquet"
    try:
        return _read_files(run_id, folder, manifest_path, grid_path)
    except (OSError, ValueError) as exc:
        raise ArtifactsMissingError(
            f"run {run_id!r} could not be read, probably while an export rewrites "
            "it; retry in a moment"
        ) from exc


def _read_files(run_id: str, folder: Path, manifest_path: Path, grid_path: Path) -> Run:
    return Run(
        run_id=run_id,
        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        forecasts=pd.read_parquet(folder / "forecasts.parquet"),
        grid=pd.read_parquet(grid_path) if grid_path.exists() else None,
        hour_value=pd.read_parquet(folder / "hour_value.parquet"),
        attribution=pd.read_parquet(folder / "attribution.parquet"),
        feature_importance=json.loads(
            (folder / "feature_importance.json").read_text(encoding="utf-8")
        ),
    )


def run_info(run: Run) -> dict[str, Any]:
    manifest = run.manifest
    keys = (
        "run_id",
        "created_utc",
        "source_commit",
        "model",
        "baseline_model",
        "first_day",
        "last_day",
        "holdout_start",
        "timezone",
        "reference_battery",
        "grid",
        "mode",
    )
    info = {key: manifest.get(key) for key in keys}
    # A manifest written before live runs has none of the three below: it is a
    # backtest, issued nowhere in particular, whose prices run to its last day.
    info["run_kind"] = manifest.get("run_kind") or "backtest"
    info["issued_utc"] = manifest.get("issued_utc")
    info["data_through"] = manifest.get("data_through") or manifest.get("last_day")
    info["traded_days"] = sum(1 for day in manifest["days"] if day["traded"])
    info["grid_available"] = run.grid is not None
    return info


# --- windows and strategies ------------------------------------------------------


def resolve_window(manifest: dict[str, Any], key: str) -> Window:
    """The days of a named window; ``last30`` and ``last90`` end on the last day."""
    if key not in WINDOWS:
        raise RequestError(
            f"unknown window {key!r}; choose one of {', '.join(WINDOWS)}"
        )
    days = [
        (date.fromisoformat(d["date"]), bool(d["traded"])) for d in manifest["days"]
    ]
    first_all, last_all = days[0][0], days[-1][0]
    holdout = date.fromisoformat(manifest["holdout_start"])
    if key in WINDOW_DAYS:
        first, last = last_all - timedelta(days=WINDOW_DAYS[key] - 1), last_all
    elif key == "validation":
        first, last = first_all, holdout - timedelta(days=1)
    elif key == "holdout":
        first, last = holdout, last_all
    else:
        first, last = first_all, last_all
    chosen = [(day, traded) for day, traded in days if first <= day <= last]
    traded = tuple(day for day, ok in chosen if ok)
    if not traded:
        raise RequestError(f"window {key!r} has no traded days in this run")
    return Window(
        key=key,
        first_day=max(first, first_all),
        last_day=min(last, last_all),
        traded_days=traded,
        skipped_days=tuple(day for day, ok in chosen if not ok),
    )


def _strategy_name(run: Run, key: str) -> str:
    names: dict[str, str] = run.manifest["grid"]["strategies"]
    if key not in names:
        raise RequestError(
            f"unknown strategy {key!r}; choose one of {', '.join(names)}"
        )
    return names[key]


def _strategy(name: str) -> Strategy:
    if name == MEDIAN_FORECAST.name:
        return MEDIAN_FORECAST
    if name.startswith("quantile_q"):
        return quantile_aware(int(name.removeprefix("quantile_q")) / 100)
    raise RequestError(f"unknown strategy {name!r}")


def _battery(run: Run, duration: int, degradation: float) -> Battery:
    grid = run.manifest["grid"]
    if duration not in grid["durations_h"]:
        raise RequestError(f"duration must be one of {grid['durations_h']} hours")
    if degradation not in grid["degradation_eur_per_mwh"]:
        wear = grid["degradation_eur_per_mwh"]
        raise RequestError(
            f"degradation must be a whole number of €/MWh from {min(wear)} to "
            f"{max(wear)}"
        )
    reference = run.manifest["reference_battery"]
    return Battery.model_validate(
        reference
        | {
            "power_mw": 1.0,
            "capacity_mwh": float(duration),
            "degradation_eur_per_mwh": float(degradation),
        }
    )


def _check_power(run: Run, power: float) -> None:
    limits = run.manifest["grid"]["power_mw"]
    steps = (power - limits["min"]) / limits["step"]
    if not limits["min"] <= power <= limits["max"] or abs(steps - round(steps)) > 1e-9:
        raise RequestError(
            f"power must be {limits['min']} to {limits['max']} MW in steps of "
            f"{limits['step']}"
        )


def _grid_cell(
    run: Run, window: Window, duration: int, degradation: int
) -> pd.DataFrame:
    if run.grid is None:
        raise ArtifactsMissingError(
            "the P&L grid has not been exported yet; run "
            "python -m src.export.artifacts --steps grid"
        )
    grid = run.grid
    cell = grid[
        (grid["duration_h"] == duration)
        & (grid["degradation_eur_per_mwh"] == degradation)
        & grid["target_day"].isin(window.traded_days)
    ]
    return cell


def _model_forecasts(run: Run, model: str, days: tuple[date, ...]) -> pd.DataFrame:
    frame = run.forecasts
    return frame[(frame["model"] == model) & frame["target_day"].isin(days)]


# --- endpoints -----------------------------------------------------------------------


def summary(
    run: Run,
    window_key: str,
    duration: int,
    degradation: int,
    strategy: str,
    power: float,
) -> dict[str, Any]:
    """The KPI strip for a window and battery."""
    window = resolve_window(run.manifest, window_key)
    _battery(run, duration, degradation)
    _check_power(run, power)
    name = _strategy_name(run, strategy)
    cell = _grid_cell(run, window, duration, degradation)
    mine = cell[cell["strategy"] == name]
    ceiling = float(
        cell.loc[cell["strategy"] == PERFECT_FORESIGHT.name, "pnl_eur"].sum()
    )
    pnl = float(mine["pnl_eur"].sum())
    model = _model_forecasts(run, run.manifest["model"], window.traded_days)
    baseline = _model_forecasts(run, run.manifest["baseline_model"], window.traded_days)
    return {
        "window": window.as_dict(),
        "battery": {
            "power_mw": power,
            "capacity_mwh": power * duration,
            "duration_h": duration,
            "degradation_eur_per_mwh": degradation,
            "strategy": strategy,
        },
        "kpis": {
            "pnl_eur": pnl * power,
            "perfect_foresight_pnl_eur": ceiling * power,
            "capture_ratio": pnl / ceiling if ceiling > 0 else None,
            "cycles_per_day": float(mine["cycles"].mean()) if len(mine) else None,
            "pinball_eur_mwh": _pinball(model, run.quantiles),
            "baseline_pinball_eur_mwh": _pinball(baseline, run.quantiles),
        },
    }


def _day_rows(run: Run, day: date) -> pd.DataFrame:
    rows: pd.DataFrame = run.forecasts[
        (run.forecasts["model"] == run.manifest["model"])
        & (run.forecasts["target_day"] == day)
    ].sort_index()
    if rows.empty:
        raise DayNotFoundError(f"no forecast for {day} in this run")
    return rows


def _pinball(forecasts: pd.DataFrame, quantiles: tuple[float, ...]) -> float | None:
    value = score(forecasts, quantiles).get("mean pinball")
    return None if value is None else _finite(value)


def _utc_stamps(index: pd.Index) -> list[str]:
    """Unique period keys; local labels repeat on the autumn clock change."""
    utc = pd.DatetimeIndex(index).tz_convert("UTC")
    return [ts.strftime("%Y-%m-%dT%H:%MZ") for ts in utc]


def _local_times(index: pd.Index, timezone: str) -> list[str]:
    local = pd.DatetimeIndex(index).tz_convert(timezone)
    return [ts.strftime("%H:%M") for ts in local]


def _finite(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def forecast_day(run: Run, day: date) -> dict[str, Any]:
    """The quantile fan for one delivery day."""
    rows = _day_rows(run, day)
    manifest = run.manifest
    issue_day = day - timedelta(days=1)
    columns = [quantile_column(q) for q in run.quantiles]
    periods = []
    for time_label, utc, (_, row) in zip(
        _local_times(rows.index, manifest["timezone"]),
        _utc_stamps(rows.index),
        rows.iterrows(),
        strict=True,
    ):
        period: dict[str, Any] = {"time": time_label, "utc": utc}
        period |= {column: _finite(row[column]) for column in columns}
        period["actual"] = _finite(row["actual"])
        periods.append(period)
    return {
        "date": str(day),
        "window": "holdout"
        if day >= date.fromisoformat(manifest["holdout_start"])
        else "validation",
        "issued_local": f"{issue_day} {manifest['forecast_issue_local']}",
        "gate_local": f"{issue_day} {manifest['gate_closure_local']}",
        "product_minutes": int(rows["price_product_minutes"].iloc[0]),
        "periods": periods,
    }


def dispatch(
    run: Run, day: date, duration: int, degradation: int, strategy: str, power: float
) -> dict[str, Any]:
    """One day's schedule, solved now for the chosen battery and strategy."""
    battery = _battery(run, duration, degradation)
    _check_power(run, power)
    chosen = _strategy(_strategy_name(run, strategy))
    traded = {d["date"]: d["traded"] for d in run.manifest["days"]}
    if not traded.get(str(day), False):
        raise RequestError(
            f"{day} was not traded: its prices or forecast are incomplete"
        )
    rows = _day_rows(run, day)
    started = time.perf_counter()
    try:
        result = dispatch_day(rows, chosen, battery, time_limit_s=SOLVE_TIME_LIMIT_S)
        ceiling = dispatch_day(
            rows, PERFECT_FORESIGHT, battery, time_limit_s=SOLVE_TIME_LIMIT_S
        )
    except DispatchError as exc:
        raise SolverError(f"the schedule for {day} could not be solved: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000
    schedule = result.schedule
    periods = [
        {
            "time": time_label,
            "utc": utc,
            "charge_mw": float(row["charge_mw"]) * power,
            "discharge_mw": float(row["discharge_mw"]) * power,
            "net_mw": float(row["net_mw"]) * power,
            "soc_mwh": float(row["soc_mwh"]) * power,
            "price": float(row["realised_price"]),
        }
        for time_label, utc, (_, row) in zip(
            _local_times(schedule.index, run.manifest["timezone"]),
            _utc_stamps(schedule.index),
            schedule.iterrows(),
            strict=True,
        )
    ]
    return {
        "date": str(day),
        "strategy": strategy,
        "solved_on_request": True,
        "solve_ms": round(elapsed_ms),
        "battery": {
            "power_mw": power,
            "capacity_mwh": power * duration,
            "degradation_eur_per_mwh": degradation,
        },
        "pnl_eur": result.settlement.pnl_eur * power,
        "perfect_foresight_pnl_eur": ceiling.settlement.pnl_eur * power,
        "periods": periods,
    }


def pnl_series(
    run: Run,
    window_key: str,
    duration: int,
    degradation: int,
    strategy: str,
    power: float,
) -> dict[str, Any]:
    """Cumulative P&L per day: perfect foresight, median and the selected strategy."""
    window = resolve_window(run.manifest, window_key)
    _battery(run, duration, degradation)
    _check_power(run, power)
    name = _strategy_name(run, strategy)
    cell = _grid_cell(run, window, duration, degradation)
    wide = cell.pivot(
        index="target_day", columns="strategy", values="pnl_eur"
    ).sort_index()
    columns = {
        "perfect_foresight": PERFECT_FORESIGHT.name,
        "median": MEDIAN_FORECAST.name,
        "selected": name,
    }
    daily = {key: wide[column] * power for key, column in columns.items()}
    cumulative = {key: series.cumsum() for key, series in daily.items()}
    return {
        "window": window.as_dict(),
        "selected_strategy": strategy,
        "series": [
            {"date": str(day)}
            | {key: float(cumulative[key].loc[day]) for key in columns}
            for day in wide.index
        ],
        "max_drawdown_eur": {
            key: max_drawdown(series) for key, series in daily.items()
        },
    }


def calibration(run: Run, window_key: str) -> dict[str, Any]:
    """Empirical share at or below each quantile, and central interval coverage."""
    window = resolve_window(run.manifest, window_key)
    rows = _model_forecasts(run, run.manifest["model"], window.traded_days)
    rows = rows[rows["actual"].notna()]
    actual = rows["actual"]
    return {
        "window": window.as_dict(),
        "quantiles": [
            {
                "level": level,
                "empirical": float((actual <= rows[quantile_column(level)]).mean()),
            }
            for level in run.quantiles
        ],
        "intervals": [
            {
                "nominal": nominal,
                "coverage": float(
                    (
                        (actual >= rows[quantile_column(low)])
                        & (actual <= rows[quantile_column(high)])
                    ).mean()
                ),
            }
            for nominal, low, high in INTERVALS
            if {low, high} <= set(run.quantiles)
        ],
    }


def error_analysis(run: Run, window_key: str) -> dict[str, Any]:
    """MAE and value at stake per local hour, and the attribution by block."""
    window = resolve_window(run.manifest, window_key)
    manifest = run.manifest
    rows = _model_forecasts(run, manifest["model"], window.traded_days)
    rows = rows[rows["actual"].notna()]
    hours = pd.DatetimeIndex(rows.index).tz_convert(manifest["timezone"]).hour
    mae = (rows["q50"] - rows["actual"]).abs().groupby(hours).mean()
    value = run.hour_value[run.hour_value["target_day"].isin(window.traded_days)]
    at_stake = value.groupby("hour")["cash_eur"].sum() / len(window.traded_days)
    costs = run.attribution[run.attribution["target_day"].isin(window.traded_days)]
    blocks = costs.pivot_table(
        index="block",
        columns="direction",
        values="cost_eur",
        aggfunc="sum",
        fill_value=0.0,
    )
    reference = manifest["reference_battery"]
    label = (
        f"{reference['power_mw']:g} MW / {reference['capacity_mwh']:g} MWh, "
        f"€{reference['degradation_eur_per_mwh']:g} wear, median dispatch"
    )
    return {
        "window": window.as_dict(),
        "by_hour": [
            {
                "hour": int(hour),
                "mae_eur_mwh": float(mae.get(hour, np.nan)),
                "value_at_stake_eur_per_day": float(at_stake.get(hour, 0.0)),
            }
            for hour in range(24)
            if hour in mae.index
        ],
        "asymmetry": {
            "reference": label,
            "gap_eur": float(costs["cost_eur"].sum()),
            "days": int(costs["target_day"].nunique()),
            "blocks": [
                {
                    "block": str(block),
                    "over_eur": float(blocks.loc[block].get("over", 0.0)),
                    "under_eur": float(blocks.loc[block].get("under", 0.0)),
                }
                for block in sorted(blocks.index)
            ],
        },
    }


def parse_day(text: str) -> date:
    if DATE_PATTERN.fullmatch(text):
        try:
            return date.fromisoformat(text)
        except ValueError:
            pass
    raise RequestError(f"date must look like 2025-11-21, not {text!r}")
