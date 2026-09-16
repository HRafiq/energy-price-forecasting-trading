"""The committed plan: optimized against the forecast, settled the next day.

A live day cannot be settled when it is traded. At 11:40 the schedule is chosen
from the forecast and committed before the 12:00 gate; the prices it will be paid
appear only when the auction clears. So the pipeline writes a plan for delivery day
D+1, and settles it once day D+1's prices are published:

    uv run python -m src.pipeline.plan --day 2026-09-17            # plan
    uv run python -m src.pipeline.plan --day 2026-09-17 --settle   # settle it

Plans live in ``data/processed/pipeline/plans/<day>.parquet`` with the schedule and
the metadata a later settlement needs: the chain step and model that produced the
forecast, and when it was issued. Settlement uses the same function as the
backtest, so a live day and a backtest day are valued identically.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd

from src.config import (
    PRICE_SERIES,
    PRODUCT_COLUMN,
    RESOLUTION_STEP,
    Settings,
    load_settings,
)
from src.trading.battery import Battery
from src.trading.optimizer import optimize_dispatch, product_blocks
from src.trading.settlement import Settlement, settle
from src.trading.strategies import MEDIAN_FORECAST, Strategy

__all__ = [
    "META_COLUMNS",
    "PLAN_COLUMNS",
    "Plan",
    "load_plan",
    "main",
    "plan_day",
    "plan_path",
    "save_plan",
    "settle_plan",
]

PLAN_COLUMNS = (
    "charge_mw",
    "discharge_mw",
    "net_mw",
    "soc_mwh",
    "sell_price",
    "buy_price",
)
META_COLUMNS = (
    "target_day",
    "strategy",
    "model",
    "step",
    "issued_utc",
    "product_minutes",
    "planned_value_eur",
    "solve_seconds",
)


@dataclass(frozen=True)
class Plan:
    """One delivery day's committed schedule, before its prices are known."""

    target_day: date
    strategy: str
    model: str
    step: str
    issued_utc: datetime
    product_minutes: int
    schedule: pd.DataFrame
    planned_value_eur: float
    solve_seconds: float

    def as_frame(self) -> pd.DataFrame:
        """The schedule with the metadata repeated per period, ready for parquet."""
        frame = self.schedule[list(PLAN_COLUMNS)].copy()
        frame["target_day"] = self.target_day
        frame["strategy"] = self.strategy
        frame["model"] = self.model
        frame["step"] = self.step
        frame["issued_utc"] = self.issued_utc
        frame["product_minutes"] = self.product_minutes
        frame["planned_value_eur"] = self.planned_value_eur
        frame["solve_seconds"] = self.solve_seconds
        frame.index.name = "timestamp_utc"
        return frame

    @classmethod
    def from_frame(cls, frame: pd.DataFrame) -> Plan:
        missing = (set(PLAN_COLUMNS) | set(META_COLUMNS)) - set(frame.columns)
        if missing:
            raise ValueError(f"plan frame lacks columns {sorted(missing)}")
        first = frame.iloc[0]
        issued = pd.Timestamp(first["issued_utc"])
        return cls(
            target_day=first["target_day"],
            strategy=str(first["strategy"]),
            model=str(first["model"]),
            step=str(first["step"]),
            issued_utc=issued.to_pydatetime(),
            product_minutes=int(first["product_minutes"]),
            schedule=frame[list(PLAN_COLUMNS)].copy(),
            planned_value_eur=float(first["planned_value_eur"]),
            solve_seconds=float(first["solve_seconds"]),
        )


def plan_day(
    values: pd.DataFrame,
    products: pd.Series,
    settings: Settings,
    *,
    target_day: date,
    model: str,
    step: str,
    battery: Battery | None = None,
    strategy: Strategy = MEDIAN_FORECAST,
    issued_utc: datetime | None = None,
    time_limit_s: float | None = None,
) -> Plan:
    """Optimize one delivery day against the forecast and commit the schedule."""
    if not values.index.equals(products.index):
        raise ValueError("products must have the same index as the forecast")
    if products.nunique() != 1:
        raise ValueError("one delivery day cannot mix product lengths")
    sell, buy = values[strategy.sell_column], values[strategy.buy_column]
    result = optimize_dispatch(
        sell,
        battery or settings.battery,
        buy_prices=buy,
        products=product_blocks(pd.DatetimeIndex(values.index), products),
        time_limit_s=time_limit_s or settings.trading.solver_time_limit_s,
    )
    schedule = result.schedule.assign(
        sell_price=sell.to_numpy(dtype=float),
        buy_price=buy.to_numpy(dtype=float),
    )
    return Plan(
        target_day=target_day,
        strategy=strategy.name,
        model=model,
        step=step,
        issued_utc=issued_utc or datetime.now(UTC),
        product_minutes=int(products.iloc[0]),
        schedule=schedule,
        planned_value_eur=result.objective_eur,
        solve_seconds=result.solve_seconds,
    )


def plan_path(settings: Settings, day: date) -> Path:
    return settings.data.processed_path / "pipeline" / "plans" / f"{day}.parquet"


def save_plan(plan: Plan, settings: Settings) -> Path:
    """Write the plan whole, so a reader never sees half a schedule."""
    path = plan_path(settings, plan.target_day)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    plan.as_frame().to_parquet(partial)
    os.replace(partial, path)
    return path


def load_plan(settings: Settings, day: date) -> Plan:
    path = plan_path(settings, day)
    if not path.exists():
        raise FileNotFoundError(f"no plan for {day} at {path}")
    return Plan.from_frame(pd.read_parquet(path))


def settle_plan(plan: Plan, prices: pd.Series, battery: Battery) -> Settlement:
    """Value a committed plan at the prices the auction actually cleared."""
    aligned = prices.reindex(plan.schedule.index)
    if aligned.isna().any():
        missing = int(aligned.isna().sum())
        raise ValueError(
            f"{plan.target_day}: {missing} of {len(aligned)} periods have no price yet"
        )
    return settle(plan.schedule, aligned, battery)


def _forecast_for(settings: Settings, day: date) -> tuple[pd.DataFrame, str]:
    path = settings.data.processed_path / "forecasts" / "production" / f"{day}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"no forecast for {day} at {path}")
    frame = pd.read_parquet(path)
    model = str(frame["model"].iloc[0]) if "model" in frame.columns else "unknown"
    quantiles = [c for c in frame.columns if c[:1] == "q" and c[1:].isdigit()]
    return frame[quantiles], model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plan a delivery day, or settle a committed plan."
    )
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--settle", action="store_true", help="settle a saved plan")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    inputs = pd.read_parquet(settings.data.inputs_path)

    if args.settle:
        plan = load_plan(settings, args.day)
        prices = inputs[PRICE_SERIES]
        settlement = settle_plan(plan, prices, settings.battery)
        print(
            json.dumps(
                {
                    "target_day": str(args.day),
                    "step": plan.step,
                    "planned_value_eur": round(plan.planned_value_eur, 2),
                    "pnl_eur": round(settlement.pnl_eur, 2),
                    "revenue_eur": round(settlement.revenue_eur, 2),
                    "degradation_eur": round(settlement.degradation_eur, 2),
                    "cycles": round(settlement.cycles, 3),
                },
                indent=2,
            )
        )
        return 0

    values, model = _forecast_for(settings, args.day)
    products = inputs[PRODUCT_COLUMN].reindex(values.index)
    if products.isna().any():
        # A day beyond the dataset has no product flag yet; use the modeling step.
        step_minutes = int(
            RESOLUTION_STEP[settings.data.modeling_resolution] / pd.Timedelta(minutes=1)
        )
        products = products.fillna(step_minutes)
    plan = plan_day(
        values,
        products.astype("int64"),
        settings,
        target_day=args.day,
        model=model,
        step="production",
    )
    path = save_plan(plan, settings)
    print(
        f"{args.day}: planned value €{plan.planned_value_eur:,.2f} "
        f"in {plan.solve_seconds:.2f} s, written to {path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
