"""The committed plan: optimizing against a forecast, saving it, settling it later."""

from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.pipeline import plan as pl
from src.trading.strategies import MEDIAN_FORECAST

TARGET = date(2025, 11, 20)


def _local(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    return settings.model_copy(update={"data": data})


def _index(periods: int = 32) -> pd.DatetimeIndex:
    return pd.date_range(
        "2025-11-19 23:00",
        periods=periods,
        freq="15min",
        tz="UTC",
        name="timestamp_utc",
    )


def _values(curve: list[float]) -> pd.DataFrame:
    index = _index(len(curve))
    columns = {MEDIAN_FORECAST.sell_column: curve, MEDIAN_FORECAST.buy_column: curve}
    return pd.DataFrame(columns, index=index)


def _products(index: pd.DatetimeIndex, minutes: int = 15) -> pd.Series:
    return pd.Series(minutes, index=index, dtype="int64")


def test_the_plan_charges_when_cheap_and_discharges_when_dear(
    settings: Settings,
) -> None:
    curve = [10.0] * 16 + [200.0] * 16
    values = _values(curve)

    plan = pl.plan_day(
        values,
        _products(pd.DatetimeIndex(values.index)),
        settings,
        target_day=TARGET,
        model="lightgbm_conformal",
        step="production",
    )

    cheap, dear = slice(0, 16), slice(16, 32)
    charge = plan.schedule["charge_mw"].to_numpy()
    discharge = plan.schedule["discharge_mw"].to_numpy()
    assert charge[cheap].sum() > 0 and discharge[dear].sum() > 0
    assert charge[dear].sum() == pytest.approx(0.0, abs=1e-6)
    assert discharge[cheap].sum() == pytest.approx(0.0, abs=1e-6)
    assert plan.planned_value_eur > 0
    assert plan.product_minutes == 15
    assert (plan.strategy, plan.step) == (MEDIAN_FORECAST.name, "production")
    assert plan.issued_utc.utcoffset() is not None


def test_a_plan_survives_the_round_trip_to_parquet(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    values = _values([20.0] * 16 + [150.0] * 16)
    issued = datetime(2026, 9, 16, 9, 40, tzinfo=UTC)
    plan = pl.plan_day(
        values,
        _products(pd.DatetimeIndex(values.index)),
        local,
        target_day=TARGET,
        model="lightgbm_conformal",
        step="no_weather",
        issued_utc=issued,
    )

    path = pl.save_plan(plan, local)
    loaded = pl.load_plan(local, TARGET)

    assert path == pl.plan_path(local, TARGET) and path.exists()
    assert not list(path.parent.glob(".*.partial"))
    assert loaded.target_day == TARGET and loaded.step == "no_weather"
    assert loaded.model == "lightgbm_conformal"
    assert loaded.issued_utc == issued
    assert loaded.planned_value_eur == pytest.approx(plan.planned_value_eur)
    # Parquet does not carry the index frequency, which the schedule never uses.
    pd.testing.assert_frame_equal(
        loaded.schedule,
        plan.schedule[list(pl.PLAN_COLUMNS)],
        check_names=False,
        check_freq=False,
    )


def test_missing_plan_and_broken_frame_are_refused(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    with pytest.raises(FileNotFoundError, match="no plan for"):
        pl.load_plan(local, TARGET)
    with pytest.raises(ValueError, match="lacks columns"):
        pl.Plan.from_frame(pd.DataFrame({"charge_mw": [0.0]}))


def test_settlement_values_the_committed_schedule_at_realised_prices(
    settings: Settings,
) -> None:
    index = _index(4)
    schedule = pd.DataFrame(
        {
            "charge_mw": [1.0, 0.0, 0.0, 0.0],
            "discharge_mw": [0.0, 0.0, 1.0, 0.0],
            "net_mw": [-1.0, 0.0, 1.0, 0.0],
            "soc_mwh": [1.24, 1.24, 0.98, 0.98],
            "sell_price": [0.0, 0.0, 100.0, 0.0],
            "buy_price": [0.0, 0.0, 100.0, 0.0],
        },
        index=index,
    )
    plan = pl.Plan(
        target_day=TARGET,
        strategy=MEDIAN_FORECAST.name,
        model="m",
        step="production",
        issued_utc=datetime(2025, 11, 19, 10, 40, tzinfo=UTC),
        product_minutes=15,
        schedule=schedule,
        planned_value_eur=23.0,
        solve_seconds=0.1,
    )
    battery = settings.battery
    prices = pd.Series([0.0, 0.0, 100.0, 0.0], index=index)

    settlement = pl.settle_plan(plan, prices, battery)

    # A quarter-hour at 1 MW moves 0.25 MWh: sold once at 100, bought once at 0.
    assert settlement.revenue_eur == pytest.approx(25.0)
    assert settlement.discharged_mwh == pytest.approx(0.25)
    assert settlement.degradation_eur == pytest.approx(
        0.25 * battery.degradation_eur_per_mwh
    )
    assert settlement.pnl_eur == pytest.approx(
        25.0 - 0.25 * battery.degradation_eur_per_mwh
    )
    assert settlement.cycles == pytest.approx(
        0.25 / battery.discharge_efficiency / battery.capacity_mwh
    )


def test_settlement_waits_for_every_price(settings: Settings) -> None:
    index = _index(4)
    plan = pl.Plan(
        target_day=TARGET,
        strategy=MEDIAN_FORECAST.name,
        model="m",
        step="production",
        issued_utc=datetime(2025, 11, 19, 10, 40, tzinfo=UTC),
        product_minutes=15,
        schedule=pd.DataFrame(
            {
                "charge_mw": [0.0] * 4,
                "discharge_mw": [0.0] * 4,
                "net_mw": [0.0] * 4,
                "soc_mwh": [1.0] * 4,
                "sell_price": [0.0] * 4,
                "buy_price": [0.0] * 4,
            },
            index=index,
        ),
        planned_value_eur=0.0,
        solve_seconds=0.0,
    )
    partial = pd.Series([10.0, np.nan, 10.0, np.nan], index=index)

    with pytest.raises(ValueError, match="2 of 4 periods have no price yet"):
        pl.settle_plan(plan, partial, settings.battery)


def test_one_day_cannot_mix_products_or_lose_its_index(settings: Settings) -> None:
    values = _values([50.0] * 8)
    index = pd.DatetimeIndex(values.index)
    mixed = pd.Series([15] * 4 + [60] * 4, index=index, dtype="int64")

    with pytest.raises(ValueError, match="mix product lengths"):
        pl.plan_day(
            values, mixed, settings, target_day=TARGET, model="m", step="production"
        )
    with pytest.raises(ValueError, match="same index as the forecast"):
        pl.plan_day(
            values,
            _products(index[:4]),
            settings,
            target_day=TARGET,
            model="m",
            step="production",
        )
