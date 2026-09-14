"""Dashboard artifacts: labels, manifest, grid checkpoints and aggregations."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.export import artifacts as art
from src.trading.battery import Battery
from src.trading.optimizer import optimize_dispatch
from src.trading.run_strategies import run_strategies
from src.trading.strategies import MEDIAN_FORECAST, dispatch_day, quantile_aware
from tests.test_trading import day_frame, prices


def _local(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    return settings.model_copy(update={"data": data})


def test_feature_labels_are_readable() -> None:
    assert art.feature_label("price_lag_1d") == "Price, same quarter-hour yesterday"
    assert (
        art.feature_label("wind_onshore_forecast_mw_lag_1d")
        == "Grid operator onshore wind forecast, yesterday"
    )
    assert art.feature_label("new_signal_mw_1d") == "New signal MW yesterday"
    assert art.feature_label("x") == "X"


def test_dashboard_strategies_follow_the_config(settings: Settings) -> None:
    dial = art.dashboard_strategies(settings)
    assert list(dial) == ["median", "q25", "q10"]
    assert (dial["q25"].sell_column, dial["q25"].buy_column) == ("q25", "q75")


def test_power_scales_the_optimal_value_exactly() -> None:
    """With duration fixed, a price taker's problem is homogeneous in power."""
    day = prices(list(np.random.default_rng(5).normal(60, 70, 32).round(2)))
    base = Battery(
        power_mw=1,
        capacity_mwh=2,
        round_trip_efficiency=0.9,
        degradation_eur_per_mwh=8,
        max_cycles_per_day=2,
    )
    bigger = Battery.model_validate(
        base.model_dump() | {"power_mw": 2.5, "capacity_mwh": 5.0}
    )
    small = optimize_dispatch(day, base)
    large = optimize_dispatch(day, bigger)
    assert large.objective_eur == pytest.approx(2.5 * small.objective_eur, abs=1e-5)


def test_manifest_lists_days_windows_and_the_grid(settings: Settings) -> None:
    holdout = settings.evaluation.holdout_start
    traded = [holdout - timedelta(days=1), holdout + timedelta(days=1)]
    manifest = art.build_manifest(
        settings, "backtest-x", traded, {holdout: "missing price or forecast"}
    )
    assert [d["window"] for d in manifest["days"]] == [
        "validation",
        "holdout",
        "holdout",
    ]
    assert [d["traded"] for d in manifest["days"]] == [True, False, True]
    assert manifest["grid"]["strategies"] == {
        "median": "median_forecast",
        "q25": "quantile_q25",
        "q10": "quantile_q10",
    }
    assert manifest["first_day"] == str(traded[0])
    assert manifest["last_day"] == str(traded[-1])


def test_pnl_grid_saves_and_reuses_cells(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    days = [date(2025, 11, 20), date(2025, 11, 21)]
    forecasts = pd.concat(
        [day_frame(settings, day, seed=i) for i, day in enumerate(days)]
    )
    grid = art.pnl_grid(
        forecasts,
        days,
        settings,
        durations=(1, 2),
        degradations=(8,),
        checkpoint_dir=tmp_path,
    )
    assert list(grid.columns) == list(art.GRID_COLUMNS)
    assert len(grid) == 2 * 2 * 4
    ceiling = grid[grid["strategy"] == "perfect_foresight"]
    by_duration = ceiling.groupby("duration_h")["pnl_eur"].sum()
    assert by_duration[2] >= by_duration[1] - 1e-6
    assert not list(tmp_path.glob(".*.partial"))

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("saved cells must not be solved again")

    monkeypatch.setattr(art, "run_strategies", forbidden)
    again = art.pnl_grid(
        forecasts,
        days,
        settings,
        durations=(1, 2),
        degradations=(8,),
        checkpoint_dir=tmp_path,
    )
    pd.testing.assert_frame_equal(
        again.reset_index(drop=True), grid.reset_index(drop=True)
    )

    shifted = forecasts["actual"].to_numpy(copy=True)
    shifted[0] += 50.0
    changed = forecasts.assign(actual=shifted)
    with pytest.raises(AssertionError, match="must not be solved again"):
        art.pnl_grid(
            changed,
            days,
            settings,
            durations=(1,),
            degradations=(8,),
            checkpoint_dir=tmp_path,
        )
    monkeypatch.setattr(art, "run_strategies", run_strategies)
    art.pnl_grid(
        changed,
        days,
        settings,
        durations=(1,),
        degradations=(8,),
        checkpoint_dir=tmp_path,
    )
    assert [p.name for p in tmp_path.glob("*.parquet")] == ["duration1_wear8.parquet"]


def _two_days(settings: Settings) -> tuple[list[date], pd.DataFrame]:
    days = [date(2025, 11, 20), date(2025, 11, 21)]
    frame = pd.concat([day_frame(settings, day, seed=i) for i, day in enumerate(days)])
    return days, frame


def test_pnl_grid_retries_failed_days_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    days, forecasts = _two_days(settings)
    calls: list[list[date]] = []

    def flaky(
        frame: pd.DataFrame, chosen: list[date], *args: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[date, str]]:
        calls.append(list(chosen))
        dispatch, pnl, failed = run_strategies(frame, chosen, *args, **kwargs)
        if len(calls) == 1:
            pnl = pnl[pnl["target_day"] != chosen[0]]
            failed = {chosen[0]: "solver failed: solver status Not Solved"}
        return dispatch, pnl, failed

    monkeypatch.setattr(art, "run_strategies", flaky)
    grid = art.pnl_grid(forecasts, days, settings, durations=(1,), degradations=(8,))
    assert calls == [days, [days[0]]]
    assert len(grid) == 2 * 4 and set(grid["target_day"]) == set(days)


def test_pnl_grid_raises_when_the_retry_fails_too(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    days, forecasts = _two_days(settings)

    def broken(
        frame: pd.DataFrame, chosen: list[date], *args: Any, **kwargs: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[date, str]]:
        dispatch, pnl, _ = run_strategies(frame, chosen, *args, **kwargs)
        return dispatch, pnl.iloc[0:0], {day: "solver failed" for day in chosen}

    monkeypatch.setattr(art, "run_strategies", broken)
    with pytest.raises(RuntimeError, match="solver failed for duration 1 h"):
        art.pnl_grid(forecasts, days, settings, durations=(1,), degradations=(8,))


def test_real_schedules_scale_with_power(settings: Settings) -> None:
    """Settled P&L, schedule and cycles, not only the objective, scale with power."""
    base = settings.battery.model_copy(update={"power_mw": 1.0, "capacity_mwh": 2.0})
    bigger = base.model_copy(update={"power_mw": 2.5, "capacity_mwh": 5.0})
    cases = [(date(2025, 11, 20), 15), (date(2025, 9, 20), 60)]
    for day, minutes in cases:
        frame = day_frame(settings, day, seed=3, product_minutes=minutes)
        for strategy in (MEDIAN_FORECAST, quantile_aware(0.25)):
            small = dispatch_day(frame, strategy, base)
            large = dispatch_day(frame, strategy, bigger)
            assert large.settlement.pnl_eur == pytest.approx(
                2.5 * small.settlement.pnl_eur, abs=1e-4
            )
            assert large.settlement.cycles == pytest.approx(
                small.settlement.cycles, abs=1e-6
            )
            np.testing.assert_allclose(
                large.schedule["net_mw"], 2.5 * small.schedule["net_mw"], atol=1e-5
            )


def test_combined_forecasts_and_traded_days(settings: Settings, tmp_path: Path) -> None:
    local = _local(settings, tmp_path)
    start, holdout = (
        settings.evaluation.validation_start,
        settings.evaluation.holdout_start,
    )
    folder = local.data.processed_path / "forecasts"
    (folder / "comparison").mkdir(parents=True)
    (folder / "holdout").mkdir(parents=True)
    before, first, last_validation = (
        start - timedelta(days=1),
        start,
        holdout - timedelta(days=1),
    )
    comparison = pd.concat(
        day_frame(settings, day) for day in (before, first, last_validation, holdout)
    ).assign(model="m")
    comparison.to_parquet(folder / "comparison" / "m.parquet")
    late = day_frame(settings, holdout + timedelta(days=1))
    gap = late["actual"].to_numpy(copy=True)
    gap[5] = np.nan
    late = late.assign(actual=gap)
    pd.concat([day_frame(settings, holdout), late]).assign(model="m").to_parquet(
        folder / "holdout" / "m.parquet"
    )
    frame = art.combined_forecasts(local, "m")
    windows = frame.groupby("target_day")["window"].agg(set)
    assert windows.to_dict() == {
        first: {"validation"},
        last_validation: {"validation"},
        holdout: {"holdout"},
        holdout + timedelta(days=1): {"holdout"},
    }
    days, skipped = art.traded_days(frame, local)
    assert days == [first, last_validation, holdout]
    assert holdout + timedelta(days=1) in skipped


def test_hour_value_sums_perfect_foresight_cash_by_local_hour(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    folder = local.data.processed_path / "backtest" / "validation"
    folder.mkdir(parents=True)
    index = pd.date_range(
        "2025-11-19 23:00", periods=8, freq="15min", tz="UTC", name="timestamp_utc"
    )
    ceiling = pd.DataFrame(
        {
            "model": "m",
            "target_day": date(2025, 11, 20),
            "strategy": "perfect_foresight",
            "net_mw": [1.0, -1.0, 0.0, 0.0, 0.5, 0.5, 0.0, 0.0],
            "realised_price": [100.0] * 8,
        },
        index=index,
    )
    pd.concat([ceiling, ceiling.assign(strategy="median_forecast")]).to_parquet(
        folder / "dispatch.parquet"
    )
    table = art.hour_value_table(local, "m")
    assert table["hour"].tolist() == [0, 1]
    assert table["cash_eur"].tolist() == [50.0, 25.0]


def test_attribution_table_keeps_median_dispatch_from_both_windows(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    backtest = local.data.processed_path / "backtest"
    (backtest / "attribution").mkdir(parents=True)
    (backtest / "holdout").mkdir(parents=True)
    row = {
        "target_day": date(2025, 11, 20),
        "block": "18-20",
        "direction": "under",
        "periods": 4,
        "mean_abs_error_eur_mwh": 5.0,
        "cost_eur": 10.0,
    }
    pd.DataFrame(
        [row | {"strategy": "median_forecast"}, row | {"strategy": "quantile_q25"}]
    ).to_parquet(backtest / "attribution" / "costs.parquet")
    pd.DataFrame(
        [row | {"strategy": "median_forecast", "target_day": date(2026, 6, 2)}]
    ).to_parquet(backtest / "holdout" / "attribution_costs.parquet")
    table = art.attribution_table(local)
    assert len(table) == 2
    assert list(table.columns) == [
        "target_day",
        "block",
        "direction",
        "periods",
        "cost_eur",
    ]
