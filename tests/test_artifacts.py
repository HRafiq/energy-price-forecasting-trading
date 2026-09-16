"""Dashboard artifacts: labels, manifest, grid checkpoints and aggregations."""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.export import artifacts as art
from src.health.incidents import (
    Incident,
    default_path,
    make_incident_id,
    upsert_incidents,
)
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
    # A run is a backtest unless the live pipeline says otherwise, and a backtest's
    # prices run to its last day.
    assert manifest["run_kind"] == "backtest"
    assert manifest["issued_utc"] is None
    assert manifest["data_through"] == str(traded[-1])


def test_manifest_records_a_live_run_and_the_prices_it_has(settings: Settings) -> None:
    holdout = settings.evaluation.holdout_start
    days = [holdout + timedelta(days=1), holdout + timedelta(days=2)]
    manifest = art.build_manifest(
        settings,
        "live-x",
        days,
        {},
        run_kind="live",
        issued_utc="2026-09-16T09:40:00+00:00",
        data_through=days[0],
    )
    assert manifest["run_kind"] == "live"
    assert manifest["issued_utc"] == "2026-09-16T09:40:00+00:00"
    # The last day it forecasts is a day ahead of the last published price.
    assert manifest["data_through"] == str(days[0])
    assert manifest["last_day"] == str(days[-1])


def test_a_live_run_is_named_after_the_day_it_ran(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A live export is named for the day it ran, not for the backtest it reads.

    The first real export took its name from the last backtest forecast, two days
    earlier, and pointed the dashboard at it. Only the command decides the name,
    so only the command can be tested for it.
    """
    local = _local(settings, tmp_path)
    monkeypatch.setattr(art, "load_settings", lambda config=None: local)
    days = [date(2025, 11, 20), date(2025, 11, 21)]
    saved = pd.concat(
        day_frame(local, day, seed=i) for i, day in enumerate(days)
    ).assign(model=local.forecasting.production_model, window="validation")
    comparison = local.data.processed_path / "forecasts" / "comparison"
    comparison.mkdir(parents=True)
    saved.to_parquet(comparison / f"{local.forecasting.production_model}.parquet")

    art.main(
        [
            "--steps",
            "health",
            "--run-kind",
            "live",
            "--issued-utc",
            "2026-09-16T22:40:00+00:00",
        ]
    )
    art.main(
        [
            "--steps",
            "health",
            "--run-kind",
            "live",
            "--issued-utc",
            "2026-09-17T05:00:00",
        ]
    )
    art.main(["--steps", "health"])

    runs = sorted(
        p.name
        for p in (local.data.processed_path / "dashboard").iterdir()
        if p.is_dir()
    )
    # 22:40 UTC is the next day in Berlin; the naive value is read as UTC.
    assert runs == ["backtest-2025-11-21", "live-2026-09-17"]


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


def _health_sources(local: Settings) -> Path:
    processed = local.data.processed_path
    (processed / "experiments").mkdir(parents=True)
    (processed / "experiments" / "m1_regime_experiment.json").write_text(
        '{"experiment": "m1_regime_shift"}', encoding="utf-8"
    )
    day = date(2025, 11, 20)
    incident = Incident(
        incident_id=make_incident_id("observed", "tail_miss", day),
        delivery_day=day,
        detected_utc=datetime(2025, 11, 20, 23, tzinfo=UTC),
        type="tail_miss",
        severity="warning",
        detail="Realised price left the 90% range.",
        action="Logged for forecast review.",
        status="review",
        source="observed",
    )
    upsert_incidents([incident], default_path(local))
    run_dir = processed / "dashboard" / "backtest-x"
    run_dir.mkdir(parents=True)
    return run_dir


def test_health_step_copies_present_files_and_skips_missing(
    settings: Settings, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    local = _local(settings, tmp_path)
    run_dir = _health_sources(local)
    index = art.export_health(local, run_dir)
    folder = run_dir / "health"
    source = local.data.processed_path / "experiments" / "m1_regime_experiment.json"
    assert (folder / "m1_regime_experiment.json").read_bytes() == source.read_bytes()
    assert not (folder / "m2_drift.json").exists()
    assert not (folder / "d5_deadline.json").exists()
    out = capsys.readouterr().out
    assert "m2_drift.json not found, skipped" in out
    assert "d5_deadline.json not found, skipped" in out

    records = json.loads((folder / "incidents.json").read_text(encoding="utf-8"))
    assert isinstance(records, list) and len(records) == 1
    assert records[0]["type"] == "tail_miss"
    listed = {entry["file"]: entry for entry in index["files"]}
    assert set(listed) == {"m1_regime_experiment.json", "incidents.json"}
    assert listed["incidents.json"]["records"] == 1
    assert listed["m1_regime_experiment.json"]["source"] == (
        "experiments/m1_regime_experiment.json"
    )
    assert json.loads((folder / "index.json").read_text(encoding="utf-8")) == index
    assert not list(folder.glob(".*.partial"))


def test_health_step_removes_stale_copies_and_swaps_files_whole(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _local(settings, tmp_path)
    run_dir = _health_sources(local)
    art.export_health(local, run_dir)
    source = local.data.processed_path / "experiments" / "m1_regime_experiment.json"
    copy = run_dir / "health" / "m1_regime_experiment.json"
    before = copy.read_bytes()

    source.write_text('{"experiment": "m1_regime_shift", "rerun": true}')

    def broken(src: Path, dst: Path) -> None:
        Path(dst).write_text('{"experiment": "m1_re', encoding="utf-8")
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copyfile", broken)
    with pytest.raises(OSError, match="disk full"):
        art.export_health(local, run_dir)
    assert copy.read_bytes() == before
    monkeypatch.undo()
    # The failed copy leaves no partial file behind.
    assert not list((run_dir / "health").glob(".*.partial"))

    # A source that disappears takes its copy with it: no silent stale data.
    source.unlink()
    index = art.export_health(local, run_dir)
    assert copy.name not in {e["file"] for e in index["files"]}
    assert not copy.exists()
    default_path(local).unlink()
    index = art.export_health(local, run_dir)
    assert index["files"] == []
    assert not (run_dir / "health" / "incidents.json").exists()


def test_health_index_records_times_and_flags_a_drift_from_another_run(
    settings: Settings, tmp_path: Path
) -> None:
    local = _local(settings, tmp_path)
    run_dir = _health_sources(local)
    experiments = local.data.processed_path / "experiments"
    series = [{"target_day": "2026-09-13"}, {"target_day": "2026-09-14"}]
    (experiments / "m2_drift.json").write_text(
        json.dumps({"series": series}), encoding="utf-8"
    )
    (experiments / "d5_deadline.json").write_text(
        json.dumps({"generated_utc": "2026-09-01T08:00:00+00:00"}), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps({"last_day": "2026-09-14"}), encoding="utf-8"
    )
    listed = {e["file"]: e for e in art.export_health(local, run_dir)["files"]}
    for entry in listed.values():
        assert entry["generated_utc"] and entry["source_modified_utc"]
        assert entry["exported_utc"]
    deadline = listed["d5_deadline.json"]
    assert deadline["generated_utc"] == "2026-09-01T08:00:00+00:00"
    assert deadline["source_modified_utc"] != deadline["generated_utc"]
    drift = listed["m2_drift.json"]
    assert drift["matches_run"] is True and drift["last_target_day"] == "2026-09-14"

    (run_dir / "manifest.json").write_text(
        json.dumps({"last_day": "2026-10-31"}), encoding="utf-8"
    )
    listed = {e["file"]: e for e in art.export_health(local, run_dir)["files"]}
    drift = listed["m2_drift.json"]
    assert drift["matches_run"] is False and drift["run_last_day"] == "2026-10-31"


def test_a_failed_write_leaves_no_partial_file(tmp_path: Path) -> None:
    path = tmp_path / "out" / "file.json"

    def half(target: Path) -> None:
        target.write_text("{", encoding="utf-8")
        raise OSError("disk full")

    with pytest.raises(OSError, match="disk full"):
        art._replace_atomically(path, half)
    assert not path.exists()
    assert not list(path.parent.glob(".*.partial"))
    art.write_json({"ok": True}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert not list(path.parent.glob(".*.partial"))


def test_a_live_export_carries_the_days_own_run_record(
    settings: Settings, tmp_path: Path
) -> None:
    """Without this, a live run shows only backtest artifacts under a live label."""
    local = _local(settings, tmp_path)
    day = date(2026, 9, 17)
    runs = local.data.processed_path / "pipeline" / "runs"
    runs.mkdir(parents=True)
    record = {
        "target_day": str(day),
        "issued_utc": "2026-09-16T09:40:00+00:00",
        "gate_utc": "2026-09-16T10:00:00+00:00",
        "on_time": True,
        "minutes_before_gate": 20.0,
        "step": "seasonal_naive",
        "model": "seasonal_naive_previous_week",
        "model_version": None,
        "readiness": {"ready": False, "missing": ["weather"], "feeds": []},
        "attempts": [{"step": "production", "used": False, "detail": "skipped"}],
        "planned_value_eur": 783.13,
        "solve_seconds": 0.08,
        "incidents": ["abc123"],
    }
    (runs / f"{day}.json").write_text(json.dumps(record), encoding="utf-8")
    run_dir = local.data.processed_path / "dashboard" / "live-2026-09-16"

    index = art.export_health(local, run_dir, live_day=day)

    summary = json.loads((run_dir / "health" / art.HEALTH_LIVE_DAY).read_text())
    assert summary["target_day"] == str(day) and summary["step"] == "seasonal_naive"
    assert summary["planned_value_eur"] == 783.13 and summary["on_time"] is True
    assert art.HEALTH_LIVE_DAY in [f["file"] for f in index["files"]]

    # A backtest export has no such day, and leaves no stale copy behind.
    again = art.export_health(local, run_dir, live_day=None)
    assert not (run_dir / "health" / art.HEALTH_LIVE_DAY).exists()
    assert art.HEALTH_LIVE_DAY not in [f["file"] for f in again["files"]]
