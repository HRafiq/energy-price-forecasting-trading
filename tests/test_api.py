"""The dashboard API over a small synthetic run."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import service
from api.main import app
from src.config import Settings
from src.export.artifacts import build_manifest
from src.trading.optimizer import DispatchError
from tests.test_trading import day_frame

RUN = "backtest-test"
STRATEGY_PNL = {
    "perfect_foresight": 100.0,
    "median_forecast": 80.0,
    "quantile_q25": 70.0,
    "quantile_q10": 60.0,
}


def _days(settings: Settings) -> tuple[list[date], date]:
    holdout = settings.evaluation.holdout_start
    days = [
        holdout - timedelta(days=2),
        holdout - timedelta(days=1),
        holdout,
        holdout + timedelta(days=1),
    ]
    return days, holdout


def _daily_pnl(strategy: str, pnl: float, day_number: int) -> float:
    """Constant P&L, except a losing second day for the q10 strategy."""
    if strategy == "quantile_q10" and day_number == 1:
        return -30.0
    return pnl


def _write_run(settings: Settings, root: Path) -> None:
    days, skipped_day = _days(settings)
    traded = [day for day in days if day != skipped_day]
    folder = root / RUN
    folder.mkdir(parents=True)
    frames = []
    for i, day in enumerate(days):
        frame = day_frame(settings, day, seed=i)
        frames.append(frame.assign(model="lightgbm_conformal"))
        frames.append(frame.assign(model="naive_previous_day", q50=frame["q50"] + 40))
    pd.concat(frames).to_parquet(folder / "forecasts.parquet")
    manifest = build_manifest(
        settings, RUN, traded, {skipped_day: "missing price or forecast"}
    )
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    rows = [
        {
            "duration_h": duration,
            "degradation_eur_per_mwh": 8,
            "strategy": strategy,
            "target_day": day,
            "pnl_eur": _daily_pnl(strategy, pnl, i) * duration,
            "revenue_eur": _daily_pnl(strategy, pnl, i) * duration + 8.0,
            "degradation_eur": 8.0,
            "discharged_mwh": 1.0,
            "cycles": 0.5 * duration,
        }
        for duration in (2, 4)
        for i, day in enumerate(traded)
        for strategy, pnl in STRATEGY_PNL.items()
    ]
    pd.DataFrame(rows).to_parquet(folder / "pnl_grid.parquet")
    pd.DataFrame(
        [
            {"target_day": day, "hour": hour, "cash_eur": float(hour)}
            for day in traded
            for hour in range(24)
        ]
    ).to_parquet(folder / "hour_value.parquet")
    pd.DataFrame(
        [
            {
                "target_day": day,
                "block": "18-20",
                "direction": direction,
                "periods": 4,
                "cost_eur": cost,
            }
            for day in traded
            for direction, cost in (("under", 6.0), ("over", 2.0))
        ]
    ).to_parquet(folder / "attribution.parquet")
    (folder / "feature_importance.json").write_text(
        json.dumps(
            {
                "model": "lightgbm_conformal",
                "trained_for_day": str(traded[-1]),
                "importance": "gain",
                "features": [
                    {"feature": "price_lag_1d", "label": "Price", "gain_share": 1.0}
                ],
            }
        ),
        encoding="utf-8",
    )
    (root / "latest.json").write_text(json.dumps({"run_id": RUN}), encoding="utf-8")


@pytest.fixture
def root(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "dashboard"
    _write_run(settings, folder)
    monkeypatch.setenv("DASHBOARD_ROOT", str(folder))
    service.clear_cache()
    return folder


@pytest.fixture
def client(root: Path) -> Iterator[TestClient]:
    yield TestClient(app)
    service.clear_cache()


def test_health_runs_and_days(client: TestClient) -> None:
    health = client.get("/api/health").json()
    assert health["status"] == "ok" and health["run"] == RUN
    assert health["traded_days"] == 3 and health["grid_available"] is True
    (run,) = client.get("/api/runs").json()
    assert run["run_id"] == RUN and "mode" in run
    days = client.get("/api/days").json()["days"]
    assert [d["traded"] for d in days] == [True, True, False, True]


def test_summary_scales_with_power_and_reads_the_chosen_strategy(
    client: TestClient,
) -> None:
    base = client.get(f"/api/runs/{RUN}/summary").json()
    assert base["window"]["traded_days"] == 3
    assert len(base["window"]["skipped_days"]) == 1
    kpis = base["kpis"]
    assert kpis["pnl_eur"] == pytest.approx(80.0 * 2 * 3)
    assert kpis["capture_ratio"] == pytest.approx(0.8)
    assert kpis["cycles_per_day"] == pytest.approx(1.0)
    assert kpis["baseline_pinball_eur_mwh"] > kpis["pinball_eur_mwh"]

    scaled = client.get(
        f"/api/runs/{RUN}/summary",
        params={"power": 2.5, "strategy": "q25", "duration": 4},
    ).json()
    assert scaled["battery"]["capacity_mwh"] == pytest.approx(10.0)
    assert scaled["kpis"]["pnl_eur"] == pytest.approx(70.0 * 4 * 3 * 2.5)
    assert scaled["kpis"]["capture_ratio"] == pytest.approx(0.7)


@pytest.mark.parametrize(
    ("window", "days"), [("validation", 2), ("holdout", 1), ("all", 3)]
)
def test_windows_select_their_days(client: TestClient, window: str, days: int) -> None:
    body = client.get(f"/api/runs/{RUN}/summary", params={"window": window}).json()
    assert body["window"]["traded_days"] == days


@pytest.mark.parametrize(
    "params",
    [{"duration": 5}, {"power": 0.7}, {"strategy": "q30"}, {"window": "last7"}],
)
def test_invalid_parameters_are_refused(
    client: TestClient, params: dict[str, object]
) -> None:
    assert client.get(f"/api/runs/{RUN}/summary", params=params).status_code == 422


def test_forecast_and_dispatch_for_a_day(
    client: TestClient, settings: Settings
) -> None:
    days, skipped_day = _days(settings)
    fan = client.get("/api/forecast", params={"date": str(days[0])}).json()
    assert len(fan["periods"]) == 96 and fan["periods"][0]["time"] == "00:00"
    assert fan["issued_local"] == f"{days[0] - timedelta(days=1)} 11:40"
    assert fan["window"] == "validation"

    one = client.get("/api/dispatch", params={"date": str(days[0])}).json()
    two = client.get("/api/dispatch", params={"date": str(days[0]), "power": 2}).json()
    assert one["solved_on_request"] is True and len(one["periods"]) == 96
    assert two["pnl_eur"] == pytest.approx(2 * one["pnl_eur"], abs=1e-6)
    assert [p["net_mw"] for p in two["periods"]] == pytest.approx(
        [2 * p["net_mw"] for p in one["periods"]], abs=1e-6
    )
    assert one["perfect_foresight_pnl_eur"] >= one["pnl_eur"] - 1e-6
    refused = client.get("/api/dispatch", params={"date": str(skipped_day)})
    assert refused.status_code == 422
    for text in ("21/11/2025", "2025-1-5", "2025-02-30"):
        assert client.get("/api/dispatch", params={"date": text}).status_code == 422
    assert client.get("/api/forecast", params={"date": "2020-01-01"}).status_code == 404


def test_pnl_calibration_error_analysis_and_features(client: TestClient) -> None:
    pnl = client.get("/api/pnl", params={"strategy": "q10"}).json()
    assert len(pnl["series"]) == 3
    assert [s["selected"] for s in pnl["series"]] == pytest.approx([120, 60, 180])
    assert pnl["series"][-1]["perfect_foresight"] == pytest.approx(100.0 * 2 * 3)
    assert pnl["max_drawdown_eur"]["selected"] == pytest.approx(60.0)
    assert pnl["max_drawdown_eur"]["median"] == 0.0

    calibration = client.get("/api/calibration").json()
    assert [q["level"] for q in calibration["quantiles"]] == pytest.approx(
        [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    )
    assert [i["nominal"] for i in calibration["intervals"]] == [0.5, 0.8, 0.9]
    assert all(0 <= q["empirical"] <= 1 for q in calibration["quantiles"])

    errors = client.get("/api/error-analysis").json()
    assert len(errors["by_hour"]) == 24
    assert errors["by_hour"][18]["value_at_stake_eur_per_day"] == pytest.approx(18.0)
    asymmetry = errors["asymmetry"]
    assert asymmetry["gap_eur"] == pytest.approx(3 * 8.0)
    assert asymmetry["blocks"] == [
        {"block": "18-20", "over_eur": 6.0, "under_eur": 18.0}
    ]

    features = client.get("/api/feature-importance").json()
    assert features["features"][0]["feature"] == "price_lag_1d"


def test_missing_runs_and_grid_are_reported(
    client: TestClient, root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert client.get("/api/runs/nope/summary").status_code == 404
    (root / RUN / "pnl_grid.parquet").unlink()
    missing = client.get(f"/api/runs/{RUN}/summary")
    assert missing.status_code == 503 and "grid" in missing.json()["detail"]
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("DASHBOARD_ROOT", str(empty))
    assert client.get("/api/health").status_code == 503


def test_a_new_export_is_picked_up_without_a_restart(
    client: TestClient, root: Path
) -> None:
    before = client.get("/api/feature-importance").json()
    importance = before | {"trained_for_day": "2099-01-01"}
    path = root / RUN / "feature_importance.json"
    path.write_text(json.dumps(importance), encoding="utf-8")
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))
    after = client.get("/api/feature-importance").json()
    assert after["trained_for_day"] == "2099-01-01"


def test_a_failed_solve_and_a_half_written_run_answer_503(
    client: TestClient,
    root: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    days, _ = _days(settings)

    def fail(*args: object, **kwargs: object) -> None:
        raise DispatchError("not optimal")

    monkeypatch.setattr(service, "dispatch_day", fail)
    failed = client.get("/api/dispatch", params={"date": str(days[0])})
    assert (
        failed.status_code == 503 and "could not be solved" in failed.json()["detail"]
    )

    manifest = root / RUN / "manifest.json"
    manifest.write_text("{", encoding="utf-8")
    stat = manifest.stat()
    os.utime(manifest, (stat.st_atime, stat.st_mtime + 10))
    assert client.get("/api/health").status_code == 503


def test_last30_and_last90_end_on_the_last_day(settings: Settings) -> None:
    last = settings.evaluation.holdout_start + timedelta(days=20)
    every_day = [last - timedelta(days=n) for n in range(119, -1, -1)]
    skipped = last - timedelta(days=5)
    manifest = build_manifest(
        settings,
        "r",
        [day for day in every_day if day != skipped],
        {skipped: "missing price or forecast"},
    )
    last30 = service.resolve_window(manifest, "last30")
    assert (last30.first_day, last30.last_day) == (last - timedelta(days=29), last)
    assert len(last30.traded_days) == 29 and last30.skipped_days == (skipped,)
    last90 = service.resolve_window(manifest, "last90")
    assert last90.first_day == last - timedelta(days=89)
    assert len(last90.traded_days) == 89
    holdout = service.resolve_window(manifest, "holdout")
    assert holdout.first_day == settings.evaluation.holdout_start
    assert len(holdout.traded_days) == 20


def test_the_autumn_clock_change_has_unique_period_keys(settings: Settings) -> None:
    day = date(2025, 10, 26)
    frame = day_frame(settings, day).assign(model="lightgbm_conformal")
    run = service.Run(
        run_id="r",
        manifest=build_manifest(settings, "r", [day], {}),
        forecasts=frame,
        grid=None,
        hour_value=pd.DataFrame(),
        attribution=pd.DataFrame(),
        feature_importance={},
    )
    fan = service.forecast_day(run, day)["periods"]
    schedule = service.dispatch(run, day, 2, 8, "median", 1.0)["periods"]
    for periods in (fan, schedule):
        assert len(periods) == 100
        assert len({p["utc"] for p in periods}) == 100
        assert [p["time"] for p in periods].count("02:15") == 2
