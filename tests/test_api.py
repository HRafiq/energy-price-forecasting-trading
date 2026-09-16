"""The dashboard API over a small synthetic run."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api import health as model_health
from api import service
from api.main import app
from src.config import Settings
from src.export.artifacts import build_manifest
from src.health.incidents import Incident, IncidentType, make_incident_id
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
    assert health["run_kind"] == "backtest"
    (run,) = client.get("/api/runs").json()
    assert run["run_id"] == RUN and "mode" in run
    assert run["run_kind"] == "backtest" and run["issued_utc"] is None
    assert run["data_through"] == run["last_day"]
    days = client.get("/api/days").json()["days"]
    assert [d["traded"] for d in days] == [True, True, False, True]


def test_a_live_run_is_served_and_an_older_manifest_is_a_backtest(
    client: TestClient, root: Path
) -> None:
    path = root / RUN / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    live = manifest | {
        "run_kind": "live",
        "issued_utc": "2026-09-16T09:40:00+00:00",
        "data_through": "2026-09-15",
    }
    path.write_text(json.dumps(live), encoding="utf-8")
    _bump(path)
    (run,) = client.get("/api/runs").json()
    assert run["run_kind"] == "live"
    assert run["issued_utc"] == "2026-09-16T09:40:00+00:00"
    assert run["data_through"] == "2026-09-15"
    assert run["timezone"] == manifest["timezone"]
    assert client.get("/api/health").json()["run_kind"] == "live"

    older = {
        key: value
        for key, value in manifest.items()
        if key not in {"run_kind", "issued_utc", "data_through"}
    }
    path.write_text(json.dumps(older), encoding="utf-8")
    _bump(path)
    (run,) = client.get("/api/runs").json()
    assert run["run_kind"] == "backtest" and run["issued_utc"] is None
    assert run["data_through"] == run["last_day"]
    assert client.get("/api/health").json()["run_kind"] == "backtest"


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


# --- Model health ---------------------------------------------------------------


def _incident(
    day: date, kind: IncidentType, source: str, **metrics: float
) -> dict[str, object]:
    return Incident(
        incident_id=make_incident_id(source, kind, day),
        delivery_day=day,
        detected_utc=datetime(day.year, day.month, day.day, 10, tzinfo=UTC),
        type=kind,
        severity="warning",
        detail=f"{kind} on {day}.",
        action="Logged.",
        status="review" if source == "observed" else "resolved",
        source=source,
        metrics=metrics,
    ).model_dump(mode="json")


HEALTH_INCIDENTS = [
    _incident(date(2024, 12, 12), "drift", "m2_drift", in_sample=1.0),
    _incident(date(2024, 12, 13), "tail_miss", "observed"),
    _incident(date(2025, 3, 1), "data_gap", "observed", fallback_periods=4.0),
    _incident(date(2025, 3, 2), "late_data", "d5_deadline"),
    _incident(date(2025, 3, 3), "pipeline", "d5_deadline"),
    _incident(date(2026, 7, 2), "drift", "m2_drift", in_sample=0.0),
]


def _write_health(folder: Path) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    dates = ["2021-01-28", "2021-01-29", "2021-01-30"]
    regime = {
        "experiment": "m1_regime_shift",
        "setup": {
            "model": "lightgbm_conformal",
            "training_days": 730,
            "calibration_days": 42,
            "feature_groups": ["calendar", "price_history"],
            "first_target_day": "2021-01-01",
            "last_target_day": "2023-12-31",
            "arms": [{"name": "frozen", "refit_every_days": None}],
        },
        "coverage_target": 0.9,
        "coverage_alert_threshold": 0.82,
        "arms": ["frozen", "monthly"],
        "rolling_coverage_90": {
            "window_days": 28,
            "dates": dates,
            "arms": {"frozen": [0.8883931, 0.7, 0.6], "monthly": [0.9, 0.91, 0.92]},
        },
        "min_rolling_coverage_90": {
            "frozen": {"value": 0.6, "window_end": "2021-01-30"}
        },
        "quarterly": [{"arm": "frozen", "period": "2021Q1"}],
        "yearly": [
            {"arm": arm, "period": "2021", "coverage_90": 0.5, "capture_ratio": 0.4}
            for arm in ("frozen", "monthly")
        ],
        "total": [
            {
                "arm": "frozen",
                "period": "total",
                "coverage_90": 0.3,
                "capture_ratio": 0.3,
            }
        ],
    }
    series = [
        {
            "target_day": day,
            "window": window,
            "coverage_90": 0.9,
            "pinball": 5.0,
            "rolling_coverage_90": coverage,
            "rolling_pinball_ratio": ratio,
            "coverage_alert": coverage < 0.74,
            "pinball_alert": ratio > 1.5,
        }
        for day, window, coverage, ratio in (
            ("2026-05-30", "validation", 0.85, 1.0),
            ("2026-05-31", "validation", 0.73, 1.6),
            ("2026-06-01", "holdout", 0.8, 1.2),
            ("2026-06-02", "holdout", 0.674851, 1.937204),
        )
    ]
    drift = {
        "model": "lightgbm_conformal",
        "window_days": 28,
        "holdout_start": "2026-06-01",
        "rule": "fixed on validation days",
        "thresholds": {"coverage": 0.74, "pinball_ratio": 1.5},
        "windows": {"validation": {"days": 2}, "holdout": {"days": 2}},
        "episodes": [
            {"signal": "coverage", "window": "validation", "start": "2026-05-31"},
            {"signal": "pinball", "window": "holdout", "start": "2026-06-02"},
        ],
        "series": series,
    }
    deadline = {
        "setup": {"issue_local": "11:40", "gate_local": "12:00", "seed": 6}
        | {"days": ["2024-06-01", "2026-05-31"]}
        | {"failure_rates": {"weather_late": 0.1, "model_fails": 0.03}},
        "realised_failure_rates": {"weather_late": 0.0876, "model_fails": 0.0274},
        "runtimes_s": {"full": 0.1},
        "summary": {
            "days": 730,
            "on_time_share_with_chain": 1.0,
            "on_time_share_without_chain": 0.869863,
            "fallback_days": 95,
            "fallback_by_step": {"fallback_no_weather": 64},
            "latest_submission_minutes_after_issue": 10.0,
            "capture_chain": 0.896,
        },
    }
    for name, payload in (
        ("m1_regime_experiment.json", regime),
        ("m2_drift.json", drift),
        ("d5_deadline.json", deadline),
        ("incidents.json", HEALTH_INCIDENTS),
        (
            "index.json",
            {
                "files": [
                    {"file": name, "generated_utc": f"2026-09-15T1{i}:00"}
                    for i, name in enumerate(
                        (
                            "m1_regime_experiment.json",
                            "m2_drift.json",
                            "d5_deadline.json",
                            "incidents.json",
                        )
                    )
                ]
            },
        ),
    ):
        (folder / name).write_text(json.dumps(payload), encoding="utf-8")


@pytest.fixture
def health_client(client: TestClient, root: Path) -> Iterator[TestClient]:
    _write_health(root / RUN / "health")
    model_health.clear_cache()
    yield client
    model_health.clear_cache()


def _bump(path: Path) -> None:
    stat = path.stat()
    os.utime(path, (stat.st_atime, stat.st_mtime + 10))


def test_regime_serves_the_backtest_without_the_stale_alert_line(
    health_client: TestClient,
) -> None:
    body = health_client.get("/api/model-health/regime").json()
    assert "coverage_alert_threshold" not in json.dumps(body)
    assert body["coverage_target"] == 0.9 and body["historical_backtest"] is True
    assert body["setup"]["weather_features"] is False
    rolling = body["rolling_coverage_90"]
    assert rolling["dates"] == ["2021-01-28", "2021-01-29", "2021-01-30"]
    # Served as written: the dashboard formats once, so nothing is pre-rounded.
    assert rolling["arms"]["frozen"][0] == 0.8883931
    assert body["generated_utc"] == "2026-09-15T10:00"
    assert [(p["arm"], p["period"]) for p in body["periods"]] == [
        ("frozen", "2021"),
        ("monthly", "2021"),
        ("frozen", "total"),
    ]
    assert "quarterly" not in body


@pytest.mark.parametrize(
    ("window", "days", "episodes"),
    [("all", 4, 2), ("validation", 2, 1), ("holdout", 2, 1)],
)
def test_drift_filters_by_window(
    health_client: TestClient, window: str, days: int, episodes: int
) -> None:
    body = health_client.get(
        "/api/model-health/drift", params={"window": window}
    ).json()
    assert len(body["series"]) == days and len(body["episodes"]) == episodes
    assert body["thresholds"] == {"coverage": 0.74, "pinball_ratio": 1.5}
    assert body["generated_utc"] == "2026-09-15T11:00"
    assert body["matches_run"] is True and body["last_target_day"] == "2026-06-02"
    assert body["run_last_day"] == "2026-06-02"
    if window != "all":
        assert set(body["windows"]) == {window}
        assert {row["window"] for row in body["series"]} == {window}
    bad = health_client.get("/api/model-health/drift", params={"window": "x"})
    assert bad.status_code == 422


def test_incidents_newest_first_with_faceted_counts(
    health_client: TestClient,
) -> None:
    body = health_client.get("/api/model-health/incidents").json()
    assert body["total"] == 6
    days = [i["delivery_day"] for i in body["incidents"]]
    assert days == sorted(days, reverse=True)
    assert body["counts"]["type"]["drift"] == 2
    assert body["counts"]["type"]["drawdown"] == 0
    assert body["counts"]["source"] == {"d5_deadline": 2, "m2_drift": 2, "observed": 2}
    assert body["counts"]["provenance"] == {
        "observed": 2,
        "measured": 2,
        "simulated": 2,
    }
    assert body["source_provenance"] == {
        "d5_deadline": "simulated",
        "m2_drift": "measured",
        "observed": "observed",
    }
    assert body["generated_utc"] == "2026-09-15T13:00"
    by_day = {i["delivery_day"]: i for i in body["incidents"]}
    # Drift alerts are measured on real saved forecasts, not injected.
    assert by_day["2026-07-02"]["provenance"] == "measured"
    assert by_day["2026-07-02"]["in_sample"] is False
    assert by_day["2024-12-12"]["in_sample"] is True
    assert by_day["2025-03-02"]["provenance"] == "simulated"
    assert by_day["2025-03-02"]["in_sample"] is None
    assert "injected" not in by_day["2025-03-02"]

    observed = health_client.get(
        "/api/model-health/incidents", params={"source": "observed"}
    ).json()
    assert observed["total"] == 2
    assert {i["provenance"] for i in observed["incidents"]} == {"observed"}
    assert observed["counts"]["provenance"]["measured"] == 0
    # Type counts follow the source filter; source counts ignore it.
    assert observed["counts"]["type"]["drift"] == 0
    assert observed["counts"]["source"]["m2_drift"] == 2

    both = health_client.get(
        "/api/model-health/incidents",
        params=[("type", "drift"), ("type", "pipeline"), ("start", "2025-01-01")],
    ).json()
    assert [i["type"] for i in both["incidents"]] == ["drift", "pipeline"]
    assert both["counts"]["source"] == {"d5_deadline": 1, "m2_drift": 1}

    page = health_client.get(
        "/api/model-health/incidents",
        params={"limit": 2, "offset": 2, "end": "2025-12-31"},
    ).json()
    assert page["total"] == 5 and page["limit"] == 2 and page["offset"] == 2
    assert [i["delivery_day"] for i in page["incidents"]] == [
        "2025-03-01",
        "2024-12-13",
    ]


@pytest.mark.parametrize(
    "params",
    [
        {"type": "outage"},
        {"source": "M2 drift"},
        {"start": "2025-02-30"},
        {"start": "2025-03-02", "end": "2025-03-01"},
        {"limit": 0},
        {"limit": 501},
        {"offset": -1},
    ],
)
def test_invalid_incident_queries_are_refused(
    health_client: TestClient, params: dict[str, object]
) -> None:
    response = health_client.get("/api/model-health/incidents", params=params)
    assert response.status_code == 422 and response.json()["detail"]


def test_ops_tiles_and_the_missing_deadline_experiment(
    health_client: TestClient, root: Path
) -> None:
    body = health_client.get("/api/model-health/ops").json()
    deadline = body["deadline"]
    assert deadline["available"] is True and deadline["simulation"] is True
    assert deadline["on_time_share_without_chain"] == 0.869863
    assert deadline["generated_utc"] == "2026-09-15T12:00"
    assert deadline["failure_rates"] == {"weather_late": 0.1, "model_fails": 0.03}
    assert deadline["seed"] == 6
    assert deadline["realised_failure_rates"]["weather_late"] == 0.0876
    fallbacks = body["fallbacks"]
    # Simulated D5 days and observed incidents cover different periods: never summed.
    assert (fallbacks["simulated_days"], fallbacks["observed"]) == (95, 1)
    assert "total" not in fallbacks
    assert fallbacks["simulated_period"] == {
        "first_day": "2024-06-01",
        "last_day": "2026-05-31",
    }
    assert fallbacks["observed_period"]["last_day"] == "2026-06-02"
    assert fallbacks["observed_generated_utc"] == "2026-09-15T13:00"
    drift = body["drift"]
    assert drift["as_of"] == "2026-06-02" and drift["holdout_days"] == 2
    assert drift["coverage"] == {"value": 0.674851, "threshold": 0.74, "alert": True}
    assert drift["pinball_ratio"]["value"] == 1.937204
    assert drift["generated_utc"] == "2026-09-15T11:00" and drift["matches_run"]

    (root / RUN / "health" / "d5_deadline.json").unlink()
    body = health_client.get("/api/model-health/ops").json()
    assert body["deadline"] == {"available": False, "detail": "D5 not exported yet"}
    assert body["fallbacks"]["available"] is False
    assert "simulated_days" not in body["fallbacks"]
    assert body["fallbacks"]["observed"] == 1
    assert body["drift"]["available"] is True


def test_missing_health_files_answer_503(health_client: TestClient, root: Path) -> None:
    folder = root / RUN / "health"
    (folder / "m1_regime_experiment.json").unlink()
    missing = health_client.get("/api/model-health/regime")
    assert missing.status_code == 503 and "M1" in missing.json()["detail"]
    assert health_client.get("/api/model-health/drift").status_code == 200
    for path in folder.iterdir():
        path.unlink()
    folder.rmdir()
    for endpoint in ("regime", "drift", "incidents", "ops"):
        response = health_client.get(f"/api/model-health/{endpoint}")
        assert response.status_code == 503, endpoint
        assert response.json()["detail"]
    assert health_client.get("/api/model-health/ops?run=nope").status_code == 404


def test_a_new_health_export_is_picked_up_without_a_restart(
    health_client: TestClient, root: Path
) -> None:
    assert health_client.get("/api/model-health/incidents").json()["total"] == 6
    path = root / RUN / "health" / "incidents.json"
    path.write_text(json.dumps(HEALTH_INCIDENTS[:2]), encoding="utf-8")
    _bump(path)
    assert health_client.get("/api/model-health/incidents").json()["total"] == 2
    path.write_text("[{", encoding="utf-8")
    _bump(path)
    assert health_client.get("/api/model-health/incidents").status_code == 503


def test_drift_that_does_not_end_on_the_run_last_day_is_flagged(
    health_client: TestClient, root: Path
) -> None:
    path = root / RUN / "health" / "m2_drift.json"
    drift = json.loads(path.read_text(encoding="utf-8"))
    drift["series"] = drift["series"][:-1]
    path.write_text(json.dumps(drift), encoding="utf-8")
    _bump(path)
    body = health_client.get("/api/model-health/drift").json()
    assert body["matches_run"] is False
    assert (body["last_target_day"], body["run_last_day"]) == (
        "2026-06-01",
        "2026-06-02",
    )
    assert (
        health_client.get("/api/model-health/ops").json()["drift"]["matches_run"]
        is False
    )


def test_a_same_timestamp_rewrite_is_picked_up_by_its_size(
    health_client: TestClient, root: Path
) -> None:
    path = root / RUN / "health" / "incidents.json"
    stat = path.stat()
    assert health_client.get("/api/model-health/incidents").json()["total"] == 6
    path.write_text(json.dumps(HEALTH_INCIDENTS[:3]), encoding="utf-8")
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    assert path.stat().st_mtime_ns == stat.st_mtime_ns
    assert health_client.get("/api/model-health/incidents").json()["total"] == 3


@pytest.mark.parametrize(
    "params",
    [{"limit": 0}, {"type": "outage"}, {"start": "2025-02-30"}, {"offset": -1}],
)
def test_a_bad_incident_query_is_refused_before_the_missing_export(
    client: TestClient, params: dict[str, object]
) -> None:
    model_health.clear_cache()
    response = client.get("/api/model-health/incidents", params=params)
    assert response.status_code == 422, response.json()
    assert client.get("/api/model-health/incidents").status_code == 503
