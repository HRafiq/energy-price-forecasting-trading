"""One live day: the guard, the chain it used, the plan, incidents and settling."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings
from src.forecasting.base import QuantileForecast, make_forecast
from src.forecasting.information import InformationSet
from src.health.incidents import default_path, load_incidents
from src.pipeline import daily_run as dr
from src.pipeline.model_source import RefitOutcome
from src.pipeline.plan import load_plan, plan_path
from tests.fakes import synthetic_market

LIVE_DAY = date(2026, 9, 17)


def _local(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    return settings.model_copy(update={"data": data})


@dataclass
class FakeModel:
    """A forecaster that answers with a cheap night and a dear evening."""

    settings: Settings
    name: str = "fake_production"
    fitted: list[date] = field(default_factory=list)

    @property
    def lookback_days(self) -> int | None:
        return 3

    @property
    def fit_lookback_days(self) -> int | None:
        return 3

    def fit(self, info: InformationSet) -> None:
        self.fitted.append(info.target_day)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        quantiles = self.settings.forecasting.quantiles
        hours = np.asarray(info.target_index.tz_convert(info.tz).hour, dtype="float64")
        curve = np.where((hours >= 17) & (hours < 21), 200.0, 20.0)
        values = pd.DataFrame(
            {f"q{int(q * 100):02d}": curve + 2 * q for q in quantiles},
            index=info.target_index,
        )
        return make_forecast(self.name, info, values, quantiles)


@pytest.fixture(autouse=True)
def _no_real_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """No test here refits into, or reads, the repository's MLflow store."""
    monkeypatch.setattr(
        dr,
        "refresh_production_model",
        lambda settings, day, frame, **kwargs: RefitOutcome(
            "not_due", "fake registry", "3", "3"
        ),
    )


@pytest.fixture
def market(settings: Settings) -> pd.DataFrame:
    return synthetic_market(settings, LIVE_DAY - timedelta(days=40), 42)


def _run(
    settings: Settings,
    market: pd.DataFrame,
    *,
    now: datetime | None = None,
    model: FakeModel | None = None,
    monkeypatch: pytest.MonkeyPatch | None = None,
    replace: bool = False,
) -> dr.RunRecord:
    if monkeypatch is not None and model is not None:
        monkeypatch.setattr(dr, "load_model", lambda s, **k: (model, "3"))
    return dr.run_day(
        settings,
        LIVE_DAY,
        frame=market,
        use_registry=model is not None,
        now_utc=now or datetime(2026, 9, 16, 9, 40, tzinfo=UTC),
        replace=replace,
    )


def test_a_day_before_live_from_is_refused(
    settings: Settings, tmp_path: Path, market: pd.DataFrame
) -> None:
    local = _local(settings, tmp_path)
    with pytest.raises(ValueError, match="before live_from"):
        dr.run_day(local, settings.evaluation.holdout_last_day, frame=market)


def test_a_ready_day_uses_the_registered_model_and_commits_a_plan(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    model = FakeModel(local)

    record = _run(local, market, model=model, monkeypatch=monkeypatch)

    assert record.step == "production" and not record.degraded
    assert record.on_time and record.model_version == "3"
    assert model.fitted == []
    assert record.incidents == ()
    assert record.planned_value_eur > 0

    plan = load_plan(local, LIVE_DAY)
    assert plan.step == "production" and plan.product_minutes == 15
    assert plan.schedule["discharge_mw"].sum() > 0
    saved = json.loads(dr.record_path(local, LIVE_DAY).read_text())
    assert saved["step"] == "production" and saved["minutes_before_gate"] == 20.0
    assert [a["step"] for a in saved["attempts"]] == ["production"]
    forecast = pd.read_parquet(
        local.data.processed_path / "forecasts" / "production" / f"{LIVE_DAY}.parquet"
    )
    assert len(forecast) == 96 and forecast["chain_step"].iloc[0] == "production"


def test_a_late_feed_steps_down_and_writes_one_incident(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    late = market.copy()
    start = local.market.local_midnight_utc(LIVE_DAY)
    weather = [c for c in late.columns if str(c).startswith("wx_")]
    late.loc[late.index >= start, weather] = np.nan
    model = FakeModel(local)

    record = _run(local, late, model=model, monkeypatch=monkeypatch)

    assert record.degraded and record.step != "production"
    assert model.fitted == []
    # The baseline that ran is not the registered model, so it carries no version.
    assert record.model_version is None
    assert len(record.incidents) == 1

    (incident,) = load_incidents(default_path(local))
    assert incident.type == "late_data" and incident.source == "pipeline"
    assert incident.status == "resolved" and incident.severity == "warning"
    assert "weather" in incident.detail
    assert "forecast issued 11:40" in incident.action
    assert incident.metrics["minutes_before_gate"] == 20.0


def test_a_run_after_the_gate_is_recorded_as_late(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    model = FakeModel(local)

    record = _run(
        local,
        market,
        model=model,
        monkeypatch=monkeypatch,
        now=datetime(2026, 9, 16, 10, 30, tzinfo=UTC),
    )

    assert not record.on_time and record.step == "production"
    (incident,) = load_incidents(default_path(local))
    assert incident.severity == "critical" and incident.status == "review"
    assert "after the 12:00 gate" in incident.detail
    # Nothing failed and nothing was late to arrive: the run itself was late.
    assert "every feed had arrived" in incident.detail
    assert "could not run" not in incident.detail
    assert incident.action.startswith("Forecast issued")


def test_settling_values_the_plan_at_published_prices(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    model = FakeModel(local)
    _run(local, market, model=model, monkeypatch=monkeypatch)

    settled = dr.settle_day(local, LIVE_DAY, frame=market)

    plan = load_plan(local, LIVE_DAY)
    assert settled["target_day"] == str(LIVE_DAY)
    assert settled["step"] == "production"
    assert settled["pnl_eur"] == pytest.approx(
        settled["revenue_eur"] - settled["degradation_eur"], abs=0.01
    )
    assert settled["discharged_mwh"] > 0
    saved = json.loads(
        (
            local.data.processed_path / "pipeline" / "settlements" / f"{LIVE_DAY}.json"
        ).read_text()
    )
    assert saved["planned_value_eur"] == pytest.approx(plan.planned_value_eur, abs=0.01)


def test_settling_waits_for_prices(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    model = FakeModel(local)
    _run(local, market, model=model, monkeypatch=monkeypatch)
    unpriced = market.copy()
    start = local.market.local_midnight_utc(LIVE_DAY)
    unpriced.loc[unpriced.index >= start, PRICE_SERIES] = np.nan

    with pytest.raises(ValueError, match="have no price yet"):
        dr.settle_day(local, LIVE_DAY, frame=unpriced)


def test_a_missed_deadline_writes_one_incident(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Airflow's deadline alert calls this, so a late run reaches the health log."""
    local = _local(settings, tmp_path)
    monkeypatch.setattr(dr, "load_settings", lambda config=None: local)

    first = dr.deadline_missed(str(LIVE_DAY))
    log = default_path(local)
    written = log.read_bytes()
    again = dr.deadline_missed(str(LIVE_DAY))

    assert first == again, "the same day must not pile up incidents"
    assert log.read_bytes() == written
    (incident,) = load_incidents(log)
    assert incident.type == "pipeline" and incident.severity == "critical"
    assert incident.status == "review" and incident.source == "pipeline"
    assert incident.delivery_day == LIVE_DAY
    assert "passed its deadline" in incident.detail


def test_the_deadline_check_reports_only_a_late_run(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    model = FakeModel(local)
    _run(local, market, model=model, monkeypatch=monkeypatch)

    assert dr.check_deadline(local, LIVE_DAY) is True
    assert not default_path(local).exists()

    # The same day again, after the gate: replacing on purpose to make a late run.
    late = _run(
        local,
        market,
        model=model,
        monkeypatch=monkeypatch,
        now=datetime(2026, 9, 16, 10, 30, tzinfo=UTC),
        replace=True,
    )
    assert not late.on_time
    assert dr.check_deadline(local, LIVE_DAY) is False
    kinds = [i.type for i in load_incidents(default_path(local))]
    assert kinds.count("pipeline") == 1

    with pytest.raises(FileNotFoundError, match="no run record"):
        dr.check_deadline(local, date(2026, 9, 20))


def test_nothing_is_defined_after_the_main_guard() -> None:
    """A def below the guard exists on import but not when run as a command.

    Every unit test here imports the module, so such a definition looks fine and
    then fails in the DAG with a NameError. This checks the file's shape instead.
    """
    source = Path(dr.__file__).read_text(encoding="utf-8")
    guard = source.index('if __name__ == "__main__":')
    after = source[guard:].splitlines()[1:]
    stray = [line for line in after if line and not line.startswith((" ", "\t"))]
    assert stray == [], f"defined after the main guard: {stray}"


def test_a_rerun_never_replaces_a_committed_schedule(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    model = FakeModel(local)
    first = _run(local, market, model=model, monkeypatch=monkeypatch)
    plan_file, record_file = plan_path(local, LIVE_DAY), dr.record_path(local, LIVE_DAY)
    plan_bytes, record_bytes = plan_file.read_bytes(), record_file.read_bytes()
    afternoon = datetime(2026, 9, 16, 14, 0, tzinfo=UTC)

    # Airflow starting in the afternoon runs the slot it missed: same day, after the
    # gate. The schedule on record was on time and must stay exactly as it was.
    with pytest.raises(dr.PlanExistsError, match="already committed"):
        _run(local, market, model=model, monkeypatch=monkeypatch, now=afternoon)

    assert first.on_time
    assert plan_file.read_bytes() == plan_bytes
    assert record_file.read_bytes() == record_bytes
    assert not default_path(local).exists(), "a refused rerun writes no incident"

    replaced = _run(
        local, market, model=model, monkeypatch=monkeypatch, now=afternoon, replace=True
    )
    assert not replaced.on_time
    assert json.loads(record_file.read_text(encoding="utf-8"))["on_time"] is False


def test_the_cli_leaves_a_committed_schedule_alone_and_exits_cleanly(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = _local(settings, tmp_path)
    monkeypatch.setattr(dr, "load_settings", lambda config=None: local)
    asked_to_replace: list[bool] = []

    def refuse(settings: Settings, day: date, **kwargs: object) -> dr.RunRecord:
        asked_to_replace.append(bool(kwargs.get("replace")))
        raise dr.PlanExistsError(f"a schedule for {day} is already committed")

    monkeypatch.setattr(dr, "run_day", refuse)

    # The DAG's forecast task must succeed, so the export, the deadline check and
    # settlement still run against the schedule on record.
    assert dr.main(["--day", str(LIVE_DAY)]) == 0
    assert "not replaced" in capsys.readouterr().out
    assert dr.main(["--day", str(LIVE_DAY), "--replace"]) == 0
    assert asked_to_replace == [False, True]


def test_a_plan_without_its_run_record_is_left_for_a_person(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    model = FakeModel(local)
    _run(local, market, model=model, monkeypatch=monkeypatch)
    plan_file = plan_path(local, LIVE_DAY)
    plan_bytes = plan_file.read_bytes()
    # A run that crashed after committing its plan, before writing its record.
    dr.record_path(local, LIVE_DAY).unlink()

    with pytest.raises(dr.IncompleteRunError, match="run record"):
        _run(local, market, model=model, monkeypatch=monkeypatch)

    assert plan_file.read_bytes() == plan_bytes
    assert not dr.record_path(local, LIVE_DAY).exists()


@pytest.mark.parametrize(
    ("status", "incident_status"),
    [
        ("failed", "review"),
        ("postponed", "resolved"),
        ("not_due", None),
        ("refitted", None),
    ],
)
def test_a_refit_that_did_not_happen_is_recorded_and_the_day_still_trades(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
    status: Literal["failed", "postponed", "not_due", "refitted"],
    incident_status: str | None,
) -> None:
    local = _local(settings, tmp_path)
    monkeypatch.setattr(
        dr,
        "refresh_production_model",
        lambda settings, day, frame, **kwargs: RefitOutcome(
            status,
            f"{status} for the test",
            "3",
            "3",
        ),
    )

    record = _run(local, market, model=FakeModel(local), monkeypatch=monkeypatch)

    saved = json.loads(dr.record_path(local, LIVE_DAY).read_text(encoding="utf-8"))
    assert saved["refit"]["status"] == status
    assert record.step == "production" and load_plan(local, LIVE_DAY) is not None
    refits = [
        incident
        for incident in load_incidents(default_path(local))
        if "Scheduled refit" in incident.detail
    ]
    if incident_status is None:
        assert refits == [] and not record.incidents
    else:
        assert [incident.status for incident in refits] == [incident_status]
        assert refits[0].incident_id in record.incidents


def test_a_refit_that_raises_does_not_stop_the_day(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)

    def explode(*args: object, **kwargs: object) -> RefitOutcome:
        raise RuntimeError("store locked")

    monkeypatch.setattr(dr, "refresh_production_model", explode)

    record = _run(local, market, model=FakeModel(local), monkeypatch=monkeypatch)

    saved = json.loads(dr.record_path(local, LIVE_DAY).read_text(encoding="utf-8"))
    assert saved["refit"]["status"] == "failed"
    assert "RuntimeError: store locked" in saved["refit"]["detail"]
    assert record.step == "production"


def test_no_refit_is_attempted_without_the_registry(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)

    def refuse(*args: object, **kwargs: object) -> RefitOutcome:
        raise AssertionError("refit attempted without the registry")

    monkeypatch.setattr(dr, "refresh_production_model", refuse)

    _run(local, market)

    saved = json.loads(dr.record_path(local, LIVE_DAY).read_text(encoding="utf-8"))
    assert saved["refit"] is None


def test_the_refit_sees_the_missing_feeds_and_runs_before_the_model_is_loaded(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    calls: list[tuple[str, object]] = []

    def refit(
        settings: Settings, day: date, frame: pd.DataFrame, **kwargs: object
    ) -> RefitOutcome:
        calls.append(("refit", kwargs.get("missing_feeds")))
        return RefitOutcome("postponed", "waiting", "3", "3")

    model = FakeModel(local)

    def load(settings: Settings, **kwargs: object) -> tuple[FakeModel, str]:
        calls.append(("load", None))
        return model, "3"

    monkeypatch.setattr(dr, "refresh_production_model", refit)
    monkeypatch.setattr(dr, "load_model", load)
    late_weather = market.copy()
    weather = [column for column in late_weather.columns if column.startswith("wx_")]
    start = pd.Timestamp(LIVE_DAY, tz=local.market.timezone).tz_convert("UTC")
    late_weather.loc[late_weather.index >= start, weather] = np.nan
    assert weather

    dr.run_day(
        local,
        LIVE_DAY,
        frame=late_weather,
        now_utc=datetime(2026, 9, 16, 9, 40, tzinfo=UTC),
    )

    assert [name for name, _ in calls] == ["refit", "load"]
    assert calls[0][1] == ("weather",)


def test_the_issue_time_is_taken_after_the_refit(
    settings: Settings,
    tmp_path: Path,
    market: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = _local(settings, tmp_path)
    clock = {"now": datetime(2026, 9, 16, 9, 50, tzinfo=UTC)}

    class Clock(datetime):
        @classmethod
        def now(cls, tz: object = None) -> datetime:  # type: ignore[override]
            return clock["now"]

    def slow_refit(*args: object, **kwargs: object) -> RefitOutcome:
        clock["now"] = datetime(2026, 9, 16, 10, 5, tzinfo=UTC)  # 12:05 Berlin
        return RefitOutcome("refitted", "took a while", "3", "4")

    monkeypatch.setattr(dr, "datetime", Clock)
    monkeypatch.setattr(dr, "refresh_production_model", slow_refit)
    monkeypatch.setattr(dr, "load_model", lambda s, **k: (FakeModel(local), "4"))

    record = dr.run_day(local, LIVE_DAY, frame=market)

    assert record.issued_utc == datetime(2026, 9, 16, 10, 5, tzinfo=UTC)
    assert record.on_time is False
