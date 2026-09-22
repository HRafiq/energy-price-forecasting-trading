"""The scheduled runner: what it carries, and that it runs the same pipeline."""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.pipeline import state

REPO = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO / ".github" / "workflows" / "daily.yml"
WORKFLOW = WORKFLOW_PATH.read_text(encoding="utf-8")
PARSED = yaml.safe_load(WORKFLOW)
#: PyYAML reads the unquoted key ``on`` as the boolean True.
TRIGGERS = PARSED[True] if True in PARSED else PARSED["on"]
STEPS = PARSED["jobs"]["bid"]["steps"]


def _step(name: str) -> dict[str, Any]:
    step: dict[str, Any] = next(
        s for s in STEPS if str(s.get("name", "")).startswith(name)
    )
    return step


#: The 12:00 Berlin gate in UTC. Summer is the tight case: Berlin is UTC+2 then,
#: so the gate falls an hour earlier in UTC than it does in winter.
SUMMER_GATE_UTC_MINUTES = 10 * 60
#: What a run needs: the download, the fit or the chain, and the solve, with room
#: for the delay GitHub's scheduler adds under load.
NEEDED_MINUTES = 60


def test_every_run_has_time_to_bid_before_the_summer_gate() -> None:
    """The gate is 10:00 UTC in summer, not 11:00; a retry after it is useless."""
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    assert crons, "no schedule"
    for cron in crons:
        minute, hour, *rest = cron.split()
        assert rest == ["*", "*", "*"] and minute.isdigit() and hour.isdigit()
        starts = int(hour) * 60 + int(minute)
        slack = SUMMER_GATE_UTC_MINUTES - starts
        assert slack >= NEEDED_MINUTES, (
            f"cron {cron} leaves {slack} min before the summer gate"
        )


def test_a_second_run_is_a_free_retry() -> None:
    """A day with a committed schedule is refused, so the retry cannot rebid."""
    assert len(TRIGGERS["schedule"]) == 2
    assert "workflow_dispatch" in TRIGGERS
    assert PARSED["concurrency"]["cancel-in-progress"] is False


def test_it_shells_into_the_same_modules_the_dag_does() -> None:
    """Neither is the authority on what a run does, so they must not drift apart.

    Derived from the DAG rather than listed here, so a step added there and not
    here is a failure instead of something nobody notices.
    """
    dag = (REPO / "dags" / "daily_pipeline.py").read_text(encoding="utf-8")
    in_dag = set(re.findall(r"src\.(?:ingest|pipeline|export)\.[a-z_]+", dag))
    in_workflow = set(re.findall(r"src\.(?:ingest|pipeline|export)\.[a-z_]+", WORKFLOW))
    #: The dashboard is exported where the backtest artifacts are, not on a runner.
    only_local = {"src.export.artifacts"}

    assert in_dag - only_local <= in_workflow, (
        f"the DAG runs {sorted(in_dag - only_local - in_workflow)} and the "
        "workflow does not"
    )
    assert "src.export.artifacts" not in in_workflow


def test_the_settled_day_is_the_day_before_the_delivery_day() -> None:
    day = _step("Work out the delivery day")
    assert 'date -u -d "tomorrow"' in str(day["run"])
    assert 'date -u -d "$day -1 day"' in str(day["run"])
    settle = _step("Settle yesterday")
    assert "--day ${{ steps.day.outputs.settle }} --settle" in str(settle["run"])


def test_a_missing_feed_does_not_stop_the_bid() -> None:
    """The fallback chain exists for this; a lesser rung beats no bid."""
    assert _step("Check the feeds")["continue-on-error"] is True
    assert "continue-on-error" not in _step("Bid")


def test_the_state_is_saved_even_when_the_bid_failed() -> None:
    """A failed run still made incidents worth keeping."""
    for name in ("Settle yesterday", "Record the days", "Save the desk", "Commit the"):
        assert _step(name)["if"] == "always()"
    assert _step("Open an issue")["if"] == "failure()"


def test_the_state_branch_is_checked_out_and_pushed_back() -> None:
    checkout = _step("Check out the desk's state")
    assert checkout["with"]["ref"] == "live-state"
    assert checkout["with"]["path"] == ".live-state"
    commit = _step("Commit the state")
    assert commit["working-directory"] == ".live-state"
    # A concurrent push must not be overwritten.
    assert "--rebase" in str(commit["run"]) and "--force" not in str(commit["run"])


def test_the_workflow_needs_no_secrets() -> None:
    """The daily feeds are all keyless; ENTSOE_API_KEY belongs to a cross-check."""
    body = "\n".join(
        line for line in WORKFLOW.splitlines() if not line.lstrip().startswith("#")
    )
    assert "secrets." not in body and "ENTSOE" not in body
    assert "env:" not in body


def test_the_state_carries_the_desk_and_nothing_redistributable() -> None:
    """The fuel prices' source does not licence redistribution."""
    assert "data/processed/pipeline" in state.PATHS
    assert "mlflow.db" in state.PATHS and "mlruns" in state.PATHS
    for excluded in (
        "data/processed/fuels_daily.parquet",
        "data/processed/smard_quarterhour.parquet",
        "data/processed/model_inputs_quarterhour.parquet",
        "data/processed/weather_forecast_quarterhour.parquet",
        "data/processed/dashboard",
    ):
        assert excluded not in state.PATHS
        assert not any(excluded.startswith(f"{p}/") for p in state.PATHS)


def test_moving_state_replaces_what_is_there_and_skips_what_is_missing(
    tmp_path: Path,
) -> None:
    source, target = tmp_path / "from", tmp_path / "to"
    (source / "data" / "processed" / "pipeline" / "runs").mkdir(parents=True)
    (source / "data" / "processed" / "pipeline" / "runs" / "a.json").write_text("new")
    (source / "mlflow.db").write_text("fresh")
    stale = target / "data" / "processed" / "pipeline" / "runs"
    stale.mkdir(parents=True)
    (stale / "gone.json").write_text("stale")

    moved = state.move_state(source, target)

    assert moved == ["data/processed/pipeline", "mlflow.db"]
    assert (target / "data" / "processed" / "pipeline" / "runs" / "a.json").exists()
    assert not (stale / "gone.json").exists()
    assert (target / "mlflow.db").read_text() == "fresh"


def test_a_first_run_with_an_empty_store_is_not_an_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = tmp_path / "store"
    store.mkdir()
    assert state.main(["restore", "--store", str(store)]) == 0
    assert "nothing to move" in capsys.readouterr().out


def test_the_runner_sizes_its_download_instead_of_fetching_everything() -> None:
    """A fresh machine pays for every byte, on every run, twice a day."""
    sizing = _step("Work out how much history")
    assert "src.pipeline.history" in str(sizing["run"])
    fetch = _step("Rebuild the market data")
    assert "--history-days" in str(fetch["run"])
    for module in ("src.ingest.build_dataset", "src.ingest.open_meteo"):
        assert module in str(fetch["run"])
    assert str(fetch["run"]).count("--history-days") == 2


def test_the_state_is_staged_despite_the_pipeline_gitignore() -> None:
    """The repo ignores data/ and the registry; on this branch they are the point."""
    commit = _step("Commit the state")
    assert "git add -Af" in str(commit["run"])


def test_a_save_refuses_to_publish_a_machine_path(tmp_path: Path) -> None:
    """The branch is public, and a home directory has no business on it."""
    store = tmp_path / "store"
    (store / "data" / "processed" / "health").mkdir(parents=True)
    (store / "data" / "processed" / "health" / "incidents.jsonl").write_text(
        '{"detail": "no plan at /Users/someone/repo/data/x.parquet"}\n',
        encoding="utf-8",
    )

    assert "/Users/someone" in state.private_paths(store)


def test_a_store_with_nothing_private_passes(tmp_path: Path) -> None:
    store = tmp_path / "store"
    (store / "data" / "processed" / "pipeline").mkdir(parents=True)
    (store / "data" / "processed" / "pipeline" / "a.json").write_text(
        '{"target_day": "2026-09-22", "kind": "live"}\n', encoding="utf-8"
    )

    assert state.private_paths(store) == []


def test_a_move_interrupted_partway_does_not_empty_the_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The caller commits whatever it finds, so a half-move must not be found."""
    source, target = tmp_path / "from", tmp_path / "to"
    for root, name in ((source, "new.json"), (target, "old.json")):
        (root / "data" / "processed" / "pipeline").mkdir(parents=True)
        (root / "data" / "processed" / "pipeline" / name).write_text("{}")

    def _boom(*args: object, **kwargs: object) -> None:
        raise KeyboardInterrupt("cancelled midway")

    monkeypatch.setattr(shutil, "copytree", _boom)
    with pytest.raises(KeyboardInterrupt):
        state.move_state(source, target)

    kept = target / "data" / "processed" / "pipeline" / "old.json"
    assert kept.exists(), "the target was emptied before the copy finished"


def test_the_runner_uses_the_interpreter_the_model_was_pickled_under() -> None:
    """Across Python minor versions the load segfaults, which writes no incident."""
    pinned = (REPO / ".python-version").read_text(encoding="utf-8").strip()
    assert pinned, ".python-version must pin the interpreter"
    uv = _step("Install uv")
    assert uv["with"]["python-version-file"] == ".python-version"
    requires = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    # The pin must be inside what the project allows, or uv resolves elsewhere.
    assert f'>={pinned}' in requires
