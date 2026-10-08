"""The scheduled runner: what it carries, and that it runs the same pipeline."""

from __future__ import annotations

import re
import shutil
from itertools import pairwise
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
#: What a run needs before the gate. GitHub does not fire these on time: delays
#: measured on this repository ran from 4h 14m to 5h 23m, and then drifted to
#: 7h 39m and cost three bids. The schedule no longer leans on one early cron
#: absorbing that, so this is a floor on the earliest attempt rather than the
#: whole defence.
NEEDED_MINUTES = 5 * 60
#: The worst delay the schedule is built to survive, past which nothing a cron
#: can do helps: a cron early enough to cover it would fire the evening before,
#: and an evening run bids for a day whose gate shut that morning.
WORST_DELAY_MINUTES = 8 * 60


def test_the_first_attempt_can_absorb_a_long_scheduler_delay() -> None:
    """GitHub fires these hours late, and all the attempts shift together.

    A delay moves every attempt by roughly the same amount, so the earliest has
    to be early enough that it still lands inside the gate on its own, and no
    attempt may be scheduled after the gate it is bidding into.
    """
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    assert crons, "no schedule"
    starts = []
    for cron in crons:
        minute, hour, *rest = cron.split()
        assert rest == ["*", "*", "*"] and minute.isdigit() and hour.isdigit()
        starts.append(int(hour) * 60 + int(minute))
    assert min(starts) <= SUMMER_GATE_UTC_MINUTES - NEEDED_MINUTES, (
        f"the earliest attempt starts {SUMMER_GATE_UTC_MINUTES - min(starts)} min "
        "before the summer gate, too late to absorb a five-hour delay"
    )
    for start in starts:
        assert start < SUMMER_GATE_UTC_MINUTES, "an attempt starts after the gate"


def test_more_than_one_scheduled_attempt_because_a_schedule_can_be_dropped() -> None:
    """A run that never starts reports nothing: not an issue, not an incident.

    GitHub delays and drops scheduled runs, and this workflow's own first two
    were never started at all. Repetition is the only defence available from
    inside the workflow, and it is safe because a day with a committed schedule
    is refused, so the later attempts cannot bid twice.
    """
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    assert len(crons) >= 3, "one dropped run should not cost the gate"
    assert len(set(crons)) == len(crons), "attempts must be at different times"
    assert "workflow_dispatch" in TRIGGERS
    # Serialised, not cancelled: a later attempt must not kill one mid-bid.
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
    """The chain exists for this; a lesser rung beats no bid, once waiting fails."""
    assert "continue-on-error" not in _step("Bid")
    # The wait ends at its cutoff rather than failing the run, so a feed that
    # never arrives still produces a committed schedule on a lower rung.
    assert "break" in str(_step("Wait for the feeds")["run"])


def test_the_state_is_saved_even_when_the_bid_failed() -> None:
    """A failed run still made incidents worth keeping."""
    for name in ("Settle yesterday", "Record the days", "Save the desk", "Commit the"):
        # always(), so a failed bid still saves what the run learned. The only
        # other condition allowed is the early exit, and a run that stopped there
        # has nothing to save.
        clause = str(_step(name)["if"])
        assert clause.startswith("always()"), name
        assert clause.replace("always()", "").strip(" &") in (
            "",
            "steps.bid_already.outputs.done != 'true'",
        ), name
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
    """Across Python minor versions the load segfaults, which writes no incident.

    setup-uv has no python-version-file input. Passing one is accepted with a
    warning and then ignored, which is how this pin was silently doing nothing
    while appearing to be set, so the version is read in a step and handed to
    the action explicitly.
    """
    pinned = (REPO / ".python-version").read_text(encoding="utf-8").strip()
    assert pinned, ".python-version must pin the interpreter"
    uv = _step("Install uv")
    assert "python-version-file" not in uv["with"], "that input does not exist"
    assert uv["with"]["python-version"] == "${{ steps.python.outputs.version }}"
    reader = _step("Read the pinned interpreter")
    assert ".python-version" in str(reader["run"])
    requires = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    # The pin must be inside what the project allows, or uv resolves elsewhere.
    assert f">={pinned}" in requires


def _wait_thresholds() -> tuple[int, int]:
    """The two thresholds the wait step is built on, in minutes before the gate."""
    run = str(_step("Wait for the feeds")["run"])
    too_early = re.search(r"too_early=\$\(\(\s*(\d+)\s*\*\s*60\s*\)\)", run)
    cutoff = re.search(r"cutoff=\$\(\(\s*(\d+)\s*\)\)", run)
    assert too_early and cutoff, "the wait needs both a stand-down and a cutoff"
    return int(too_early.group(1)) * 60, int(cutoff.group(1))


def test_the_run_waits_for_the_feeds_instead_of_bidding_without_them() -> None:
    """An early start must not cost a fallback bid, which is what it cost before."""
    wait = _step("Wait for the feeds")
    run = str(wait["run"])
    assert "src.pipeline.readiness" in run
    # Each poke rebuilds, or a feed published since the last one stays invisible.
    assert "src.ingest.build_dataset" in run and "src.ingest.build_inputs" in run
    _, cutoff = _wait_thresholds()
    # It measures against the delivery day's own gate as an instant, not the
    # clock, so a delayed run that has watched the date roll over still counts
    # from the right moment. And it gives up long enough before that gate for a
    # bid to fit in what is left: a daily_run takes minutes, so a cutoff of one
    # or two would miss the gate anyway.
    assert "gate_day" in run and "+%s" in run, (
        "the gate must be an instant, not a clock"
    )
    assert cutoff >= 15, "the cutoff leaves too little time to bid"
    assert "continue-on-error" not in wait


def test_a_run_that_lands_far_from_the_gate_stands_down_instead_of_bidding() -> None:
    """The feeds do not exist yet, so polling for them would burn the job limit."""
    run = str(_step("Wait for the feeds")["run"])
    too_early, cutoff = _wait_thresholds()
    assert too_early > cutoff, "the polling window would be empty"
    assert "proceed=false" in run and "proceed=true" in run
    # Bounded by construction: polling starts no earlier than too_early before
    # the gate and ends at the cutoff, so it cannot approach the six hours at
    # which a hosted job is killed whatever timeout-minutes claims.
    assert too_early - cutoff < 6 * 60


def test_only_a_run_that_bid_goes_on_to_bid_or_to_judge_the_deadline() -> None:
    """A stand-down is not a missed day, and must not be recorded as one."""
    assert "steps.wait.outputs.proceed == 'true'" in str(_step("Bid")["if"])
    deadline = str(_step("Check the deadline")["if"])
    assert "steps.wait.outputs.proceed == 'true'" in deadline
    # Still always(), so a failed bid is still judged.
    assert "always()" in deadline


def test_the_attempts_are_close_enough_that_no_delay_slips_between_them() -> None:
    """One early cron cannot cover a delay that moves; density is what covers it.

    A run is useful if it lands inside the window between the stand-down and
    the cutoff. Whatever the delay shifts the whole schedule by, some attempt
    has to land in that window, which bounds how far apart they may sit.
    """
    too_early, cutoff = _wait_thresholds()
    window = too_early - cutoff
    starts = sorted(
        int(h) * 60 + int(m)
        for m, h, *_ in (entry["cron"].split() for entry in TRIGGERS["schedule"])
    )
    gaps = [b - a for a, b in pairwise(starts)]
    assert max(gaps) <= window, (
        f"a {max(gaps)} min gap in a {window} min window lets a delay slip through"
    )
    # And the earliest must be early enough that the worst delay still lands it
    # inside the window rather than past the far edge.
    assert starts[0] <= SUMMER_GATE_UTC_MINUTES - cutoff - WORST_DELAY_MINUTES


def test_the_attempts_avoid_the_most_contended_minutes() -> None:
    """The top and the half of the hour are where every cron on GitHub piles up."""
    minutes = [int(entry["cron"].split()[0]) for entry in TRIGGERS["schedule"]]
    assert not {0, 30} & set(minutes), "scheduled on a contended minute"


def test_the_wait_resolves_the_real_gate_rather_than_pinning_a_summer_constant() -> (
    None
):
    """Berlin is UTC+2 in summer and UTC+1 in winter, and the desk trades both.

    Pinning the summer value makes the cutoff fire an hour early all winter,
    giving up on a load forecast that still had an hour to arrive. The DAG and
    daily_run both resolve 12:00 Europe/Berlin, and this has to agree with them.
    """
    run = str(_step("Wait for the feeds")["run"])
    assert "TZ=Europe/Berlin" in run, "the gate must be resolved in Berlin time"
    assert "12:00" in run, "the gate is 12:00 local, not a UTC constant"
    assert "-1 day" in run, "the gate falls the day before the delivery day"
    assert '"$gate_day 10:00"' not in run, "that is the summer gate, wrong all winter"


def test_a_slow_rebuild_cannot_carry_the_run_past_the_gate_and_still_bid() -> None:
    """The clock was read once a lap, so one bad poke could overshoot the cutoff."""
    run = str(_step("Wait for the feeds")["run"])
    assert run.count("timeout ") >= 2, "the in-loop rebuilds must be capped"
    _, after = run.split("sleep 300", 1)
    assert "<= cutoff" in after, "the cutoff must be re-checked after a rebuild"


def test_a_broken_readiness_is_loud_rather_than_mistaken_for_a_late_feed() -> None:
    """Exit 1 is 'not ready'. A crash must not look like a feed that never came."""
    run = str(_step("Wait for the feeds")["run"])
    assert "ready != 1" in run, "every non-zero exit is being read as not ready"
    assert 'exit "$ready"' in run, "a broken readiness must fail the step"


def test_the_schedule_is_the_one_the_readme_and_the_coverage_argument_describe() -> (
    None
):
    crons = [entry["cron"] for entry in TRIGGERS["schedule"]]
    assert len(crons) == 8, "the README and the coverage argument both say eight"


def test_the_concurrency_group_is_fixed_so_two_runs_can_never_bid_at_once() -> None:
    """A per-run group would let eight attempts race for one committed schedule."""
    assert "${{" not in str(PARSED["concurrency"]["group"])


def test_the_poll_refetches_the_feeds_that_can_arrive_while_it_waits() -> None:
    """Rebuilding is not fetching, and a feed nobody fetches again never arrives.

    The loop rebuilt the dataset every five minutes but never re-ran the weather
    fetch, so a run that caught the weather four quarter-hours short rebuilt that
    same shortfall for three hours and bid a fallback rung. The shortfall is this
    repository's own doing: open_meteo masks anything stamped past as_of plus the
    lead minus the archive lag, and as_of only moves when the fetch is repeated.

    Fuels is deliberately not in here. It is fetched with an exclusive end of
    today, so within one day it can only ever return the same settlement, and it
    writes its frame wholesale rather than merging, so a degraded response would
    shrink it rather than add to it.
    """
    polled = set(
        re.findall(r"src\.ingest\.\w+", str(_step("Wait for the feeds")["run"]))
    )
    assert "src.ingest.open_meteo" in polled, (
        "the poll waits on weather without ever fetching it again"
    )
    assert "src.ingest.build_dataset" in polled and "src.ingest.build_inputs" in polled
    assert "src.ingest.fuels" not in polled, (
        "fuels cannot change within a day and overwrites rather than merges"
    )


def test_one_poll_lap_fits_inside_the_cutoff_it_is_racing() -> None:
    """A lap that overruns the cutoff bids past the gate and logs that it did not.

    The clock is read at the top of the lap, so a lap beginning one minute inside
    the cutoff must still end before the gate.
    """
    run = str(_step("Wait for the feeds")["run"])
    _, cutoff = _wait_thresholds()
    body = run.split("sleep 300", 1)[1]
    commands = [
        line.strip() for line in body.splitlines() if "uv run python -m" in line
    ]
    assert commands, "no commands found after the sleep"
    caps = [int(m) for m in re.findall(r"timeout (\d+)", run)]
    assert len(caps) == len(commands), f"an uncapped command in the poll: {commands}"
    lap = 300 + sum(caps)
    assert lap <= cutoff * 60, (
        f"a {lap // 60} min lap can start {cutoff} min out and end past the gate"
    )


def test_a_day_already_bid_on_time_stops_before_anything_expensive() -> None:
    """Eight attempts, one of which has work to do; the rest must be cheap.

    Each attempt used to rebuild the whole market dataset before it could learn
    that the day was already traded, so five runs a day did ten minutes of work
    for nothing and committed the state again at the end of it.
    """
    check = _step("Stop here if this day is already bid")
    names = [str(s.get("name", "")) for s in STEPS]
    # Before the toolchain, not merely before the rebuild: installing uv and
    # syncing the project is most of what a skipped run would otherwise cost.
    for later in ("Install uv", "Install the project", "Rebuild the market data"):
        assert names.index(check["name"]) < names.index(later), later
    # It reads the state branch's own checkout, which is why it can run this early.
    assert ".live-state/" in str(check["run"])
    # And only a live bid that made its gate counts: a reconstruction leaves the
    # day unbid, and a late bid is still a day worth a record.
    run = str(check["run"])
    assert "'live'" in run and "on_time" in run


def test_everything_costly_is_skipped_once_the_day_is_bid() -> None:
    gate = "steps.bid_already.outputs.done != 'true'"
    for name in (
        "Read the pinned interpreter",
        "Install uv",
        "Install the project",
        "Restore the desk's state",
        "Work out how much history this run needs",
        "Rebuild the market data",
        "Wait for the feeds, until the cutoff or the stand-down",
        "Bid",
        "Settle yesterday",
        "Record the days with no bid",
        "Run the drift monitor",
        "Save the desk's state",
        "Commit the state",
    ):
        assert gate in str(_step(name).get("if", "")), f"{name} still runs for nothing"
    # The issue step must stay unconditional on failure, or a real break goes quiet.
    assert _step("Open an issue if the desk failed")["if"] == "failure()"
