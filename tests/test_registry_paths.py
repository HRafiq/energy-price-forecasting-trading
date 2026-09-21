"""A model registry that survives being moved, and never carries a home path."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.pipeline import registry_paths as rp

LAPTOP = "/Users/someone/work/repo"
RUNNER = "/home/runner/work/repo/repo"


def _store(base: Path, root: str = LAPTOP) -> Path:
    """A registry shaped like MLflow's, with ``root`` written into it."""
    base.mkdir(parents=True, exist_ok=True)
    artifacts = base / rp.ARTIFACT_DIR / "models" / "m-1" / "artifacts"
    artifacts.mkdir(parents=True)
    (artifacts / "MLmodel").write_text(
        f"artifact_path: file://{root}/mlruns/models/m-1/artifacts\n", encoding="utf-8"
    )
    (artifacts / "python_model.pkl").write_bytes(b"\x80\x04 not text")
    with sqlite3.connect(base / rp.DB_NAME) as db:
        db.execute("create table model_versions (name text, storage_location text)")
        db.execute("create table experiments (artifact_location text)")
        db.execute(
            "insert into model_versions values (?, ?)",
            ("price-quantile-gbm", f"file://{root}/mlruns/models/m-1/artifacts"),
        )
        db.execute("insert into experiments values (?)", (f"{root}/mlruns",))
        db.commit()
    return base


def _values(store: Path) -> list[str]:
    """Every path the store records, from the database and the text files.

    Compared instead of the raw file: SQLite rewrites its header on any write,
    so identical content does not mean identical bytes.
    """
    with sqlite3.connect(store / rp.DB_NAME) as db:
        rows = [
            str(v) for (v,) in db.execute("select storage_location from model_versions")
        ]
        rows += [
            str(v) for (v,) in db.execute("select artifact_location from experiments")
        ]
    rows += [
        path.read_text(encoding="utf-8")
        for path in sorted((store / rp.ARTIFACT_DIR).rglob("MLmodel"))
    ]
    return rows


def test_a_store_made_portable_holds_no_machine_path(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    assert rp.machine_paths(store)

    rp.make_portable(store, Path(LAPTOP))

    assert rp.machine_paths(store) == []
    assert rp.TOKEN in (store / rp.DB_NAME).read_bytes().decode("utf-8", "ignore")


def test_rooting_it_somewhere_else_points_at_that_somewhere(tmp_path: Path) -> None:
    """The failure this prevents: MLflow looking under the laptop's root."""
    store = _store(tmp_path / "store")
    rp.make_portable(store, Path(LAPTOP))

    rp.make_rooted(store, Path(RUNNER))

    with sqlite3.connect(store / rp.DB_NAME) as db:
        (location,) = next(db.execute("select storage_location from model_versions"))
    assert location == f"file://{RUNNER}/mlruns/models/m-1/artifacts"
    assert LAPTOP not in location
    text = (
        store / rp.ARTIFACT_DIR / "models" / "m-1" / "artifacts" / "MLmodel"
    ).read_text()
    assert RUNNER in text and LAPTOP not in text


def test_the_round_trip_returns_the_store_it_started_from(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    before = _values(store)

    rp.make_portable(store, Path(LAPTOP))
    rp.make_rooted(store, Path(LAPTOP))

    assert _values(store) == before


def test_making_it_portable_twice_changes_nothing_the_second_time(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path / "store")
    rp.make_portable(store, Path(LAPTOP))
    once = _values(store)

    rp.make_portable(store, Path(LAPTOP))

    assert _values(store) == once


def test_a_home_path_anywhere_in_the_store_is_reported(tmp_path: Path) -> None:
    """Not only artifact locations: any home directory is a leak."""
    store = _store(tmp_path / "store")
    rp.make_portable(store, Path(LAPTOP))
    stray = store / rp.ARTIFACT_DIR / "models" / "m-1" / "artifacts" / "conda.yaml"
    stray.write_text("prefix: /Users/someone/miniconda3/envs/x\n", encoding="utf-8")

    assert "/Users/someone" in rp.machine_paths(store)


def test_a_binary_payload_is_left_alone(tmp_path: Path) -> None:
    store = _store(tmp_path / "store")
    payload = (
        store / rp.ARTIFACT_DIR / "models" / "m-1" / "artifacts" / "python_model.pkl"
    )
    before = payload.read_bytes()

    rp.make_portable(store, Path(LAPTOP))

    assert payload.read_bytes() == before


def test_the_check_command_fails_when_a_path_is_still_there(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    store = _store(tmp_path / "store")

    assert rp.main(["check", "--base", str(store)]) == 1
    assert "machine path" in capsys.readouterr().out

    rp.make_portable(store, Path(LAPTOP))
    assert rp.main(["check", "--base", str(store)]) == 0


def test_an_empty_store_is_not_a_leak(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    assert rp.machine_paths(empty) == []
    assert rp.make_portable(empty, Path(LAPTOP)) == 0
