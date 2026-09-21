"""The live desk's state, moved between a run and the branch that keeps it.

A scheduled runner is a fresh machine every time, so whatever the desk has to
remember must be carried in and out. Almost nothing is: the price dataset is
rebuilt from SMARD on every run, the weather from Open-Meteo and the fuel
prices from their own source, so none of that is state. What remains is what
the desk itself produced and could not reproduce:

* the run records, committed plans and settlements: the live record;
* the forecasts those runs saved, which the drift monitor scores;
* the incident log and the drift summary;
* the thresholds the M2 experiment fixed on validation, which the monitor reads;
* the model registry, so the version that has been serving keeps serving rather
  than a fresh one being fitted every day.

``PATHS`` is the whole of it, and it is deliberately a list rather than a rule
over the processed folder. Two things there must never be carried out to a
public branch: the rebuilt market data, which is large and reproducible, and
``fuels_daily.parquet``, whose source does not licence redistribution. Listing
what goes makes that a decision rather than an oversight.

Paths are relative to the repository root and may be a file or a folder; a path
that does not exist is skipped, so a first run carries out whatever it made.

Two things happen around the copy. The model registry records absolute paths,
so it is rewritten on the way out and on the way back in
(:mod:`src.pipeline.registry_paths`); without that the restored registry points
at the machine that logged it and the chain quietly fits a model of its own.
And a store on its way out is read for machine paths of any kind: the branch
that keeps it is public, and a home directory has no business on it. Finding one
stops the save rather than publishing it.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from pathlib import Path

from src.pipeline.registry_paths import machine_paths, make_portable, make_rooted

__all__ = [
    "PATHS",
    "REPO_ROOT",
    "PublishRefused",
    "main",
    "move_state",
    "private_paths",
]

#: Any absolute home directory, which must never reach a public branch.
_HOME = re.compile(r"/(?:Users|home)/[^/\s\"']+")
#: Files worth reading for a leak. The rest are model payloads and parquet.
_READABLE = (".json", ".jsonl", ".yaml", ".yml", ".txt", ".md", ".toml", ".lock", "")


class PublishRefused(RuntimeError):
    """The store holds a path belonging to the machine that wrote it."""


REPO_ROOT = Path(__file__).resolve().parents[2]

#: Everything the desk must remember, relative to the repository root.
PATHS: tuple[str, ...] = (
    "data/processed/pipeline",
    "data/processed/forecasts/production",
    "data/processed/health",
    "data/processed/experiments/m2_drift.json",
    "mlflow.db",
    "mlruns",
)


def move_state(source_root: Path, target_root: Path) -> list[str]:
    """Copy every path in ``PATHS`` from one root to the other, replacing it.

    Returns the paths that were there to copy.
    """
    moved: list[str] = []
    for name in PATHS:
        source = source_root / name
        if not source.exists():
            continue
        target = target_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        _replace(source, target)
        moved.append(name)
    return moved


def _drop(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _replace(source: Path, target: Path) -> None:
    """Put ``source`` at ``target``, without a moment where neither is there.

    Deleting the target and then copying would, if the run were cancelled in
    between, leave a store holding half of what it held before; the caller
    commits whatever it finds, so the branch would lose the rest. The copy is
    made beside the target and swapped in, and the old copy is only dropped once
    the new one is in place.
    """
    partial = target.parent / f".{target.name}.partial"
    previous = target.parent / f".{target.name}.previous"
    _drop(partial)
    _drop(previous)
    if source.is_dir():
        shutil.copytree(source, partial)
    else:
        shutil.copy2(source, partial)
    if target.exists() or target.is_symlink():
        os.replace(target, previous)
    os.replace(partial, target)
    _drop(previous)


def private_paths(store: Path) -> list[str]:
    """Machine paths anywhere in the store, so a save can refuse to publish one.

    The registry is read by :func:`~src.pipeline.registry_paths.machine_paths`,
    which understands its database; everything else is read as text. Parquet and
    the model payloads are left alone: they hold numbers, not paths.
    """
    found: set[str] = set(machine_paths(store))
    for name in PATHS:
        target = store / name
        files = (
            sorted(f for f in target.rglob("*") if f.is_file())
            if target.is_dir()
            else [target]
        )
        for path in files:
            if not path.exists() or path.suffix.lower() not in _READABLE:
                continue
            if store / "mlruns" in path.parents:
                continue  # already read, and read better, by machine_paths
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            found.update(match.group(0) for match in _HOME.finditer(text))
    return sorted(found)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Move the live desk's state between the repository and a store."
    )
    parser.add_argument(
        "direction",
        choices=("restore", "save"),
        help="restore: store into the repository, before a run. "
        "save: repository into the store, after one.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        required=True,
        help="checkout of the branch that keeps the state",
    )
    args = parser.parse_args(argv)

    store = args.store.resolve()
    if args.direction == "restore":
        source, target = store, REPO_ROOT
    else:
        source, target = REPO_ROOT, store
    if not store.exists():
        raise SystemExit(f"no state store at {store}")
    moved = move_state(source, target)
    if not moved:
        print(f"{args.direction}: nothing to move (a first run has no state yet)")
        return 0
    if args.direction == "restore":
        # The registry records absolute paths. Without this the restored store
        # points at the machine that logged the model, load_model fails, and the
        # chain fits one of its own and calls it the production model.
        rewritten = make_rooted(REPO_ROOT, REPO_ROOT)
    else:
        rewritten = make_portable(store, REPO_ROOT)
        leaks = private_paths(store)
        if leaks:
            raise PublishRefused(
                f"{len(leaks)} machine path(s) in the store, which is published: "
                + ", ".join(leaks[:5])
                + ". Nothing was left for the caller to commit."
            )
    print(
        f"{args.direction}: {', '.join(moved)} ({rewritten} registry path(s) rewritten)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
