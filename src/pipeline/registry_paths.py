"""The model registry, made independent of the machine that wrote it.

MLflow records where an artifact lives as an absolute path, in the tracking
database and again inside each model's ``MLmodel`` file. That is fine while the
store never moves. It is not fine here, for two reasons:

* a scheduled runner restores the store under its own root, and MLflow would
  still look for the artifacts under the root of the laptop that logged them,
  so the registered model cannot be loaded and the chain quietly fits one of
  its own instead;
* the store is kept on a branch of a public repository, and those paths carry
  the home directory of whoever logged the model.

So a stored copy holds :data:`TOKEN` where the repository root belongs, and the
root is written back in when the store is restored. Nothing else is touched:
the token stands in for the root only where a path points into ``mlruns``.

:func:`machine_paths` is the check that makes this safe rather than hopeful. It
reads a store and reports every absolute path still in it, so a caller can
refuse to publish one instead of finding out later.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from pathlib import Path

__all__ = [
    "ARTIFACT_DIR",
    "DB_NAME",
    "TOKEN",
    "machine_paths",
    "main",
    "make_portable",
    "make_rooted",
]

#: Stands in for the repository root in a stored copy of the registry.
TOKEN = "/__repo_root__"
DB_NAME = "mlflow.db"
ARTIFACT_DIR = "mlruns"
#: Files under ``mlruns`` that record a path. The model payloads do not.
TEXT_SUFFIXES = (".yaml", ".yml", ".json", ".txt", ".toml", ".lock", "")
#: An absolute path, with or without a file URI, pointing into the artifact dir.
_POINTS_AT_ARTIFACTS = re.compile(
    rf"(?<![\w/])(/[^\s\"']*?)/{ARTIFACT_DIR}(?=[/\s\"']|$)"
)
#: A home directory, which is what must never reach a public branch.
_HOME = re.compile(r"/(?:Users|home)/[^/\s\"']+")


def _text_files(base: Path) -> list[Path]:
    artifacts = base / ARTIFACT_DIR
    if not artifacts.exists():
        return []
    return [
        path
        for path in sorted(artifacts.rglob("*"))
        if path.is_file() and path.suffix.lower() in TEXT_SUFFIXES
    ]


def _swap_in_db(db: Path, before: str, after: str) -> int:
    """Replace ``before`` with ``after`` in every text value, return how many."""
    if not db.exists():
        return 0
    changed = 0
    with sqlite3.connect(db) as connection:
        tables = [
            row[0]
            for row in connection.execute(
                "select name from sqlite_master where type='table'"
            )
        ]
        for table in tables:
            columns = [
                row[1] for row in connection.execute(f"pragma table_info({table})")
            ]
            for column in columns:
                quoted = f'"{table}"."{column}"'
                try:
                    cursor = connection.execute(
                        f'update "{table}" set "{column}" = replace('
                        f"cast({quoted} as text), ?, ?) "
                        f"where cast({quoted} as text) like ?",
                        (before, after, f"%{before}%"),
                    )
                except sqlite3.OperationalError:
                    continue  # a view, or a column that cannot be updated
                changed += cursor.rowcount if cursor.rowcount > 0 else 0
        connection.commit()
    return changed


def _swap_in_files(base: Path, before: str, after: str) -> int:
    changed = 0
    for path in _text_files(base):
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if before not in text:
            continue
        path.write_text(text.replace(before, after), encoding="utf-8")
        changed += 1
    return changed


def make_portable(base: Path, root: Path) -> int:
    """Replace ``root`` with :data:`TOKEN` throughout the store at ``base``."""
    return _swap_in_db(base / DB_NAME, str(root), TOKEN) + _swap_in_files(
        base, str(root), TOKEN
    )


def make_rooted(base: Path, root: Path) -> int:
    """Write ``root`` back in place of :data:`TOKEN` throughout ``base``."""
    return _swap_in_db(base / DB_NAME, TOKEN, str(root)) + _swap_in_files(
        base, TOKEN, str(root)
    )


def machine_paths(base: Path) -> list[str]:
    """Absolute paths still in the store at ``base``, home directories first.

    A store about to be published must have none. The list is deduplicated and
    each entry is the path as it appears, so a caller can say what it found.
    """
    found: set[str] = set()
    db = base / DB_NAME
    if db.exists():
        with sqlite3.connect(db) as connection:
            tables = [
                row[0]
                for row in connection.execute(
                    "select name from sqlite_master where type='table'"
                )
            ]
            for table in tables:
                columns = [
                    row[1] for row in connection.execute(f"pragma table_info({table})")
                ]
                for column in columns:
                    try:
                        rows = connection.execute(
                            f'select distinct cast("{column}" as text) from "{table}" '
                            f'where cast("{column}" as text) like ?',
                            ("%/%",),
                        )
                    except sqlite3.OperationalError:
                        continue
                    for (value,) in rows:
                        found.update(_absolute_in(str(value)))
    for path in _text_files(base):
        try:
            found.update(_absolute_in(path.read_text(encoding="utf-8")))
        except (UnicodeDecodeError, OSError):
            continue
    return sorted(found, key=lambda p: (not _HOME.match(p), p))


def _absolute_in(text: str) -> set[str]:
    # A file URI leaves the token behind extra slashes ("file:///__repo_root__"),
    # so compare on the collapsed form rather than reporting it as a machine path.
    hits = {
        re.sub(r"^/+", "/", match.group(1))
        for match in _POINTS_AT_ARTIFACTS.finditer(text)
    }
    hits |= {match.group(0) for match in _HOME.finditer(text)}
    return {hit for hit in hits if hit and hit != TOKEN}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Make a model registry independent of the machine that wrote it."
    )
    parser.add_argument("action", choices=("portable", "rooted", "check"))
    parser.add_argument(
        "--base", type=Path, required=True, help="folder holding the store"
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="repository root, for portable and rooted",
    )
    args = parser.parse_args(argv)

    if args.action == "check":
        leaks = machine_paths(args.base)
        if leaks:
            print(f"{len(leaks)} machine path(s) in {args.base}:")
            for leak in leaks[:10]:
                print(f"  {leak}")
            return 1
        print(f"no machine paths in {args.base}")
        return 0
    if args.root is None:
        raise SystemExit(f"--root is required for {args.action}")
    changed = (
        make_portable(args.base, args.root)
        if args.action == "portable"
        else make_rooted(args.base, args.root)
    )
    print(f"{args.action}: {changed} value(s) rewritten in {args.base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
