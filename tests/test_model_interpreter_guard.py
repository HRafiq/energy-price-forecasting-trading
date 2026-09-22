"""Refusing a model pickled under another Python, before it can crash the process."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
import yaml
from mlflow.entities.model_registry import ModelVersion

from src.pipeline import model_source as ms


def _registered(source: str, version: str = "1") -> ModelVersion:
    """An MLflow ModelVersion carrying only what the guard reads."""
    return ModelVersion(
        name=ms.REGISTERED_NAME, version=version, creation_timestamp=0, source=source
    )


def _model_dir(tmp_path: Path, python_version: str | None) -> Path:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir(parents=True)
    flavors: dict[str, Any] = {"python_function": {"loader_module": "mlflow.pyfunc"}}
    if python_version is not None:
        flavors["python_function"]["python_version"] = python_version
    (artifacts / "MLmodel").write_text(yaml.safe_dump({"flavors": flavors}))
    return artifacts


def _running() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def test_a_model_from_another_python_is_refused_before_it_is_unpickled(
    tmp_path: Path,
) -> None:
    """Unpickling across minor versions segfaults; a segfault writes no incident."""
    other = "3.99.0" if not _running().startswith("3.99") else "3.98.0"
    artifacts = _model_dir(tmp_path, other)

    with pytest.raises(ms.ModelSourceError, match="pickled under Python 3.9"):
        ms._refuse_foreign_interpreter(_registered(str(artifacts)))


def test_the_same_minor_version_is_allowed_through(tmp_path: Path) -> None:
    """A patch release is not a pickle boundary."""
    artifacts = _model_dir(tmp_path, f"{_running()}.99")

    ms._refuse_foreign_interpreter(_registered(str(artifacts)))


def test_a_file_uri_source_is_read_too(tmp_path: Path) -> None:
    artifacts = _model_dir(tmp_path, "3.99.0")

    with pytest.raises(ms.ModelSourceError):
        ms._refuse_foreign_interpreter(_registered(artifacts.as_uri()))


def test_a_model_that_records_nothing_is_left_to_the_loader(tmp_path: Path) -> None:
    """Refusing on missing metadata would refuse every older model."""
    ms._refuse_foreign_interpreter(_registered(str(_model_dir(tmp_path, None))))
    ms._refuse_foreign_interpreter(_registered(str(tmp_path / "absent")))


def test_an_unreadable_descriptor_is_not_fatal(tmp_path: Path) -> None:
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "MLmodel").write_text("{ not: valid: yaml: [")

    ms._refuse_foreign_interpreter(_registered(str(artifacts)))


def test_the_registered_model_matches_this_interpreter(settings: object) -> None:
    """The repository's own model, against the interpreter the project pins."""
    pinned = Path(__file__).resolve().parents[1] / ".python-version"
    assert pinned.read_text(encoding="utf-8").strip() == _running()
