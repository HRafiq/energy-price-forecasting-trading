"""A time limit on one pipeline step, so a hung model cannot hold the gate.

A model step can hang inside compiled code, a LightGBM fit for instance, where a
Python signal is never delivered. So the step runs on a daemon thread and the
pipeline waits for it at most ``seconds``. A step that has not finished by then is
abandoned, not killed: its thread keeps running in the background until the
process exits, which ends daemon threads, and nothing it returns afterwards is
used. The pipeline moves on to the next rung of the chain as it would after an
error, so the cost of a hang is ``seconds`` of the time before the gate.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import TypeVar

from src.forecasting.base import ForecastError

__all__ = ["StepTimeoutError", "run_with_time_limit"]

T = TypeVar("T")


class StepTimeoutError(ForecastError):
    """A step ran past its time limit and was abandoned."""


def run_with_time_limit(step: Callable[[], T], seconds: float, label: str) -> T:
    """``step()``, or ``StepTimeoutError`` if it has not returned within ``seconds``.

    An exception raised by the step is raised here, in the caller's thread.
    """
    if seconds <= 0:
        raise ValueError("a time limit must be positive")
    outcome: dict[str, T] = {}
    failure: list[BaseException] = []

    def target() -> None:
        try:
            outcome["value"] = step()
        except BaseException as exc:  # handed back to the caller below
            failure.append(exc)

    worker = threading.Thread(target=target, name=f"step-{label}", daemon=True)
    worker.start()
    worker.join(seconds)
    if worker.is_alive():
        raise StepTimeoutError(f"{label} still running after {seconds:g} s; abandoned")
    if failure:
        raise failure[0]
    return outcome["value"]
