"""The fallback chain: the best forecast the pipeline can still make today.

The gate closes at 12:00 whether or not every feed arrived, so the pipeline walks
down a chain of forecasters and submits the first one it can run:

1. ``production``: the configured model with every feature group;
2. ``no_weather``: the same model without the weather group, for a late weather
   feed. On validation it cost 1.2 capture points against 2.3 for running the
   full model on empty weather inputs (D1);
3. ``seasonal_naive``: last week's prices, which need no feed published today;
4. ``naive``: yesterday's prices.

Seasonal naive comes before naive previous day because it earned more on the
validation window, 79.7% of perfect foresight against 77.6%, although its pinball
loss is worse. This project judges a forecast by the money it makes. The choice is
recorded in the Phase 7 decisions entry; the D5 experiment keeps the order it was
frozen with, and its published numbers stand.

Each step declares the feeds it needs, so a missing feed removes it from the chain
before anything is fitted. A step that raises is recorded and the chain moves on.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

import pandas as pd

from src.config import Settings
from src.features.build import FEATURE_GROUPS
from src.forecasting.base import Forecaster, ForecastError, QuantileForecast
from src.forecasting.baselines import build_baselines
from src.forecasting.information import build_information_set
from src.forecasting.models.common import feature_names
from src.forecasting.production import build_production_model
from src.pipeline.readiness import Readiness

__all__ = [
    "STEPS",
    "Attempt",
    "ChainResult",
    "ChainStep",
    "build_step",
    "run_chain",
    "steps_for",
]


def _no_weather_model(settings: Settings) -> Forecaster:
    """The production model without the weather group, the first fallback."""
    from src.forecasting.models.gradient_boosting import LightGBMConformalModel

    groups = [group for group in FEATURE_GROUPS if group != "weather"]
    return LightGBMConformalModel(
        settings,
        name="lightgbm_conformal_no_weather",
        _names=feature_names(groups),
    )


@dataclass(frozen=True)
class ChainStep:
    """One rung of the chain: what it needs, and how to build it."""

    name: str
    label: str
    needs: frozenset[str]
    build: Callable[[Settings], Forecaster]


STEPS: tuple[ChainStep, ...] = (
    ChainStep(
        "production",
        "production model",
        frozenset({"prices", "load_forecast", "weather"}),
        build_production_model,
    ),
    ChainStep(
        "no_weather",
        "model without weather",
        frozenset({"prices", "load_forecast"}),
        _no_weather_model,
    ),
    ChainStep(
        "seasonal_naive",
        "seasonal naive previous week",
        frozenset(),
        lambda settings: build_baselines(settings)[1],
    ),
    ChainStep(
        "naive",
        "naive previous day",
        frozenset({"prices"}),
        lambda settings: build_baselines(settings)[0],
    ),
)


@dataclass(frozen=True)
class Attempt:
    """What happened on one rung."""

    step: str
    used: bool
    detail: str
    seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "used": self.used,
            "detail": self.detail,
            "seconds": round(self.seconds, 3),
        }


@dataclass(frozen=True)
class ChainResult:
    """The forecast the pipeline will submit, and how it got there."""

    step: str
    forecast: QuantileForecast
    attempts: tuple[Attempt, ...]

    @property
    def degraded(self) -> bool:
        return self.step != STEPS[0].name

    @property
    def label(self) -> str:
        return next(s.label for s in STEPS if s.name == self.step)

    @property
    def seconds(self) -> float:
        return sum(attempt.seconds for attempt in self.attempts)


def build_step(name: str, settings: Settings) -> Forecaster:
    """A fresh forecaster for one rung of the chain."""
    for step in STEPS:
        if step.name == name:
            return step.build(settings)
    raise ValueError(f"unknown chain step {name!r}")


def steps_for(readiness: Readiness) -> tuple[ChainStep, ...]:
    """The rungs whose feeds all arrived, in chain order."""
    missing = set(readiness.missing)
    return tuple(step for step in STEPS if not (step.needs & missing))


def run_chain(
    frame: pd.DataFrame,
    target_day: date,
    settings: Settings,
    readiness: Readiness,
    *,
    model: Forecaster | None = None,
) -> ChainResult:
    """Walk the chain and return the first forecast that succeeds.

    ``model`` replaces the production rung with an already-fitted forecaster, which
    is how the pipeline serves a model from the registry. It is never refitted.
    """
    missing = set(readiness.missing)
    attempts: list[Attempt] = []
    for step in STEPS:
        blocked = step.needs & missing
        if blocked:
            attempts.append(
                Attempt(
                    step.name,
                    False,
                    f"skipped, waiting on {', '.join(sorted(blocked))}",
                )
            )
            continue
        started = time.perf_counter()
        try:
            # A model handed in by the pipeline comes from the registry and is
            # already fitted; only a model this chain builds needs training.
            forecaster: Forecaster
            if model is not None and step is STEPS[0]:
                forecaster = model
            else:
                forecaster = step.build(settings)
                forecaster.fit(
                    build_information_set(
                        frame, target_day, settings, forecaster.fit_lookback_days
                    )
                )
            forecast = forecaster.forecast(
                build_information_set(
                    frame, target_day, settings, forecaster.lookback_days
                )
            )
        except (ForecastError, ValueError) as exc:
            attempts.append(
                Attempt(
                    step.name,
                    False,
                    f"failed: {exc}",
                    time.perf_counter() - started,
                )
            )
            continue
        attempts.append(
            Attempt(step.name, True, "forecast issued", time.perf_counter() - started)
        )
        return ChainResult(step.name, forecast, tuple(attempts))
    raise ForecastError(
        f"no chain step could forecast {target_day}: "
        + "; ".join(f"{a.step} {a.detail}" for a in attempts)
    )
