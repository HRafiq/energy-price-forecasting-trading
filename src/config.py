"""Typed loader for ``config/settings.yaml``.

Config is validated once, at load time, so a typo in a threshold or a SMARD
series id fails immediately instead of silently changing a backtest three
phases later.
"""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.trading.battery import Battery

__all__ = [
    "CARBON_COLUMN",
    "CONFIG_DIR",
    "DEFAULT_SETTINGS_PATH",
    "DERIVED_COLUMNS",
    "FUEL_COLUMNS",
    "GAS_COLUMN",
    "PRICE_SERIES",
    "PRODUCT_COLUMN",
    "REPO_ROOT",
    "RESOLUTION_STEP",
    "AvailabilityConfig",
    "AvailabilityRule",
    "BaselinesConfig",
    "DataConfig",
    "EvaluationConfig",
    "ForecastingConfig",
    "FuelsConfig",
    "MarketConfig",
    "ProductionModel",
    "Resolution",
    "Settings",
    "SmardConfig",
    "TradingConfig",
    "WeatherConfig",
    "load_settings",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.yaml"

Resolution = Literal["hour", "quarterhour"]
AvailabilityRule = Literal[
    "before_target_day", "through_target_day", "before_issue_lag"
]
#: Models that can run on their own as the production forecaster.
ProductionModel = Literal[
    "lightgbm_conformal",
    "lightgbm_quantile",
    "quantile_forest",
    "lear",
    "mstl",
    "naive_previous_day",
    "seasonal_naive_previous_week",
]

RESOLUTION_STEP: dict[str, pd.Timedelta] = {
    "hour": pd.Timedelta(hours=1),
    "quarterhour": pd.Timedelta(minutes=15),
}

PRICE_SERIES = "price_eur_mwh"
PRODUCT_COLUMN = "price_product_minutes"
#: Columns the dataset builder adds to the downloaded SMARD series.
DERIVED_COLUMNS = (
    "residual_load_actual_mw",
    "residual_load_forecast_mw",
    PRODUCT_COLUMN,
)
GAS_COLUMN = "gas_ttf_eur_mwh"
CARBON_COLUMN = "carbon_eua_eur_t"
FUEL_COLUMNS = (GAS_COLUMN, CARBON_COLUMN)

_CLOCK_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"


def _under_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _clock_minutes(clock: str) -> int:
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketConfig(_Frozen):
    bidding_zone: str
    timezone: str
    gate_closure_local: str = Field(pattern=_CLOCK_PATTERN)
    forecast_issue_local: str = Field(pattern=_CLOCK_PATTERN)
    zone_start: date
    quarter_hour_products_from: date

    @field_validator("timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"unknown timezone {value!r}") from exc
        return value

    @model_validator(mode="after")
    def _issue_before_gate(self) -> MarketConfig:
        if _clock_minutes(self.forecast_issue_local) >= _clock_minutes(
            self.gate_closure_local
        ):
            raise ValueError("forecast_issue_local must be before gate_closure_local")
        return self

    def local_midnight_utc(self, day: date) -> pd.Timestamp:
        """The UTC instant at which local calendar ``day`` begins."""
        return pd.Timestamp(day).tz_localize(self.timezone).tz_convert("UTC")


class DataConfig(_Frozen):
    source: Literal["smard"]
    modeling_resolution: Resolution
    start: date
    raw_dir: Path
    processed_dir: Path
    dataset_file: str = Field(pattern=r"^[\w.-]+\.parquet$")
    inputs_file: str = Field(pattern=r"^[\w.-]+\.parquet$")

    @property
    def raw_path(self) -> Path:
        return _under_repo(self.raw_dir)

    @property
    def processed_path(self) -> Path:
        return _under_repo(self.processed_dir)

    @property
    def dataset_path(self) -> Path:
        return self.processed_path / self.dataset_file

    @property
    def inputs_path(self) -> Path:
        return self.processed_path / self.inputs_file


class SmardConfig(_Frozen):
    base_url: str = Field(pattern=r"^https://")
    region: str
    resolution: Resolution
    timeout_s: float = Field(gt=0)
    max_workers: int = Field(ge=1, le=8)
    refresh_recent_chunks: int = Field(ge=1)
    series: dict[str, int]

    @field_validator("series")
    @classmethod
    def _series_ok(cls, value: dict[str, int]) -> dict[str, int]:
        if PRICE_SERIES not in value:
            raise ValueError(f"series must include {PRICE_SERIES!r}")
        ids = list(value.values())
        if len(set(ids)) != len(ids):
            raise ValueError("each SMARD filter id may appear only once in series")
        return value


class WeatherConfig(_Frozen):
    base_url: str = Field(pattern=r"^https://")
    #: Forecast issued this many days before each valid time.
    lead_days: int = Field(ge=2, le=7)
    start: date
    timeout_s: float = Field(gt=0)
    request_days: int = Field(ge=1, le=92)
    variables: tuple[str, ...] = Field(min_length=1)
    points: dict[str, tuple[float, float]] = Field(min_length=1)

    @field_validator("points")
    @classmethod
    def _points_ok(
        cls, value: dict[str, tuple[float, float]]
    ) -> dict[str, tuple[float, float]]:
        for name, (lat, lon) in value.items():
            if not name.isidentifier() or not name.islower():
                raise ValueError(f"point name {name!r} must be lower_snake_case")
            if not (47.0 <= lat <= 56.0 and 5.0 <= lon <= 16.0):
                raise ValueError(f"point {name!r} at {lat}, {lon} is outside Germany")
        return value

    def column(self, point: str, variable: str) -> str:
        return f"wx_{point}_{variable}"

    @property
    def columns(self) -> list[str]:
        return [self.column(p, v) for p in self.points for v in self.variables]


class FuelsConfig(_Frozen):
    ttf_ticker: str
    eua_archive_url: str = Field(pattern=r"^https://")
    eua_archive_last_year: int = Field(ge=2018)
    eua_year_url: str = Field(pattern=r"^https://.*\{year\}")
    timeout_s: float = Field(gt=0)


class AvailabilityConfig(_Frozen):
    actuals_lag_minutes: int = Field(ge=0)
    columns: dict[str, AvailabilityRule]


class EvaluationConfig(_Frozen):
    holdout_start: date
    holdout_last_day: date
    live_from: date
    first_target_day: date
    validation_start: date
    spike_threshold_eur_mwh: float = Field(gt=0)


class ForecastingConfig(_Frozen):
    quantiles: tuple[float, ...]
    production_model: ProductionModel

    @field_validator("quantiles")
    @classmethod
    def _quantiles_ok(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not 0.0 < q < 1.0 for q in value):
            raise ValueError("quantiles must be non-empty and strictly between 0 and 1")
        if any(b <= a for a, b in pairwise(value)):
            raise ValueError("quantiles must be strictly increasing")
        if any(abs(q * 100 - round(q * 100)) > 1e-9 for q in value):
            raise ValueError(
                "quantiles must be whole percentages, such as 0.05, so every "
                "quantile gets its own column name"
            )
        if 0.5 not in value:
            raise ValueError("quantiles must include the median, 0.5")
        return value


class BaselinesConfig(_Frozen):
    error_window_days: int = Field(ge=7)
    min_error_days: int = Field(ge=1)

    @model_validator(mode="after")
    def _enough_window(self) -> BaselinesConfig:
        if self.min_error_days > self.error_window_days:
            raise ValueError("min_error_days cannot exceed error_window_days")
        return self


class TradingConfig(_Frozen):
    """Strategy levels and solver limits for battery dispatch."""

    #: Quantile-aware levels: selling valued at q_level, buying at q_(1 - level).
    dispatch_quantiles: tuple[float, ...] = Field(min_length=1)
    solver_time_limit_s: float = Field(gt=0)
    #: Wear prices the optimizer is given in the degradation sweep (T3).
    degradation_sweep_eur_per_mwh: tuple[float, ...] = Field(min_length=1)

    @field_validator("degradation_sweep_eur_per_mwh")
    @classmethod
    def _sweep_values(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(wear < 0 for wear in value) or len(set(value)) != len(value):
            raise ValueError("degradation sweep values must be unique and non-negative")
        return tuple(sorted(value))

    @field_validator("dispatch_quantiles")
    @classmethod
    def _pessimistic_levels(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0 < level < 0.5 for level in value):
            raise ValueError("dispatch_quantiles must lie strictly between 0 and 0.5")
        if len(set(value)) != len(value):
            raise ValueError("dispatch_quantiles must be unique")
        return value


class PipelineConfig(_Frozen):
    """How the live pipeline keeps its model fresh and its steps bounded."""

    #: Days between refits of the served model, counted as the walk-forward counts
    #: them: from the first delivery day a version forecast.
    refit_every_days: int = Field(ge=1)
    #: Longest a model fit, or one rung of the fallback chain, may take before the
    #: pipeline gives up on it and moves on.
    step_time_limit_s: float = Field(gt=0)


class Settings(_Frozen):
    """Whole-file model for ``config/settings.yaml``."""

    market: MarketConfig
    data: DataConfig
    smard: SmardConfig
    weather: WeatherConfig
    fuels: FuelsConfig
    availability: AvailabilityConfig
    evaluation: EvaluationConfig
    forecasting: ForecastingConfig
    baselines: BaselinesConfig
    battery: Battery
    trading: TradingConfig
    pipeline: PipelineConfig

    @model_validator(mode="after")
    def _consistent(self) -> Settings:
        if self.data.start < self.market.zone_start:
            raise ValueError(
                f"data.start {self.data.start} is before the DE-LU zone existed "
                f"({self.market.zone_start}); earlier prices are a different market"
            )
        if self.evaluation.holdout_start <= self.data.start:
            raise ValueError("evaluation.holdout_start must be after data.start")
        if self.smard.resolution != self.data.modeling_resolution:
            raise ValueError(
                "smard.resolution must equal data.modeling_resolution: no "
                "resampling policy exists yet (D4)"
            )
        ev = self.evaluation
        if not (
            self.data.start
            < ev.first_target_day
            < ev.validation_start
            < ev.holdout_start
        ):
            raise ValueError(
                "evaluation dates must satisfy data.start < first_target_day < "
                "validation_start < holdout_start"
            )
        # The longest baseline lag is 14 days; the error window needs that much
        # history before its first day.
        needed = timedelta(days=self.baselines.error_window_days + 14)
        if ev.first_target_day < self.data.start + needed:
            raise ValueError(
                f"first_target_day must be at least {needed.days} days after "
                "data.start so the baselines have a full error window"
            )
        if self.weather.start < self.data.start:
            raise ValueError("weather.start cannot be before data.start")
        expected = (
            set(self.smard.series)
            | set(DERIVED_COLUMNS)
            | set(FUEL_COLUMNS)
            | set(self.weather.columns)
        )
        declared = set(self.availability.columns)
        if declared != expected:
            raise ValueError(
                "availability.columns must list exactly the dataset columns; "
                f"missing {sorted(expected - declared)}, "
                f"unknown {sorted(declared - expected)}"
            )
        forecast_levels = {round(q, 6) for q in self.forecasting.quantiles}
        for level in self.trading.dispatch_quantiles:
            if not {round(level, 6), round(1 - level, 6)} <= forecast_levels:
                raise ValueError(
                    f"dispatch quantile {level} needs forecast quantiles {level} and "
                    f"{round(1 - level, 6)} in forecasting.quantiles"
                )
        return self


def load_settings(path: Path | None = None) -> Settings:
    """Load and validate settings; defaults to ``config/settings.yaml``."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    with settings_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{settings_path} must contain a mapping at the top level")
    return Settings.model_validate(raw)
