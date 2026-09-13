"""Typed loader for ``config/settings.yaml``.

Config is validated once, at load time, so a typo in a threshold or a SMARD
series id fails immediately instead of silently changing a backtest three
phases later.
"""

from __future__ import annotations

from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "CONFIG_DIR",
    "DEFAULT_SETTINGS_PATH",
    "PRICE_SERIES",
    "REPO_ROOT",
    "RESOLUTION_STEP",
    "BatteryConfig",
    "DataConfig",
    "EvaluationConfig",
    "ForecastingConfig",
    "HealthConfig",
    "MarketConfig",
    "Resolution",
    "Settings",
    "SmardConfig",
    "load_settings",
]

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "config"
DEFAULT_SETTINGS_PATH = CONFIG_DIR / "settings.yaml"

Resolution = Literal["hour", "quarterhour"]

RESOLUTION_STEP: dict[str, pd.Timedelta] = {
    "hour": pd.Timedelta(hours=1),
    "quarterhour": pd.Timedelta(minutes=15),
}

PRICE_SERIES = "price_eur_mwh"


def _under_repo(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MarketConfig(_Frozen):
    bidding_zone: str
    timezone: str
    gate_closure_local: str = Field(pattern=r"^([01]\d|2[0-3]):[0-5]\d$")
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

    @property
    def raw_path(self) -> Path:
        return _under_repo(self.raw_dir)

    @property
    def processed_path(self) -> Path:
        return _under_repo(self.processed_dir)

    @property
    def dataset_path(self) -> Path:
        return self.processed_path / self.dataset_file


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


class EvaluationConfig(_Frozen):
    holdout_start: date


class ForecastingConfig(_Frozen):
    quantiles: tuple[float, ...]

    @field_validator("quantiles")
    @classmethod
    def _quantiles_ok(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not 0.0 < q < 1.0 for q in value):
            raise ValueError("quantiles must be non-empty and strictly between 0 and 1")
        if any(b <= a for a, b in pairwise(value)):
            raise ValueError("quantiles must be strictly increasing")
        if 0.5 not in value:
            raise ValueError("quantiles must include the median, 0.5")
        return value


class BatteryConfig(_Frozen):
    power_mw: float = Field(gt=0)
    capacity_mwh: float = Field(gt=0)
    round_trip_efficiency: float = Field(gt=0, le=1)
    degradation_eur_per_mwh: float = Field(ge=0)


class HealthConfig(_Frozen):
    coverage_alert_threshold: float = Field(gt=0, lt=1)


class Settings(_Frozen):
    """Whole-file model for ``config/settings.yaml``."""

    market: MarketConfig
    data: DataConfig
    smard: SmardConfig
    evaluation: EvaluationConfig
    forecasting: ForecastingConfig
    battery: BatteryConfig
    health: HealthConfig

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
        return self


def load_settings(path: Path | None = None) -> Settings:
    """Load and validate settings; defaults to ``config/settings.yaml``."""
    settings_path = path or DEFAULT_SETTINGS_PATH
    with settings_path.open(encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{settings_path} must contain a mapping at the top level")
    return Settings.model_validate(raw)
