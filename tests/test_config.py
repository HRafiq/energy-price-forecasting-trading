from __future__ import annotations

from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from src.config import DEFAULT_SETTINGS_PATH, Settings, load_settings


def _raw() -> dict[str, Any]:
    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def test_repository_settings_are_valid() -> None:
    settings = load_settings()
    assert settings.market.bidding_zone == "DE-LU"
    assert 0.5 in settings.forecasting.quantiles
    assert settings.evaluation.holdout_start > settings.data.start
    assert settings.data.dataset_path.name == "smard_quarterhour.parquet"


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        ("forecasting", "quantiles", [0.5, 0.25], "increasing"),
        ("forecasting", "quantiles", [0.25, 0.75], "median"),
        ("battery", "round_trip_efficiency", 1.2, "less than or equal"),
        ("data", "start", "2018-01-01", "DE-LU zone"),
        ("market", "timezone", "Europe/Berlinn", "unknown timezone"),
        ("smard", "resolution", "hour", "resampling policy"),
        ("evaluation", "holdout_start", "2018-10-01", "holdout_start"),
        ("battery", "capacity_kwh", 2000, "Extra inputs"),
    ],
)
def test_invalid_settings_are_rejected(
    section: str, key: str, value: object, match: str
) -> None:
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ValidationError, match=match):
        Settings.model_validate(raw)


def test_series_must_include_price_and_unique_ids() -> None:
    raw = _raw()
    del raw["smard"]["series"]["price_eur_mwh"]
    with pytest.raises(ValidationError, match="price_eur_mwh"):
        Settings.model_validate(raw)

    raw = _raw()
    series = raw["smard"]["series"]
    series["duplicate_of_price"] = series["price_eur_mwh"]
    with pytest.raises(ValidationError, match="only once"):
        Settings.model_validate(raw)
