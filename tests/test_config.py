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


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        (
            "evaluation",
            "validation_start",
            "2027-01-01",
            "validation_start < holdout_start",
        ),
        (
            "evaluation",
            "first_target_day",
            "2018-01-01",
            "data.start < first_target_day",
        ),
        ("market", "forecast_issue_local", "12:30", "before gate_closure_local"),
        ("baselines", "error_window_days", 3, "greater than or equal"),
        ("availability", "actuals_lag_minutes", -5, "greater than or equal"),
    ],
)
def test_invalid_evaluation_and_availability_settings_are_rejected(
    section: str, key: str, value: object, match: str
) -> None:
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ValidationError, match=match):
        Settings.model_validate(raw)


def test_availability_rules_must_cover_exactly_the_dataset_columns() -> None:
    raw = _raw()
    del raw["availability"]["columns"]["solar_actual_mw"]
    with pytest.raises(ValidationError, match="missing \\['solar_actual_mw'\\]"):
        Settings.model_validate(raw)

    raw = _raw()
    raw["availability"]["columns"]["secret"] = "before_target_day"
    with pytest.raises(ValidationError, match="unknown \\['secret'\\]"):
        Settings.model_validate(raw)

    raw = _raw()
    raw["availability"]["columns"]["price_eur_mwh"] = "whenever"
    with pytest.raises(ValidationError, match="before_target_day"):
        Settings.model_validate(raw)


@pytest.mark.parametrize(
    ("section", "key", "value", "match"),
    [
        (
            "forecasting",
            "quantiles",
            [0.02, 0.025, 0.5, 0.975, 0.98],
            "whole percentages",
        ),
        ("evaluation", "first_target_day", "2018-10-03", "full error window"),
        ("baselines", "min_error_days", 40, "cannot exceed error_window_days"),
    ],
)
def test_settings_that_would_mislead_results_are_rejected(
    section: str, key: str, value: object, match: str
) -> None:
    raw = _raw()
    raw[section][key] = value
    with pytest.raises(ValidationError, match=match):
        Settings.model_validate(raw)
