from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import pytest

from src.config import Settings
from src.forecasting.models.qra import QuantileRegressionAveraging
from src.forecasting.run_comparison import (
    first_scored_day,
    main,
    qra_target_days,
    remove_stale_qra,
)
from src.forecasting.walkforward import day_range


def test_qra_waits_for_a_full_calibration_window_of_member_forecasts() -> None:
    days = day_range(date(2026, 3, 28), date(2026, 5, 31))

    qra_days = qra_target_days(days, calibration_days=56)

    assert qra_days[0] == date(2026, 5, 23)
    assert qra_days[-1] == date(2026, 5, 31)


def test_full_run_warm_up_lets_qra_score_the_whole_validation_window(
    settings: Settings,
) -> None:
    calibration = QuantileRegressionAveraging(settings, {}).calibration_days
    first = settings.evaluation.validation_start - timedelta(days=calibration + 7)
    days = day_range(first, settings.evaluation.holdout_start - timedelta(days=1))

    assert qra_target_days(days, calibration)[0] <= settings.evaluation.validation_start


def test_quick_runs_score_only_the_requested_days(settings: Settings) -> None:
    days = day_range(date(2026, 3, 28), date(2026, 5, 31))

    assert first_scored_day(days, settings, last_days=2) == date(2026, 5, 30)
    assert (
        first_scored_day(days, settings, last_days=None)
        == settings.evaluation.validation_start
    )


@pytest.mark.parametrize("value", ["0", "-60", "100000"])
def test_quick_runs_need_at_least_one_day(value: str) -> None:
    with pytest.raises(SystemExit):
        main(["--last-days", value])


def test_rerunning_a_member_removes_stale_qra_results(tmp_path: Path) -> None:
    (tmp_path / "qra.parquet").write_bytes(b"old")
    (tmp_path / "validation_scores.json").write_text(
        json.dumps({"qra": {}, "lear": {}})
    )

    assert remove_stale_qra(tmp_path, ["lear"])

    assert not (tmp_path / "qra.parquet").exists()
    assert json.loads((tmp_path / "validation_scores.json").read_text()) == {"lear": {}}
    assert not remove_stale_qra(tmp_path, ["lear", "qra"])
    assert not remove_stale_qra(tmp_path, ["naive_previous_day"])
