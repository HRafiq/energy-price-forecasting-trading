"""The hold-out runner: explicit confirmation, the right days, no scores. Synthetic."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import pytest

import src.forecasting.run_holdout as holdout_module
from src.config import PRICE_SERIES, Settings
from src.forecasting.base import (
    Forecaster,
    QuantileForecast,
    make_forecast,
    quantile_column,
)
from src.forecasting.information import InformationSet
from src.forecasting.run_comparison import CANDIDATES, REFIT_EVERY_DAYS
from src.forecasting.run_holdout import (
    default_models,
    holdout_days,
    incomplete_price_days,
    last_complete_price_day,
    main,
    run_holdout,
)
from src.forecasting.walkforward import day_range, run_walk_forward
from tests.fakes import day_clock_price, synthetic_market

SCORE_WORDS = ("mae", "rmse", "pinball", "crps", "coverage", "winkler", "skill")


class RecordingForecaster:
    """Forecasts its quantile levels and records every day it is asked about."""

    name = "lightgbm_quantile"
    lookback_days: int | None = 2
    fit_lookback_days: int | None = 5

    def __init__(self, settings: Settings) -> None:
        self.quantiles = settings.forecasting.quantiles
        self.fit_days: list[date] = []
        self.forecast_days: list[date] = []

    def fit(self, info: InformationSet) -> None:
        self.fit_days.append(info.target_day)

    def forecast(self, info: InformationSet) -> QuantileForecast:
        self.forecast_days.append(info.target_day)
        raw = pd.DataFrame(
            {quantile_column(q): 50.0 + 100.0 * q for q in self.quantiles},
            index=info.target_index,
        )
        return make_forecast(self.name, info, raw, self.quantiles)


def _in_tmp(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    return settings.model_copy(update={"data": data})


def _market(settings: Settings) -> pd.DataFrame:
    """Forty-seven days before the hold-out and ten inside it."""
    return synthetic_market(
        settings, date(2026, 4, 15), 57, day_clock_price(settings.market.timezone)
    )


def _without_prices_from(
    frame: pd.DataFrame, settings: Settings, local: str
) -> pd.DataFrame:
    out = frame.copy()
    cut = pd.Timestamp(local).tz_localize(settings.market.timezone).tz_convert("UTC")
    out.loc[out.index >= cut, PRICE_SERIES] = np.nan
    return out


def test_refuses_without_the_confirmation_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("no data may be read without confirmation")

    monkeypatch.setattr(holdout_module, "load_settings", forbidden)
    monkeypatch.setattr(pd, "read_parquet", forbidden)
    monkeypatch.setattr(holdout_module, "run_holdout", forbidden)

    assert main(["--last", "2026-06-03"]) != 0
    assert "--confirm-holdout" in capsys.readouterr().err


def test_default_models_put_the_production_model_first(settings: Settings) -> None:
    models = default_models(settings)

    assert models[0] == settings.forecasting.production_model
    assert {"lightgbm_quantile", "quantile_forest", "naive_previous_day"} <= set(models)
    assert len(models) == len(set(models))
    assert all(name in CANDIDATES for name in models)


def test_days_start_at_the_holdout_and_end_at_the_last_day(
    settings: Settings,
) -> None:
    days = holdout_days(settings, date(2026, 6, 10))

    assert days[0] == settings.evaluation.holdout_start
    assert days[-1] == date(2026, 6, 10)
    assert days == day_range(settings.evaluation.holdout_start, date(2026, 6, 10))
    with pytest.raises(ValueError, match="before the hold-out start"):
        holdout_days(settings, date(2026, 5, 31))


def test_last_day_skips_a_final_day_with_missing_prices(settings: Settings) -> None:
    frame = _without_prices_from(_market(settings), settings, "2026-06-05 13:00")

    assert last_complete_price_day(frame, settings) == date(2026, 6, 4)


def test_last_day_skips_a_final_day_whose_rows_are_not_there_yet(
    settings: Settings,
) -> None:
    frame = _market(settings)
    cut = settings.market.local_midnight_utc(date(2026, 6, 8)) + pd.Timedelta(hours=3)

    assert last_complete_price_day(frame.loc[frame.index < cut], settings) == date(
        2026, 6, 7
    )
    assert last_complete_price_day(frame, settings) == date(2026, 6, 10)


@pytest.mark.parametrize(
    ("first", "change_day", "periods"),
    [(date(2026, 10, 10), date(2026, 10, 25), 100), (date(2026, 3, 15), None, 92)],
)
def test_last_day_counts_every_period_of_a_clock_change_day(
    settings: Settings, first: date, change_day: date | None, periods: int
) -> None:
    tz = settings.market.timezone
    day = change_day or date(2026, 3, 29)
    frame = synthetic_market(settings, first, (day - first).days + 2)
    frame = _without_prices_from(frame, settings, str(day + timedelta(days=1)))
    local_days = pd.DatetimeIndex(frame.index).tz_convert(tz).date
    assert int((local_days == day).sum()) == periods

    assert last_complete_price_day(frame, settings) == day

    last_period = frame.index[local_days == day][-1]
    frame.loc[last_period, PRICE_SERIES] = np.nan
    assert last_complete_price_day(frame, settings) == day - timedelta(days=1)


def test_last_day_needs_some_prices(settings: Settings) -> None:
    frame = _market(settings)
    frame[PRICE_SERIES] = np.nan

    with pytest.raises(ValueError, match="no realised prices"):
        last_complete_price_day(frame, settings)


def test_runs_walk_forward_on_holdout_days_only_with_holdout_access(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    made: list[RecordingForecaster] = []

    def spy(
        frame: pd.DataFrame,
        forecaster: Forecaster,
        target_days: Sequence[date],
        settings: Settings,
        **kwargs: Any,
    ) -> pd.DataFrame:
        calls.append({"days": list(target_days), "end": frame.index.max(), **kwargs})
        return run_walk_forward(frame, forecaster, target_days, settings, **kwargs)

    def factory(s: Settings) -> Forecaster:
        made.append(RecordingForecaster(s))
        return made[-1]

    monkeypatch.setattr(holdout_module, "run_walk_forward", spy)
    last = date(2026, 6, 4)

    runs = run_holdout(
        _market(settings),
        settings,
        ["lightgbm_quantile"],
        last,
        tmp_path,
        factories={"lightgbm_quantile": factory},
        log=False,
    )

    holdout = settings.evaluation.holdout_start
    (call,) = calls
    assert call["allow_holdout"] is True
    assert call["refit_every_days"] == REFIT_EVERY_DAYS["lightgbm_quantile"]
    assert call["days"] == day_range(holdout, last)
    assert call["end"] < settings.market.local_midnight_utc(last + timedelta(days=1))
    (model,) = made
    assert model.forecast_days == day_range(holdout, last)
    assert model.fit_days == [holdout]
    assert min(model.fit_days + model.forecast_days) >= holdout
    (run,) = runs
    assert (run.days, run.periods) == (4, 4 * 96)
    saved = pd.read_parquet(run.path)
    assert set(saved["target_day"]) == set(day_range(holdout, last))


def test_output_schema_matches_the_comparison(
    settings: Settings, tmp_path: Path
) -> None:
    frame = _market(settings)
    (run,) = run_holdout(
        frame,
        settings,
        ["lightgbm_quantile"],
        date(2026, 6, 3),
        tmp_path,
        factories={"lightgbm_quantile": RecordingForecaster},
        log=False,
    )
    comparison_path = tmp_path / "comparison.parquet"
    run_walk_forward(
        frame,
        RecordingForecaster(settings),
        day_range(date(2026, 5, 29), date(2026, 5, 31)),
        settings,
        refit_every_days=REFIT_EVERY_DAYS["lightgbm_quantile"],
    ).to_parquet(comparison_path)

    holdout = pd.read_parquet(run.path)
    comparison = pd.read_parquet(comparison_path)

    assert run.path == tmp_path / "lightgbm_quantile.parquet"
    assert list(holdout.columns) == list(comparison.columns)
    pd.testing.assert_series_equal(holdout.dtypes, comparison.dtypes)
    assert holdout.index.name == comparison.index.name == "timestamp_utc"
    assert str(holdout.index.dtype) == str(comparison.index.dtype)


def test_logs_parameters_and_run_time_but_no_scores(
    settings: Settings,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    experiment = mlflow.create_experiment(
        holdout_module.EXPERIMENT, artifact_location=(tmp_path / "mlruns").as_uri()
    )
    mlflow.set_experiment(holdout_module.EXPERIMENT)

    run_holdout(
        _market(settings),
        settings,
        ["lightgbm_quantile"],
        date(2026, 6, 2),
        tmp_path / "out",
        factories={"lightgbm_quantile": RecordingForecaster},
    )

    (logged,) = mlflow.search_runs([experiment], output_format="list")
    assert set(logged.data.metrics) == {"run_seconds"}
    assert logged.data.params["first_day"] == "2026-06-01"
    assert logged.data.params["last_day"] == "2026-06-02"
    assert logged.data.params["refit_every_days"] == "28"
    assert logged.data.tags["window"] == "holdout"
    assert logged.data.tags["model"] == "lightgbm_quantile"
    printed = capsys.readouterr().out.lower()
    assert "2 days, 192 periods" in printed
    assert not any(word in printed for word in SCORE_WORDS)


def test_confirmed_cli_forecasts_to_the_last_complete_day(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    local = _in_tmp(settings, tmp_path)
    local.data.processed_path.mkdir(parents=True)
    frame = _without_prices_from(_market(local), local, "2026-06-03 09:00")
    frame.to_parquet(local.data.inputs_path)
    monkeypatch.setattr(holdout_module, "load_settings", lambda path: local)
    monkeypatch.setattr(
        holdout_module, "TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}"
    )
    monkeypatch.setattr(holdout_module, "ARTIFACTS", tmp_path / "mlruns")

    code = main(["--confirm-holdout", "--models", "naive_previous_day"])

    assert code == 0
    saved = pd.read_parquet(
        local.data.processed_path
        / "forecasts"
        / "holdout"
        / "naive_previous_day.parquet"
    )
    assert sorted(set(saved["target_day"])) == [date(2026, 6, 1), date(2026, 6, 2)]
    assert saved["actual"].notna().all()
    printed = capsys.readouterr().out.lower()
    assert "2026-06-01 to 2026-06-02 (2 days)" in printed
    assert not any(word in printed for word in SCORE_WORDS)
    runs = mlflow.search_runs(
        experiment_names=[holdout_module.EXPERIMENT], output_format="list"
    )
    assert all(set(r.data.metrics) == {"run_seconds"} for r in runs)


def test_incomplete_price_days_lists_a_gap_inside_the_holdout(
    settings: Settings,
) -> None:
    frame = _market(settings)
    tz = settings.market.timezone
    start = pd.Timestamp("2026-06-03 10:00").tz_localize(tz).tz_convert("UTC")
    frame.loc[
        (frame.index >= start) & (frame.index < start + pd.Timedelta(hours=1)),
        PRICE_SERIES,
    ] = np.nan
    days = day_range(date(2026, 6, 1), date(2026, 6, 5))

    assert incomplete_price_days(frame, settings, days) == [date(2026, 6, 3)]


def test_confirmed_cli_refuses_to_overwrite_existing_forecasts(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = _in_tmp(settings, tmp_path)
    local.data.processed_path.mkdir(parents=True)
    _market(local).to_parquet(local.data.inputs_path)
    out = local.data.processed_path / "forecasts" / "holdout"
    out.mkdir(parents=True)
    (out / "naive_previous_day.parquet").write_bytes(b"existing")
    monkeypatch.setattr(holdout_module, "load_settings", lambda path: local)

    with pytest.raises(SystemExit):
        main(["--confirm-holdout", "--models", "naive_previous_day"])
    assert (out / "naive_previous_day.parquet").read_bytes() == b"existing"
