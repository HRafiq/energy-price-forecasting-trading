from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import PRICE_SERIES, Settings
from src.ingest.build_dataset import PRODUCT_COLUMN, build_dataset, publish
from src.ingest.quality import GranularityError, QualityReport
from src.ingest.smard import SmardClient
from src.timegrid import HOUR
from tests.fakes import FakeSmard, hourly_chunks

QUARTER = pd.Timedelta(minutes=15)
PERIODS = 192  # two local days of quarter-hours
CHUNK = 96
PRICE_PUBLISHED = 172  # the last 20 quarter-hours are not yet published
WIND_GAP = 3


def _raw_value(settings: Settings, name: str, i: int) -> float:
    k = list(settings.smard.series).index(name)
    return 1000.0 * (k + 1) + i


def _market(
    settings: Settings, step: pd.Timedelta = QUARTER, periods: int = PERIODS
) -> FakeSmard:
    fake = FakeSmard(settings.smard.model_copy(update={"max_workers": 1}))
    start = settings.market.local_midnight_utc(settings.data.start)
    for name, filter_id in settings.smard.series.items():

        def value(i: int, name: str = name) -> float | None:
            if name == PRICE_SERIES and i >= PRICE_PUBLISHED:
                return None
            if name == "wind_onshore_actual_mw" and i == WIND_GAP:
                return None
            return _raw_value(settings, name, i)

        fake.add_series(filter_id, hourly_chunks(start, periods, CHUNK, value, step))
    return fake


def _build(settings: Settings, fake: FakeSmard, tmp_path: Path) -> pd.DataFrame:
    return build_dataset(settings, SmardClient(fake.config, tmp_path, fake))


def test_dataset_is_quarter_hourly_utc_and_trimmed_to_last_price(
    settings: Settings, tmp_path: Path
) -> None:
    frame = _build(settings, _market(settings), tmp_path)

    index = pd.DatetimeIndex(frame.index)
    assert index.name == "timestamp_utc"
    assert str(index.tz) == "UTC"
    assert index[0] == pd.Timestamp("2018-09-30 22:00", tz="UTC")
    assert (index[1:] - index[:-1] == QUARTER).all()
    assert len(frame) == PRICE_PUBLISHED
    assert frame[PRICE_SERIES].notna().all()


def test_volumes_become_average_mw_and_prices_stay_in_eur_per_mwh(
    settings: Settings, tmp_path: Path
) -> None:
    frame = _build(settings, _market(settings), tmp_path)

    assert frame["load_actual_mw"].iloc[10] == 4 * _raw_value(
        settings, "load_actual_mw", 10
    )
    assert frame[PRICE_SERIES].iloc[10] == _raw_value(settings, PRICE_SERIES, 10)


def test_residual_load_subtracts_renewables_and_propagates_gaps(
    settings: Settings, tmp_path: Path
) -> None:
    frame = _build(settings, _market(settings), tmp_path)

    renewables = [
        "wind_onshore_actual_mw",
        "wind_offshore_actual_mw",
        "solar_actual_mw",
    ]
    expected = frame["load_actual_mw"] - frame[renewables].sum(axis=1)
    ok = frame.index != frame.index[WIND_GAP]
    np.testing.assert_allclose(frame.loc[ok, "residual_load_actual_mw"], expected[ok])
    assert np.isnan(frame["residual_load_actual_mw"].iloc[WIND_GAP])
    assert frame["residual_load_forecast_mw"].notna().all()


def test_price_product_minutes_marks_the_switch_to_quarter_hour_products(
    settings: Settings, tmp_path: Path
) -> None:
    switched = settings.model_copy(
        update={
            "market": settings.market.model_copy(
                update={"quarter_hour_products_from": date(2018, 10, 2)}
            )
        }
    )
    frame = _build(switched, _market(switched), tmp_path)

    assert (frame[PRODUCT_COLUMN].iloc[:96] == 60).all()
    assert (frame[PRODUCT_COLUMN].iloc[96:] == 15).all()


def test_hourly_source_data_is_rejected_when_quarter_hours_are_configured(
    settings: Settings, tmp_path: Path
) -> None:
    fake = _market(settings, step=HOUR, periods=48)
    with pytest.raises(GranularityError):
        _build(settings, fake, tmp_path)


def test_missing_required_series_is_a_config_error(
    settings: Settings, tmp_path: Path
) -> None:
    series = dict(settings.smard.series)
    del series["solar_forecast_mw"]
    trimmed = settings.model_copy(
        update={"smard": settings.smard.model_copy(update={"series": series})}
    )
    with pytest.raises(ValueError, match="solar_forecast_mw"):
        _build(trimmed, _market(trimmed), tmp_path)


def test_series_without_a_known_unit_suffix_is_rejected(
    settings: Settings, tmp_path: Path
) -> None:
    series = {**settings.smard.series, "gas_price": 9999}
    odd = settings.model_copy(
        update={"smard": settings.smard.model_copy(update={"series": series})}
    )
    with pytest.raises(ValueError, match="gas_price"):
        _build(odd, FakeSmard(odd.smard), tmp_path)


def _report(duplicates: int) -> QualityReport:
    moment = datetime(2024, 1, 1, tzinfo=UTC)
    return QualityReport(
        start_utc=moment,
        end_utc=moment,
        resolution="0 days 00:15:00",
        n_rows=1,
        expected_rows=1,
        missing_timestamps=0,
        duplicate_timestamps=duplicates,
        is_monotonic=True,
        day_length_issues=[],
        hourly_product_mismatches=0,
        columns=[],
        price=None,
    )


def test_failed_build_never_replaces_the_last_good_dataset(tmp_path: Path) -> None:
    out = tmp_path / "market.parquet"
    index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01", tz="UTC")], name="timestamp_utc"
    )
    good = pd.DataFrame({"x": [1.0]}, index=index)

    assert publish(good, _report(duplicates=0), out)
    assert not publish(good.assign(x=[99.0]), _report(duplicates=1), out)

    assert pd.read_parquet(out)["x"].iloc[0] == 1.0
    kept = json.loads((tmp_path / "market_quality.json").read_text())
    assert kept["duplicate_timestamps"] == 0
    assert (tmp_path / "market_quality_rejected.json").exists()


def test_passing_build_clears_a_stale_rejection(tmp_path: Path) -> None:
    out = tmp_path / "market.parquet"
    index = pd.DatetimeIndex(
        [pd.Timestamp("2024-01-01", tz="UTC")], name="timestamp_utc"
    )
    frame = pd.DataFrame({"x": [1.0]}, index=index)

    assert not publish(frame, _report(duplicates=1), out)
    assert (tmp_path / "market_quality_rejected.json").exists()
    assert publish(frame, _report(duplicates=0), out)

    assert not (tmp_path / "market_quality_rejected.json").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "market.parquet",
        "market_quality.json",
    ]


def test_price_like_name_with_a_wrong_unit_suffix_is_rejected(
    settings: Settings, tmp_path: Path
) -> None:
    series = {**settings.smard.series, "neighbour_price_eur_mw": 9999}
    odd = settings.model_copy(
        update={"smard": settings.smard.model_copy(update={"series": series})}
    )
    with pytest.raises(ValueError, match="neighbour_price_eur_mw"):
        _build(odd, FakeSmard(odd.smard), tmp_path)
