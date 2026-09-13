from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Settings, SmardConfig
from src.ingest.smard import SmardClient, SmardError, parse_chunk
from tests.fakes import FakeSmard, hourly_chunks

T0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")


def test_parse_chunk_maps_null_to_nan_on_a_utc_index() -> None:
    payload = {"series": [[1711839600000, 10.5], [1711843200000, None]]}
    series = parse_chunk(payload, "p")
    assert str(pd.DatetimeIndex(series.index).tz) == "UTC"
    assert series.index[0] == pd.Timestamp("2024-03-30 23:00", tz="UTC")
    assert series.iloc[0] == 10.5
    assert np.isnan(series.iloc[1])


@pytest.mark.parametrize("payload", [{}, {"series": None}, {"series": [["x", 1]]}, []])
def test_parse_chunk_rejects_malformed_payloads(payload: object) -> None:
    with pytest.raises(SmardError):
        parse_chunk(payload, "p")


def test_series_skips_chunks_ending_before_start(
    tmp_path: Path, smard_config: SmardConfig
) -> None:
    fake = FakeSmard(smard_config)
    starts = fake.add_series(4169, hourly_chunks(T0, hours=72, chunk_hours=24))
    client = SmardClient(smard_config, tmp_path, fake)

    start = T0 + pd.Timedelta(hours=30)
    series = client.series(4169, "price", start=start)

    assert series.index[0] == start
    assert len(series) == 72 - 30
    assert fake.chunk_url(4169, starts[0]) not in fake.calls


def test_series_respects_end(tmp_path: Path, smard_config: SmardConfig) -> None:
    fake = FakeSmard(smard_config)
    starts = fake.add_series(4169, hourly_chunks(T0, hours=72, chunk_hours=24))
    client = SmardClient(smard_config, tmp_path, fake)

    series = client.series(4169, "price", start=T0, end=T0 + pd.Timedelta(hours=24))

    assert len(series) == 24
    assert fake.chunk_url(4169, starts[2]) not in fake.calls


def test_cached_chunks_are_reused_except_the_newest(
    tmp_path: Path, smard_config: SmardConfig
) -> None:
    fake = FakeSmard(smard_config)
    starts = fake.add_series(4169, hourly_chunks(T0, hours=72, chunk_hours=24))
    SmardClient(smard_config, tmp_path, fake).series(4169, "price", start=T0)

    fake.calls.clear()
    again = SmardClient(smard_config, tmp_path, fake).series(4169, "price", start=T0)

    chunk_calls = [url for url in fake.calls if "index_" not in url]
    assert chunk_calls == [fake.chunk_url(4169, starts[-1])]
    assert len(again) == 72


def test_identical_values_on_chunk_boundaries_are_merged(
    tmp_path: Path, smard_config: SmardConfig
) -> None:
    first, second = hourly_chunks(T0, hours=48, chunk_hours=24)
    overlapping = pd.concat([first.iloc[-1:], second])
    fake = FakeSmard(smard_config)
    fake.add_series(4169, [first, overlapping])

    series = SmardClient(smard_config, tmp_path, fake).series(4169, "price", start=T0)

    assert len(series) == 48
    assert series.index.is_unique


def test_conflicting_values_on_chunk_boundaries_raise(
    tmp_path: Path, smard_config: SmardConfig
) -> None:
    first, second = hourly_chunks(T0, hours=48, chunk_hours=24)
    clash = first.iloc[-1:] + 100.0
    fake = FakeSmard(smard_config)
    fake.add_series(4169, [first, pd.concat([clash, second])])

    with pytest.raises(SmardError, match="conflicting"):
        SmardClient(smard_config, tmp_path, fake).series(4169, "price", start=T0)


def test_series_requires_timezone_aware_bounds(
    tmp_path: Path, smard_config: SmardConfig
) -> None:
    client = SmardClient(smard_config, tmp_path, FakeSmard(smard_config))
    with pytest.raises(ValueError, match="timezone-aware"):
        client.series(4169, "price", start=pd.Timestamp("2024-01-01"))


def test_chunk_cached_while_incomplete_is_refetched_after_it_settles(
    tmp_path: Path, smard_config: SmardConfig
) -> None:
    """A week cached before it was complete must not keep its gaps forever."""
    week1, week2, week3 = hourly_chunks(T0, hours=72, chunk_hours=24)
    incomplete_week2 = week2.copy()
    incomplete_week2.iloc[12:] = np.nan

    fake = FakeSmard(smard_config)
    fake.add_series(4169, [week1, incomplete_week2])
    first = SmardClient(smard_config, tmp_path, fake).series(4169, "p", start=T0)
    assert first.isna().sum() == 12

    # Later: week 2 is complete and week 3 exists, so week 2 is no longer newest.
    fake.add_series(4169, [week1, week2, week3])
    fake.calls.clear()
    later = SmardClient(smard_config, tmp_path, fake).series(4169, "p", start=T0)

    assert later.notna().all()
    assert len(later) == 72
    assert fake.chunk_url(4169, _ms(week2)) in fake.calls
    assert fake.chunk_url(4169, _ms(week1)) not in fake.calls


def _ms(chunk: pd.Series) -> int:
    return int(pd.DatetimeIndex(chunk.index)[0].value // 1_000_000)


def test_settle_window_uses_the_configured_four_weeks(
    tmp_path: Path, settings: Settings
) -> None:
    config = settings.smard.model_copy(update={"max_workers": 1})
    assert config.refresh_recent_chunks == 4
    fake = FakeSmard(config)
    starts = fake.add_series(4169, hourly_chunks(T0, hours=6 * 24, chunk_hours=24))
    SmardClient(config, tmp_path, fake).series(4169, "p", start=T0)

    fake.calls.clear()
    SmardClient(config, tmp_path, fake).series(4169, "p", start=T0)

    # Chunks 0 and 1 had at least four newer chunks when cached; 2 to 5 did not.
    refetched = [url for url in fake.calls if "index_" not in url]
    assert refetched == [fake.chunk_url(4169, s) for s in starts[2:]]
