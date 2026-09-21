"""Laying a downloaded window over a longer file, without losing or faking rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ingest.merge import merge_into


def _frame(start: str, periods: int, value: float) -> pd.DataFrame:
    index = pd.date_range(start, periods=periods, freq="15min", tz="UTC")
    index.name = "timestamp_utc"
    return pd.DataFrame({"price": value, "load": value * 2}, index=index)


def test_with_no_file_there_the_window_is_the_whole_table(tmp_path: Path) -> None:
    window = _frame("2026-09-01", 4, 1.0)
    assert merge_into(window, tmp_path / "absent.parquet").equals(window)


def test_rows_outside_the_window_are_kept(tmp_path: Path) -> None:
    """The backtest reads the older rows; a window must never shorten the file."""
    path = tmp_path / "data.parquet"
    _frame("2026-01-01", 96, 1.0).to_parquet(path)
    window = _frame("2026-09-01", 4, 2.0)

    merged = merge_into(window, path)

    assert len(merged) == 100
    assert merged.index[0] == pd.Timestamp("2026-01-01", tz="UTC")
    assert merged.index.is_monotonic_increasing


def test_a_revised_value_replaces_the_one_it_revises(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    _frame("2026-09-01", 8, 1.0).to_parquet(path)
    window = _frame("2026-09-01", 4, 9.0)

    merged = merge_into(window, path)

    assert len(merged) == 8
    assert merged["price"].iloc[0] == 9.0 and merged["price"].iloc[-1] == 1.0


def test_a_changed_column_set_is_refused_rather_than_back_filled(
    tmp_path: Path,
) -> None:
    """Concatenating would leave a cliff of NaN that reads as real missing data."""
    path = tmp_path / "data.parquet"
    _frame("2026-01-01", 4, 1.0).to_parquet(path)
    window = _frame("2026-09-01", 4, 1.0)
    window["wind"] = 3.0

    with pytest.raises(ValueError, match=r"added \['wind'\]"):
        merge_into(window, path)


def test_a_dropped_column_is_refused_too(tmp_path: Path) -> None:
    path = tmp_path / "data.parquet"
    _frame("2026-01-01", 4, 1.0).to_parquet(path)
    window = _frame("2026-09-01", 4, 1.0).drop(columns=["load"])

    with pytest.raises(ValueError, match=r"missing \['load'\]"):
        merge_into(window, path)
