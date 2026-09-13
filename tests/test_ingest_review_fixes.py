"""Fixes from the Phase 2 data review: weather cache settling and fuel parsing."""

from __future__ import annotations

from datetime import date

import pandas as pd

from src.ingest.fuels import _as_date, ttf_settlements
from src.ingest.open_meteo import is_settled, settle_reference

AS_OF = pd.Timestamp("2026-09-13 10:00", tz="UTC")


def test_chunks_settle_against_the_download_day_not_a_future_end() -> None:
    assert settle_reference(date(2026, 9, 30), AS_OF) == date(2026, 9, 13)
    assert settle_reference(date(2025, 1, 31), AS_OF) == date(2025, 1, 31)
    # The reviewer's case: a recent chunk requested with a future end date.
    assert not is_settled(
        date(2026, 9, 21), settle_reference(date(2026, 9, 30), AS_OF), lead_days=2
    )


def test_iso_date_text_keeps_month_and_day() -> None:
    assert _as_date("2019-01-03") == date(2019, 1, 3)
    assert _as_date("03/01/2019") == date(2019, 1, 3)
    assert _as_date("2019-01-03 11:00:00") == date(2019, 1, 3)


def test_malformed_iso_date_text_is_skipped() -> None:
    assert _as_date("2019-02-30") is None
    assert _as_date("2019-01-03foo") is None


def test_flat_zero_volume_bars_are_kept_as_settlements() -> None:
    """On TTF=F most flat zero-volume bars are real settlement-only days.

    In 2022, 183 of 251 bars were flat with zero volume while the price fell
    from 214 to 126 EUR/MWh within three days, so dropping them would erase the
    gas crisis from the features.
    """
    index = pd.DatetimeIndex(["2022-03-08", "2022-03-09", "2022-03-10"])
    prices = [214.55, 155.88, 126.40]
    bars = pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": 0.0},
        index=index,
    )

    series = ttf_settlements(bars, "TTF=F")

    assert list(series.to_numpy()) == prices
