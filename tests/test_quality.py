from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.ingest.quality import (
    GranularityError,
    assert_resolution,
    build_quality_report,
    column_gaps,
    day_length_issues,
    hourly_product_mismatches,
)
from src.timegrid import HOUR

TZ = "Europe/Berlin"


def _hourly(start: str, end: str) -> pd.DatetimeIndex:
    return pd.date_range(
        pd.Timestamp(start, tz="UTC"),
        pd.Timestamp(end, tz="UTC"),
        freq="h",
        inclusive="left",
    )


def _frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    return pd.DataFrame({"price_eur_mwh": rng.normal(60, 30, len(index))}, index=index)


def test_clean_frame_across_both_dst_changes_passes() -> None:
    # Local midnight 2024-03-30 to local midnight 2024-10-28.
    frame = _frame(_hourly("2024-03-29 23:00", "2024-10-27 23:00"))
    report = build_quality_report(frame, TZ, HOUR)
    assert report.structural_ok
    assert report.n_rows == report.expected_rows
    assert report.resolution == "0 days 01:00:00"


def test_missing_timestamps_are_counted() -> None:
    frame = _frame(_hourly("2024-06-01 22:00", "2024-06-04 22:00"))
    frame = frame.drop(frame.index[30:33])
    report = build_quality_report(frame, TZ, HOUR)
    assert report.missing_timestamps == 3
    assert not report.structural_ok
    assert [i.delivery_date for i in report.day_length_issues] == [date(2024, 6, 3)]


def test_local_grid_shifted_forward_on_spring_day_is_caught() -> None:
    """The quiet version of the DST bug: no exception, just a duplicated hour."""
    naive = pd.date_range("2024-03-30 00:00", "2024-04-01 23:00", freq="h")
    index = naive.tz_localize(TZ, nonexistent="shift_forward").tz_convert("UTC")
    report = build_quality_report(_frame(index), TZ, HOUR)
    assert report.duplicate_timestamps == 1
    assert [
        (i.delivery_date, i.expected_periods, i.observed_periods)
        for i in report.day_length_issues
    ] == [(date(2024, 3, 31), 23, 24)]
    assert not report.structural_ok


def test_truncated_edge_days_are_not_flagged() -> None:
    index = _hourly("2024-06-01 10:00", "2024-06-03 05:00")
    assert day_length_issues(index, TZ, HOUR) == []


def test_quarter_hour_data_is_rejected_when_hourly_is_expected() -> None:
    index = pd.date_range(pd.Timestamp("2025-10-01", tz="UTC"), periods=8, freq="15min")
    with pytest.raises(GranularityError, match="15:00"):
        assert_resolution(index, HOUR)


def test_trailing_gaps_are_separated_from_interior_gaps() -> None:
    series = pd.Series(
        [1.0, np.nan, np.nan, 2.0, np.nan, 3.0, np.nan, np.nan],
        index=_hourly("2024-06-01 00:00", "2024-06-01 08:00"),
        name="x",
    )
    gaps = column_gaps(series)
    assert gaps.missing_before_last_value == 3
    assert gaps.longest_gap_periods == 2
    assert gaps.trailing_missing == 2


def test_negative_prices_are_reported_not_failed() -> None:
    frame = _frame(_hourly("2024-05-31 22:00", "2024-06-01 22:00"))
    frame.iloc[10:14, 0] = -25.0
    report = build_quality_report(frame, TZ, HOUR)
    assert report.structural_ok
    assert report.price is not None
    assert report.price.negative_periods >= 4
    assert report.price.min_eur_mwh <= -25.0


def test_single_row_frame_gets_a_report_instead_of_a_crash() -> None:
    frame = _frame(_hourly("2024-06-01 10:00", "2024-06-01 11:00"))
    report = build_quality_report(frame, TZ, HOUR)
    assert report.n_rows == 1
    assert report.resolution == "undetermined"


QUARTER = pd.Timedelta(minutes=15)
SWITCH = pd.Timestamp("2025-09-30 22:00", tz="UTC")  # local midnight, 1 Oct 2025


def _quarter_prices(start_utc: str, periods: int) -> pd.Series:
    index = pd.date_range(
        pd.Timestamp(start_utc, tz="UTC"), periods=periods, freq="15min"
    )
    hourly = np.repeat(np.arange(periods // 4, dtype=float), 4)
    return pd.Series(hourly, index=index, name="price_eur_mwh")


def test_repeated_quarter_hour_prices_before_the_switch_pass() -> None:
    prices = _quarter_prices("2025-09-29 22:00", 96)
    assert hourly_product_mismatches(prices, QUARTER, SWITCH) == 0
    report = build_quality_report(
        prices.to_frame(), TZ, QUARTER, quarter_hour_products_from=SWITCH
    )
    assert report.structural_ok


def test_differing_quarter_hours_inside_an_hourly_product_fail() -> None:
    prices = _quarter_prices("2025-09-29 22:00", 96)
    prices.iloc[5] += 1.0
    assert hourly_product_mismatches(prices, QUARTER, SWITCH) == 1
    report = build_quality_report(
        prices.to_frame(), TZ, QUARTER, quarter_hour_products_from=SWITCH
    )
    assert report.hourly_product_mismatches == 1
    assert not report.structural_ok


def test_quarter_hour_prices_may_differ_after_the_switch() -> None:
    prices = _quarter_prices("2025-09-30 22:00", 96)
    prices.iloc[5] += 1.0
    assert hourly_product_mismatches(prices, QUARTER, SWITCH) == 0


def test_hourly_data_skips_the_product_check() -> None:
    frame = _frame(_hourly("2025-09-28 22:00", "2025-09-29 22:00"))
    assert hourly_product_mismatches(frame["price_eur_mwh"], HOUR, SWITCH) == 0


def test_sub_hourly_prices_without_a_switch_date_are_refused() -> None:
    prices = _quarter_prices("2025-09-29 22:00", 8)
    with pytest.raises(ValueError, match="quarter_hour_products_from"):
        hourly_product_mismatches(prices, QUARTER, None)


def test_quarter_hour_frame_across_the_autumn_dst_day_passes() -> None:
    # Local 27 Oct 2024 runs from 26 Oct 22:00 UTC to 27 Oct 23:00 UTC: 25 hours.
    index = pd.date_range(
        pd.Timestamp("2024-10-26 22:00", tz="UTC"),
        pd.Timestamp("2024-10-27 23:00", tz="UTC"),
        freq="15min",
        inclusive="left",
    )
    frame = pd.DataFrame({"price_eur_mwh": np.repeat(np.arange(25.0), 4)}, index=index)
    report = build_quality_report(frame, TZ, QUARTER, quarter_hour_products_from=SWITCH)
    assert len(frame) == 100
    assert report.structural_ok


def test_quarter_hour_grid_shifted_forward_on_the_spring_day_is_caught() -> None:
    naive = pd.date_range("2024-03-30 00:00", "2024-04-01 23:45", freq="15min")
    index = naive.tz_localize(TZ, nonexistent="shift_forward").tz_convert("UTC")
    frame = pd.DataFrame({"price_eur_mwh": 50.0}, index=index)
    report = build_quality_report(frame, TZ, QUARTER, quarter_hour_products_from=SWITCH)
    assert report.duplicate_timestamps == 4
    assert [
        (i.expected_periods, i.observed_periods) for i in report.day_length_issues
    ] == [(92, 96)]
    assert not report.structural_ok
