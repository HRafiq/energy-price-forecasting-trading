"""Data-quality checks for the market dataset (Phase 0).

Structural checks decide pass or fail, because a backtest built on a misaligned
index produces confident nonsense. Value checks are reported, never enforced:
negative prices and spikes are real market behaviour, not data errors.

Covered: granularity (D4), missing and duplicate timestamps, DST day lengths
(D3), per-column gaps, and hourly-product consistency: before the switch to
15-minute products, every quarter-hour of an hour must carry the same price.
Revision detection (D2) comes later.
"""

from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from src.config import PRICE_SERIES
from src.timegrid import HOUR, delivery_calendar, ensure_utc_index

__all__ = [
    "ColumnGaps",
    "DayLengthIssue",
    "GranularityError",
    "PriceSummary",
    "QualityReport",
    "assert_resolution",
    "build_quality_report",
    "column_gaps",
    "day_length_issues",
    "hourly_product_mismatches",
    "infer_resolution",
    "missing_timestamps",
]


class GranularityError(ValueError):
    """The data's time step differs from the configured resolution (D4)."""


class _Report(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DayLengthIssue(_Report):
    delivery_date: date
    expected_periods: int
    observed_periods: int


class ColumnGaps(_Report):
    column: str
    #: Missing values up to the column's last observed value, leading gaps included.
    missing_before_last_value: int
    longest_gap_periods: int
    #: Missing values after the last observed value, typically not yet published.
    trailing_missing: int


class PriceSummary(_Report):
    min_eur_mwh: float
    max_eur_mwh: float
    negative_periods: int
    negative_share: float


class QualityReport(_Report):
    start_utc: datetime
    end_utc: datetime
    resolution: str
    n_rows: int
    expected_rows: int
    missing_timestamps: int
    duplicate_timestamps: int
    is_monotonic: bool
    day_length_issues: list[DayLengthIssue]
    #: Hours traded as hourly products whose sub-hourly prices differ.
    hourly_product_mismatches: int
    columns: list[ColumnGaps]
    price: PriceSummary | None

    @property
    def structural_ok(self) -> bool:
        return (
            self.missing_timestamps == 0
            and self.duplicate_timestamps == 0
            and self.is_monotonic
            and not self.day_length_issues
            and self.hourly_product_mismatches == 0
        )


def infer_resolution(index: pd.DatetimeIndex) -> pd.Timedelta:
    """The most common step between consecutive distinct timestamps."""
    idx = ensure_utc_index(index).unique().sort_values()
    if len(idx) < 2:
        raise ValueError("need at least two distinct timestamps to infer a resolution")
    steps = pd.Series(idx[1:] - idx[:-1])
    return pd.Timedelta(steps.mode().iloc[0])


def assert_resolution(index: pd.DatetimeIndex, expected: pd.Timedelta) -> None:
    found = infer_resolution(index)
    if found != expected:
        raise GranularityError(
            f"expected a {expected} step but the data is at {found}. Resample "
            "explicitly under a documented policy instead of mixing granularities."
        )


def missing_timestamps(index: pd.DatetimeIndex, step: pd.Timedelta) -> pd.DatetimeIndex:
    """Timestamps absent from a regular grid between the first and last one."""
    idx = ensure_utc_index(index)
    if idx.empty:
        return idx
    full = pd.date_range(idx.min(), idx.max(), freq=step)
    return full.difference(idx)


def day_length_issues(
    index: pd.DatetimeIndex, tz: str, step: pd.Timedelta
) -> list[DayLengthIssue]:
    """Local delivery days whose row count differs from what DST implies.

    A short first or last day is a truncated dataset edge, not corruption, so it
    is not reported. Any other mismatch, including a surplus, is.
    """
    calendar = delivery_calendar(index, tz, step)
    if calendar.empty:
        return []
    by_day = calendar.groupby("delivery_date")["periods_in_day"].agg(["size", "first"])
    first_day, last_day = by_day.index.min(), by_day.index.max()
    issues: list[DayLengthIssue] = []
    for day, row in by_day.iterrows():
        observed, expected = int(row["size"]), int(row["first"])
        if observed == expected:
            continue
        if day in (first_day, last_day) and observed < expected:
            continue
        issues.append(
            DayLengthIssue(
                delivery_date=day,
                expected_periods=expected,
                observed_periods=observed,
            )
        )
    return issues


def hourly_product_mismatches(
    price: pd.Series,
    step: pd.Timedelta,
    quarter_hour_products_from: pd.Timestamp | None,
) -> int:
    """Count hours before the 15-minute switch whose sub-hourly prices differ.

    Only meaningful when the data is finer than hourly. Hours are UTC hours,
    which match local hourly products because Europe/Berlin is a whole number
    of hours from UTC.
    """
    if step >= HOUR:
        return 0
    if quarter_hour_products_from is None:
        raise ValueError(
            "sub-hourly prices need quarter_hour_products_from; without it the "
            "hourly-product check would be skipped silently"
        )
    idx = ensure_utc_index(price.index)
    hourly_era = price[idx < quarter_hour_products_from].dropna()
    if hourly_era.empty:
        return 0
    hours = pd.DatetimeIndex(hourly_era.index).floor("h")
    distinct = hourly_era.groupby(hours).nunique()
    return int((distinct > 1).sum())


def column_gaps(series: pd.Series) -> ColumnGaps:
    name = str(series.name)
    observed = np.flatnonzero(series.notna().to_numpy())
    if observed.size == 0:
        return ColumnGaps(
            column=name,
            missing_before_last_value=0,
            longest_gap_periods=0,
            trailing_missing=len(series),
        )
    body = series.iloc[: int(observed[-1]) + 1]
    missing = body.isna()
    run_id = (missing != missing.shift()).cumsum()
    longest = int(missing.groupby(run_id).sum().max())
    return ColumnGaps(
        column=name,
        missing_before_last_value=int(missing.sum()),
        longest_gap_periods=longest,
        trailing_missing=len(series) - len(body),
    )


def build_quality_report(
    frame: pd.DataFrame,
    tz: str,
    step: pd.Timedelta,
    price_column: str = PRICE_SERIES,
    quarter_hour_products_from: pd.Timestamp | None = None,
) -> QualityReport:
    idx = ensure_utc_index(frame.index)
    if idx.empty:
        raise ValueError("cannot assess an empty dataset")
    price: PriceSummary | None = None
    mismatches = 0
    if price_column in frame.columns:
        mismatches = hourly_product_mismatches(
            frame[price_column], step, quarter_hour_products_from
        )
        values = frame[price_column].dropna()
        if not values.empty:
            negative = values < 0
            price = PriceSummary(
                min_eur_mwh=float(values.min()),
                max_eur_mwh=float(values.max()),
                negative_periods=int(negative.sum()),
                negative_share=float(negative.mean()),
            )
    return QualityReport(
        start_utc=idx.min().to_pydatetime(),
        end_utc=idx.max().to_pydatetime(),
        resolution=(
            str(infer_resolution(idx)) if idx.nunique() >= 2 else "undetermined"
        ),
        n_rows=len(frame),
        expected_rows=len(pd.date_range(idx.min(), idx.max(), freq=step)),
        missing_timestamps=len(missing_timestamps(idx, step)),
        duplicate_timestamps=int(idx.duplicated().sum()),
        is_monotonic=bool(idx.is_monotonic_increasing),
        day_length_issues=day_length_issues(idx, tz, step),
        hourly_product_mismatches=mismatches,
        columns=[column_gaps(frame[c]) for c in frame.columns],
        price=price,
    )
