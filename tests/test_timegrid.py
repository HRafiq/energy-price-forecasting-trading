"""D3: daylight-saving days must not corrupt the delivery-day grid."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from src.timegrid import (
    delivery_calendar,
    delivery_periods,
    ensure_utc_index,
    expected_periods,
)

TZ = "Europe/Berlin"
SPRING = date(2024, 3, 31)  # clocks jump 02:00 -> 03:00, a 23-hour day
AUTUMN = date(2024, 10, 27)  # clocks fall 03:00 -> 02:00, a 25-hour day
ORDINARY = date(2024, 6, 15)


@pytest.mark.parametrize(("day", "n"), [(SPRING, 23), (AUTUMN, 25), (ORDINARY, 24)])
def test_expected_periods_follow_dst(day: date, n: int) -> None:
    assert expected_periods(day, TZ) == n
    assert expected_periods(day, TZ, pd.Timedelta(minutes=15)) == 4 * n


def test_delivery_periods_are_contiguous_utc_instants() -> None:
    periods = delivery_periods(SPRING, TZ)
    assert len(periods) == 23
    assert periods[0] == pd.Timestamp("2024-03-30 23:00", tz="UTC")
    assert periods[-1] == pd.Timestamp("2024-03-31 21:00", tz="UTC")
    assert (periods[1:] - periods[:-1] == pd.Timedelta(hours=1)).all()


def test_spring_day_has_no_local_two_oclock() -> None:
    cal = delivery_calendar(delivery_periods(SPRING, TZ), TZ)
    assert 2 not in set(cal["local_hour"])
    assert list(cal["period"]) == list(range(23))


def test_autumn_day_repeats_local_two_oclock_but_periods_stay_unique() -> None:
    cal = delivery_calendar(delivery_periods(AUTUMN, TZ), TZ)
    assert list(cal["local_hour"]).count(2) == 2
    assert list(cal["period"]) == list(range(25))
    assert set(cal["periods_in_day"]) == {25}


def test_calendar_restarts_period_ordinal_each_local_day() -> None:
    index = pd.date_range(
        pd.Timestamp("2024-10-25 22:00", tz="UTC"),
        pd.Timestamp("2024-10-28 23:00", tz="UTC"),
        freq="h",
        inclusive="left",
    )
    cal = delivery_calendar(index, TZ)
    for day, group in cal.groupby("delivery_date"):
        assert isinstance(day, date)
        assert list(group["period"]) == list(range(expected_periods(day, TZ)))


def test_naive_local_hour_grid_cannot_even_be_built_on_dst_days() -> None:
    """Why the rules exist: '24 local hours per day' is wrong twice a year."""
    spring = pd.date_range("2024-03-31 00:00", periods=24, freq="h")
    with pytest.raises(Exception, match="2024-03-31 02:00:00") as spring_exc:
        spring.tz_localize(TZ)
    assert "NonExistentTimeError" in type(spring_exc.value).__name__

    autumn = pd.date_range("2024-10-27 00:00", periods=24, freq="h")
    with pytest.raises(Exception, match="2024-10-27 02:00:00") as autumn_exc:
        autumn.tz_localize(TZ)
    assert "AmbiguousTimeError" in type(autumn_exc.value).__name__


def test_non_utc_indexes_are_rejected() -> None:
    naive = pd.date_range("2024-01-01", periods=3, freq="h")
    with pytest.raises(ValueError, match="naive"):
        ensure_utc_index(naive)
    with pytest.raises(ValueError, match="UTC"):
        ensure_utc_index(naive.tz_localize(TZ))


def test_off_grid_timestamps_are_rejected() -> None:
    index = pd.DatetimeIndex([pd.Timestamp("2024-01-01 10:30", tz="UTC")])
    with pytest.raises(ValueError, match="not aligned"):
        delivery_calendar(index, TZ)
