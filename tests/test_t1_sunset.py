"""Sun-anchored windows for the T1 split: the sun, the labels and the reading."""

from __future__ import annotations

import pickle
from datetime import date, timedelta
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.health.experiments import t1_sunset as t1

BERLIN = (52.52, 13.405)
TZ = "Europe/Berlin"


def _minutes(timestamp: pd.Timestamp) -> float:
    return timestamp.hour * 60 + timestamp.minute + timestamp.second / 60


@pytest.mark.parametrize(
    ("day", "sunrise_utc", "sunset_utc"),
    [
        # Published Berlin times: 04:43 and 21:33 summer time,
        # 08:15 and 15:54 winter time.
        (date(2024, 6, 21), 2 * 60 + 43, 19 * 60 + 33),
        (date(2024, 12, 21), 7 * 60 + 15, 14 * 60 + 54),
    ],
)
def test_sun_times_match_published_berlin_solstices(
    day: date, sunrise_utc: int, sunset_utc: int
) -> None:
    sunrise, sunset = t1.sun_times(day, *BERLIN)

    assert _minutes(sunrise) == pytest.approx(sunrise_utc, abs=5)
    assert _minutes(sunset) == pytest.approx(sunset_utc, abs=5)


def _labels(day: str) -> pd.Series:
    start = pd.Timestamp(day, tz=TZ)
    end = pd.Timestamp(date.fromisoformat(day) + timedelta(days=1), tz=TZ)
    index = pd.date_range(start, end, freq="15min", inclusive="left")
    return pd.Series(t1.sun_windows(index, timezone=TZ), index=index.tz_convert("UTC"))


def test_window_boundaries_on_a_known_winter_day() -> None:
    # At the centre of Germany sunrise is 07:20 UTC and sunset 15:13 UTC, rounded
    # to 07:00 and 15:00.
    labels = _labels("2024-12-21")

    def at(clock: str) -> str:
        return str(labels.at[pd.Timestamp(f"2024-12-21 {clock}", tz="UTC")])

    assert [at("05:45"), at("06:00"), at("09:45"), at("10:00")] == [
        "night",
        "morning",
        "morning",
        "midday",
    ]
    assert [at("11:45"), at("12:00"), at("14:45"), at("15:00")] == [
        "midday",
        "pre_sunset",
        "pre_sunset",
        "post_sunset",
    ]
    assert [at("17:45"), at("18:00")] == ["post_sunset", "night"]


def test_the_same_clock_hour_falls_in_different_windows_by_season() -> None:
    december, june = _labels("2024-12-21"), _labels("2024-06-21")

    local = pd.Timestamp("2024-12-21 17:00", tz=TZ).tz_convert("UTC")
    assert december.at[local] == "post_sunset"
    assert (
        june.at[pd.Timestamp("2024-06-21 17:00", tz=TZ).tz_convert("UTC")] == "midday"
    )
    assert (
        june.at[pd.Timestamp("2024-06-21 20:00", tz=TZ).tz_convert("UTC")]
        == "pre_sunset"
    )


def test_a_june_sunset_window_is_cut_at_the_end_of_the_delivery_day() -> None:
    hours = t1.window_hours(date(2024, 6, 21), TZ)

    assert hours["pre_sunset"] == 3.0
    assert hours["post_sunset"] == 2.0
    assert sum(hours.values()) == 24.0


@pytest.mark.parametrize(("day", "hours"), [("2024-10-27", 25.0), ("2024-03-31", 23.0)])
def test_clock_change_days_are_labelled_in_full(day: str, hours: float) -> None:
    labelled = t1.window_hours(date.fromisoformat(day), TZ)

    assert sum(labelled.values()) == hours
    assert labelled["pre_sunset"] == 3.0


def test_a_sunset_scheme_survives_the_trip_to_a_worker_process() -> None:
    label = partial(
        t1.sun_windows, timezone=TZ, latitude=t1.LATITUDE, longitude=t1.LONGITUDE
    )
    index = pd.date_range("2025-03-10", periods=96, freq="15min", tz="UTC")

    restored = pickle.loads(pickle.dumps(label))

    np.testing.assert_array_equal(restored(index), label(index))


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    days: list[dict[str, Any]] = []
    for day, gap, sunset_cost in (
        (date(2025, 1, 10), 100.0, 40.0),
        (date(2025, 7, 10), 50.0, 20.0),
    ):
        days.append({"target_day": day, "gap_eur": gap})
        rows += [
            {
                "target_day": day,
                "block": "pre_sunset",
                "direction": "under",
                "cost_eur": sunset_cost,
            },
            {
                "target_day": day,
                "block": "midday",
                "direction": "over",
                "cost_eur": gap - sunset_cost,
            },
        ]
    return pd.DataFrame(rows), pd.DataFrame(days)


def test_share_of_gap_reads_windows_months_and_direction() -> None:
    costs, days = _frames()

    assert t1.share_of_gap(costs, days, t1.SUNSET) == pytest.approx(60 / 150)
    assert t1.share_of_gap(costs, days, t1.SUNSET, months=t1.SUMMER) == pytest.approx(
        0.4
    )
    assert t1.share_of_gap(costs, days, ["midday"], direction="under") == pytest.approx(
        0.0
    )


def test_both_criteria_must_hold() -> None:
    held = t1.judge(
        sunset_summer=0.42, sunset_winter=0.38, sunset_all=0.41, clock_all=0.397
    )
    unstable = t1.judge(
        sunset_summer=0.50, sunset_winter=0.30, sunset_all=0.41, clock_all=0.397
    )
    diluted = t1.judge(
        sunset_summer=0.36, sunset_winter=0.34, sunset_all=0.35, clock_all=0.397
    )

    assert held["supported"] and held["season_difference_points"] == pytest.approx(4.0)
    assert not unstable["stable"] and unstable["concentrated"]
    assert diluted["stable"] and not diluted["concentrated"]


def _per_day(sunset: list[float], clock: list[float]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {"gap": [10.0] * len(sunset), "sunset": sunset, "clock": clock},
        index=pd.date_range("2025-05-01", periods=len(sunset), freq="D"),
    )
    frame["summer"] = pd.DatetimeIndex(frame.index).month.isin(sorted(t1.SUMMER))
    return frame


def test_equal_windows_differ_by_nothing() -> None:
    point, low, high = t1.share_difference_interval(_per_day([4.0] * 60, [4.0] * 60))

    assert (point, low, high) == pytest.approx((0.0, 0.0, 0.0))


def test_a_window_that_always_takes_more_clears_zero() -> None:
    point, low, high = t1.share_difference_interval(_per_day([5.0] * 60, [3.0] * 60))

    assert point == pytest.approx(20.0)
    assert low > 0


def test_the_seasonal_difference_reads_each_season_from_its_own_days() -> None:
    # 1 May to 29 June: May days take 20% of their gap, June days 50%.
    frame = _per_day([2.0] * 31 + [5.0] * 29, [0.0] * 60)

    point, low, high = t1.season_difference_interval(frame, "sunset")

    assert point == pytest.approx(30.0)
    assert low > 0 and high > 0


def _summary(**intervals: tuple[float, float]) -> dict[str, Any]:
    def row(value: float) -> dict[str, float]:
        return {"all": value, "summer": value, "winter": value, "too_low": value}

    keys = (
        "sunset_minus_clock",
        "sunset_season",
        "clock_season",
        "pre_sunset_season",
        "post_sunset_season",
        "clock_15_17_season",
        "clock_18_20_season",
    )
    bounds = {key: intervals.get(key, (-5.0, 5.0)) for key in keys}
    return {
        "days": 730,
        "first_day": "2024-06-01",
        "last_day": "2026-05-31",
        "gap_eur": 16424.0,
        "gap_max_difference_eur": 0.0,
        "sunset_all": 0.413,
        "clock_all": 0.398,
        "verdict": {
            "season_difference_points": 7.0,
            "stable": True,
            "concentrated": True,
            "supported": True,
        },
        "intervals": {
            key: {"mean": (low + high) / 2, "low": low, "high": high}
            for key, (low, high) in bounds.items()
        },
        "truncated": {"days": 70, "months": [6, 7], "summer_gap_share": 0.387},
        "shares": {
            "sun": {name: row(0.2) for name in t1.WINDOW_NAMES},
            "clock": {"15-17": row(0.2), "18-20": row(0.2)},
        },
        "by_month": [],
    }


def test_an_estimate_inside_the_noise_is_read_as_indistinguishable() -> None:
    page = t1.results_markdown(_summary(sunset_minus_clock=(-4.3, 6.5)))

    assert "cannot tell the sunset window from the clock window" in page
    assert "beyond sampling noise" not in page
    assert "Criterion 1 cannot separate the hypotheses" in page
    assert "No single window's seasonal shift is clear of zero." in page
    assert "On 70 days, in months 6, 7" in page


def test_a_clear_difference_and_clear_shifts_are_reported_as_such() -> None:
    page = t1.results_markdown(
        _summary(
            sunset_minus_clock=(0.5, 6.5),
            pre_sunset_season=(8.8, 36.1),
            post_sunset_season=(-44.1, -14.4),
            clock_15_17_season=(-28.2, -2.9),
            clock_18_20_season=(5.2, 39.9),
        )
    )

    assert "take more of the gap than the clock 15:00 to 20:59 window" in page
    assert "cannot tell" not in page
    assert "pre_sunset" in page and "+8.8 to +36.1" in page
    assert "later on the clock (from 15:00 to 17:59 into 18:00 to 20:59)" in page
    assert "earlier against sunset (from after sunset to before it)" in page


def test_one_clear_shift_alone_is_listed_without_a_direction_claim() -> None:
    page = t1.results_markdown(_summary(pre_sunset_season=(8.8, 36.1)))

    assert "pre_sunset +" in page and "(+8.8 to +36.1)." in page
    assert "In June to September the loss sits" not in page
