"""How the 15:00 to 21:00 window loses money: the readings and what they may say."""

from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.health.experiments import t1_mechanism as mech
from src.trading.backtest import block_bootstrap_index, paired_bootstrap_ci

TZ = "Europe/Berlin"
DAY = date(2025, 6, 10)  # summer time: local = UTC + 2


def _schedule(
    day: date,
    *,
    discharge: dict[int, float] | None = None,
    charge: dict[int, float] | None = None,
    price: dict[int, float] | None = None,
    forecast: dict[int, float] | None = None,
    start_soc: float = 1.0,
) -> pd.DataFrame:
    """96 quarter-hours of one schedule, set by local hour."""
    index = pd.date_range(
        pd.Timestamp(day, tz=TZ), periods=96, freq="15min"
    ).tz_convert("UTC")
    hours = index.tz_convert(TZ).hour
    discharge_mw = np.array([(discharge or {}).get(h, 0.0) for h in hours])
    charge_mw = np.array([(charge or {}).get(h, 0.0) for h in hours])
    realised = np.array([(price or {}).get(h, 50.0) for h in hours])
    soc = start_soc + np.cumsum(0.25 * (charge_mw - discharge_mw))
    return pd.DataFrame(
        {
            "target_day": day,
            "charge_mw": charge_mw,
            "discharge_mw": discharge_mw,
            "soc_mwh": soc,
            "sell_price": [(forecast or {}).get(h, 50.0) for h in hours],
            "realised_price": realised,
        },
        index=index,
    )


def test_the_window_is_read_in_local_time_with_stored_energy_at_its_edges() -> None:
    frame = _schedule(
        DAY,
        charge={12: 1.0},
        discharge={14: 1.0, 19: 1.0, 21: 1.0},
        price={12: 20.0, 14: 90.0, 19: 150.0, 21: 70.0},
    )

    row = mech.window_schedule(frame, TZ).loc[DAY]

    # Only 19:00 falls inside; 14:00 is before and 21:00 after.
    assert row["discharged_mwh"] == pytest.approx(1.0)
    assert row["discharge_eur"] == pytest.approx(150.0)
    assert row["charged_mwh"] == 0.0 and row["charge_eur"] == 0.0
    # 1.0 + 1.0 charged at 12:00 - 1.0 sold at 14:00, then 1.0 sold at 19:00.
    assert row["soc_at_start_mwh"] == pytest.approx(1.0)
    assert row["soc_at_end_mwh"] == pytest.approx(0.0)


def _window(**columns: float) -> pd.DataFrame:
    base = {
        "discharged_mwh": 0.0,
        "discharge_eur": 0.0,
        "charged_mwh": 0.0,
        "charge_eur": 0.0,
        "soc_at_start_mwh": 1.0,
        "soc_at_end_mwh": 1.0,
    }
    return pd.DataFrame([base | columns], index=[DAY])


def test_the_same_energy_at_a_better_price_is_all_price() -> None:
    median = _window(discharged_mwh=1.0, discharge_eur=100.0)
    foresight = _window(discharged_mwh=1.0, discharge_eur=160.0)

    parts = mech.decompose(median, foresight, 8.0).loc[DAY]

    assert parts["discharge_price"] == pytest.approx(60.0)
    assert parts["discharge_volume"] == 0.0 and parts["wear"] == 0.0
    assert parts["cash_gap_eur"] == pytest.approx(60.0)


def test_a_side_one_schedule_never_trades_is_all_volume() -> None:
    median = _window()
    foresight = _window(
        discharged_mwh=2.0, discharge_eur=300.0, charged_mwh=0.5, charge_eur=10.0
    )

    parts = mech.decompose(median, foresight, 8.0).loc[DAY]

    assert parts["discharge_volume"] == pytest.approx(300.0)
    assert parts["discharge_price"] == 0.0
    assert parts["charge_volume"] == pytest.approx(-10.0)
    assert parts["charge_price"] == 0.0
    assert parts["wear"] == pytest.approx(-16.0)
    assert parts["cash_gap_eur"] == pytest.approx(300.0 - 10.0 - 16.0)


def test_the_five_parts_add_up_to_the_cash_gap_on_any_days() -> None:
    rng = np.random.default_rng(3)
    days = pd.date_range("2025-01-01", periods=200).date

    def random_windows() -> pd.DataFrame:
        energy = rng.choice([0.0, 0.25, 1.0, 1.7], size=(200, 2))
        prices = rng.normal(80, 60, size=(200, 2))
        return pd.DataFrame(
            {
                "discharged_mwh": energy[:, 0],
                "discharge_eur": energy[:, 0] * prices[:, 0],
                "charged_mwh": energy[:, 1],
                "charge_eur": energy[:, 1] * prices[:, 1],
                "soc_at_start_mwh": 1.0,
                "soc_at_end_mwh": 1.0,
            },
            index=days,
        )

    parts = mech.decompose(random_windows(), random_windows(), 8.0)

    assert parts[list(mech.EFFECTS)].sum(axis=1).to_numpy() == pytest.approx(
        parts["cash_gap_eur"].to_numpy()
    )


def test_decompose_refuses_schedules_over_different_days() -> None:
    other = _window().set_axis([date(2025, 6, 11)], axis=0)

    with pytest.raises(ValueError, match="different days"):
        mech.decompose(_window(), other, 8.0)


@pytest.mark.parametrize(
    ("median", "foresight", "cycles", "expected"),
    [
        # Same energy, perfect foresight gets the better price: timing.
        (
            {"discharged_mwh": 1.0, "discharge_eur": 100.0},
            {"discharged_mwh": 1.2, "discharge_eur": 150.0},
            (1.0, 1.0),
            {"timing"},
        ),
        # 0.25 MWh apart is not "similar energy".
        (
            {"discharged_mwh": 1.0, "discharge_eur": 100.0},
            {"discharged_mwh": 1.25, "discharge_eur": 150.0},
            (1.0, 1.0),
            set(),
        ),
        # Exactly 0.25 MWh emptier at 15:00 and fuller at 21:00, half a cycle fewer.
        (
            {"soc_at_start_mwh": 0.75, "soc_at_end_mwh": 1.25},
            {"soc_at_start_mwh": 1.0, "soc_at_end_mwh": 1.0},
            (1.0, 1.5),
            {"emptier", "held_back", "fewer_cycles"},
        ),
        (
            {"soc_at_start_mwh": 0.8, "soc_at_end_mwh": 1.2},
            {"soc_at_start_mwh": 1.0, "soc_at_end_mwh": 1.0},
            (1.0, 1.49),
            set(),
        ),
    ],
)
def test_flags_use_the_thresholds_fixed_in_advance(
    median: dict[str, float],
    foresight: dict[str, float],
    cycles: tuple[float, float],
    expected: set[str],
) -> None:
    flags = mech.flag_days(
        _window(**median),
        _window(**foresight),
        pd.Series([cycles[0]], index=[DAY]),
        pd.Series([cycles[1]], index=[DAY]),
    ).loc[DAY]

    assert {name for name in mech.FLAGS if flags[name]} == expected


def test_the_peak_offset_is_forecast_hour_minus_realised_hour() -> None:
    early = _schedule(DAY, forecast={17: 120.0}, price={19: 200.0, 20: 180.0})
    tied = _schedule(
        date(2025, 6, 11), forecast={16: 90.0, 18: 90.0}, price={17: 99.0, 19: 99.0}
    )

    offsets = mech.peak_offsets(pd.concat([early, tied]), TZ)

    assert offsets.loc[DAY] == -2
    # Ties go to the earlier hour on both sides: 16 against 17.
    assert offsets.loc[date(2025, 6, 11)] == -1
    with pytest.raises(ValueError, match="no peak offset"):
        mech.offset_bucket(pd.Series([0, np.nan]))
    buckets = mech.offset_bucket(pd.Series([0, 1, -1, 2, -4]))
    assert buckets.tolist() == [
        "same_hour",
        "one_hour",
        "one_hour",
        "two_or_more_hours",
        "two_or_more_hours",
    ]


def test_block_cash_adds_up_to_the_days_profit() -> None:
    frame = _schedule(
        DAY,
        charge={3: 1.0, 12: 1.0},
        discharge={8: 1.0, 19: 1.0, 22: 0.5},
        price={3: 10.0, 8: 70.0, 12: 5.0, 19: 140.0, 22: 90.0},
    )

    cash = mech.block_cash(frame, TZ, 8.0).loc[DAY]

    discharged = 0.25 * frame["discharge_mw"]
    profit = (
        (discharged - 0.25 * frame["charge_mw"]) * frame["realised_price"]
        - 8.0 * discharged
    ).sum()
    assert list(cash.index) == list(mech.BLOCKS)
    assert cash.sum() == pytest.approx(profit)
    assert cash["18-20"] == pytest.approx(140.0 - 8.0)
    assert cash["00-05"] == pytest.approx(-10.0)


def test_a_share_interval_resamples_both_sums() -> None:
    days = pd.date_range("2025-01-01", periods=140).date
    rng = np.random.default_rng(1)
    cost = pd.Series(rng.gamma(2.0, 5.0, size=140), index=days)
    mask = pd.Series(rng.random(140) < 0.4, index=days)

    share = mech.share_interval(cost, mask, draws=500)
    everything = mech.share_interval(cost, mask | True, draws=500)

    assert share["share_pct"] == pytest.approx(100 * cost[mask].sum() / cost.sum())
    assert share["low"] < share["share_pct"] < share["high"]
    assert share["days"] == int(mask.sum())
    assert everything["share_pct"] == pytest.approx(100.0)
    assert everything["low"] == pytest.approx(100.0)


def test_a_seasonal_mean_draws_whole_blocks_but_averages_its_own_days() -> None:
    days = pd.date_range("2025-01-01", periods=70).date
    values = pd.Series(np.where(np.arange(70) % 2 == 0, 10.0, 0.0), index=days)
    even = pd.Series(np.arange(70) % 2 == 0, index=days)

    seasonal = mech.mean_interval(values, even, draws=300)
    overall = mech.mean_interval(values, draws=300)

    assert seasonal == {
        "mean": 10.0,
        "low": 10.0,
        "high": 10.0,
        "days": 35,
        "total": 350.0,
    }
    mean, low, high = paired_bootstrap_ci(values, draws=300)
    assert (overall["mean"], overall["low"], overall["high"]) == pytest.approx(
        (mean, low, high)
    )


def test_the_shared_resampling_keeps_blocks_of_consecutive_days() -> None:
    index = block_bootstrap_index(20, block_days=5, draws=50, seed=2)

    assert index.shape == (50, 20)
    steps = np.diff(index[:, :5], axis=1)
    assert (steps == 1).all()
    with pytest.raises(ValueError, match="no days"):
        block_bootstrap_index(0)


def _interval(mean: float, low: float, high: float) -> dict[str, float]:
    return {"mean": mean, "low": low, "high": high, "days": 730, "total": 730 * mean}


def _share(share: float, low: float, high: float, days: int = 100) -> dict[str, Any]:
    return {
        "share_pct": share,
        "low": low,
        "high": high,
        "days": days,
        "eur": 10 * share,
    }


def _summary(**changes: Any) -> dict[str, Any]:
    both = {"all": _interval(0.0, -1.0, 1.0)}
    both |= {"summer": _interval(0.0, -1.0, 1.0), "winter": _interval(0.0, -1.0, 1.0)}
    summary: dict[str, Any] = {
        "model": "lightgbm_conformal",
        "wear_eur_per_mwh": 8.0,
        "days": 730,
        "first_day": "2024-06-01",
        "last_day": "2026-05-31",
        "gap_eur": 16000.0,
        "window_cost": {"eur": 6500.0, "share_pct": 40.0, "low": 32.0, "high": 47.0},
        "negative_window_days": 136,
        "energy": {
            "discharged": _interval(-0.06, -0.2, 0.1),
            "charged": _interval(0.1, 0.05, 0.15),
        },
        "persistence": {"same_hour": 478, "days": 729},
        "window_direction": {
            "over": _share(10.0, -1.0, 20.0),
            "under": _share(90.0, 80.0, 101.0),
        },
        "decomposition": {name: dict(both) for name in (*mech.EFFECTS, "cash_gap_eur")},
        "blocks": {
            name: {
                "cash": _interval(0.0, -1.0, 1.0),
                "shapley_eur_per_day": 1.0,
                "over_eur_per_day": 0.6 if name in ("00-05", "11-14") else 0.2,
                "under_eur_per_day": 0.4 if name in ("00-05", "11-14") else 0.8,
            }
            for name in mech.BLOCKS
        },
        "stored": {
            edge: {
                "median": 1.0,
                "foresight": 1.0,
                "difference": _interval(0, -0.1, 0.1),
            }
            for edge in ("start", "end")
        },
        "flags": {
            "timing": _share(45.0, 31.0, 59.0),
            "emptier": _share(16.0, 4.0, 34.0),
            "held_back": _share(10.0, 2.0, 17.0),
            "fewer_cycles": _share(20.0, 7.0, 35.0),
            "no_flag": _share(27.0, 18.0, 39.0),
        },
        "offsets": {
            "same_hour": _share(58.0, 45.0, 71.0, 510),
            "one_hour": _share(26.0, 15.0, 38.0, 179),
            "two_or_more_hours": _share(16.0, 9.0, 24.0, 41),
            "off_by_an_hour_or_more": _share(42.0, 29.0, 55.0, 220),
        },
        "offset_direction": {"early": 112, "late": 108},
    }
    for key, value in changes.items():
        summary[key] = value
    return summary


def test_the_reading_names_no_main_mechanism_while_intervals_overlap() -> None:
    sentences = mech.reading(_summary())

    assert any("no single mechanism can be named" in s for s in sentences)
    assert any("cannot be told from the median schedule's" in s for s in sentences)
    assert any("forecasts below the realised price: 90.0%" in s for s in sentences)
    assert not any("comes from forecasts above" in s for s in sentences)
    # Nothing clears zero, so no part and no block is claimed.
    assert not any(s.startswith("Perfect foresight sells") for s in sentences)
    assert not any("beyond noise in" in s for s in sentences)


def test_the_reading_claims_only_what_clears_its_interval() -> None:
    summary = _summary()
    summary["flags"]["timing"] = _share(60.0, 50.0, 70.0)
    summary["decomposition"]["discharge_price"] = {
        "all": _interval(4.1, 2.6, 6.1),
        "summer": _interval(2.0, 0.4, 3.9),
        "winter": _interval(5.1, -0.1, 7.9),
    }
    summary["blocks"]["11-14"]["cash"] = _interval(12.0, 5.8, 18.3)
    summary["stored"]["end"]["difference"] = _interval(0.07, 0.03, 0.11)

    sentences = mech.reading(summary)
    page = mech.results_markdown(summary)

    assert any(
        "The timing flag carries more of the window's cost than any other" in s
        for s in sentences
    )
    assert any(
        "sells its window energy at better average prices" in s
        and "clear of zero June to September" in s
        for s in sentences
    )
    assert any("comes out ahead beyond noise in 11-14" in s for s in sentences)
    assert any("more energy stored at 21:00" in s for s in sentences)
    assert "| 11-14 | +12.00 (+5.80 to +18.30) | +1.00 |" in page
    assert "## Reading" in page and "\u2014" not in page and "\u2013" not in page


def test_the_window_edges_hold_on_the_days_the_clocks_change() -> None:
    for day, periods in ((date(2025, 3, 30), 92), (date(2025, 10, 26), 100)):
        index = pd.date_range(
            pd.Timestamp(day, tz=TZ), periods=periods, freq="15min"
        ).tz_convert("UTC")
        hours = index.tz_convert(TZ).hour
        discharge = np.where(hours == 19, 1.0, 0.0)
        charge = np.where(hours == 2, 1.0, 0.0)
        frame = pd.DataFrame(
            {
                "target_day": day,
                "charge_mw": charge,
                "discharge_mw": discharge,
                "soc_mwh": 1.0 + np.cumsum(0.25 * (charge - discharge)),
                "sell_price": 50.0,
                "realised_price": np.where(hours == 19, 120.0, 40.0),
            },
            index=index,
        )

        row = mech.window_schedule(frame, TZ).loc[day]

        # 02:00 does not exist on the spring day, so nothing charges there.
        stored = 1.0 if periods == 92 else (2.0 if periods == 96 else 3.0)
        assert row["soc_at_start_mwh"] == pytest.approx(stored), day
        assert row["soc_at_end_mwh"] == pytest.approx(stored - 1.0), day
        assert row["discharge_eur"] == pytest.approx(120.0), day


def test_the_two_direction_shares_mirror_each_other_past_100() -> None:
    days = pd.date_range("2025-01-01", periods=70).date
    total = pd.Series(np.tile([10.0, -2.0], 35), index=days)
    over = pd.Series(np.tile([-1.0, 0.5], 35), index=days)

    above = mech._cost_share_interval(total, over, draws=300)
    below = mech._cost_share_interval(total, total - over, draws=300)

    assert above["share_pct"] == pytest.approx(-6.25)
    assert above["share_pct"] + below["share_pct"] == pytest.approx(100.0)
    assert above["low"] + below["high"] == pytest.approx(100.0)
    assert below["share_pct"] > 100


def test_the_reading_does_not_clear_the_window_when_its_cash_could_hold_the_cost() -> (
    None
):
    summary = _summary()
    summary["decomposition"]["cash_gap_eur"]["all"] = _interval(-3.0, -10.0, 9.5)

    sentences = mech.reading(summary)

    assert any("cannot be told from the median schedule's" in s for s in sentences)
    assert not any("does not show up as cash" in s for s in sentences)


def test_volume_and_wear_sentences_follow_the_energy_not_the_sign_of_the_part() -> None:
    summary = _summary()
    # A negative average price flips the volume part's sign; the energy decides.
    summary["decomposition"]["charge_volume"] = {
        "all": _interval(2.0, 1.0, 3.0),
        "summer": _interval(3.0, 1.0, 5.0),
        "winter": _interval(-2.0, -4.0, -0.5),
    }
    summary["decomposition"]["wear"] = {
        "all": _interval(0.5, 0.2, 0.9),
        "summer": _interval(0.8, 0.2, 1.4),
        "winter": _interval(0.4, -0.1, 0.9),
    }

    sentences = mech.reading(summary)

    assert any(
        "buys more energy in the window" in s
        and "clear of zero June to September" in s
        and "and October to May clears it the other way" in s
        for s in sentences
    )
    assert any("discharges less in the window, saving wear" in s for s in sentences)


def test_the_page_says_why_a_share_can_pass_100_and_compares_the_peak_hour() -> None:
    page = mech.results_markdown(_summary())

    assert "on 136 days here, so a share and its interval" in page
    assert "| 11-14 | +0.00 (-1.00 to +1.00) | +1.00 | +0.60 | +0.40 |" in page
    assert "same hour as the day before's on 478 of 729 days" in page
    assert "carry 42.0% (29.0 to 55.0) of the window's cost" in page
    assert "forecasts above it in 00-05, 11-14." in page
