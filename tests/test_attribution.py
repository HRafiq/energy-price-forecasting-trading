"""Cost of forecast error by local hour block and direction, split by Shapley.

The hand-built days use a battery whose arithmetic is easy to follow: 1 MW / 1 MWh,
no losses, €5 wear per MWh discharged, starting and ending every day empty. On a
flat €50 day it stays idle, since buying and selling at €50 loses the €5 wear. A
full cycle buys 1 MWh at price b and sells it at price s for s - b - 5 euros, and
a full hour at 1 MW moves exactly 1 MWh.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

from src.config import Settings
from src.forecasting.base import quantile_column
from src.forecasting.walkforward import HoldoutAccessError
from src.trading import attribution as attribution_module
from src.trading.attribution import (
    DIRECTIONS,
    HOUR_BLOCKS,
    DayAttribution,
    attribute_day,
    attribute_days,
    counterfactual_frame,
    counterfactual_pnl,
    local_hour_blocks,
    ordering_seed,
    shapley_orderings,
)
from src.trading.battery import Battery
from src.trading.strategies import (
    MEDIAN_FORECAST,
    PERFECT_FORESIGHT,
    Strategy,
    dispatch_day,
    quantile_aware,
)

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
Z_SCORES = (-1.645, -1.2816, -0.6745, 0.0, 0.6745, 1.2816, 1.645)
QUANTILE_COLUMNS = [quantile_column(q) for q in QUANTILES]
SIMPLE = Battery(
    power_mw=1,
    capacity_mwh=1,
    round_trip_efficiency=1.0,
    degradation_eur_per_mwh=5,
    initial_soc_fraction=0,
)
DAY = date(2025, 11, 20)


def flat_day(settings: Settings, day: date, price: float = 50.0) -> pd.DataFrame:
    """Every forecast column and the realised price equal ``price`` all day."""
    start = settings.market.local_midnight_utc(day)
    end = settings.market.local_midnight_utc(day + timedelta(days=1))
    index = pd.date_range(
        start, end, freq="15min", inclusive="left", name="timestamp_utc"
    )
    frame = pd.DataFrame({column: price for column in QUANTILE_COLUMNS}, index=index)
    frame["actual"] = price
    frame["target_day"] = day
    frame["price_product_minutes"] = 15
    return frame


def shaped_day(
    settings: Settings,
    day: date,
    seed: int = 0,
    product_minutes: int = 15,
    band_scale: float = 25.0,
) -> pd.DataFrame:
    """A forecast fan and noisy realised prices shaped like a German day."""
    frame = flat_day(settings, day)
    hours = np.arange(len(frame)) / 4
    median = (
        70
        + 60 * np.sin((hours - 8) / 24 * 2 * np.pi)
        - 50 * np.exp(-(((hours - 13) / 2) ** 2))
    )
    for column, z in zip(QUANTILE_COLUMNS, Z_SCORES, strict=True):
        frame[column] = median + band_scale * z
    frame["actual"] = median + np.random.default_rng(seed).normal(0, 25, len(frame))
    frame["price_product_minutes"] = product_minutes
    return frame


def local_hour(frame: pd.DataFrame, settings: Settings, hour: int) -> pd.Series:
    hours = pd.DatetimeIndex(frame.index).tz_convert(settings.market.timezone).hour
    return pd.Series(hours == hour, index=frame.index)


def spike_day(settings: Settings) -> pd.DataFrame:
    """Realised €150 from 18:00 to 19:00 local, €50 otherwise; forecast €50."""
    frame = flat_day(settings, DAY)
    frame.loc[local_hour(frame, settings, 18), "actual"] = 150.0
    return frame


def cost(table: pd.DataFrame, block: str, direction: str) -> float:
    row = table[(table["block"] == block) & (table["direction"] == direction)]
    return float(row["cost_eur"].iloc[0])


def other_costs(table: pd.DataFrame, *keep: tuple[str, str]) -> pd.Series:
    kept = pd.Series(False, index=table.index)
    for block, direction in keep:
        kept |= (table["block"] == block) & (table["direction"] == direction)
    return table.loc[~kept, "cost_eur"]


def assert_adds_up(result: DayAttribution) -> None:
    assert result.attributed_eur == pytest.approx(result.gap_eur, abs=1e-2)
    assert abs(result.residual_eur) <= 1e-2
    empty = result.costs["periods"] == 0
    assert (result.costs.loc[empty, "cost_eur"] == 0.0).all()
    assert result.players == int((~empty).sum())


def test_blocks_cover_every_hour_once() -> None:
    hours = [hour for _, block in HOUR_BLOCKS for hour in block]
    assert sorted(hours) == list(range(24))


def test_orderings_are_exact_when_few_and_antithetic_when_many() -> None:
    orderings, exact = shapley_orderings(3, permutations=4, exact_max_players=4, seed=1)
    assert exact
    assert len(orderings) == 6 == len(set(orderings))
    orderings, exact = shapley_orderings(7, permutations=4, exact_max_players=4, seed=1)
    assert not exact
    assert len(orderings) == 8
    for first, second in zip(orderings[::2], orderings[1::2], strict=True):
        assert second == first[::-1]
        assert sorted(first) == list(range(7))
    again, _ = shapley_orderings(7, permutations=4, exact_max_players=4, seed=1)
    assert again == orderings
    assert ordering_seed(DAY, "median_forecast") == ordering_seed(
        DAY, "median_forecast"
    )
    assert ordering_seed(DAY, "median_forecast") != ordering_seed(DAY, "quantile_q25")


def test_missed_evening_spike_takes_the_whole_gap(settings: Settings) -> None:
    """The median strategy sees a flat €50 day and stays idle: P&L 0.

    Perfect foresight buys 1 MWh at €50 and sells it in the spike: 150 - 50 - 5 =
    €95, the gap. The four evening periods are under-forecast and form the only
    group, so they take the whole €95; every other group is empty and costs 0.
    """
    result = attribute_day(
        spike_day(settings), MEDIAN_FORECAST, SIMPLE, timezone=settings.market.timezone
    )
    assert result.strategy_pnl_eur == pytest.approx(0.0, abs=1e-6)
    assert result.ceiling_pnl_eur == pytest.approx(95.0, abs=1e-6)
    assert result.gap_eur == pytest.approx(95.0, abs=1e-6)
    table = result.costs
    assert len(table) == len(HOUR_BLOCKS) * len(DIRECTIONS)
    assert cost(table, "18-20", "under") == pytest.approx(95.0, abs=1e-6)
    np.testing.assert_allclose(other_costs(table, ("18-20", "under")), 0.0)
    evening = table[(table["block"] == "18-20") & (table["direction"] == "under")]
    assert int(evening["periods"].iloc[0]) == 4
    assert float(evening["mean_abs_error_eur_mwh"].iloc[0]) == pytest.approx(100.0)
    assert (result.players, result.orderings, result.exact) == (1, 1, True)
    assert result.solves == 3  # baseline, ceiling, and the one-group coalition
    assert_adds_up(result)


@pytest.mark.parametrize(
    ("midday", "strategy_pnl", "midday_cost", "gap"),
    [(53.0, 0.0, 0.0, 95.0), (60.0, -5.0, 5.0, 100.0)],
)
def test_midday_over_forecast_costs_only_what_it_changes(
    settings: Settings,
    midday: float,
    strategy_pnl: float,
    midday_cost: float,
    gap: float,
) -> None:
    """The missed evening spike again, plus q50 above a flat €50 from 12:00 to 13:00.

    Two groups, midday over (M) and evening under (E); v is the settled P&L.

    At €53 midday the spread 53 - 50 = 3 does not pay the €5 wear, so the schedule
    stays idle. v() = 0, v(M) = 0, v(E) = 95, v(ME) = 95. Both orders credit M with
    0 and E with 95.

    At €60 midday the strategy plans 60 - 50 - 5 = €5 but sells at the realised
    €50: v() = -5. Fixing midday leaves a flat forecast: v(M) = 0. Fixing the
    evening alone plans two cycles, midday (still believed at €60) and the spike:
    v(E) = -5 + 95 = 90. v(ME) = 95. Order M, E credits M with 5 and E with 95;
    order E, M credits E with 95 and M with 5. Shares 5 and 95 add to the gap 100.
    """
    frame = spike_day(settings)
    frame.loc[local_hour(frame, settings, 12), "q50"] = midday
    result = attribute_day(
        frame, MEDIAN_FORECAST, SIMPLE, timezone=settings.market.timezone
    )
    table = result.costs
    assert result.strategy_pnl_eur == pytest.approx(strategy_pnl, abs=1e-6)
    assert result.gap_eur == pytest.approx(gap, abs=1e-6)
    assert cost(table, "11-14", "over") == pytest.approx(midday_cost, abs=1e-6)
    assert cost(table, "18-20", "under") == pytest.approx(95.0, abs=1e-6)
    np.testing.assert_allclose(
        other_costs(table, ("11-14", "over"), ("18-20", "under")), 0.0
    )
    assert (result.players, result.orderings, result.solves) == (2, 2, 5)
    assert_adds_up(result)


def night_and_evening_day(settings: Settings) -> pd.DataFrame:
    """The missed evening spike plus q50 = €40 from 03:00 to 04:00 (realised €50)."""
    frame = spike_day(settings)
    frame.loc[local_hour(frame, settings, 3), "q50"] = 40.0
    return frame


def test_shapley_splits_an_interaction_evenly_over_orders(settings: Settings) -> None:
    """Two groups whose values interact: night under (N) and evening under (E).

    v(): the strategy buys at a believed €40 at night, sells at a believed €50,
    and settles 50 - 50 - 5 = -5. v(N): a flat forecast, idle, 0. v(E): it buys at
    the believed €40 night hour and sells in the spike, settling 150 - 50 - 5 = 95.
    v(NE) = 95, the ceiling. Gap 95 - (-5) = 100.

    Order N, E credits N with 0 - (-5) = 5 and E with 95 - 0 = 95. Order E, N
    credits E with 95 - (-5) = 100 and N with 95 - 95 = 0. Shapley shares: N 2.5,
    E 97.5, adding to 100. One group at a time would have said 5 and 100.

    With two players, one random ordering and its reverse are both orderings, so
    antithetic sampling with a single permutation reproduces the exact value.
    """
    frame = night_and_evening_day(settings)
    timezone = settings.market.timezone
    exact = attribute_day(frame, MEDIAN_FORECAST, SIMPLE, timezone=timezone)
    assert exact.exact
    assert exact.gap_eur == pytest.approx(100.0, abs=1e-6)
    assert cost(exact.costs, "00-05", "under") == pytest.approx(2.5, abs=1e-6)
    assert cost(exact.costs, "18-20", "under") == pytest.approx(97.5, abs=1e-6)
    assert_adds_up(exact)

    sampled = attribute_day(
        frame,
        MEDIAN_FORECAST,
        SIMPLE,
        timezone=timezone,
        permutations=1,
        exact_max_players=0,
    )
    assert not sampled.exact
    assert sampled.orderings == 2
    np.testing.assert_allclose(
        sampled.costs["cost_eur"], exact.costs["cost_eur"], atol=1e-9
    )


@pytest.mark.parametrize("forecast_strategy", [MEDIAN_FORECAST, quantile_aware(0.25)])
def test_shares_add_up_to_the_gap_on_a_many_group_day(
    settings: Settings, forecast_strategy: Strategy
) -> None:
    frame = shaped_day(settings, DAY, seed=4)
    battery = settings.battery
    ceiling = dispatch_day(frame, PERFECT_FORESIGHT, battery).settlement.pnl_eur
    everything = np.ones(len(frame), dtype=bool)
    exact_prices = counterfactual_pnl(frame, forecast_strategy, battery, everything)
    assert exact_prices == pytest.approx(ceiling, abs=1e-6)

    timezone = settings.market.timezone
    result = attribute_day(
        frame, forecast_strategy, battery, timezone=timezone, permutations=2
    )
    assert result.players > 4
    assert not result.exact
    assert result.orderings == 4
    assert result.ceiling_pnl_eur == pytest.approx(ceiling, abs=1e-6)
    assert result.gap_eur >= -1e-6
    assert_adds_up(result)
    with pytest.raises(RuntimeError, match="perfect-foresight"):
        attribute_day(
            frame,
            forecast_strategy,
            battery,
            timezone=timezone,
            ceiling_pnl_eur=ceiling + 1.0,
            permutations=1,
        )


def test_each_coalition_is_solved_once(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    frame = shaped_day(settings, DAY, seed=4)
    solved: list[tuple[int, ...]] = []
    real = counterfactual_pnl

    def counting(
        day: pd.DataFrame,
        strategy: Strategy,
        battery: Battery,
        mask: NDArray[np.bool_],
        *,
        time_limit_s: float = 60.0,
    ) -> float:
        solved.append(tuple(int(i) for i in np.flatnonzero(mask)))
        return real(day, strategy, battery, mask, time_limit_s=time_limit_s)

    monkeypatch.setattr(attribution_module, "counterfactual_pnl", counting)
    result = attribute_day(
        frame,
        MEDIAN_FORECAST,
        settings.battery,
        timezone=settings.market.timezone,
        permutations=3,
    )
    assert len(solved) == len(set(solved))
    # Six orderings each end in the grand coalition, solved once.
    assert len(solved) < result.orderings * result.players
    assert result.solves == len(solved) + 2
    assert_adds_up(result)


def test_exact_forecast_has_no_gap_and_no_costs(settings: Settings) -> None:
    frame = shaped_day(settings, DAY, seed=2)
    for column in QUANTILE_COLUMNS:
        frame[column] = frame["actual"]
    result = attribute_day(
        frame, MEDIAN_FORECAST, settings.battery, timezone=settings.market.timezone
    )
    assert result.gap_eur == pytest.approx(0.0, abs=1e-6)
    assert result.residual_eur == pytest.approx(0.0, abs=1e-6)
    assert (result.costs["periods"] == 0).all()
    assert (result.costs["cost_eur"] == 0.0).all()
    assert result.costs["mean_abs_error_eur_mwh"].isna().all()
    assert (result.players, result.solves) == (0, 2)


@pytest.mark.parametrize(
    ("day", "periods", "product_minutes", "per_block"),
    [
        (date(2025, 10, 26), 100, 15, (28, 20, 16, 12, 12, 12)),
        (date(2025, 3, 30), 92, 60, (20, 20, 16, 12, 12, 12)),
    ],
)
def test_dst_days_count_local_hours_into_blocks(
    settings: Settings,
    day: date,
    periods: int,
    product_minutes: int,
    per_block: tuple[int, ...],
) -> None:
    """On the autumn day the repeated 02:00 hour stays in 00-05: 7 hours, 28 periods.

    The spring day skips 02:00, leaving 5 hours, 20 periods. q50 sits €1 above the
    realised price everywhere, so every period is an over-forecast.
    """
    frame = shaped_day(settings, day, seed=1, product_minutes=product_minutes)
    frame["q50"] = frame["actual"] + 1.0
    assert len(frame) == periods
    blocks = local_hour_blocks(pd.DatetimeIndex(frame.index), settings.market.timezone)
    assert [int((blocks == name).sum()) for name, _ in HOUR_BLOCKS] == list(per_block)

    result = attribute_day(
        frame,
        MEDIAN_FORECAST,
        settings.battery,
        timezone=settings.market.timezone,
        permutations=1,
    )
    table = result.costs
    over = table[table["direction"] == "over"]
    assert list(over["block"]) == [name for name, _ in HOUR_BLOCKS]
    assert list(over["periods"]) == list(per_block)
    assert (table.loc[table["direction"] == "under", "periods"] == 0).all()
    np.testing.assert_allclose(over["mean_abs_error_eur_mwh"], 1.0)
    assert_adds_up(result)


def test_perfect_foresight_is_rejected(settings: Settings) -> None:
    frame = shaped_day(settings, DAY)
    everything = np.ones(len(frame), dtype=bool)
    timezone = settings.market.timezone
    with pytest.raises(ValueError, match="realised prices"):
        attribute_day(frame, PERFECT_FORESIGHT, settings.battery, timezone=timezone)
    with pytest.raises(ValueError, match="realised prices"):
        counterfactual_pnl(frame, PERFECT_FORESIGHT, settings.battery, everything)
    with pytest.raises(ValueError, match="no forecast-driven strategy"):
        attribute_days(
            frame,
            [DAY],
            (PERFECT_FORESIGHT,),
            settings.battery,
            timezone=timezone,
            holdout_start=settings.evaluation.holdout_start,
        )


def test_attribute_days_refuses_the_holdout(settings: Settings) -> None:
    holdout = settings.evaluation.holdout_start
    days = [holdout - timedelta(days=1), holdout]
    forecasts = pd.concat([shaped_day(settings, day) for day in days])
    with pytest.raises(HoldoutAccessError):
        attribute_days(
            forecasts,
            days,
            (MEDIAN_FORECAST,),
            settings.battery,
            timezone=settings.market.timezone,
            holdout_start=holdout,
        )


def test_results_are_deterministic_across_runs_and_workers(
    settings: Settings,
) -> None:
    days = [date(2025, 11, 20), date(2025, 11, 21)]
    forecasts = pd.concat(
        [shaped_day(settings, day, seed=i) for i, day in enumerate(days)]
    )
    runs = [
        attribute_days(
            forecasts,
            days,
            (PERFECT_FORESIGHT, MEDIAN_FORECAST),
            settings.battery,
            timezone=settings.market.timezone,
            holdout_start=settings.evaluation.holdout_start,
            workers=workers,
            permutations=1,
        )
        for workers in (1, 1, 2)
    ]
    first = runs[0]
    assert first.failed == {}
    assert len(first.costs) == len(days) * len(HOUR_BLOCKS) * len(DIRECTIONS)
    assert set(first.costs["strategy"]) == {"median_forecast"}
    assert list(first.days["target_day"]) == days
    assert not first.days["exact"].any()
    for other in runs[1:]:
        assert other.failed == {}
        pd.testing.assert_frame_equal(first.costs, other.costs)
        pd.testing.assert_frame_equal(
            first.days.drop(columns="seconds"), other.days.drop(columns="seconds")
        )
    np.testing.assert_allclose(
        first.days["attributed_eur"], first.days["gap_eur"], atol=1e-2
    )


def test_quantile_aware_counterfactual_replaces_both_legs(settings: Settings) -> None:
    frame = shaped_day(settings, DAY, seed=6, band_scale=40.0)
    mask = (
        local_hour_blocks(pd.DatetimeIndex(frame.index), settings.market.timezone)
        == "18-20"
    )
    strategy = quantile_aware(0.25)
    assert (strategy.sell_column, strategy.buy_column) == ("q25", "q75")

    changed = counterfactual_frame(frame, strategy, mask)
    for column in ("q25", "q75"):
        np.testing.assert_array_equal(
            changed.loc[mask, column], frame.loc[mask, "actual"]
        )
        np.testing.assert_array_equal(
            changed.loc[~mask, column], frame.loc[~mask, column]
        )
    untouched = [c for c in frame.columns if c not in ("q25", "q75")]
    pd.testing.assert_frame_equal(changed[untouched], frame[untouched])

    median = counterfactual_frame(frame, MEDIAN_FORECAST, mask)
    np.testing.assert_array_equal(median.loc[mask, "q50"], frame.loc[mask, "actual"])
    pd.testing.assert_frame_equal(median.drop(columns="q50"), frame.drop(columns="q50"))
    with pytest.raises(ValueError, match="one value per period"):
        counterfactual_frame(frame, strategy, mask[:-1])


def test_hourly_products_take_the_direction_of_their_mean_error() -> None:
    from src.trading.attribution import product_directions

    index = pd.date_range("2025-03-12 10:00", periods=8, freq="15min", tz="UTC")
    realised = pd.Series([50.0] * 8, index=index)
    forecast = pd.Series([45.0, 70.0, 60.0, 65.0, 52.0, 40.0, 48.0, 44.0], index=index)
    hourly = np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    quarter = np.arange(8, dtype=np.int64)

    assert list(product_directions(forecast, realised, hourly)) == (
        ["over"] * 4 + ["under"] * 4
    )
    assert list(product_directions(forecast, realised, quarter)) == [
        "under",
        "over",
        "over",
        "over",
        "over",
        "under",
        "under",
        "under",
    ]
