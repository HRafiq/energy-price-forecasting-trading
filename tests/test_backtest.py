"""Backtest metrics, the mean forecast and the experiment suites on synthetic days."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.forecasting.base import quantile_column
from src.forecasting.walkforward import HoldoutAccessError
from src.trading import backtest as bt
from src.trading.run_strategies import run_strategies, select_days
from src.trading.strategies import MEAN, MEDIAN_FORECAST, add_mean_forecast
from tests.test_trading import Z_SCORES, day_frame

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


# --- metrics ------------------------------------------------------------------


def test_max_drawdown_counts_from_a_zero_start() -> None:
    """Cumulative 10, 5, -2, 18, -12: the worst fall is from 18 to -12."""
    assert bt.max_drawdown(pd.Series([10.0, -5.0, -7.0, 20.0, -30.0])) == 30.0
    assert bt.max_drawdown(pd.Series([-5.0, 3.0])) == 5.0
    assert bt.max_drawdown(pd.Series([1.0, 2.0])) == 0.0
    assert bt.max_drawdown(pd.Series([], dtype=float)) == 0.0


def test_losing_streak_and_concentration() -> None:
    daily = pd.Series([1.0, -1.0, -2.0, 3.0, -1.0, 5.0, 2.0, 1.0, 4.0, 8.0])
    assert bt.longest_losing_streak(daily) == 2
    assert bt.top_share(daily, 0.1) == pytest.approx(8.0 / 20.0)
    assert np.isnan(bt.top_share(pd.Series([-1.0, -2.0])))


def test_trading_metrics_extend_the_phase3_summary() -> None:
    days = [date(2025, 11, 1) + timedelta(days=i) for i in range(3)]
    pnl = pd.DataFrame(
        {
            "target_day": days * 2,
            "strategy": ["perfect_foresight"] * 3 + ["median_forecast"] * 3,
            "product_minutes": 15,
            "pnl_eur": [10.0, 20.0, 30.0, 8.0, -2.0, 25.0],
            "discharged_mwh": 1.0,
            "cycles": 0.5,
        }
    )
    metrics = bt.trading_metrics(pnl, ["perfect_foresight", "median_forecast"])
    median = metrics.loc["median_forecast"]
    assert median["capture_ratio"] == pytest.approx(31 / 60)
    assert median["max_drawdown_eur"] == pytest.approx(2.0)
    assert median["longest_losing_streak_days"] == 1
    assert metrics.loc["perfect_foresight", "max_drawdown_eur"] == 0.0
    capture = bt.monthly_capture(pnl)
    assert capture.loc[0, "median_forecast_capture"] == pytest.approx(31 / 60)


# --- mean forecast ------------------------------------------------------------


def test_mean_forecast_on_known_distributions() -> None:
    """A straight quantile function 100 x level has mean 50; a symmetric one
    has its median as mean; a long upper tail lifts the mean above the median."""
    rows = {
        "linear": {q: 100 * q for q in QUANTILES},
        "symmetric": {q: 50 + 20 * Z_SCORES[q] for q in QUANTILES},
        "right_skewed": {
            0.05: 40.0,
            0.10: 42.0,
            0.25: 46.0,
            0.50: 50.0,
            0.75: 60.0,
            0.90: 90.0,
            0.95: 160.0,
        },
    }
    frame = pd.DataFrame(
        {quantile_column(q): [rows[r][q] for r in rows] for q in QUANTILES},
        index=list(rows),
    )
    means = add_mean_forecast(frame, QUANTILES)[MEAN]
    assert means["linear"] == pytest.approx(50.0)
    assert means["symmetric"] == pytest.approx(50.0)
    assert means["right_skewed"] > 60.0


# --- synthetic forecasts ---------------------------------------------------------


def test_synthetic_curves_place_the_same_error_in_different_periods(
    settings: Settings,
) -> None:
    frame = day_frame(settings, date(2025, 11, 20), seed=1)
    traded = np.zeros(len(frame), dtype=bool)
    traded[30:40] = True
    traded[70:80] = True
    curves = bt.synthetic_curves(
        frame, traded, 12.0, np.random.default_rng(0), settings.market.timezone
    )
    actual = frame["actual"].to_numpy()
    for name in ("noise_everywhere", "noise_idle_periods", "noise_traded_periods"):
        curve = curves[name]
        assert curve is not None
        assert np.abs(curve - actual).mean() == pytest.approx(12.0)
    idle, busy = curves["noise_idle_periods"], curves["noise_traded_periods"]
    assert idle is not None and busy is not None
    np.testing.assert_array_equal(idle[traded], actual[traded])
    np.testing.assert_array_equal(busy[~traded], actual[~traded])
    level = curves["level_shift"]
    assert level is not None
    np.testing.assert_allclose(level - actual, 12.0)
    early = curves["peak_one_hour_early"]
    assert early is not None
    hours = pd.DatetimeIndex(frame.index).tz_convert(settings.market.timezone).hour
    at_six_pm = int(np.flatnonzero(hours == 18)[0])
    at_ten_am = int(np.flatnonzero(hours == 10)[0])
    assert early[at_six_pm] == actual[at_six_pm + 4]
    assert early[at_ten_am] == actual[at_ten_am]
    no_trades = bt.synthetic_curves(
        frame,
        np.zeros(len(frame), dtype=bool),
        12.0,
        np.random.default_rng(0),
        settings.market.timezone,
    )
    assert no_trades["noise_traded_periods"] is None


def test_synthetic_table_scores_variants_against_the_ceiling(
    settings: Settings,
) -> None:
    days = [date(2025, 11, 20), date(2025, 11, 21)]
    forecasts = pd.concat([day_frame(settings, d, seed=i) for i, d in enumerate(days)])
    daily, summary = bt.synthetic_table(
        forecasts, days, settings.battery, settings, 10.0
    )
    assert set(daily["variant"]) <= set(bt.SYNTHETIC_VARIANTS)
    assert (summary["capture_ratio"] <= 1 + 1e-9).all()
    noisy = summary.set_index("variant").loc["noise_everywhere"]
    assert noisy["mean_mae_eur_mwh"] == pytest.approx(10.0)
    assert noisy["days"] == 2


# --- degradation sweep --------------------------------------------------------------


def test_degradation_sweep_charges_the_true_wear(settings: Settings) -> None:
    days = [date(2025, 11, 20), date(2025, 11, 21)]
    forecasts = pd.concat([day_frame(settings, d, seed=i) for i, d in enumerate(days)])
    daily, summary = bt.degradation_table(
        forecasts, days, settings.battery, settings, (0.0, 8.0)
    )
    true_wear = settings.battery.degradation_eur_per_mwh
    np.testing.assert_allclose(
        daily["pnl_true_wear_eur"],
        daily["revenue_eur"] - true_wear * daily["discharged_mwh"],
    )
    capped = summary[summary["cycle_cap"] == settings.battery.max_cycles_per_day]
    median = capped[capped["strategy"] == "median_forecast"].set_index(
        "optimizer_wear_eur_per_mwh"
    )
    cycles = median["cycles_per_day"]
    assert float(cycles.loc[0.0]) >= float(cycles.loc[8.0])
    ceiling = capped[
        (capped["strategy"] == "perfect_foresight")
        & (capped["optimizer_wear_eur_per_mwh"] == true_wear)
    ]
    best = float(ceiling["pnl_true_wear_eur"].iloc[0])
    # Perfect foresight optimizing the true wear is the best any row can earn
    # under the same cycle cap.
    assert (capped["pnl_true_wear_eur"] <= best + 1e-6).all()
    assert summary["cycle_cap"].isna().any()


# --- hold-out discipline ------------------------------------------------------------


def test_holdout_needs_explicit_permission(settings: Settings) -> None:
    holdout = settings.evaluation.holdout_start
    days = [holdout, holdout + timedelta(days=1)]
    forecasts = pd.concat([day_frame(settings, d, seed=i) for i, d in enumerate(days)])
    columns = {"actual", "q50", "price_product_minutes"}
    with pytest.raises(HoldoutAccessError):
        select_days(forecasts, days[0], days[1], settings, columns)
    allowed, skipped = select_days(
        forecasts, days[0], days[1], settings, columns, allow_holdout=True
    )
    assert allowed == days and skipped == {}
    with pytest.raises(HoldoutAccessError):
        run_strategies(
            forecasts,
            days,
            (MEDIAN_FORECAST,),
            settings.battery,
            holdout_start=holdout,
        )
    _, pnl, _ = run_strategies(
        forecasts,
        days,
        (MEDIAN_FORECAST,),
        settings.battery,
        holdout_start=holdout,
        allow_holdout=True,
    )
    assert len(pnl) == 2


def test_holdout_suite_refuses_without_confirmation() -> None:
    with pytest.raises(SystemExit):
        bt.main(["--suite", "holdout", "--no-mlflow"])


# --- attribution summary --------------------------------------------------------------


def test_attribution_summary_columns_add_up_to_the_gap() -> None:
    day = date(2025, 11, 20)
    costs = pd.DataFrame(
        {
            "target_day": [day] * 3,
            "strategy": ["median_forecast"] * 3,
            "block": ["18-20", "11-14", "18-20"],
            "direction": ["under", "over", "over"],
            "periods": [12, 16, 4],
            "mean_abs_error_eur_mwh": [30.0, 5.0, 8.0],
            "cost_eur": [90.0, 12.0, -2.0],
            "model": "m",
        }
    )
    days = pd.DataFrame(
        {
            "target_day": [day],
            "strategy": ["median_forecast"],
            "strategy_pnl_eur": [200.0],
            "ceiling_pnl_eur": [300.0],
            "gap_eur": [100.0],
            "attributed_eur": [100.0],
            "residual_eur": [0.0],
            "model": ["m"],
        }
    )
    row = bt.attribution_summary(costs, days).iloc[0]
    assert row["cost_18-20_under_eur"] == 90.0
    assert row["cost_18-20_over_eur"] == -2.0
    assert row["cost_00-05_under_eur"] == 0.0
    assert row["cost_under_eur"] + row["cost_over_eur"] == pytest.approx(row["gap_eur"])
    group_columns = [
        c for c in row.index if c.startswith("cost_") and c.count("_") == 3
    ]
    assert len(group_columns) == 12
    assert sum(row[c] for c in group_columns) == pytest.approx(row["gap_eur"])


def test_attribution_needs_hold_out_permission(settings: Settings) -> None:
    from src.trading.attribution import attribute_days

    holdout = settings.evaluation.holdout_start
    frame = day_frame(settings, holdout, seed=2)
    with pytest.raises(HoldoutAccessError):
        attribute_days(
            frame,
            [holdout],
            (MEDIAN_FORECAST,),
            settings.battery,
            timezone=settings.market.timezone,
            holdout_start=holdout,
        )
    result = attribute_days(
        frame,
        [holdout],
        (MEDIAN_FORECAST,),
        settings.battery,
        timezone=settings.market.timezone,
        holdout_start=holdout,
        allow_holdout=True,
    )
    assert result.days["gap_eur"].iloc[0] == pytest.approx(
        result.days["attributed_eur"].iloc[0], abs=0.01
    )


# --- paired bootstrap ------------------------------------------------------------


def test_paired_bootstrap_interval() -> None:
    constant = pd.Series([2.0] * 60)
    assert bt.paired_bootstrap_ci(constant) == (2.0, 2.0, 2.0)
    alternating = pd.Series([5.0, -5.0] * 100)
    mean, low, high = bt.paired_bootstrap_ci(alternating)
    assert mean == 0.0 and low < 0.0 < high
    assert bt.paired_bootstrap_ci(alternating) == bt.paired_bootstrap_ci(alternating)
    with pytest.raises(ValueError):
        bt.paired_bootstrap_ci(pd.Series([], dtype=float))


def test_pnl_differences_pairs_days_by_model() -> None:
    days = [date(2025, 11, 1) + timedelta(days=i) for i in range(14)]
    pnl = pd.DataFrame(
        {
            "target_day": days * 2,
            "model": ["base"] * 14 + ["other"] * 14,
            "strategy": "median_forecast",
            "pnl_eur": [100.0] * 14 + [103.0] * 14,
        }
    )
    table = bt.pnl_differences(pnl, base_model="base", strategy="median_forecast")
    row = table.iloc[0]
    assert row["model"] == "other" and row["days"] == 14
    assert row["mean_daily_difference_eur"] == pytest.approx(3.0)
    assert row["total_difference_eur"] == pytest.approx(42.0)
    assert row["ci_low_eur"] == pytest.approx(3.0)


# --- review follow-ups ---------------------------------------------------------------


@pytest.mark.parametrize("day", [date(2025, 3, 30), date(2025, 10, 26)])
def test_one_hour_early_shift_is_one_hour_on_dst_days(
    settings: Settings, day: date
) -> None:
    frame = day_frame(settings, day, seed=4)
    curves = bt.synthetic_curves(
        frame,
        np.zeros(len(frame), dtype=bool),
        5.0,
        np.random.default_rng(0),
        settings.market.timezone,
    )
    early = curves["peak_one_hour_early"]
    assert early is not None
    index = pd.DatetimeIndex(frame.index)
    hours = index.tz_convert(settings.market.timezone).hour.to_numpy()
    shifted = np.flatnonzero(np.isin(hours, list(range(14, 22))))
    shifted = shifted[shifted + 4 < len(frame)]
    actual = frame["actual"].to_numpy()
    np.testing.assert_array_equal(early[shifted], actual[shifted + 4])
    assert ((index[shifted + 4] - index[shifted]) == pd.Timedelta(hours=1)).all()


def test_block_bootstrap_widens_the_interval_for_dependent_days() -> None:
    runs = pd.Series(np.repeat(np.random.default_rng(3).normal(size=60), 7))
    _, low_week, high_week = bt.paired_bootstrap_ci(runs, block_days=7)
    _, low_day, high_day = bt.paired_bootstrap_ci(runs, block_days=1)
    assert high_week - low_week > 1.5 * (high_day - low_day)


def _pnl_rows(
    model: str, days: list[date], months_pnl: float = 8.0
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "target_day": day,
            "strategy": strategy,
            "product_minutes": 15,
            "pnl_eur": value,
            "discharged_mwh": 1.0,
            "cycles": 0.5,
        }
        for day in days
        for strategy, value in (
            ("perfect_foresight", 10.0),
            ("median_forecast", months_pnl),
        )
    ]


def test_common_days_are_rechecked_after_solving() -> None:
    days = [date(2025, 11, 1) + timedelta(days=i) for i in range(3)]
    pnl = pd.DataFrame(_pnl_rows("a", days) + _pnl_rows("b", days[:2]))
    kept, metrics = bt.restrict_to_common_days(
        pnl, ["perfect_foresight", "median_forecast"]
    )
    assert set(kept["target_day"]) == set(days[:2])
    assert (metrics["days"] == 2).all()
    assert set(metrics["model"]) == {"a", "b"}


def test_capture_in_months_keeps_the_season() -> None:
    pnl = pd.DataFrame(
        _pnl_rows("m", [date(2025, 7, 1)], 9.0)
        + _pnl_rows("m", [date(2025, 8, 1)], 5.0)
    )
    july = bt.capture_in_months(pnl, {7}).set_index(["model", "strategy"])
    both = bt.capture_in_months(pnl, {7, 8}).set_index(["model", "strategy"])
    assert july.loc[("m", "median_forecast"), "capture_ratio"] == pytest.approx(0.9)
    assert both.loc[("m", "median_forecast"), "capture_ratio"] == pytest.approx(0.7)


def test_holdout_suite_refuses_a_second_run(
    settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = settings.data.model_copy(update={"processed_dir": tmp_path / "processed"})
    local = settings.model_copy(update={"data": data})
    folder = local.data.processed_path / "backtest" / "holdout"
    folder.mkdir(parents=True)
    (folder / "summary.csv").write_text("done", encoding="utf-8")
    monkeypatch.setattr(bt, "load_settings", lambda path: local)

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("the hold-out suite must not run a second time")

    monkeypatch.setitem(bt.SUITE_RUNNERS, "holdout", forbidden)
    with pytest.raises(SystemExit):
        bt.main(["--suite", "holdout", "--confirm-holdout", "--no-mlflow"])
