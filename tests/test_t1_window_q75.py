"""Selling the 15:00 to 21:00 window at q75: the curve, the criterion, the page."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.health.experiments import t1_window_q75 as q75
from src.trading.strategies import MEDIAN_FORECAST

TZ = "Europe/Berlin"


def test_only_selling_in_the_local_window_moves_to_q75() -> None:
    # 2025-10-26 has 25 hours in Berlin, so the window sits at a new UTC offset.
    index = pd.date_range(
        pd.Timestamp("2025-10-26", tz=TZ), periods=100, freq="15min"
    ).tz_convert("UTC")
    frame = pd.DataFrame({"q50": 50.0, "q75": 80.0}, index=index)

    curve = q75.with_window_sell(frame, TZ)[q75.SELL_COLUMN]

    hours = index.tz_convert(TZ).hour
    inside = (hours >= 15) & (hours < 21)
    assert (curve[inside] == 80.0).all() and (curve[~inside] == 50.0).all()
    assert int(inside.sum()) == 24
    assert q75.CANDIDATE.sell_column == q75.SELL_COLUMN
    assert q75.CANDIDATE.buy_column == MEDIAN_FORECAST.buy_column == "q50"


@pytest.mark.parametrize(
    ("low", "adopted"), [(0.01, True), (0.0, False), (-0.89, False)]
)
def test_the_criterion_needs_the_whole_interval_above_zero(
    low: float, adopted: bool
) -> None:
    assert q75.adopt({"mean": 1.0, "low": low, "high": 3.0}) is adopted


def _interval(mean: float, low: float, high: float, days: int = 730) -> dict[str, Any]:
    return {"mean": mean, "low": low, "high": high, "days": days, "total": mean * days}


def _summary(low: float, high: float, mean: float) -> dict[str, Any]:
    flat = _interval(0.0, -1.0, 1.0)
    return {
        "wear_eur_per_mwh": 8.0,
        "days": 730,
        "first_day": "2024-06-01",
        "last_day": "2026-05-31",
        "perfect_foresight_pnl_eur": 165922.0,
        "arms": {
            name: {
                "pnl_eur": 149498.0,
                "capture": 0.901,
                "cycles_per_day": 1.73,
                "sold_mwh_per_day": 3.28,
                "window_sold_mwh_per_day": 1.56,
                "plan_minus_settled_eur_per_day": -13.25,
            }
            for name in ("median_forecast", "window_q75_sell")
        },
        "difference": {
            "all": _interval(mean, low, high),
            "summer": flat,
            "winter": flat,
        },
        "window_cash": {"all": flat, "summer": flat, "winter": flat},
        "blocks": {
            name: flat
            for name in ("00-05", "06-10", "11-14", "15-17", "18-20", "21-23")
        },
    }


def test_the_page_states_a_loss_only_when_the_interval_lies_below_zero() -> None:
    losing = q75.results_markdown(_summary(-3.95, -0.89, -2.59))
    unclear = q75.results_markdown(_summary(-1.0, 2.0, 0.5))
    winning = q75.results_markdown(_summary(0.2, 2.0, 1.1))

    assert "is not adopted" in losing and "loses money" in losing
    assert "-€1,891 against median dispatch" in losing
    assert "is not adopted" in unclear and "loses money" not in unclear
    assert "is adopted: the whole interval is above zero" in winning
    assert (
        "| median_forecast | €149,498 | 90.10% | 1.73 | 3.28 | 1.56 | -13.25 |"
        in losing
    )
    assert "| cash in the window | +0.00 (-1.00 to +1.00) |" in losing
    for page in (losing, unclear, winning):
        assert "\u2014" not in page and "\u2013" not in page
        assert not np.any([line.endswith(" ") for line in page.splitlines()])


def test_a_summary_with_an_empty_season_is_written_as_strict_json() -> None:
    import json

    written = json.dumps(q75._finite({"winter": {"mean": float("nan"), "days": 0}}))

    assert written == '{"winter": {"mean": null, "days": 0}}'
    json.loads(written, parse_constant=lambda name: pytest.fail(name))
