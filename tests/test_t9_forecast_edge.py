"""What kind of day the forecast's edge lands on: the grouping and the verdicts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.health.experiments import t9_forecast_edge as fe


def _frame(values: Any, edge: Any) -> pd.DataFrame:
    days = pd.date_range("2024-06-01", periods=len(values), freq="D").date
    return pd.DataFrame(
        {
            "spread": np.asarray(values, dtype="float64"),
            "edge_eur": np.asarray(edge, dtype="float64"),
        },
        index=pd.Index(days, name="target_day"),
    )


def test_quintiles_split_into_five_and_their_shares_sum_to_one() -> None:
    frame = _frame(np.arange(100.0), np.arange(100.0))

    table = fe.quintile_table(frame, "spread", "edge_eur")

    assert [row["quintile"] for row in table] == ["Q1", "Q2", "Q3", "Q4", "Q5"]
    assert sum(row["share_of_total"] for row in table) == pytest.approx(1.0)
    assert all(row["days"] == 20 for row in table)
    # The lowest fifth holds the lowest values, the highest fifth the highest.
    assert table[0]["mean"] < table[-1]["mean"]


def test_a_characteristic_with_many_ties_makes_fewer_groups_than_five() -> None:
    """``peak_shift`` is mostly zero: forcing five equal bins is impossible."""
    ties = np.concatenate([np.zeros(60), np.array([1.0, 2.0, 3.0, 4.0] * 10)])

    table = fe.quintile_table(_frame(ties, np.arange(100.0)), "spread", "edge_eur")

    assert 1 < len(table) < fe.QUINTILES
    assert sum(row["share_of_total"] for row in table) == pytest.approx(1.0)


def test_the_verdict_needs_the_intervals_to_clear_each_other() -> None:
    def row(low: float, high: float) -> dict[str, Any]:
        return {"low": low, "high": high, "mean": (low + high) / 2}

    clear = [row(-1.0, 1.0), row(5.0, 9.0)]
    overlapping = [row(-1.0, 6.0), row(5.0, 9.0)]

    # direction +1: the edge should be larger at the top of the range.
    assert fe._verdict(clear, +1) is True
    assert fe._verdict(overlapping, +1) is False
    # direction -1 reverses which end has to win, so the same table now fails.
    assert fe._verdict(clear, -1) is False
    assert fe._verdict(list(reversed(clear)), -1) is True


def test_a_day_that_keeps_its_seasonal_ranking_scores_one() -> None:
    shape = np.arange(24.0) * 10.0

    kept = fe.characterise(shape.copy(), shape, hours=1.0, threshold=200.0)

    assert kept["shape_agreement"] == pytest.approx(1.0)
    assert kept["spread"] == pytest.approx(230.0)
    assert kept["peak_shift"] == pytest.approx(0.0)
    assert kept["spike"] is True


def test_a_day_that_reverses_its_seasonal_ranking_scores_minus_one() -> None:
    shape = np.arange(24.0) * 10.0

    broken = fe.characterise(shape[::-1].copy(), shape, hours=1.0, threshold=200.0)

    assert broken["shape_agreement"] == pytest.approx(-1.0)
    # The peak moved from the last hour to the first: 23 hours away.
    assert broken["peak_shift"] == pytest.approx(23.0)


def test_the_spike_flag_follows_the_day_maximum_not_its_mean() -> None:
    quiet = np.full(24, 199.0)
    one_spike = np.concatenate([np.zeros(23), [200.0]])

    assert fe.characterise(quiet, quiet, 1.0, 200.0)["spike"] is False
    assert fe.characterise(one_spike, quiet, 1.0, 200.0)["spike"] is True


def test_quarter_hour_days_measure_the_peak_shift_in_hours() -> None:
    shape = np.zeros(96)
    shape[40] = 100.0
    realised = np.zeros(96)
    realised[48] = 100.0  # two hours later

    moved = fe.characterise(realised, shape, hours=0.25, threshold=200.0)

    assert moved["peak_shift"] == pytest.approx(2.0)


def test_the_edge_is_the_forecast_minus_the_fixed_rule_and_not_the_reverse() -> None:
    """The sign is the whole experiment, so pin it against the real function."""
    day = pd.Timestamp("2024-06-01").date()
    joined = pd.DataFrame(
        {"median_forecast": [100.0], fe.FIXED: [60.0], "perfect_foresight": [200.0]},
        index=pd.Index([day], name="target_day"),
    )

    out = fe.add_edge(joined)

    # A forecast that beat the rule must show a POSITIVE edge.
    assert out["edge_eur"].iloc[0] == pytest.approx(40.0)
    assert out["edge_share"].iloc[0] == pytest.approx(0.2)


def test_a_forecast_that_loses_to_the_fixed_rule_shows_a_negative_edge() -> None:
    day = pd.Timestamp("2024-06-01").date()
    joined = pd.DataFrame(
        {"median_forecast": [40.0], fe.FIXED: [70.0], "perfect_foresight": [100.0]},
        index=pd.Index([day], name="target_day"),
    )

    out = fe.add_edge(joined)

    assert out["edge_eur"].iloc[0] == pytest.approx(-30.0)
    assert out["edge_share"].iloc[0] == pytest.approx(-0.3)


def test_the_page_states_each_verdict_from_the_intervals() -> None:
    """P1 is built to hold and P2 to fail, so the page must tell them apart."""

    def table(holds: bool, direction: int) -> list[dict[str, Any]]:
        ends = [(-1.0, 1.0), (5.0, 9.0)] if holds else [(-1.0, 6.0), (5.0, 9.0)]
        if direction < 0:
            ends.reverse()
        return [
            {
                "quintile": f"Q{i + 1}",
                "low": low,
                "high": high,
                "mean": (low + high) / 2,
                "low_edge": 0.0,
                "high_edge": 1.0,
                "share_of_total": 0.5,
                "days": 365,
                "total": 0.0,
            }
            for i, (low, high) in enumerate(ends)
        ]

    holding = {"shape_agreement": True, "spread": False, "peak_shift": False}
    summary: dict[str, Any] = {
        "days": 730,
        "first_day": "2024-06-01",
        "last_day": "2026-05-31",
        "fixed_arm": fe.FIXED,
        "spike_threshold_eur_mwh": 200.0,
        "total_edge_eur": 8575.65,
        "mean_edge_eur": 11.75,
        "correlations": dict.fromkeys(fe.CHARACTERISTICS, -0.21),
        "quintiles": {
            value: {
                name: table(holding[name], direction)
                for name, direction in fe.CHARACTERISTICS.items()
            }
            for value in ("edge_eur", "edge_share")
        },
        "spike": {
            value: {
                "spike_days": {"mean": 1.0, "low": 0.0, "high": 2.0, "days": 143},
                "other_days": {"mean": 1.0, "low": 0.0, "high": 2.0, "days": 587},
                "count": 143,
            }
            for value in ("edge_eur", "edge_share")
        },
    }
    summary["verdicts"] = {
        name: {
            v: fe._verdict(summary["quintiles"][v][name], d)
            for v in ("edge_eur", "edge_share")
        }
        for name, d in fe.CHARACTERISTICS.items()
    }
    summary["verdicts"]["spike"] = dict.fromkeys(("edge_eur", "edge_share"), False)
    summary["capture_by_agreement"] = [
        {
            "quintile": "Q1",
            "days": 146,
            "low_agreement": -0.39,
            "high_agreement": 0.74,
            "forecast_capture": 0.808,
            "fixed_capture": 0.673,
            "gap": 0.135,
        },
        {
            "quintile": "Q5",
            "days": 146,
            "low_agreement": 0.95,
            "high_agreement": 0.99,
            "forecast_capture": 0.929,
            "fixed_capture": 0.932,
            "gap": -0.003,
        },
    ]

    page = fe.results_markdown(summary)

    # Only P1 was built to hold, in both columns; everything else must read fails.
    assert page.count("HOLDS") == 2
    p1 = next(line for line in page.splitlines() if "**P1**" in line)
    p2 = next(line for line in page.splitlines() if "**P2**" in line)
    assert p1.endswith("| HOLDS | HOLDS |") and p2.endswith("| fails | fails |")
    assert "143 of 730 days" in page
    # The post-hoc table reports the gap in percentage points, not fractions.
    assert "+13.5 pts" in page and "-0.3 pts" in page
    assert "not pre-registered" in page


def test_capture_by_agreement_divides_each_arm_by_the_days_own_ceiling() -> None:
    days = pd.date_range("2024-06-01", periods=10, freq="D").date
    frame = pd.DataFrame(
        {
            "shape_agreement": np.linspace(0.0, 1.0, 10),
            "median_forecast": np.full(10, 90.0),
            fe.FIXED: np.concatenate([np.full(5, 50.0), np.full(5, 95.0)]),
            "perfect_foresight": np.full(10, 100.0),
        },
        index=pd.Index(days, name="target_day"),
    )

    rows = fe.capture_by_agreement(frame)

    assert [row["days"] for row in rows] == [2, 2, 2, 2, 2]
    # The arms are built so the fixed rule is behind on the days that broke
    # pattern and ahead on the ordinary ones, which is the real finding's shape.
    assert rows[0]["forecast_capture"] == pytest.approx(0.9)
    assert rows[0]["fixed_capture"] == pytest.approx(0.5)
    assert rows[0]["gap"] == pytest.approx(0.4)
    assert rows[-1]["gap"] == pytest.approx(-0.05)
