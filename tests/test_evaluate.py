from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.config import Settings
from src.forecasting.evaluate import central_intervals, pinball, score, segment_scores

QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
BANDS = {
    "q05": 0.0,
    "q10": 10.0,
    "q25": 25.0,
    "q50": 50.0,
    "q75": 75.0,
    "q90": 90.0,
    "q95": 100.0,
}


def _table(actuals: list[float], day: date, product: int = 15) -> pd.DataFrame:
    index = pd.date_range(
        pd.Timestamp(day, tz="UTC"), periods=len(actuals), freq="15min"
    )
    frame = pd.DataFrame(dict.fromkeys(BANDS, 0.0), index=index)
    for column, value in BANDS.items():
        frame[column] = value
    frame["actual"] = actuals
    frame["target_day"] = day
    frame["price_product_minutes"] = product
    return frame


def test_pinball_matches_the_primer_example() -> None:
    assert pinball(np.array([120.0]), np.array([100.0]), 0.9)[0] == pytest.approx(18.0)
    assert pinball(np.array([80.0]), np.array([100.0]), 0.9)[0] == pytest.approx(2.0)


def test_central_intervals_pair_symmetric_quantiles() -> None:
    assert central_intervals(QUANTILES) == [
        (0.05, 0.95, 90),
        (0.10, 0.90, 80),
        (0.25, 0.75, 50),
    ]


def test_score_counts_coverage_error_and_missing_actuals() -> None:
    result = score(_table([50.0, -10.0, 120.0, np.nan], date(2024, 1, 1)), QUANTILES)

    assert result["periods"] == 3
    assert result["missing actuals"] == 1
    assert result["coverage 90%"] == pytest.approx(1 / 3)
    assert result["coverage 50%"] == pytest.approx(1 / 3)
    assert result["width 90%"] == pytest.approx(100.0)
    assert result["MAE of median"] == pytest.approx(130 / 3)
    assert result["bias of median"] == pytest.approx(-10 / 3)


def test_segments_flag_negative_spike_and_quarter_hour_days(settings: Settings) -> None:
    negative_day = _table([5.0, -1.0], date(2024, 1, 1), product=60)
    spike_day = _table([300.0, 10.0], date(2024, 1, 2), product=15)

    segments = segment_scores(pd.concat([negative_day, spike_day]), settings)

    assert segments.loc["All target days", "periods"] == 4
    assert segments.loc["Days with a negative price", "periods"] == 2
    assert segments.loc["Days with a price above €200", "periods"] == 2
    assert segments.loc["15-minute products", "periods"] == 2
    assert segments.loc["Validation window", "periods"] == 0
