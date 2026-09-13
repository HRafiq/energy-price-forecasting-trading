"""Scoring for probabilistic price forecasts.

Every function works on the long table from the walk-forward harness: one row
per delivery period with quantile columns, ``actual``, ``target_day`` and
``price_product_minutes``. Periods without a published price are skipped and
counted, never scored as zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import PRODUCT_COLUMN, Settings
from src.forecasting.base import quantile_column

__all__ = [
    "central_intervals",
    "pinball",
    "score",
    "segment_scores",
    "yearly_scores",
]


def pinball(
    actual: NDArray[np.float64], forecast: NDArray[np.float64], q: float
) -> NDArray[np.float64]:
    """Pinball loss per period: q times the shortfall, 1 - q times the excess."""
    diff = actual - forecast
    return np.asarray(np.maximum(q * diff, (q - 1.0) * diff), dtype=np.float64)


def central_intervals(quantiles: tuple[float, ...]) -> list[tuple[float, float, int]]:
    """Symmetric intervals the quantiles form, widest first: (low, high, percent)."""
    intervals = []
    for low in quantiles:
        high = next(
            (h for h in quantiles if low < 0.5 and abs(h - (1 - low)) < 1e-9), None
        )
        if high is not None:
            intervals.append((low, high, round((high - low) * 100)))
    return intervals


def score(forecasts: pd.DataFrame, quantiles: tuple[float, ...]) -> dict[str, float]:
    valid = forecasts[forecasts["actual"].notna()]
    result: dict[str, float] = {
        "periods": float(len(valid)),
        "days": float(valid["target_day"].nunique()),
        "missing actuals": float(len(forecasts) - len(valid)),
    }
    if valid.empty:
        return result
    actual = valid["actual"].to_numpy(dtype="float64")
    losses = [
        float(
            pinball(
                actual, valid[quantile_column(q)].to_numpy(dtype="float64"), q
            ).mean()
        )
        for q in quantiles
    ]
    error = valid[quantile_column(0.5)].to_numpy(dtype="float64") - actual
    result["mean pinball"] = float(np.mean(losses))
    result["MAE of median"] = float(np.mean(np.abs(error)))
    result["RMSE of median"] = float(np.sqrt(np.mean(error**2)))
    result["bias of median"] = float(np.mean(error))
    for low, high, percent in central_intervals(quantiles):
        lower = valid[quantile_column(low)].to_numpy(dtype="float64")
        upper = valid[quantile_column(high)].to_numpy(dtype="float64")
        result[f"coverage {percent}%"] = float(
            np.mean((actual >= lower) & (actual <= upper))
        )
        result[f"width {percent}%"] = float(np.mean(upper - lower))
    return result


def _day_flag(forecasts: pd.DataFrame, flagged_days: pd.Series) -> NDArray[np.bool_]:
    return np.asarray(
        forecasts["target_day"].map(flagged_days).fillna(False), dtype=bool
    )


def segment_scores(forecasts: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Scores for all days and for the segments that matter for trading."""
    quantiles = settings.forecasting.quantiles
    threshold = settings.evaluation.spike_threshold_eur_mwh
    by_day = forecasts.groupby("target_day")["actual"]
    segments = {
        "All target days": np.ones(len(forecasts), dtype=bool),
        "Validation window": np.asarray(
            [
                day >= settings.evaluation.validation_start
                for day in forecasts["target_day"]
            ],
            dtype=bool,
        ),
        "15-minute products": np.asarray(forecasts[PRODUCT_COLUMN] == 15, dtype=bool),
        "Days with a negative price": _day_flag(forecasts, by_day.min() < 0),
        f"Days with a price above €{threshold:.0f}": _day_flag(
            forecasts, by_day.max() > threshold
        ),
    }
    rows = {name: score(forecasts[mask], quantiles) for name, mask in segments.items()}
    return pd.DataFrame(rows).T


def yearly_scores(
    forecasts: pd.DataFrame, quantiles: tuple[float, ...]
) -> pd.DataFrame:
    years = np.asarray([day.year for day in forecasts["target_day"]])
    rows = {
        int(year): score(forecasts[years == year], quantiles)
        for year in np.unique(years)
    }
    return pd.DataFrame(rows).T
