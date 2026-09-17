"""Drift monitoring (M2): rolling interval coverage and pinball loss with alerts.

Everything here is a pure function over a forecasts frame in the walk-forward
format: a UTC index of delivery periods, quantile columns ``q05`` to ``q95``,
``actual`` and ``target_day``.

* A **traded day** is a delivery day whose every period has a realised price.
  A day with any missing price is left out of the series, the way the backtest
  skips it, so it neither counts as a miss nor dilutes the window.
* **Daily scores:** 90% central interval coverage (share of periods with the
  price inside q05 to q95) and mean pinball loss over all quantiles
  (``src.forecasting.evaluate.daily_pinball``).
* **Rolling scores:** over the last 28 traded days, pooled over periods, so a
  25-hour day weighs a little more than a 23-hour day. A day gets no rolling
  value until 28 traded days exist.
* **Pinball ratio:** rolling mean pinball divided by the median of the rolling
  series over the validation days.

Threshold rule, fixed before any run:

* coverage: the highest threshold on a 0.5 percentage point grid such that at
  most 5% of validation days have rolling coverage below it;
* pinball: the lowest ratio on a 0.05 grid such that at most 5% of validation
  days have a ratio above it.

``fit_thresholds`` drops every row on or after ``evaluation.holdout_start``
before it computes anything, so hold-out rows cannot move a threshold. Rolling
windows only look back, so validation days never see later rows either.

An **alert episode** is a run of consecutive traded days in alert on one signal,
grouped within one window (validation or hold-out). An episode in the validation
window starts on a day that helped fit its threshold, so its incident says it is
in-sample and carries ``metrics["in_sample"] = 1.0``; hold-out episodes get 0.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import numpy as np
import pandas as pd

from src.config import Settings
from src.forecasting.base import quantile_column
from src.forecasting.evaluate import central_intervals, daily_pinball
from src.health.incidents import IN_SAMPLE_NOTE, Incident, make_incident_id

__all__ = [
    "COVERAGE_STEP",
    "MAX_ALERT_SHARE",
    "PINBALL_STEP",
    "SOURCE",
    "WINDOW_DAYS",
    "AlertEpisode",
    "Thresholds",
    "alert_episodes",
    "apply_thresholds",
    "daily_scores",
    "drift_incidents",
    "fit_thresholds",
    "longest_run",
    "rolling_scores",
    "select_coverage_threshold",
    "select_pinball_threshold",
    "window_labels",
]

WINDOW_DAYS = 28
MAX_ALERT_SHARE = 0.05
COVERAGE_STEP = 0.005
PINBALL_STEP = 0.05
SOURCE = "m2_drift"
#: Tolerance for comparing a pooled ratio with a grid value.
_EPS = 1e-12


@dataclass(frozen=True)
class Thresholds:
    coverage: float
    pinball_ratio: float
    #: Median rolling mean pinball over validation, the ratio's denominator.
    pinball_median: float
    validation_days: int
    coverage_alert_share: float
    pinball_alert_share: float


@dataclass(frozen=True)
class AlertEpisode:
    signal: str
    window: str
    start: date
    end: date
    days: int
    first_value: float
    extreme_value: float
    extreme_day: date
    #: True when the last traded day in the window is still in alert.
    open_at_end: bool


def _interval_columns(quantiles: tuple[float, ...], percent: int) -> tuple[str, str]:
    for low, high, width in central_intervals(quantiles):
        if width == percent:
            return quantile_column(low), quantile_column(high)
    raise ValueError(f"quantiles {quantiles} form no {percent}% central interval")


def daily_scores(forecasts: pd.DataFrame, quantiles: tuple[float, ...]) -> pd.DataFrame:
    """Per traded day: periods, 90% coverage and mean pinball, sorted by day."""
    low, high = _interval_columns(quantiles, 90)
    complete = forecasts.groupby("target_day")["actual"].apply(
        lambda prices: bool(prices.notna().all())
    )
    traded = forecasts[forecasts["target_day"].map(complete).astype(bool)]
    if traded.empty:
        return pd.DataFrame(
            columns=["periods", "coverage_90", "pinball"],
            index=pd.Index([], name="target_day"),
        )
    inside = (traded["actual"] >= traded[low]) & (traded["actual"] <= traded[high])
    grouped = inside.groupby(traded["target_day"])
    daily = pd.DataFrame(
        {
            "periods": grouped.size().astype("int64"),
            "coverage_90": grouped.mean().astype("float64"),
            "pinball": daily_pinball(traded, quantiles).astype("float64"),
        }
    )
    daily.index.name = "target_day"
    return daily.sort_index()


def rolling_scores(daily: pd.DataFrame, window: int = WINDOW_DAYS) -> pd.DataFrame:
    """Add pooled rolling coverage and mean pinball over ``window`` traded days."""
    periods = daily["periods"].astype("float64")
    inside = daily["coverage_90"] * periods
    loss = daily["pinball"] * periods
    total = periods.rolling(window, min_periods=window).sum()
    out = daily.copy()
    out["rolling_coverage_90"] = (
        inside.rolling(window, min_periods=window).sum() / total
    )
    out["rolling_pinball"] = loss.rolling(window, min_periods=window).sum() / total
    return out


def _grid_value(k: int, step: float) -> float:
    return round(k * step, 10)


def select_coverage_threshold(
    rolling_coverage: pd.Series,
    max_share: float = MAX_ALERT_SHARE,
    step: float = COVERAGE_STEP,
) -> float:
    """Highest grid value with at most ``max_share`` of days strictly below it."""
    values = rolling_coverage.dropna().to_numpy(dtype="float64")
    if values.size == 0:
        raise ValueError("no rolling coverage values to set a threshold from")
    for k in range(round(1.0 / step), -1, -1):
        threshold = _grid_value(k, step)
        if float(np.mean(values < threshold - _EPS)) <= max_share:
            return threshold
    return 0.0


def select_pinball_threshold(
    ratio: pd.Series,
    max_share: float = MAX_ALERT_SHARE,
    step: float = PINBALL_STEP,
) -> float:
    """Lowest grid value with at most ``max_share`` of days strictly above it."""
    values = ratio.dropna().to_numpy(dtype="float64")
    if values.size == 0:
        raise ValueError("no pinball ratio values to set a threshold from")
    last = math.ceil(float(values.max()) / step) + 1
    for k in range(0, last + 1):
        threshold = _grid_value(k, step)
        if float(np.mean(values > threshold + _EPS)) <= max_share:
            return threshold
    return _grid_value(last, step)


def window_labels(days: pd.Index, settings: Settings) -> pd.Series:
    """``validation``, ``holdout`` or ``before`` for each target day."""
    ev = settings.evaluation
    labels = [
        "holdout"
        if day >= ev.holdout_start
        else "validation"
        if day >= ev.validation_start
        else "before"
        for day in days
    ]
    return pd.Series(labels, index=days, dtype="object")


def fit_thresholds(
    forecasts: pd.DataFrame, settings: Settings, window: int = WINDOW_DAYS
) -> tuple[Thresholds, pd.DataFrame]:
    """Thresholds from validation days only, and the validation daily series.

    Rows on or after the hold-out start are removed first. Earlier rows before
    the validation window stay, to fill the first rolling windows.
    """
    ev = settings.evaluation
    before_holdout = forecasts[
        np.asarray([day < ev.holdout_start for day in forecasts["target_day"]])
    ]
    rolled = rolling_scores(
        daily_scores(before_holdout, settings.forecasting.quantiles), window
    )
    validation = rolled[window_labels(rolled.index, settings) == "validation"].copy()
    scored = validation[validation["rolling_coverage_90"].notna()]
    if scored.empty:
        raise ValueError("no validation day has a full rolling window")
    median = float(scored["rolling_pinball"].median())
    coverage = select_coverage_threshold(scored["rolling_coverage_90"])
    ratio = select_pinball_threshold(scored["rolling_pinball"] / median)
    thresholds = Thresholds(
        coverage=coverage,
        pinball_ratio=ratio,
        pinball_median=median,
        validation_days=len(scored),
        coverage_alert_share=0.0,
        pinball_alert_share=0.0,
    )
    flagged = apply_thresholds(validation, thresholds)
    thresholds = Thresholds(
        coverage=coverage,
        pinball_ratio=ratio,
        pinball_median=median,
        validation_days=len(scored),
        coverage_alert_share=float(flagged.loc[scored.index, "coverage_alert"].mean()),
        pinball_alert_share=float(flagged.loc[scored.index, "pinball_alert"].mean()),
    )
    return thresholds, flagged


def apply_thresholds(rolled: pd.DataFrame, thresholds: Thresholds) -> pd.DataFrame:
    """Add the pinball ratio and both alert flags; days without a window never alert."""
    out = rolled.copy()
    out["rolling_pinball_ratio"] = out["rolling_pinball"] / thresholds.pinball_median
    coverage = out["rolling_coverage_90"].to_numpy(dtype="float64")
    ratio = out["rolling_pinball_ratio"].to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        out["coverage_alert"] = np.nan_to_num(coverage, nan=np.inf) < (
            thresholds.coverage - _EPS
        )
        out["pinball_alert"] = np.nan_to_num(ratio, nan=-np.inf) > (
            thresholds.pinball_ratio + _EPS
        )
    return out


_SIGNALS = {
    "coverage": ("coverage_alert", "rolling_coverage_90", "min"),
    "pinball": ("pinball_alert", "rolling_pinball_ratio", "max"),
}


def alert_episodes(
    series: pd.DataFrame, signal: str, windows: pd.Series
) -> list[AlertEpisode]:
    """Runs of consecutive traded days in alert, split at window boundaries."""
    flag_column, value_column, extreme = _SIGNALS[signal]
    episodes: list[AlertEpisode] = []
    for window in pd.unique(windows):
        part = series[windows == window]
        flags = part[flag_column].astype(bool)
        run_ids = (flags != flags.shift()).cumsum()
        for _, members in part[flags].groupby(run_ids[flags]):
            values = members[value_column]
            extreme_day = values.idxmin() if extreme == "min" else values.idxmax()
            episodes.append(
                AlertEpisode(
                    signal=signal,
                    window=str(window),
                    start=members.index[0],
                    end=members.index[-1],
                    days=len(members),
                    first_value=float(values.iloc[0]),
                    extreme_value=float(values.loc[extreme_day]),
                    extreme_day=extreme_day,
                    open_at_end=members.index[-1] == part.index[-1],
                )
            )
    return sorted(episodes, key=lambda e: (e.start, e.signal))


def longest_run(flags: pd.Series) -> tuple[int, date | None, date | None]:
    """Longest run of consecutive True values: length, first and last day."""
    best: tuple[int, date | None, date | None] = (0, None, None)
    length = 0
    start: date | None = None
    for day, flag in flags.items():
        assert isinstance(day, date)
        if flag:
            if length == 0:
                start = day
            length += 1
            if length > best[0]:
                best = (length, start, day)
        else:
            length = 0
    return best


def drift_incidents(
    episodes: list[AlertEpisode],
    thresholds: Thresholds,
    timezone: str,
    window_days: int = WINDOW_DAYS,
    source: str = SOURCE,
    days_word: str = "traded days",
) -> list[Incident]:
    """One drift incident per alert episode, dated on its first alert day.

    ``source`` names the monitor that raised it: the M2 experiment by default, or
    the live pipeline's daily check, which counts ``days_word`` as scored days
    because it leaves out traded days a fallback forecast. ``detected_utc`` is the
    local midnight after the first alert day, as a batch run after delivery would
    see it; the live check replaces it with the time of the run that found it.
    """
    incidents = []
    for episode in episodes:
        detected = (
            pd.Timestamp(datetime.combine(episode.start + timedelta(days=1), time()))
            .tz_localize(timezone)
            .tz_convert("UTC")
            .to_pydatetime()
        )
        span = (
            f"still in alert on the last {days_word.removesuffix('s')}, {episode.end}"
            if episode.open_at_end
            else f"it lasted to {episode.end}"
        )
        if episode.signal == "coverage":
            threshold = thresholds.coverage
            detail = (
                f"Rolling {window_days}-day 90% interval coverage was "
                f"{episode.first_value:.1%} on {episode.start}, below the alert "
                f"threshold of {threshold:.1%}. The alert ran {episode.days} "
                f"{days_word} ({span}); lowest {episode.extreme_value:.1%} on "
                f"{episode.extreme_day}."
            )
        else:
            threshold = thresholds.pinball_ratio
            detail = (
                f"Rolling {window_days}-day mean pinball loss was "
                f"{episode.first_value:.3f} times its validation median "
                f"({thresholds.pinball_median:.2f} €/MWh) on {episode.start}, above "
                f"the alert threshold of {threshold:.2f}. The alert ran "
                f"{episode.days} {days_word} ({span}); highest "
                f"{episode.extreme_value:.2f} on {episode.extreme_day}."
            )
        in_sample = episode.window == "validation"
        if in_sample:
            detail = f"{detail} {IN_SAMPLE_NOTE}"
        incidents.append(
            Incident(
                incident_id=make_incident_id(
                    source, "drift", episode.start, episode.signal
                ),
                delivery_day=episode.start,
                detected_utc=detected,
                type="drift",
                severity="warning",
                detail=detail,
                action="Flag for retraining review",
                status="review",
                source=source,
                metrics={
                    "threshold": threshold,
                    "first_value": episode.first_value,
                    "extreme_value": episode.extreme_value,
                    "alert_days": float(episode.days),
                    "in_sample": 1.0 if in_sample else 0.0,
                },
            )
        )
    return incidents
