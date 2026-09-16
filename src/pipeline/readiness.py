"""Is tomorrow's data there yet? The check behind the pipeline's sensor.

    uv run python -m src.pipeline.readiness --day 2026-09-17

The daily forecast for delivery day D+1 is issued at 11:40 on day D and the gate
closes at 12:00. By then the model needs four things in the built dataset:

* **prices** for every period of day D, which feed the lag features;
* **load forecast** for every period of day D+1, published about two hours before
  the gate;
* **weather** for every period of day D+1, forecast two days ahead;
* **fuels**, the last gas and carbon settlement published before day D+1.

Each feed is checked separately so the pipeline can say what is missing rather
than only that something is. The command exits 0 when every feed is ready and 1
when any is not, which is what a sensor polls.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd

from src.config import (
    CARBON_COLUMN,
    GAS_COLUMN,
    PRICE_SERIES,
    RESOLUTION_STEP,
    Settings,
    load_settings,
)
from src.timegrid import delivery_periods, local_day_bounds_utc

__all__ = [
    "FEED_NAMES",
    "FeedStatus",
    "Readiness",
    "check_readiness",
    "main",
    "weather_columns",
]

WEATHER_PREFIX = "wx_"
FEED_NAMES = ("prices", "load_forecast", "weather", "fuels")


def weather_columns(frame: pd.DataFrame) -> list[str]:
    """The dataset's weather forecast columns."""
    return [c for c in frame.columns if str(c).startswith(WEATHER_PREFIX)]


@dataclass(frozen=True)
class FeedStatus:
    """One feed the forecast needs, and whether it has arrived."""

    name: str
    ready: bool
    expected: int
    present: int
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "ready": self.ready,
            "expected": self.expected,
            "present": self.present,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class Readiness:
    """What the pipeline knows about the inputs for one delivery day."""

    target_day: date
    checked_utc: datetime
    feeds: tuple[FeedStatus, ...]

    @property
    def ready(self) -> bool:
        return all(feed.ready for feed in self.feeds)

    @property
    def missing(self) -> tuple[str, ...]:
        return tuple(feed.name for feed in self.feeds if not feed.ready)

    def as_dict(self) -> dict[str, object]:
        return {
            "target_day": str(self.target_day),
            "checked_utc": self.checked_utc.isoformat(timespec="seconds"),
            "ready": self.ready,
            "missing": list(self.missing),
            "feeds": [feed.as_dict() for feed in self.feeds],
        }


def _periods_present(
    frame: pd.DataFrame, columns: list[str], index: pd.DatetimeIndex
) -> int:
    """Periods of ``index`` where every one of ``columns`` has a value."""
    if not columns:
        return 0
    rows = frame.reindex(index)[columns]
    return int(rows.notna().all(axis=1).sum())


def _feed(
    name: str,
    frame: pd.DataFrame,
    columns: list[str],
    index: pd.DatetimeIndex,
    label: str,
) -> FeedStatus:
    expected = len(index)
    present = _periods_present(frame, columns, index)
    ready = present == expected
    detail = (
        f"{label}: {present} of {expected} periods"
        if ready
        else f"{label}: {present} of {expected} periods, {expected - present} missing"
    )
    return FeedStatus(name, ready, expected, present, detail)


def _fuels(frame: pd.DataFrame, target_day: date, settings: Settings) -> FeedStatus:
    """Gas and carbon carry the last settlement published before the day."""
    start, _ = local_day_bounds_utc(target_day, settings.market.timezone)
    columns = [GAS_COLUMN, CARBON_COLUMN]
    before = frame.loc[frame.index < start, columns].dropna()
    if before.empty:
        return FeedStatus("fuels", False, 1, 0, "fuels: no gas or carbon price yet")
    last = pd.Timestamp(before.index[-1]).tz_convert(settings.market.timezone)
    age_days = (target_day - last.date()).days
    return FeedStatus(
        "fuels",
        True,
        1,
        1,
        f"fuels: last settlement {last.date()}, {age_days} days before delivery",
    )


def check_readiness(
    frame: pd.DataFrame, target_day: date, settings: Settings
) -> Readiness:
    """Check every feed the forecast for ``target_day`` needs."""
    tz = settings.market.timezone
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    target_index = delivery_periods(target_day, tz, step)
    issue_day_index = delivery_periods(target_day - timedelta(days=1), tz, step)
    feeds = (
        _feed(
            "prices",
            frame,
            [PRICE_SERIES],
            issue_day_index,
            f"prices for {target_day - timedelta(days=1)}",
        ),
        _feed(
            "load_forecast",
            frame,
            ["load_forecast_mw"],
            target_index,
            f"load forecast for {target_day}",
        ),
        _feed(
            "weather",
            frame,
            weather_columns(frame),
            target_index,
            f"weather for {target_day}",
        ),
        _fuels(frame, target_day, settings),
    )
    return Readiness(target_day, datetime.now(UTC), feeds)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a delivery day's inputs have arrived."
    )
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--json", action="store_true", help="print the result as JSON for a task log"
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    frame = pd.read_parquet(settings.data.inputs_path)
    readiness = check_readiness(frame, args.day, settings)
    if args.json:
        print(json.dumps(readiness.as_dict(), indent=2))
    else:
        for feed in readiness.feeds:
            print(f"{'ready  ' if feed.ready else 'missing'} {feed.detail}")
        state = (
            "ready" if readiness.ready else f"missing {', '.join(readiness.missing)}"
        )
        print(f"{args.day}: {state}")
    return 0 if readiness.ready else 1


if __name__ == "__main__":
    raise SystemExit(main())
