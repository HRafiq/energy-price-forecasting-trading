"""In-memory stand-ins so no test touches the network."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import PRICE_SERIES, PRODUCT_COLUMN, Settings, SmardConfig
from src.timegrid import HOUR


def _ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


class FakeSmard:
    """Serves SMARD-shaped JSON from memory and records every URL requested."""

    def __init__(self, config: SmardConfig) -> None:
        self.config = config
        self.payloads: dict[str, Any] = {}
        self.calls: list[str] = []

    def index_url(self, filter_id: int) -> str:
        c = self.config
        return f"{c.base_url}/{filter_id}/{c.region}/index_{c.resolution}.json"

    def chunk_url(self, filter_id: int, start_ms: int) -> str:
        c = self.config
        return (
            f"{c.base_url}/{filter_id}/{c.region}/"
            f"{filter_id}_{c.region}_{c.resolution}_{start_ms}.json"
        )

    def add_series(self, filter_id: int, chunks: Iterable[pd.Series]) -> list[int]:
        starts: list[int] = []
        for chunk in chunks:
            index = pd.DatetimeIndex(chunk.index)
            start_ms = _ms(index[0])
            starts.append(start_ms)
            rows = [
                [_ms(ts), None if pd.isna(value) else float(value)]
                for ts, value in zip(index, chunk.to_numpy(), strict=True)
            ]
            self.payloads[self.chunk_url(filter_id, start_ms)] = {"series": rows}
        self.payloads[self.index_url(filter_id)] = {"timestamps": starts}
        return starts

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        if url not in self.payloads:
            raise KeyError(f"unexpected URL {url}")
        return copy.deepcopy(self.payloads[url])


def hourly_chunks(
    start_utc: pd.Timestamp,
    hours: int,
    chunk_hours: int,
    value: Callable[[int], float | None] = float,
    step: pd.Timedelta = HOUR,
) -> list[pd.Series]:
    """Split ``hours`` consecutive periods from ``start_utc`` into chunks."""
    index = pd.date_range(start_utc, periods=hours, freq=step, name="timestamp_utc")
    values = [value(i) for i in range(hours)]
    full = pd.Series(values, index=index, dtype="float64")
    return [full.iloc[i : i + chunk_hours] for i in range(0, hours, chunk_hours)]


def synthetic_market(
    settings: Settings,
    first_day: date,
    days: int,
    price: Callable[[pd.DatetimeIndex], NDArray[np.float64]] | None = None,
) -> pd.DataFrame:
    """A quarter-hourly dataset with every configured column, local days aligned."""
    start = settings.market.local_midnight_utc(first_day)
    end = settings.market.local_midnight_utc(first_day + timedelta(days=days))
    index = pd.date_range(
        start, end, freq="15min", inclusive="left", name="timestamp_utc"
    )
    base = np.arange(len(index), dtype="float64")
    frame = pd.DataFrame(
        {col: base + 1000.0 * k for k, col in enumerate(settings.availability.columns)},
        index=index,
    )
    frame[PRODUCT_COLUMN] = 15
    if price is not None:
        frame[PRICE_SERIES] = price(index)
    return frame


def day_clock_price(tz: str) -> Callable[[pd.DatetimeIndex], NDArray[np.float64]]:
    """Price = 1000 per local day since 2000-01-01 plus the local clock minutes."""
    epoch = date(2000, 1, 1).toordinal()

    def price(index: pd.DatetimeIndex) -> NDArray[np.float64]:
        local = index.tz_convert(tz)
        days = np.asarray([d.toordinal() - epoch for d in local.date], dtype="float64")
        clock = np.asarray(local.hour * 60 + local.minute, dtype="float64")
        return np.asarray(1000.0 * days + clock, dtype=np.float64)

    return price


def corrupt_unpublished(
    frame: pd.DataFrame, settings: Settings, target_day: date
) -> pd.DataFrame:
    """Overwrite every value not published at the issue time for ``target_day``.

    Written independently of the information set, from the rules in config.
    """
    out = frame.astype("float64")
    idx = pd.DatetimeIndex(out.index)
    tz = settings.market.timezone
    start = settings.market.local_midnight_utc(target_day)
    end = settings.market.local_midnight_utc(target_day + timedelta(days=1))
    hours, minutes = (int(x) for x in settings.market.forecast_issue_local.split(":"))
    issue = (
        (
            pd.Timestamp(target_day - timedelta(days=1))
            + pd.Timedelta(hours=hours, minutes=minutes)
        )
        .tz_localize(tz)
        .tz_convert("UTC")
    )
    known_until = issue - pd.Timedelta(
        minutes=settings.availability.actuals_lag_minutes
    )
    garbage = -9.0e9
    out.loc[idx >= end, :] = garbage
    for column, rule in settings.availability.columns.items():
        if column not in out.columns:
            continue
        if rule == "before_target_day":
            hidden = idx >= start
        elif rule == "through_target_day":
            hidden = idx >= end
        else:
            hidden = idx + pd.Timedelta(minutes=15) > known_until
        out.loc[hidden, column] = garbage
    return out
