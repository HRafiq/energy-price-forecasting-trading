"""In-memory stand-ins so no test touches the network."""

from __future__ import annotations

import copy
from collections.abc import Callable, Iterable
from typing import Any

import pandas as pd

from src.config import SmardConfig
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
