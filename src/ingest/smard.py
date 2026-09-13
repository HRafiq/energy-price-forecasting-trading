"""SMARD (Bundesnetzagentur) chart-data client.

SMARD serves each series as weekly JSON chunks::

    {base}/{filter}/{region}/index_{resolution}.json
        -> {"timestamps": [ms, ...]}            chunk start instants, epoch ms UTC
    {base}/{filter}/{region}/{filter}_{region}_{resolution}_{ms}.json
        -> {"series": [[ms, value | null], ...]}

Checked against the live endpoint on 2026-09-13 (docs/production_notes.md):

* Timestamps are true UTC instants: DST weeks hold 167 or 169 hours, or
  668 or 676 quarter-hours.
* Volumes are energy per interval. An hourly value is the sum of its four
  quarter-hour values, so MWh per hour equals average MW.
* After the 15-minute switch the hourly price is the mean of the quarter-hours.

This endpoint backs the SMARD website. It is not a versioned public API and
could change without notice. Data licence: CC BY 4.0, Bundesnetzagentur |
SMARD.de, and attribution is required.

Chunks are cached as raw JSON. A chunk counts as settled once
``refresh_recent_chunks`` newer chunks exist. A sidecar file records how many
newer chunks existed when a chunk was downloaded, and only a chunk that was
already settled then is served from cache. Anything cached earlier, while it
may still have been incomplete or open to revision, is downloaded again (D2).
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from src.config import SmardConfig

__all__ = ["FetchJson", "SmardClient", "SmardError", "http_fetch_json", "parse_chunk"]

FetchJson = Callable[[str], Any]

_USER_AGENT = "energy-price-forecasting-trading/0.1 (public research project)"


class SmardError(RuntimeError):
    """SMARD returned something this client cannot use."""


def http_fetch_json(
    timeout_s: float, attempts: int = 3, backoff_s: float = 1.0
) -> FetchJson:
    """A fetcher that GETs JSON with a timeout and exponential backoff."""

    def fetch(url: str) -> Any:
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(
                    url, timeout=timeout_s, headers={"User-Agent": _USER_AGENT}
                )
                response.raise_for_status()
                return response.json()
            except (requests.RequestException, ValueError) as exc:
                if attempt == attempts:
                    raise SmardError(
                        f"GET {url} failed after {attempts} attempts"
                    ) from exc
                time.sleep(backoff_s * 2 ** (attempt - 1))
        raise SmardError(f"GET {url}: no attempts made")

    return fetch


def _epoch_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 1_000_000)


def _empty_series(name: str) -> pd.Series:
    index = pd.DatetimeIndex([], tz="UTC", name="timestamp_utc")
    return pd.Series([], index=index, name=name, dtype="float64")


def parse_chunk(payload: Any, name: str) -> pd.Series:
    """Turn one chunk payload into a float series on a UTC index; null -> NaN."""
    if not isinstance(payload, dict) or not isinstance(payload.get("series"), list):
        raise SmardError(f"{name}: chunk payload has no 'series' list")
    rows = payload["series"]
    try:
        stamps = [int(row[0]) for row in rows]
        values = [float("nan") if row[1] is None else float(row[1]) for row in rows]
    except (TypeError, ValueError, IndexError) as exc:
        raise SmardError(f"{name}: malformed chunk row") from exc
    index = pd.DatetimeIndex(
        pd.to_datetime(stamps, unit="ms", utc=True), name="timestamp_utc"
    )
    return pd.Series(values, index=index, name=name, dtype="float64")


def _merge_chunks(parts: list[pd.Series], name: str) -> pd.Series:
    combined = pd.concat(parts)
    if combined.index.has_duplicates:
        distinct = combined.groupby(level=0).nunique()
        conflicts = distinct[distinct > 1]
        if not conflicts.empty:
            raise SmardError(
                f"{name}: {len(conflicts)} timestamps have conflicting values "
                f"across chunks, first at {conflicts.index[0]}"
            )
        combined = combined.groupby(level=0).first()
    merged: pd.Series = combined.sort_index().rename(name)
    return merged


class SmardClient:
    """Fetches SMARD series chunk by chunk, caching the raw JSON on disk."""

    def __init__(
        self,
        config: SmardConfig,
        cache_dir: Path,
        fetch_json: FetchJson | None = None,
    ) -> None:
        self._config = config
        self._cache_dir = cache_dir
        self._fetch = fetch_json or http_fetch_json(config.timeout_s)

    def index_url(self, filter_id: int) -> str:
        c = self._config
        return f"{c.base_url}/{filter_id}/{c.region}/index_{c.resolution}.json"

    def chunk_url(self, filter_id: int, chunk_start_ms: int) -> str:
        c = self._config
        return (
            f"{c.base_url}/{filter_id}/{c.region}/"
            f"{filter_id}_{c.region}_{c.resolution}_{chunk_start_ms}.json"
        )

    def chunk_starts(self, filter_id: int) -> list[int]:
        payload = self._fetch(self.index_url(filter_id))
        stamps = payload.get("timestamps") if isinstance(payload, dict) else None
        if not isinstance(stamps, list) or not stamps:
            raise SmardError(f"filter {filter_id}: index lists no chunks")
        return sorted({int(s) for s in stamps})

    def series(
        self,
        filter_id: int,
        name: str,
        start: pd.Timestamp,
        end: pd.Timestamp | None = None,
    ) -> pd.Series:
        """Values in ``[start, end)``; ``end=None`` means everything published."""
        if start.tzinfo is None or (end is not None and end.tzinfo is None):
            raise ValueError("start and end must be timezone-aware")
        starts = self.chunk_starts(filter_id)
        start_ms = _epoch_ms(start)
        end_ms = None if end is None else _epoch_ms(end)

        wanted: list[tuple[int, int]] = []
        for i, chunk_start in enumerate(starts):
            next_start = starts[i + 1] if i + 1 < len(starts) else None
            if next_start is not None and next_start <= start_ms:
                continue
            if end_ms is not None and chunk_start >= end_ms:
                continue
            wanted.append((chunk_start, len(starts) - 1 - i))
        if not wanted:
            return _empty_series(name)

        with ThreadPoolExecutor(max_workers=self._config.max_workers) as pool:
            payloads = list(
                pool.map(lambda item: self._chunk_payload(filter_id, *item), wanted)
            )
        combined = _merge_chunks([parse_chunk(p, name) for p in payloads], name)
        keep = combined.index >= start
        if end is not None:
            keep &= combined.index < end
        return combined[keep]

    def _chunk_payload(
        self, filter_id: int, chunk_start_ms: int, newer_chunks: int
    ) -> Any:
        folder = self._cache_dir / str(filter_id) / self._config.resolution
        path = folder / f"{chunk_start_ms}.json"
        meta_path = folder / f"{chunk_start_ms}.meta.json"
        settle = self._config.refresh_recent_chunks
        if newer_chunks >= settle and path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if int(meta.get("newer_chunks_at_fetch", -1)) >= settle:
                return json.loads(path.read_text(encoding="utf-8"))

        payload = self._fetch(self.chunk_url(filter_id, chunk_start_ms))
        folder.mkdir(parents=True, exist_ok=True)
        # Payload first, then metadata: a crash in between leaves no metadata
        # claiming the new payload settled, so the next run downloads it again.
        _write_json_atomic(path, payload)
        _write_json_atomic(meta_path, {"newer_chunks_at_fetch": newer_chunks})
        return payload


def _write_json_atomic(path: Path, content: Any) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(content), encoding="utf-8")
    tmp.replace(path)
