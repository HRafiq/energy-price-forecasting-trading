"""ENTSO-E Transparency Platform client for DE-LU day-ahead and German imbalance prices.

The Web API answers ``GET {base}?securityToken=...&documentType=...`` with an
XML document, or a zip of XML documents for some data items:

* A44 day-ahead prices, ``in_Domain = out_Domain = 10Y1001A1001A82H`` (DE-LU),
  return a ``Publication_MarketDocument``. Each ``TimeSeries`` carries a
  ``classificationSequence_AttributeInstanceComponent.position``. For DE-LU,
  position 1 is the Single Day-Ahead Coupling (SDAC) price (auction at 12:00
  on the day before delivery) and position 2 is the separate 10:15 auction of
  EXAA (Energy Exchange Austria). See ``POSITION_NOTES``.
* A85 imbalance prices, ``controlArea_Domain = 10Y1001A1001A83F`` (Germany),
  return a zipped ``Balancing_MarketDocument`` with one 15-minute series per
  imbalance direction: category A04 (excess balance, the system was long) and
  A05 (insufficient balance, short). Germany uses one price, the reBAP, for
  both directions, so the two series normally hold the same values. The DE-LU
  bidding-zone code and the German TSO and LFC block codes answer "No matching
  data found" for this item (checked 2026-09-14).
* A problem, including "No matching data found", comes back as an
  ``Acknowledgement_MarketDocument``, raised here as ``AcknowledgementError``
  or its subclass ``NoMatchingDataError``.

Curve type A03 omits a point whose value repeats the previous one, so values
are carried forward within each Period up to the Period end. Curve type A01
lists every point.

The token is read from ``ENTSOE_API_KEY`` in the environment, falling back to
the repository ``.env`` file. It is sent only as a request parameter and never
written to logs, cache paths, exception text or reports: every message that
could contain the request URL passes through ``redact`` first and is raised
outside the ``except`` block, so no chained exception carries it either.

Requests are spaced at least ``min_interval_s`` apart (0.5 s, at most 120 per
minute against the platform limit of 400), and 429 and 5xx answers are retried
with exponential backoff, honouring ``Retry-After``. Raw responses are cached
under ``data/raw/entsoe/{documentType}/{domains}/{periodStart}_{periodEnd}.raw``
by local calendar month. As in ``src.ingest.smard``, a sidecar file records
whether the month had settled (ended ``settle_days`` before the download), and
only a settled download is served from cache; later runs fetch anything else
again, since recent values can still be published late or revised.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import time
import urllib.parse
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Protocol
from xml.etree import ElementTree

import numpy as np
import pandas as pd
import requests
from numpy.typing import NDArray

from src.config import PRICE_SERIES, REPO_ROOT, load_settings

__all__ = [
    "BASE_URL",
    "DE_IMBALANCE_AREA",
    "DE_LU_BIDDING_ZONE",
    "POSITION_NOTES",
    "TOKEN_ENV",
    "Acknowledgement",
    "AcknowledgementError",
    "Comparison",
    "EntsoeClient",
    "EntsoeError",
    "EntsoeHttpError",
    "EntsoeOptions",
    "HttpResponse",
    "NoMatchingDataError",
    "RateLimiter",
    "Transport",
    "build_price_check_report",
    "cache_path",
    "compare_prices",
    "day_ahead_table",
    "imbalance_table",
    "load_token",
    "main",
    "month_chunks",
    "parse_acknowledgement",
    "parse_day_ahead_document",
    "parse_env_text",
    "parse_imbalance_document",
    "payload_is_final",
    "quarter_hour_grid",
    "redact",
    "resolution_minutes",
    "retry_delay",
    "split_payload",
]

BASE_URL = "https://web-api.tp.entsoe.eu/api"
DE_LU_BIDDING_ZONE = "10Y1001A1001A82H"
DE_IMBALANCE_AREA = "10Y1001A1001A83F"
TOKEN_ENV = "ENTSOE_API_KEY"
MARKET_TIMEZONE = "Europe/Berlin"
DEFAULT_FIRST_DAY = date(2024, 6, 1)
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
QUARTER_HOUR_MINUTES = 15
TOLERANCE_EUR_MWH = 0.005

DAY_AHEAD_FILE = "day_ahead_prices_de_lu.parquet"
IMBALANCE_FILE = "imbalance_prices_de.parquet"
REPORT_PATH = Path("docs/results/entsoe_price_check.md")

IMBALANCE_CATEGORY_LABELS = {"A04": "excess", "A05": "insufficient"}

POSITION_NOTES = """\
* **Position 1 is the SDAC price.** The ENTSO-E Transparency Platform notes
  for DE-LU day-ahead prices say that prices of the Single Day-Ahead Coupling,
  gate closure 12:00 CE(S)T on the day before delivery, are published under
  "Sequence 1", and prices of the separate 10:15 CE(S)T auction of EXAA under
  "Sequence 2". The platform wording is reproduced in
  [entsoe-py issue 422](https://github.com/EnergieID/entsoe-py/issues/422);
  the platform data-view page itself was not reachable for this check. The
  data agree: position 1 matches SMARD, which publishes the SDAC price.
* **Position 2 is the EXAA 10:15 auction** (Energy Exchange Austria, German
  market area). Consistent with this, position 2 is a 15-minute series before
  2025-10-01, when SDAC still traded hourly products (position 1 is PT60M
  then): EXAA has auctioned quarter-hours in its 10:15 auction since 2014
  ([EXAA, trading with EXAA](https://www.exaa.at/en/energytrading/handel-mit-exaa/)).
* **Timing (partly unverified).** The EXAA auction closes at 10:15 local time,
  before the 11:40 forecast issue time. A secondary summary of EXAA trading
  gives order entry until 10:12 and results to participants at 10:30 on
  weekdays; this was not confirmed in first-party EXAA documents, and neither
  the time ENTSO-E publishes position 2 nor how weekend and holiday delivery
  days are auctioned was confirmed. Treat position 2 as a candidate feature
  only after checking its publication time and licence for live use.
"""

_USER_AGENT = "energy-price-forecasting-trading/0.1 (public research project)"
_REDACTED = "<token>"
_CACHE_KEYS = (
    "documentType",
    "processType",
    "businessType",
    "in_Domain",
    "out_Domain",
    "controlArea_Domain",
)
_SAFE_SEGMENT = re.compile(r"[^A-Za-z0-9-]")
_DURATION = re.compile(r"^PT(?:(\d+)H)?(?:(\d+)M)?$")

_log = logging.getLogger(__name__)


class EntsoeError(RuntimeError):
    """ENTSO-E returned something this client cannot use."""


class EntsoeHttpError(EntsoeError):
    """A request failed at the HTTP level; the message never holds the token."""

    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Acknowledgement:
    """The reason given in an ``Acknowledgement_MarketDocument``."""

    code: str
    text: str

    @property
    def no_matching_data(self) -> bool:
        return "no matching data" in self.text.lower()


class AcknowledgementError(EntsoeError):
    """The platform answered with an acknowledgement instead of data."""

    def __init__(self, acknowledgement: Acknowledgement) -> None:
        super().__init__(
            f"ENTSO-E acknowledgement {acknowledgement.code}: {acknowledgement.text}"
        )
        self.acknowledgement = acknowledgement


class NoMatchingDataError(AcknowledgementError):
    """The platform holds no data for the requested item and interval."""


# ---------------------------------------------------------------------------
# Token handling


def parse_env_text(text: str) -> dict[str, str]:
    """Parse ``KEY=value`` lines; ``#`` comments, ``export`` and quotes allowed."""
    values: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        else:
            value = value.split(" #", 1)[0].rstrip()
        if key:
            values[key] = value
    return values


def load_token(
    environ: Mapping[str, str] | None = None, env_path: Path | None = None
) -> str:
    """The API token: the environment variable wins, then the ``.env`` file."""
    env = os.environ if environ is None else environ
    value = env.get(TOKEN_ENV, "").strip()
    if value:
        return value
    path = REPO_ROOT / ".env" if env_path is None else env_path
    if path.is_file():
        parsed = parse_env_text(path.read_text(encoding="utf-8"))
        value = parsed.get(TOKEN_ENV, "").strip()
        if value:
            return value
    raise EntsoeError(f"{TOKEN_ENV} is not set in the environment or in .env")


def redact(text: str, secret: str) -> str:
    """Replace the secret, raw or URL-encoded, with a placeholder."""
    if not secret:
        return text
    out = text.replace(secret, _REDACTED)
    for encoded in {urllib.parse.quote(secret, safe=""), urllib.parse.quote(secret)}:
        if encoded != secret:
            out = out.replace(encoded, _REDACTED)
    return out


# ---------------------------------------------------------------------------
# HTTP, rate limiting, retries


class HttpResponse(Protocol):
    """The parts of a ``requests.Response`` this client reads."""

    @property
    def status_code(self) -> int: ...

    @property
    def content(self) -> bytes: ...

    @property
    def headers(self) -> Mapping[str, str]: ...


Transport = Callable[[str, Mapping[str, str], float], HttpResponse]


def requests_transport(
    url: str, params: Mapping[str, str], timeout_s: float
) -> HttpResponse:
    """GET with ``requests``; urllib3 debug logging would print the URL."""
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    return requests.get(
        url,
        params=dict(params),
        timeout=timeout_s,
        headers={"User-Agent": _USER_AGENT},
    )


class RateLimiter:
    """Keeps at least ``min_interval_s`` between the starts of two requests."""

    def __init__(
        self,
        min_interval_s: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if min_interval_s < 0:
            raise ValueError("min_interval_s must not be negative")
        self._min_interval_s = min_interval_s
        self._clock = clock
        self._sleep = sleep
        self._last_start: float | None = None

    def wait(self) -> float:
        """Sleep if the previous request started too recently; return the delay."""
        delay = 0.0
        if self._last_start is not None:
            delay = self._last_start + self._min_interval_s - self._clock()
            if delay > 0:
                self._sleep(delay)
            else:
                delay = 0.0
        self._last_start = self._clock()
        return delay


def retry_delay(
    attempt: int, base_s: float, cap_s: float, retry_after: str | None = None
) -> float:
    """Seconds to wait after failed attempt ``attempt`` (1-based)."""
    if retry_after is not None:
        try:
            seconds = float(retry_after)
        except ValueError:
            seconds = math.nan
        if math.isfinite(seconds):
            return min(max(seconds, 0.0), cap_s)
    return float(min(base_s * 2 ** (attempt - 1), cap_s))


@dataclass(frozen=True)
class EntsoeOptions:
    """Client settings; defaults suit the public Web API."""

    base_url: str = BASE_URL
    timeout_s: float = 60.0
    min_interval_s: float = 0.5
    attempts: int = 5
    backoff_base_s: float = 2.0
    backoff_cap_s: float = 60.0
    settle_days: int = 7
    #: reBAP is revised after first publication, so imbalance months settle later.
    imbalance_settle_days: int = 45


# ---------------------------------------------------------------------------
# Periods and cache keys


def local_midnight_utc(day: date, timezone: str = MARKET_TIMEZONE) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(timezone).tz_convert("UTC")


def month_chunks(
    first: date, last: date, timezone: str = MARKET_TIMEZONE
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """UTC ``[start, end)`` bounds of local calendar months covering the days."""
    if last < first:
        raise ValueError("last must not be before first")
    stop = last + timedelta(days=1)
    chunks: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    day = first
    while day < stop:
        next_month = date(day.year + (day.month == 12), day.month % 12 + 1, 1)
        chunk_end = min(next_month, stop)
        chunks.append(
            (local_midnight_utc(day, timezone), local_midnight_utc(chunk_end, timezone))
        )
        day = chunk_end
    return chunks


def quarter_hour_grid(
    first: date, last: date, timezone: str = MARKET_TIMEZONE
) -> pd.DatetimeIndex:
    """Every UTC quarter-hour of the local delivery days ``first`` to ``last``."""
    return pd.date_range(
        local_midnight_utc(first, timezone),
        local_midnight_utc(last + timedelta(days=1), timezone),
        freq="15min",
        inclusive="left",
        name="timestamp_utc",
    )


def api_time(ts: pd.Timestamp) -> str:
    """The ``yyyyMMddHHmm`` UTC form the API expects."""
    return ts.tz_convert("UTC").strftime("%Y%m%d%H%M")


def cache_path(cache_dir: Path, params: Mapping[str, str]) -> Path:
    """Cache file for a request, built only from non-secret parameters."""
    domains = [
        _SAFE_SEGMENT.sub("_", params[key]) for key in _CACHE_KEYS[1:] if key in params
    ]
    doc_type = _SAFE_SEGMENT.sub("_", params["documentType"])
    start = _SAFE_SEGMENT.sub("_", params["periodStart"])
    end = _SAFE_SEGMENT.sub("_", params["periodEnd"])
    return cache_dir / doc_type / "_".join(domains or ["none"]) / f"{start}_{end}.raw"


def _describe(params: Mapping[str, str]) -> str:
    keys = (*_CACHE_KEYS, "periodStart", "periodEnd")
    return " ".join(f"{key}={params[key]}" for key in keys if key in params)


def split_payload(payload: bytes) -> list[bytes]:
    """A zip holds one XML document per member; anything else is one document."""
    if not payload.startswith(b"PK"):
        return [payload]
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            return [archive.read(name) for name in sorted(archive.namelist())]
    except zipfile.BadZipFile:
        raise EntsoeError("response looks like a zip but cannot be opened") from None


# ---------------------------------------------------------------------------
# Client


class EntsoeClient:
    """Fetches ENTSO-E documents month by month, caching settled raw responses."""

    def __init__(
        self,
        token: str,
        cache_dir: Path,
        options: EntsoeOptions | None = None,
        transport: Transport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now_utc: Callable[[], pd.Timestamp] | None = None,
    ) -> None:
        if not token:
            raise EntsoeError("an API token is required")
        self._token = token
        self._cache_dir = cache_dir
        self._options = options or EntsoeOptions()
        self._transport = transport or requests_transport
        self._sleep = sleep
        self._limiter = RateLimiter(self._options.min_interval_s, clock, sleep)
        self._now_utc = now_utc or (lambda: pd.Timestamp.now(tz="UTC"))
        self.requests_made = 0
        self.cache_hits = 0
        self.empty_chunks: list[str] = []

    def __repr__(self) -> str:
        return f"EntsoeClient(cache_dir={str(self._cache_dir)!r})"

    def documents(
        self, params: Mapping[str, str], period_end: pd.Timestamp
    ) -> list[bytes]:
        """XML documents for one request, from cache when settled when cached."""
        path = cache_path(self._cache_dir, params)
        meta_path = path.with_suffix(".meta.json")
        if path.exists() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("settled") is True:
                self.cache_hits += 1
                return split_payload(path.read_bytes())

        payload = self._get(params)
        fetched_at = self._now_utc()
        settle_days = (
            self._options.imbalance_settle_days
            if params.get("documentType") == "A85"
            else self._options.settle_days
        )
        # An error answer, even one sent with HTTP 200, is never kept as final.
        settled = period_end + pd.Timedelta(days=settle_days) <= fetched_at
        settled = settled and payload_is_final(payload)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Payload first, then metadata: a crash in between leaves no metadata
        # claiming the new payload settled, so the next run downloads it again.
        _write_bytes_atomic(path, payload)
        _write_json_atomic(
            meta_path, {"settled": settled, "fetched_at_utc": fetched_at.isoformat()}
        )
        return split_payload(payload)

    def day_ahead_prices(
        self,
        first: date,
        last: date,
        domain: str = DE_LU_BIDDING_ZONE,
        timezone: str = MARKET_TIMEZONE,
    ) -> pd.DataFrame:
        """Long frame of A44 prices for local delivery days ``first`` to ``last``."""
        frames: list[pd.DataFrame] = []
        for start, end in month_chunks(first, last, timezone):
            params = {
                "documentType": "A44",
                "in_Domain": domain,
                "out_Domain": domain,
                "periodStart": api_time(start),
                "periodEnd": api_time(end),
            }
            frames.extend(self._chunk(params, end, parse_day_ahead_document))
        return _concat_long(frames, "position")

    def imbalance_prices(
        self,
        first: date,
        last: date,
        area: str = DE_IMBALANCE_AREA,
        timezone: str = MARKET_TIMEZONE,
    ) -> pd.DataFrame:
        """Long frame of A85 imbalance prices for local days ``first`` to ``last``."""
        frames: list[pd.DataFrame] = []
        for start, end in month_chunks(first, last, timezone):
            params = {
                "documentType": "A85",
                "controlArea_Domain": area,
                "periodStart": api_time(start),
                "periodEnd": api_time(end),
            }
            frames.extend(self._chunk(params, end, parse_imbalance_document))
        return _concat_long(frames, "category")

    def _chunk(
        self,
        params: Mapping[str, str],
        period_end: pd.Timestamp,
        parser: Callable[[bytes], pd.DataFrame],
    ) -> list[pd.DataFrame]:
        try:
            return [parser(doc) for doc in self.documents(params, period_end)]
        except NoMatchingDataError:
            self.empty_chunks.append(_describe(params))
            return []

    def _get(self, params: Mapping[str, str]) -> bytes:
        query = {**params, "securityToken": self._token}
        what = _describe(params)
        attempts = max(self._options.attempts, 1)
        failure = "no attempt made"
        status: int | None = None
        for attempt in range(1, attempts + 1):
            self._limiter.wait()
            self.requests_made += 1
            retry_after: str | None = None
            try:
                response = self._transport(
                    self._options.base_url, query, self._options.timeout_s
                )
            except (requests.RequestException, OSError) as exc:
                # Keep only a redacted string; the exception object itself may
                # carry the URL with the token.
                failure = redact(f"{type(exc).__name__}: {exc}", self._token)
                status = None
            else:
                status = response.status_code
                if status == 200:
                    return bytes(response.content)
                if status not in RETRY_STATUSES:
                    return self._raise_client_error(what, status, response.content)
                failure = f"HTTP {status}"
                retry_after = response.headers.get("Retry-After")
            if attempt < attempts:
                delay = retry_delay(
                    attempt,
                    self._options.backoff_base_s,
                    self._options.backoff_cap_s,
                    retry_after,
                )
                _log.warning("ENTSO-E %s: %s, retrying in %.1f s", what, failure, delay)
                self._sleep(delay)
        message = f"ENTSO-E {what} failed after {attempts} attempts: {failure}"
        raise EntsoeHttpError(redact(message, self._token), status)

    def _raise_client_error(self, what: str, status: int, content: bytes) -> bytes:
        acknowledgement: Acknowledgement | None = None
        try:
            root = ElementTree.fromstring(content)
            if _local(root.tag) == "Acknowledgement_MarketDocument":
                acknowledgement = parse_acknowledgement(root)
        except ElementTree.ParseError:
            acknowledgement = None
        if acknowledgement is not None:
            safe = Acknowledgement(
                redact(acknowledgement.code, self._token),
                redact(acknowledgement.text, self._token),
            )
            raise _acknowledgement_error(safe)
        body = content[:300].decode("utf-8", "replace")
        message = f"ENTSO-E {what} failed with HTTP {status}: {body}"
        raise EntsoeHttpError(redact(message, self._token), status)


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _write_json_atomic(path: Path, content: object) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(content), encoding="utf-8")
    tmp.replace(path)


# ---------------------------------------------------------------------------
# XML parsing


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _children(element: ElementTree.Element, name: str) -> list[ElementTree.Element]:
    return [child for child in element if _local(child.tag) == name]


def _child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    for child in element:
        if _local(child.tag) == name:
            return child
    return None


def _text(element: ElementTree.Element, name: str) -> str | None:
    child = _child(element, name)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _required_text(element: ElementTree.Element, name: str) -> str:
    value = _text(element, name)
    if not value:
        raise EntsoeError(f"<{_local(element.tag)}> has no <{name}>")
    return value


def parse_acknowledgement(root: ElementTree.Element) -> Acknowledgement:
    """Code and text of the first ``Reason`` in an acknowledgement document."""
    reason = _child(root, "Reason")
    if reason is None:
        return Acknowledgement(code="", text="acknowledgement without a reason")
    return Acknowledgement(
        code=_text(reason, "code") or "", text=_text(reason, "text") or ""
    )


def _acknowledgement_error(acknowledgement: Acknowledgement) -> AcknowledgementError:
    if acknowledgement.no_matching_data:
        return NoMatchingDataError(acknowledgement)
    return AcknowledgementError(acknowledgement)


def payload_is_final(payload: bytes) -> bool:
    """True when a response holds only market documents or a "no matching data" answer.

    Anything else, such as a limit or maintenance acknowledgement, an HTML error page
    or broken XML, may change on the next request and must not be cached as settled.
    """
    try:
        documents = split_payload(payload)
    except EntsoeError:
        return False
    for content in documents:
        try:
            root = ElementTree.fromstring(content)
        except ElementTree.ParseError:
            return False
        name = _local(root.tag)
        if name == "Acknowledgement_MarketDocument":
            if not parse_acknowledgement(root).no_matching_data:
                return False
        elif not name.endswith("_MarketDocument"):
            return False
    return bool(documents)


def _document_root(
    content: bytes, root_name: str, doc_type: str
) -> ElementTree.Element:
    try:
        root = ElementTree.fromstring(content)
    except ElementTree.ParseError as exc:
        message = f"response is not valid XML: {exc}"
        raise EntsoeError(message) from None
    name = _local(root.tag)
    if name == "Acknowledgement_MarketDocument":
        raise _acknowledgement_error(parse_acknowledgement(root))
    if name != root_name:
        raise EntsoeError(f"expected {root_name}, got {name}")
    kind = _text(root, "type")
    if kind != doc_type:
        raise EntsoeError(f"expected document type {doc_type}, got {kind}")
    return root


def resolution_minutes(duration: str) -> int:
    """Minutes in an ISO 8601 resolution such as ``PT15M``, ``PT60M`` or ``PT1H``."""
    match = _DURATION.match(duration.strip())
    if match is None or not any(match.groups()):
        raise EntsoeError(f"unsupported resolution {duration!r}")
    hours, minutes = (int(group) if group else 0 for group in match.groups())
    total = hours * 60 + minutes
    if total <= 0:
        raise EntsoeError(f"unsupported resolution {duration!r}")
    return total


def _utc(text: str) -> pd.Timestamp:
    ts = pd.Timestamp(text)
    if ts.tzinfo is None:
        raise EntsoeError(f"timestamp {text!r} has no time zone")
    return ts.tz_convert("UTC")


_Row = tuple[pd.Timestamp, object, int, float]


def _expand_period(
    period: ElementTree.Element,
    curve_type: str,
    amount_tag: str,
    key: Callable[[ElementTree.Element], object],
) -> list[_Row]:
    """One row per interval; A03 carries values forward to the period end."""
    interval = _child(period, "timeInterval")
    if interval is None:
        raise EntsoeError("<Period> has no <timeInterval>")
    start = _utc(_required_text(interval, "start"))
    end = _utc(_required_text(interval, "end"))
    minutes = resolution_minutes(_required_text(period, "resolution"))
    step = pd.Timedelta(minutes=minutes)
    span = end - start
    if span <= pd.Timedelta(0) or span % step != pd.Timedelta(0):
        raise EntsoeError(f"period {start} to {end} is not a multiple of {minutes} min")
    count = int(span // step)

    given: dict[int, tuple[float, object]] = {}
    for point in _children(period, "Point"):
        position = int(_required_text(point, "position"))
        if not 1 <= position <= count:
            raise EntsoeError(
                f"point position {position} outside a period of {count} intervals"
            )
        given[position] = (float(_required_text(point, amount_tag)), key(point))

    if curve_type == "A03" and given and 1 not in given:
        raise EntsoeError("curve type A03 period does not start with position 1")
    rows: list[_Row] = []
    current: tuple[float, object] | None = None
    for position in range(1, count + 1):
        if position in given:
            current = given[position]
        elif curve_type != "A03" or current is None:
            continue
        rows.append((start + (position - 1) * step, current[1], minutes, current[0]))
    return rows


def _curve_type(series: ElementTree.Element) -> str:
    curve = _text(series, "curveType") or "A01"
    if curve not in {"A01", "A03"}:
        raise EntsoeError(f"unsupported curve type {curve}")
    return curve


def _constant_key(value: object) -> Callable[[ElementTree.Element], object]:
    def key(_point: ElementTree.Element) -> object:
        return value

    return key


def _long_frame(rows: list[_Row], key: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["timestamp_utc", key, "resolution_minutes", "price_eur_mwh"]
    )
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame["resolution_minutes"] = frame["resolution_minutes"].astype("int64")
    frame["price_eur_mwh"] = frame["price_eur_mwh"].astype("float64")
    return frame


def _concat_long(frames: list[pd.DataFrame], key: str) -> pd.DataFrame:
    if not frames:
        return _long_frame([], key)
    return pd.concat(frames, ignore_index=True)


def parse_day_ahead_document(content: bytes) -> pd.DataFrame:
    """Rows of an A44 document: timestamp, classification position, resolution, price.

    Every TimeSeries is kept. A series without a classification position is
    given position 1.
    """
    root = _document_root(content, "Publication_MarketDocument", "A44")
    rows: list[_Row] = []
    for series in _children(root, "TimeSeries"):
        currency = _text(series, "currency_Unit.name")
        unit = _text(series, "price_Measure_Unit.name")
        if currency not in (None, "EUR") or unit not in (None, "MWH"):
            raise EntsoeError(f"unexpected price unit {currency}/{unit}")
        position_text = _text(
            series, "classificationSequence_AttributeInstanceComponent.position"
        )
        position = int(position_text) if position_text else 1
        curve = _curve_type(series)
        for period in _children(series, "Period"):
            rows.extend(
                _expand_period(period, curve, "price.amount", _constant_key(position))
            )
    frame = _long_frame(rows, "position")
    frame["position"] = frame["position"].astype("int64")
    return frame


def parse_imbalance_document(content: bytes) -> pd.DataFrame:
    """Rows of an A85 document: timestamp, price category, resolution, price.

    The category is ``imbalance_Price.category`` of each point (A04 excess
    balance, A05 insufficient balance), or ``none`` when absent. Under curve
    type A03 an omitted point repeats the previous point's price and category.
    """
    root = _document_root(content, "Balancing_MarketDocument", "A85")
    rows: list[_Row] = []
    for series in _children(root, "TimeSeries"):
        currency = _text(series, "currency_Unit.name")
        unit = _text(series, "price_Measure_Unit.name")
        if currency not in (None, "EUR") or unit not in (None, "MWH"):
            raise EntsoeError(f"unexpected price unit {currency}/{unit}")
        curve = _curve_type(series)
        for period in _children(series, "Period"):
            rows.extend(
                _expand_period(
                    period,
                    curve,
                    "imbalance_Price.amount",
                    lambda point: _text(point, "imbalance_Price.category") or "none",
                )
            )
    frame = _long_frame(rows, "category")
    frame["category"] = frame["category"].astype("str")
    return frame


# ---------------------------------------------------------------------------
# Quarter-hour tables


def _quarter_hour_pivot(
    long: pd.DataFrame,
    key: str,
    grid: pd.DatetimeIndex,
    names: Callable[[object], tuple[str, str]],
) -> pd.DataFrame:
    """Spread each key onto the quarter-hour grid; coarser products repeat.

    Where one key has values at two resolutions for the same quarter-hour the
    finer one is kept. Different values at the same resolution raise.
    """
    table = pd.DataFrame(index=grid)
    if long.empty:
        return table
    frame = long.reset_index(drop=True)
    if (frame["resolution_minutes"] % QUARTER_HOUR_MINUTES != 0).any():
        raise EntsoeError("a resolution is not a multiple of 15 minutes")
    repeats = (frame["resolution_minutes"] // QUARTER_HOUR_MINUTES).to_numpy("int64")
    source_rows = np.repeat(np.arange(len(frame)), repeats)
    expanded = frame.iloc[source_rows].reset_index(drop=True)
    offsets = np.concatenate([np.arange(n, dtype="int64") for n in repeats])
    expanded["timestamp_utc"] = expanded["timestamp_utc"] + pd.to_timedelta(
        offsets * QUARTER_HOUR_MINUTES, unit="min"
    )

    groups = ["timestamp_utc", key, "resolution_minutes"]
    distinct = expanded.groupby(groups)["price_eur_mwh"].nunique()
    conflicts = distinct[distinct > 1]
    if not conflicts.empty:
        raise EntsoeError(
            f"{len(conflicts)} quarter-hours have conflicting {key} values, "
            f"first at {conflicts.index[0]}"
        )
    ordered = expanded.sort_values(groups).drop_duplicates(
        ["timestamp_utc", key], keep="first"
    )
    for value in sorted(ordered[key].unique()):
        part = ordered[ordered[key] == value].set_index("timestamp_utc")
        price_name, resolution_name = names(value)
        table[price_name] = part["price_eur_mwh"].reindex(grid).astype("float64")
        table[resolution_name] = (
            part["resolution_minutes"].reindex(grid).astype("Int64")
        )
    return table


def day_ahead_table(long: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Columns ``price_position_N`` and ``resolution_minutes_position_N``."""
    return _quarter_hour_pivot(
        long,
        "position",
        grid,
        lambda v: (f"price_position_{v}", f"resolution_minutes_position_{v}"),
    )


def imbalance_table(long: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    """Columns ``imbalance_price_{excess|insufficient}_eur_mwh`` and resolutions."""

    def names(value: object) -> tuple[str, str]:
        label = IMBALANCE_CATEGORY_LABELS.get(str(value), str(value).lower())
        return f"imbalance_price_{label}_eur_mwh", f"resolution_minutes_{label}"

    return _quarter_hour_pivot(long, "category", grid, names)


# ---------------------------------------------------------------------------
# Cross-check report


@dataclass(frozen=True)
class Comparison:
    """How one ENTSO-E price series compares with SMARD on shared quarter-hours."""

    periods: int
    identical_share: float
    max_abs_diff: float
    mean_diff: float
    mismatch_days: list[date]


def compare_prices(
    entsoe: pd.Series,
    smard: pd.Series,
    timezone: str = MARKET_TIMEZONE,
    tolerance: float = TOLERANCE_EUR_MWH,
) -> Comparison:
    """Compare two price series where both have a value."""
    joined = pd.concat({"entsoe": entsoe, "smard": smard}, axis=1).dropna()
    if joined.empty:
        return Comparison(0, math.nan, math.nan, math.nan, [])
    diff = joined["entsoe"] - joined["smard"]
    mismatch = (diff.abs() > tolerance).to_numpy()
    stamps = pd.DatetimeIndex(joined.index)[mismatch]
    days = sorted(set(stamps.tz_convert(timezone).date))
    return Comparison(
        periods=len(joined),
        identical_share=float(1.0 - mismatch.mean()),
        max_abs_diff=float(diff.abs().max()),
        mean_diff=float(diff.mean()),
        mismatch_days=days,
    )


def _pct(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{100.0 * value:.2f}%"


def _num(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:,.2f}"


def _day_list(days: list[date], limit: int) -> str:
    if not days:
        return "none"
    shown = ", ".join(day.isoformat() for day in days[:limit])
    extra = len(days) - limit
    return shown if extra <= 0 else f"{shown}, and {extra} more"


def _missing_days(series: pd.Series, timezone: str) -> list[date]:
    stamps = pd.DatetimeIndex(series.index)[series.isna().to_numpy()]
    return sorted(set(stamps.tz_convert(timezone).date))


def _eras(
    index: pd.DatetimeIndex, switch_utc: pd.Timestamp
) -> list[tuple[str, NDArray[np.bool_]]]:
    return [
        ("Hourly products (before 2025-10-01)", np.asarray(index < switch_utc)),
        ("15-minute products (from 2025-10-01)", np.asarray(index >= switch_utc)),
    ]


def build_price_check_report(
    day_ahead: pd.DataFrame,
    imbalance: pd.DataFrame | None,
    smard_price: pd.Series,
    first: date,
    last: date,
    quarter_hour_from: date = date(2025, 10, 1),
    timezone: str = MARKET_TIMEZONE,
) -> str:
    """Markdown comparing ENTSO-E A44 positions with SMARD, plus imbalance prices."""
    grid = quarter_hour_grid(first, last, timezone)
    switch_utc = local_midnight_utc(quarter_hour_from, timezone)
    da = day_ahead.reindex(grid)
    smard = smard_price.reindex(grid)
    lines = [
        "# ENTSO-E price cross-check",
        "",
        "Generated by `uv run python -m src.ingest.entsoe --what report` from",
        "`data/processed/entsoe/` and the SMARD quarter-hour dataset. Delivery "
        f"days {first.isoformat()} to {last.isoformat()} ({timezone}), compared "
        "on the UTC 15-minute grid. Hourly prices are repeated over their four "
        "quarter-hours on both sides. Identical means an absolute difference of "
        f"at most {TOLERANCE_EUR_MWH} EUR/MWh.",
        "",
        "Sources: ENTSO-E Transparency Platform, day-ahead prices [12.1.D] "
        f"(A44, DE-LU {DE_LU_BIDDING_ZONE}) and imbalance prices [17.1.G] "
        f"(A85, Germany {DE_IMBALANCE_AREA}); SMARD, Bundesnetzagentur, "
        "day-ahead price DE-LU (CC BY 4.0).",
        "",
        "## What classification positions 1 and 2 are",
        "",
        POSITION_NOTES.rstrip(),
        "",
    ]

    comparisons: dict[tuple[int, str], Comparison] = {}
    for position in (1, 2):
        column = f"price_position_{position}"
        lines += [f"## Position {position} against SMARD", ""]
        if column not in da.columns:
            lines += ["No values for this position.", ""]
            continue
        lines += [
            "| Era | Quarter-hours on grid | ENTSO-E values | SMARD values "
            "| Compared | Identical | Max abs diff (EUR/MWh) "
            "| Mean diff (EUR/MWh) | Days with a mismatch |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        day_notes: list[str] = []
        for era, mask in _eras(grid, switch_utc):
            comparison = compare_prices(da[column][mask], smard[mask], timezone)
            comparisons[(position, era)] = comparison
            entsoe_values = int(da[column][mask].notna().sum())
            lines.append(
                f"| {era} | {int(mask.sum()):,} | {entsoe_values:,}"
                f" | {int(smard[mask].notna().sum()):,} | {comparison.periods:,}"
                f" | {_pct(comparison.identical_share)}"
                f" | {_num(comparison.max_abs_diff)} | {_num(comparison.mean_diff)}"
                f" | {len(comparison.mismatch_days)} |"
            )
            limit = 60 if position == 1 else 10
            day_notes.append(f"* {era}: {_day_list(comparison.mismatch_days, limit)}")
        lines += ["", "Days with any mismatch (local delivery days):", ""]
        lines += [*day_notes, ""]
        lines += [
            "Days with at least one quarter-hour missing (local delivery days): "
            f"ENTSO-E {_day_list(_missing_days(da[column], timezone), 20)}; "
            f"SMARD {_day_list(_missing_days(smard, timezone), 20)}.",
            "",
        ]

    lines += _imbalance_section(imbalance, da, grid)
    lines += ["## Finding", "", *_finding(comparisons), ""]
    return "\n".join(lines)


def _imbalance_section(
    imbalance: pd.DataFrame | None, day_ahead: pd.DataFrame, grid: pd.DatetimeIndex
) -> list[str]:
    lines = ["## Imbalance prices (Germany, reBAP)", ""]
    if imbalance is None or imbalance.empty:
        return [*lines, "No imbalance prices were available.", ""]
    table = imbalance.reindex(grid)
    price_columns = [c for c in table.columns if c.startswith("imbalance_price_")]
    present = table[price_columns].notna().any(axis=1)
    stamps = pd.DatetimeIndex(grid[present.to_numpy()])
    if stamps.empty:
        return [*lines, "No imbalance prices were available.", ""]
    lines += [
        f"Coverage: {int(present.sum()):,} of {len(grid):,} quarter-hours, "
        f"{stamps[0].isoformat()} to {stamps[-1].isoformat()} (UTC start times).",
        "",
        "| Series | Values | Min | Max | Mean | Share above day-ahead position 1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    reference = day_ahead.get("price_position_1")
    for column in price_columns:
        series = table[column]
        share = math.nan
        if reference is not None:
            both = pd.concat({"imb": series, "da": reference}, axis=1).dropna()
            if not both.empty:
                share = float((both["imb"] > both["da"]).mean())
        lines.append(
            f"| {column} | {int(series.notna().sum()):,} | {_num(series.min())}"
            f" | {_num(series.max())} | {_num(series.mean())} | {_pct(share)} |"
        )
    excess = "imbalance_price_excess_eur_mwh"
    short = "imbalance_price_insufficient_eur_mwh"
    if excess in table.columns and short in table.columns:
        both = table[[excess, short]].dropna()
        if not both.empty:
            same = float(
                ((both[excess] - both[short]).abs() <= TOLERANCE_EUR_MWH).mean()
            )
            lines += [
                "",
                "Categories A04 (excess balance) and A05 (insufficient balance) "
                f"carry the same price in {_pct(same)} of {len(both):,} "
                "quarter-hours, as expected for the single German reBAP.",
            ]
    return [*lines, ""]


def _finding(comparisons: dict[tuple[int, str], Comparison]) -> list[str]:
    sentences: list[str] = []
    for position, label in ((1, "SDAC"), (2, "EXAA 10:15 auction")):
        parts = []
        for (pos, era), comparison in comparisons.items():
            if pos != position or comparison.periods == 0:
                continue
            parts.append(
                f"{_pct(comparison.identical_share)} of {comparison.periods:,} "
                f"quarter-hours in the {era.split(' (')[0].lower()} era "
                f"({len(comparison.mismatch_days)} days with a mismatch, largest "
                f"gap {_num(comparison.max_abs_diff)} EUR/MWh)"
            )
        if parts:
            sentences.append(
                f"* Position {position} ({label}) equals SMARD in "
                + "; ".join(parts)
                + "."
            )
    sentences.append(
        "* Position 1 is the series to use as the ENTSO-E day-ahead price; it is "
        "the same SDAC clearing price SMARD publishes. Position 2 is a different "
        "auction and must not be mixed with it."
    )
    return sentences


# ---------------------------------------------------------------------------
# Command line


def _parse_day(text: str) -> date:
    return datetime.strptime(text, "%Y-%m-%d").date()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m src.ingest.entsoe",
        description="Download ENTSO-E prices or write the SMARD cross-check report.",
    )
    parser.add_argument(
        "--what", choices=["day-ahead", "imbalance", "report"], required=True
    )
    parser.add_argument("--first", type=_parse_day, default=DEFAULT_FIRST_DAY)
    parser.add_argument("--last", type=_parse_day, default=None)
    args = parser.parse_args(None if argv is None else list(argv))

    settings = load_settings()
    timezone = settings.market.timezone
    first: date = args.first
    last: date = args.last or pd.Timestamp.now(tz=timezone).date()
    processed = settings.data.processed_path / "entsoe"

    if args.what == "report":
        day_ahead = pd.read_parquet(processed / DAY_AHEAD_FILE)
        imbalance_path = processed / IMBALANCE_FILE
        imbalance = pd.read_parquet(imbalance_path) if imbalance_path.exists() else None
        smard = pd.read_parquet(settings.data.dataset_path, columns=[PRICE_SERIES])
        report = build_price_check_report(
            day_ahead,
            imbalance,
            smard[PRICE_SERIES],
            first,
            last,
            settings.market.quarter_hour_products_from,
            timezone,
        )
        target = REPO_ROOT / REPORT_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(report, encoding="utf-8")
        print(f"wrote {REPORT_PATH}")
        return 0

    client = EntsoeClient(load_token(), settings.data.raw_path / "entsoe")
    grid = quarter_hour_grid(first, last, timezone)
    started = time.monotonic()
    if args.what == "day-ahead":
        table = day_ahead_table(
            client.day_ahead_prices(first, last, timezone=timezone), grid
        )
        name = DAY_AHEAD_FILE
    else:
        table = imbalance_table(
            client.imbalance_prices(first, last, timezone=timezone), grid
        )
        name = IMBALANCE_FILE
    processed.mkdir(parents=True, exist_ok=True)
    table.to_parquet(processed / name)
    elapsed = time.monotonic() - started
    filled = {
        column: int(table[column].notna().sum())
        for column in table.columns
        if "price" in column
    }
    print(
        f"wrote data/processed/entsoe/{name}: {len(table):,} quarter-hours, "
        f"values per column {filled}; {client.requests_made} requests, "
        f"{client.cache_hits} cache hits, {len(client.empty_chunks)} months "
        f"without data, {elapsed:.0f} s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
