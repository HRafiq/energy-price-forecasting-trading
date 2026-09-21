"""Open-Meteo Previous Runs client: archived weather forecasts at a fixed lead.

Grid operators publish their wind and solar forecasts for day D+1 after the
11:40 issue time on day D, so the models use archived weather-model forecasts
issued ``lead_days`` before each valid time instead. The endpoint is::

    {base_url}?latitude=54.4,48.8&longitude=6.8,11.8
        &minutely_15=wind_speed_100m_previous_day2,...
        &start_date=YYYY-MM-DD&end_date=YYYY-MM-DD&timezone=UTC
        -> [{"latitude": .., "longitude": .., "utc_offset_seconds": 0,
             "minutely_15": {"time": ["2024-03-01T00:00", ...],
                             "wind_speed_100m_previous_day2": [x | null, ...]}},
            ...]                                  one object per location, in order

``{variable}_previous_day{n}`` is the value predicted ``n`` days (``24 * n``
hours) before its valid time.

Checked against the live API on 2026-09-13, around 14:00 UTC:

* ``minutely_15`` with the ``_previous_day2`` suffix returned 96 non-null values
  per day for all three configured variables (wind_speed_100m,
  shortwave_radiation, temperature_2m) at 54.4,6.8 and 48.8,11.8 on 2024-02-28,
  2024-03-01, 2024-07-15, 2025-01-10, 2025-10-26, 2026-03-29, 2026-09-12,
  2026-09-14 and 2026-09-15. All 12 configured points for 2025-06-01 to
  2025-07-01 gave 2,976 rows each with no nulls, in one request of 1.7 s and
  1.2 MB. No variable needs an hourly fallback, so this client requests only
  native 15-minute data. ``cell_selection`` default, ``nearest`` and ``sea``
  chose the same grid cells with identical values for all 12 points, offshore
  ones included, so the default is kept.
* The 15-minute data is native, not interpolated. Wind and temperature at
  :15, :30 and :45 differ from linear interpolation between full hours by up to
  30 km/h and 4.2 degrees C. The hourly endpoint's value equals the 15-minute
  value at :00.
* With ``timezone=UTC`` the times are naive ``YYYY-MM-DDTHH:MM`` strings in UTC
  and ``utc_offset_seconds`` is 0. Units: km/h, W/m2, degrees C.
* Wind and temperature are instantaneous: the value is the state at its stamp.
  Shortwave radiation is the mean over the 15 minutes ENDING at its stamp. The
  hourly value stamped HH:00 matched the mean of the 15-minute values stamped
  HH-1:15 to HH:00, with a mean absolute error of 0.7 W/m2 (2024-07-15,
  48.8,11.8) and 2.3 W/m2 (2025-04-10, 52.3,13.5). The following hour's mean
  was off by 55 and 41 W/m2. Sunset agrees: the last non-zero 15-minute value was
  stamped 19:15 for a 19:11 sunset and 18:00 for a 17:55 sunset.
* Near the present, ``_previous_day2`` is silently filled from newer runs. At
  14:01 UTC on 2026-09-13 it differed from the latest forecast only up to valid
  times 2026-09-15 09:30 (wind) and 12:45 (temperature). Later valid times
  equalled the newest run, forecasts with less than 48 hours of lead, and none
  were null. ``download_weather_forecasts(..., as_of=...)`` masks those values.

Alignment rule. Output rows are labelled by the UTC START ``t`` of the period
``[t, t + 15 min)``. An instantaneous variable takes the value stamped ``t``. A
variable in ``PRECEDING_PERIOD_VARIABLES`` (radiation means, precipitation sums)
takes the value stamped ``t + 15 min``, the one that describes ``[t, t + 15 min)``.
One extra day is downloaded after ``end`` so the last period of ``end`` has it.

Masking rule. With ``as_of`` given, a value stamped ``s`` is kept only if
``s <= as_of + lead_days - ARCHIVE_LAG``. The observed fill-in started 1.3 hours
(temperature) and 4.5 hours (wind) before ``as_of + 48 h``. Ten hours leaves
room for a six-hourly model published a few hours after its run, and still
keeps every period of local D+1 at the 11:40 issue on day D.

Caching. Requests cover ``request_days``-long ranges anchored at
``weather.start``, all points in one request, cached as raw JSON under
``data/raw/open_meteo/``. A chunk whose end date is at least ``lead_days + 7``
days before the download day, or before ``end`` if earlier, is settled. A
sidecar records the request URL
and whether the chunk was already settled when downloaded. Only a chunk that is
settled now, was settled then, and was requested with the same URL (points,
variables, lead) is served from cache. Newer chunks are downloaded on every run.

Free non-commercial use: under 10,000 calls a day, 5,000 an hour and 600 a
minute. Open-Meteo may count a request with many locations or a long range as
several calls, so requests are paused and retried with backoff. Data licence:
CC BY 4.0, attribution "Weather data by Open-Meteo.com".
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import requests

from src.config import Settings, WeatherConfig, load_settings
from src.ingest.merge import merge_into

__all__ = [
    "ARCHIVE_LAG",
    "CACHE_SUBDIR",
    "OUTPUT_FILE",
    "PRECEDING_PERIOD_VARIABLES",
    "QUARTER_HOUR",
    "SETTLE_EXTRA_DAYS",
    "FetchJson",
    "OpenMeteoError",
    "align_to_period_start",
    "api_variable",
    "build_request_url",
    "chunk_ranges",
    "download_weather_forecasts",
    "http_fetch_json",
    "is_settled",
    "main",
    "mask_unpublished",
    "parse_response",
    "settle_reference",
]

FetchJson = Callable[[str], Any]
Sleep = Callable[[float], None]

QUARTER_HOUR = pd.Timedelta(minutes=15)
#: Variables Open-Meteo reports as a mean or sum over the period ending at the
#: stamp. Every other variable is treated as instantaneous.
PRECEDING_PERIOD_VARIABLES: frozenset[str] = frozenset(
    {
        "shortwave_radiation",
        "direct_radiation",
        "diffuse_radiation",
        "direct_normal_irradiance",
        "global_tilted_irradiance",
        "terrestrial_radiation",
        "shortwave_radiation_instant_free",
        "precipitation",
        "rain",
        "showers",
        "snowfall",
        "sunshine_duration",
        "lightning_potential",
    }
)
#: Values stamped later than ``as_of + lead_days - ARCHIVE_LAG`` are masked.
ARCHIVE_LAG = pd.Timedelta(hours=10)
#: Days beyond ``lead_days`` after which a chunk counts as settled.
SETTLE_EXTRA_DAYS = 7
CACHE_SUBDIR = "open_meteo"
OUTPUT_FILE = "weather_forecast_quarterhour.parquet"
DEFAULT_PAUSE_S = 2.0

_TIME_FORMAT = "%Y-%m-%dT%H:%M"
_COORD_TOLERANCE_DEG = 0.1
_USER_AGENT = "energy-price-forecasting-trading/0.1 (public research project)"


class OpenMeteoError(RuntimeError):
    """Open-Meteo returned something this client cannot use."""


def http_fetch_json(
    timeout_s: float,
    attempts: int = 4,
    backoff_s: float = 5.0,
    sleep: Sleep = time.sleep,
) -> FetchJson:
    """A fetcher that GETs JSON, retrying 429, 5xx and network errors."""

    def fetch(url: str) -> Any:
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(
                    url, timeout=timeout_s, headers={"User-Agent": _USER_AGENT}
                )
            except requests.RequestException as exc:
                error: Exception = exc
            else:
                status = response.status_code
                if status == 200:
                    try:
                        return response.json()
                    except ValueError as exc:
                        error = exc
                elif status == 429 or status >= 500:
                    error = OpenMeteoError(f"HTTP {status}: {_reason(response)}")
                else:
                    raise OpenMeteoError(
                        f"GET {url} -> HTTP {status}: {_reason(response)}"
                    )
            if attempt == attempts:
                raise OpenMeteoError(
                    f"GET {url} failed after {attempts} attempts"
                ) from error
            sleep(backoff_s * 2 ** (attempt - 1))
        raise OpenMeteoError(f"GET {url}: no attempts made")

    return fetch


def _reason(response: requests.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict) and "reason" in body:
        return str(body["reason"])
    return str(body)[:200]


def api_variable(variable: str, lead_days: int) -> str:
    """The API name of ``variable`` as forecast ``lead_days`` before valid time."""
    return f"{variable}_previous_day{lead_days}"


def build_request_url(config: WeatherConfig, start: date, end: date) -> str:
    """URL for all points and variables at 15 minutes, ``start``..``end`` inclusive."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    coords = list(config.points.values())
    params = {
        "latitude": ",".join(str(float(lat)) for lat, _ in coords),
        "longitude": ",".join(str(float(lon)) for _, lon in coords),
        "minutely_15": ",".join(
            api_variable(v, config.lead_days) for v in config.variables
        ),
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "timezone": "UTC",
    }
    return f"{config.base_url}?{urlencode(params, safe=',')}"


def chunk_ranges(start: date, end: date, request_days: int) -> list[tuple[date, date]]:
    """Inclusive date ranges of ``request_days`` anchored at ``start``."""
    if end < start:
        raise ValueError(f"end {end} is before start {start}")
    if request_days < 1:
        raise ValueError("request_days must be at least 1")
    ranges: list[tuple[date, date]] = []
    first = start
    while first <= end:
        last = min(first + timedelta(days=request_days - 1), end)
        ranges.append((first, last))
        first += timedelta(days=request_days)
    return ranges


def is_settled(chunk_end: date, requested_end: date, lead_days: int) -> bool:
    """Settled once the chunk ends ``lead_days + 7`` days before ``requested_end``."""
    return chunk_end <= requested_end - timedelta(days=lead_days + SETTLE_EXTRA_DAYS)


def settle_reference(end: date, as_of: pd.Timestamp) -> date:
    """The day chunks are judged settled against: ``end`` or the download day.

    Comparing with the requested ``end`` alone would let a run that asks for
    future dates cache recent chunks as settled while they hold fresher values.
    """
    return min(end, as_of.tz_convert("UTC").date())


def _to_float(value: Any, key: str) -> float:
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise OpenMeteoError(f"{key}: non-numeric value {value!r}")
    return float(value)


def parse_response(payload: Any, config: WeatherConfig) -> pd.DataFrame:
    """Turn a multi-location response into one wide frame; null -> NaN.

    The index is the API's own UTC stamp (``valid_time_utc``), not yet aligned
    to period starts. Columns are ``config.columns`` in order.
    """
    if isinstance(payload, dict) and payload.get("error"):
        raise OpenMeteoError(f"API error: {payload.get('reason')}")
    points = list(config.points.items())
    locations = [payload] if isinstance(payload, dict) and len(points) == 1 else payload
    if not isinstance(locations, list) or len(locations) != len(points):
        got = len(locations) if isinstance(locations, list) else type(payload).__name__
        raise OpenMeteoError(f"expected {len(points)} locations, got {got}")

    columns: dict[str, list[float]] = {}
    index: pd.DatetimeIndex | None = None
    for (name, (lat, lon)), location in zip(points, locations, strict=True):
        if not isinstance(location, dict):
            raise OpenMeteoError(f"{name}: location entry is not an object")
        if location.get("utc_offset_seconds", 0) != 0:
            raise OpenMeteoError(f"{name}: response is not in UTC")
        _check_coordinates(name, lat, lon, location)
        block = location.get("minutely_15")
        if not isinstance(block, dict) or not isinstance(block.get("time"), list):
            raise OpenMeteoError(f"{name}: response has no minutely_15 time list")
        times = block["time"]
        try:
            stamps = pd.DatetimeIndex(
                pd.to_datetime(pd.Index(times), format=_TIME_FORMAT, utc=True),
                name="valid_time_utc",
            )
        except (TypeError, ValueError) as exc:
            raise OpenMeteoError(f"{name}: malformed time values") from exc
        if index is None:
            index = stamps
        elif not stamps.equals(index):
            raise OpenMeteoError(f"{name}: time axis differs from the first location")
        for variable in config.variables:
            key = api_variable(variable, config.lead_days)
            raw = block.get(key)
            if not isinstance(raw, list) or len(raw) != len(times):
                raise OpenMeteoError(f"{name}: {key} missing or of the wrong length")
            columns[config.column(name, variable)] = [_to_float(v, key) for v in raw]
    if index is None:
        raise OpenMeteoError("response has no locations")
    return pd.DataFrame(columns, index=index, columns=config.columns, dtype="float64")


def _check_coordinates(
    name: str, lat: float, lon: float, location: dict[str, Any]
) -> None:
    got_lat, got_lon = location.get("latitude"), location.get("longitude")
    if not isinstance(got_lat, int | float) or not isinstance(got_lon, int | float):
        raise OpenMeteoError(f"{name}: response has no coordinates")
    if (
        abs(got_lat - lat) > _COORD_TOLERANCE_DEG
        or abs(got_lon - lon) > _COORD_TOLERANCE_DEG
    ):
        raise OpenMeteoError(
            f"{name}: requested {lat},{lon} but got {got_lat},{got_lon}; "
            "locations out of order?"
        )


def mask_unpublished(
    frame: pd.DataFrame, as_of: pd.Timestamp, lead_days: int
) -> pd.DataFrame:
    """NaN every value stamped after ``as_of + lead_days - ARCHIVE_LAG``.

    ``frame`` is indexed by API stamps, before :func:`align_to_period_start`.
    """
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    cutoff = as_of.tz_convert("UTC") + pd.Timedelta(days=lead_days) - ARCHIVE_LAG
    late = pd.DatetimeIndex(frame.index) > cutoff
    masked = frame.copy()
    masked.loc[late, :] = float("nan")
    return masked


def align_to_period_start(frame: pd.DataFrame, config: WeatherConfig) -> pd.DataFrame:
    """Relabel API stamps as 15-minute period starts (``timestamp_utc``).

    Values of ``PRECEDING_PERIOD_VARIABLES`` describe the 15 minutes ending at
    their stamp, so they move 15 minutes earlier. Instantaneous values stay.
    """
    parts: list[pd.Series] = []
    for point in config.points:
        for variable in config.variables:
            column = config.column(point, variable)
            series = frame[column]
            if variable in PRECEDING_PERIOD_VARIABLES:
                series = series.set_axis(pd.DatetimeIndex(series.index) - QUARTER_HOUR)
            parts.append(series)
    aligned = pd.concat(parts, axis=1).sort_index()
    aligned.index.name = "timestamp_utc"
    return aligned[config.columns]


def _utc_midnight(day: date) -> pd.Timestamp:
    return pd.Timestamp(day.isoformat()).tz_localize("UTC")


def _check_grid(index: pd.Index, expected: pd.DatetimeIndex, what: str) -> None:
    stamps = pd.DatetimeIndex(index)
    if stamps.has_duplicates:
        first = stamps[stamps.duplicated()][0]
        raise OpenMeteoError(f"{what}: duplicate timestamps, first at {first}")
    missing = expected.difference(stamps)
    if len(missing):
        raise OpenMeteoError(
            f"{what}: {len(missing)} timestamps missing, first at {missing[0]}"
        )
    extra = stamps.difference(expected)
    if len(extra):
        raise OpenMeteoError(
            f"{what}: {len(extra)} unexpected timestamps, first at {extra[0]}"
        )
    if not stamps.equals(expected):
        raise OpenMeteoError(f"{what}: timestamps are not in order")


def _cached_settled_payload(path: Path, meta_path: Path, url: str) -> Any | None:
    if not (path.exists() and meta_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except ValueError:
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get("url") != url or meta.get("settled_at_fetch") is not True:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json_atomic(path: Path, content: Any) -> None:
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(json.dumps(content), encoding="utf-8")
    tmp.replace(path)


def download_weather_forecasts(
    settings: Settings,
    end: date,
    *,
    fetch_json: FetchJson | None = None,
    as_of: pd.Timestamp | None = None,
    pause_s: float = DEFAULT_PAUSE_S,
    sleep: Sleep = time.sleep,
    since: date | None = None,
) -> pd.DataFrame:
    """Forecasts issued ``lead_days`` ahead, per UTC quarter-hour.

    Covers UTC days ``settings.weather.start`` through ``end`` inclusive, or
    ``since`` through ``end`` when ``since`` is later. A machine with no cache
    pays for every chunk it asks for, so a scheduled run asks only for the days
    it needs; the caller keeps whatever longer file it already had.

    The index ``timestamp_utc`` holds period starts, columns are
    ``settings.weather.columns`` as floats, NaN where the API returned null. When
    ``as_of`` is given, values not genuinely archived at that instant are NaN
    too (see the module docstring).
    """
    config = settings.weather
    if end < config.start:
        raise ValueError(f"end {end} is before weather.start {config.start}")
    first_day = config.start if since is None else max(since, config.start)
    if end < first_day:
        raise ValueError(f"end {end} is before since {first_day}")
    if as_of is not None and as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    fetch = fetch_json or http_fetch_json(config.timeout_s, sleep=sleep)
    cache_dir = settings.data.raw_path / CACHE_SUBDIR
    # The value closing the last period of ``end`` is stamped 00:00 the next day.
    fetch_end = end + timedelta(days=1)

    frames: list[pd.DataFrame] = []
    network_calls = 0
    for first, last in chunk_ranges(first_day, fetch_end, config.request_days):
        url = build_request_url(config, first, last)
        path = cache_dir / f"{first.isoformat()}_{last.isoformat()}.json"
        meta_path = path.with_name(f"{path.stem}.meta.json")
        # Without a download time nothing counts as settled, so no chunk is cached
        # for good while it might still hold fresher-than-lead forecasts.
        settled = as_of is not None and is_settled(
            last, settle_reference(end, as_of), config.lead_days
        )
        payload = _cached_settled_payload(path, meta_path, url) if settled else None
        if payload is not None:
            frames.append(parse_response(payload, config))
            continue
        if network_calls and pause_s > 0:
            sleep(pause_s)
        payload = fetch(url)
        network_calls += 1
        frames.append(parse_response(payload, config))
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Payload first, then metadata: a crash in between leaves no metadata
        # claiming the new payload settled, so the next run downloads it again.
        _write_json_atomic(path, payload)
        _write_json_atomic(meta_path, {"url": url, "settled_at_fetch": settled})

    raw = pd.concat(frames)
    stamps = pd.date_range(
        _utc_midnight(first_day),
        _utc_midnight(fetch_end + timedelta(days=1)),
        freq=QUARTER_HOUR,
        inclusive="left",
    )
    _check_grid(raw.index, stamps, "downloaded forecasts")
    if as_of is not None:
        raw = mask_unpublished(raw, as_of, config.lead_days)

    grid = pd.date_range(
        _utc_midnight(first_day),
        _utc_midnight(end + timedelta(days=1)),
        freq=QUARTER_HOUR,
        inclusive="left",
        name="timestamp_utc",
    )
    result = align_to_period_start(raw, config).reindex(grid).astype("float64")
    _check_grid(result.index, grid, "aligned forecasts")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download archived Open-Meteo weather forecasts."
    )
    parser.add_argument("--config", type=Path, default=None, help="settings YAML path")
    parser.add_argument(
        "--end",
        type=date.fromisoformat,
        default=None,
        help="last UTC day, YYYY-MM-DD (default: today plus two days)",
    )
    parser.add_argument(
        "--history-days",
        type=int,
        default=None,
        help="download only this many days before --end instead of everything "
        "from weather.start. Rows already in the file are kept, so this shortens "
        "the download, never the file. Use src.pipeline.history to size it",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    as_of = pd.Timestamp.now(tz="UTC")
    end: date = args.end or as_of.date() + timedelta(days=2)
    since = (
        None if args.history_days is None else end - timedelta(days=args.history_days)
    )
    frame = download_weather_forecasts(settings, end, as_of=as_of, since=since)

    out = settings.data.processed_path / OUTPUT_FILE
    if since is not None:
        # A windowed download must not shorten a longer file: the backtest reads
        # the older rows. New rows win, so a revised value replaces its revision.
        frame = merge_into(frame, out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(f"{out.name}.tmp")
    frame.to_parquet(tmp)
    tmp.replace(out)

    complete = frame.notna().all(axis=1)
    first_complete = frame.index[complete.to_numpy()]
    cutoff = as_of + pd.Timedelta(days=settings.weather.lead_days) - ARCHIVE_LAG
    print(f"rows            {len(frame)}")
    print(f"first (UTC)     {frame.index[0]}")
    print(f"last (UTC)      {frame.index[-1]}")
    print(f"as of (UTC)     {as_of}; stamps after {cutoff} masked")
    print(f"first complete  {first_complete[0] if len(first_complete) else 'none'}")
    print("null counts")
    for column, nulls in frame.isna().sum().items():
        print(f"  {column:<48} {nulls}")
    print(f"wrote           {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
