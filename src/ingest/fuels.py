"""Daily fuel prices: TTF gas settlements and EU ETS auction clearing prices.

    uv run python -m src.ingest.fuels

Writes ``data/processed/fuels_daily.parquet``: one row per calendar date with at
least one observation, columns ``gas_ttf_eur_mwh`` and ``carbon_eua_eur_t``, NaN
where that series has no observation that date. Models must not read these rows
directly: :func:`last_known_before` maps them to delivery days so that a price
observed on day x never counts for day x.

TTF gas (checked against Yahoo Finance on 2026-09-13):

* Ticker ``TTF=F`` is the continuous front-month Dutch TTF future in EUR/MWh.
  The daily bar's Close is used as the settlement. Its index is midnight
  America/New_York and its date is the trading date. Close equals Adj Close.
* Bars follow the US exchange calendar: US holidays (Martin Luther King Day,
  Presidents' Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Good
  Friday) have no bar even though ICE Endex trades TTF then. The front month
  rolls, so there are jumps at contract expiry.
* During a trading day the newest bar is a live price, not a settlement, so
  ``end`` is exclusive and the CLI passes today's date.
* Licence: Yahoo Finance data is provided for personal use under Yahoo's terms.
  yfinance is an unofficial client with no affiliation to Yahoo, so the data is
  not for redistribution or commercial use without a licence.

EUA carbon (EEX primary auction reports, checked on 2026-09-13):

* The 2012-2025 zip holds one workbook per year: ``.xls`` up to 2019 (xlrd),
  ``.xlsx`` from 2020. Later years are separate ``.xlsx`` downloads and the
  current year's file grows as auctions happen. A year with no file yet
  returns HTTP 404.
* Sheet 0 is "Primary Market Auction". Row 5 (0-based) is the header and
  column 0 is empty. Columns include Date, Time, Auction Name, Contract, a
  Status column from 2020 on ("successful" or "cancelled"; cancelled rows have
  no price), "Auction Price €/tCO2" and "Auction Volume tCO2". Data rows run
  newest first from row 6.
* Contract ``T3PA`` holds general allowances (EU, DE, PL and NIR auctions) and
  ``EAA3`` holds aviation allowances (EUAA). Only general allowances are kept:
  EUAA is a separate product that cleared on average 0.38 EUR/t away from the
  same day's EUA auction (at most 1.28), and 23 dates from 2022 to 2024 had
  only an EUAA auction.
* After that filter no date from 2018 to 2026 has two general auctions. If one
  ever does, the day's price is the volume-weighted mean of the clearing
  prices, which is the average price paid per allowance that day. A plain mean
  is used if a volume is missing.
* Auctions clear at about 11:00 CET on most weekdays. Each January there is a
  pause of about three weeks, and in 2021 auctions restarted only on 29 January.
* Licence: EEX publishes auction results as required by the EU Auctioning
  Regulation. Redistribution of the data is governed by EEX's website terms of
  use, which grant no open licence, so cite EEX and check the terms before
  republishing.
"""

from __future__ import annotations

import argparse
import io
import re
import time
import zipfile
from collections.abc import Callable, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

from src.config import CARBON_COLUMN, GAS_COLUMN, Settings, load_settings

__all__ = [
    "FUELS_FILE",
    "FetchBytes",
    "FetchTtf",
    "FuelsError",
    "daily_carbon_prices",
    "http_fetch_bytes",
    "last_known_before",
    "load_daily_fuel_prices",
    "main",
    "parse_auction_report",
    "ttf_settlements",
    "yahoo_fetch_ttf",
]

#: (ticker, first date, exclusive end date or None) -> daily bars with a Close.
FetchTtf = Callable[[str, date, date | None], pd.DataFrame]
#: URL -> body bytes, or None when the server answers 404.
FetchBytes = Callable[[str], bytes | None]

FUELS_FILE = "fuels_daily.parquet"
INDEX_NAME = "observation_date"
GENERAL_CONTRACT = "T3PA"

_USER_AGENT = "energy-price-forecasting-trading/0.1 (public research project)"
_YEAR_FILE = re.compile(r"auction-report-(\d{4})-data\.xlsx?$")
_HEADER_SEARCH_ROWS = 20


class FuelsError(RuntimeError):
    """A fuel price source returned something this module cannot use."""


def yahoo_fetch_ttf(ticker: str, start: date, end: date | None) -> pd.DataFrame:
    """Daily bars from Yahoo Finance via yfinance; ``end`` is exclusive."""
    import yfinance as yf

    frame = yf.Ticker(ticker).history(
        start=start.isoformat(),
        end=None if end is None else end.isoformat(),
        interval="1d",
        auto_adjust=False,
        actions=False,
    )
    if not isinstance(frame, pd.DataFrame):
        raise FuelsError(f"yfinance returned {type(frame).__name__} for {ticker}")
    return frame


def http_fetch_bytes(
    timeout_s: float, attempts: int = 3, backoff_s: float = 1.0
) -> FetchBytes:
    """A fetcher that GETs bytes with retries; a 404 returns None at once."""

    def fetch(url: str) -> bytes | None:
        for attempt in range(1, attempts + 1):
            try:
                response = requests.get(
                    url, timeout=timeout_s, headers={"User-Agent": _USER_AGENT}
                )
                if response.status_code == 404:
                    return None
                response.raise_for_status()
                return response.content
            except requests.RequestException as exc:
                if attempt == attempts:
                    raise FuelsError(
                        f"GET {url} failed after {attempts} attempts"
                    ) from exc
                time.sleep(backoff_s * 2 ** (attempt - 1))
        raise FuelsError(f"GET {url}: no attempts made")

    return fetch


# --------------------------------------------------------------------------- TTF


def ttf_settlements(bars: pd.DataFrame, ticker: str) -> pd.Series:
    """Close per trading date from yfinance-style daily bars, as float."""
    if "Close" not in bars.columns:
        raise FuelsError(f"{ticker}: download has no Close column")
    if not isinstance(bars.index, pd.DatetimeIndex):
        raise FuelsError(f"{ticker}: download is not indexed by timestamps")
    # The date is taken in the index's own timezone: Yahoo stamps each bar at
    # local exchange midnight, so converting to UTC first could shift it.
    days = [ts.date() for ts in bars.index]
    close = pd.to_numeric(bars["Close"], errors="coerce").to_numpy(dtype="float64")
    # Flat bars with zero volume are kept: on this ticker they are mostly
    # settlement-only days with real price moves (183 of 251 bars in 2022).
    series = pd.Series(close, index=pd.Index(days, name=INDEX_NAME), name=GAS_COLUMN)
    series = series.dropna()
    return _one_value_per_date(series, ticker)


def _one_value_per_date(series: pd.Series, source: str) -> pd.Series:
    if series.index.has_duplicates:
        distinct = series.groupby(level=0).nunique()
        conflicts = distinct[distinct > 1]
        if not conflicts.empty:
            raise FuelsError(
                f"{source}: {len(conflicts)} dates have conflicting values, "
                f"first {conflicts.index[0]}"
            )
        series = series.groupby(level=0).first()
    result: pd.Series = series.sort_index()
    return result


def _load_gas(
    settings: Settings, fetch_ttf: FetchTtf, end: date | None, raw_dir: Path
) -> pd.Series:
    ticker = settings.fuels.ttf_ticker
    bars = fetch_ttf(ticker, settings.data.start, end)
    if bars.empty:
        raise FuelsError(f"{ticker}: download is empty")
    safe = re.sub(r"[^\w.-]", "_", ticker)
    _write_bytes_atomic(
        raw_dir / f"ttf_{safe}_daily.csv", bars.to_csv().encode("utf-8")
    )
    series = ttf_settlements(bars, ticker)
    keep = [d >= settings.data.start and (end is None or d < end) for d in series.index]
    kept: pd.Series = series[keep]
    return kept


# --------------------------------------------------------------------------- EUA


def _find_header_row(raw: pd.DataFrame, source: str) -> int:
    for row in range(min(_HEADER_SEARCH_ROWS, len(raw))):
        cells = [str(v).strip() for v in raw.iloc[row].tolist() if pd.notna(v)]
        if "Date" in cells and any(c.startswith("Auction Price") for c in cells):
            return row
    raise FuelsError(f"{source}: no header row with Date and Auction Price")


def _column_starting(header: list[str], prefix: str) -> int | None:
    for position, name in enumerate(header):
        if name.startswith(prefix):
            return position
    return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value.strip():
        text = value.strip()
        iso = re.fullmatch(r"(\d{4}-\d{2}-\d{2})(?:[ T][\d:.]+)?", text)
        if iso:
            try:
                return date.fromisoformat(iso.group(1))
            except ValueError:
                return None
        if re.match(r"\d{4}-", text):
            return None
        parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
        if isinstance(parsed, pd.Timestamp) and not pd.isna(parsed):
            return parsed.date()
    return None


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float | np.integer | np.floating):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip().replace(",", ""))
        except ValueError:
            return None
    else:
        return None
    return None if np.isnan(number) else number


def parse_auction_report(
    content: bytes, source: str = "auction report"
) -> pd.DataFrame:
    """Successful auctions from one EEX yearly workbook (``.xls`` or ``.xlsx``).

    Returns columns ``date`` (datetime.date), ``auction_name``, ``contract``,
    ``price_eur_t`` and ``volume_t`` (NaN when the workbook has no volume),
    oldest first. Rows without a valid date and a numeric price are skipped, and
    so are rows whose Status is anything other than "successful".
    """
    try:
        raw = pd.read_excel(io.BytesIO(content), sheet_name=0, header=None)
    except Exception as exc:  # xlrd and openpyxl raise many unrelated types
        raise FuelsError(f"{source}: not a readable workbook") from exc
    header_row = _find_header_row(raw, source)
    header = [
        "" if pd.isna(v) else str(v).strip() for v in raw.iloc[header_row].tolist()
    ]

    def required(name: str) -> int:
        position = _column_starting(header, name)
        if position is None:
            raise FuelsError(f"{source}: no {name!r} column")
        return position

    date_col = header.index("Date") if "Date" in header else required("Date")
    price_col = required("Auction Price")
    name_col = required("Auction Name")
    contract_col = required("Contract")
    status_col = header.index("Status") if "Status" in header else None
    volume_col = _column_starting(header, "Auction Volume")

    records: list[dict[str, Any]] = []
    for values in raw.iloc[header_row + 1 :].itertuples(index=False, name=None):
        day = _as_date(values[date_col])
        price = _as_float(values[price_col])
        if day is None or price is None:
            continue
        if status_col is not None:
            status = values[status_col]
            if not isinstance(status, str) or status.strip().lower() != "successful":
                continue
        volume = None if volume_col is None else _as_float(values[volume_col])
        records.append(
            {
                "date": day,
                "auction_name": str(values[name_col]).strip(),
                "contract": str(values[contract_col]).strip(),
                "price_eur_t": price,
                "volume_t": np.nan if volume is None else volume,
            }
        )
    columns = ["date", "auction_name", "contract", "price_eur_t", "volume_t"]
    frame = pd.DataFrame.from_records(records, columns=columns)
    frame["price_eur_t"] = frame["price_eur_t"].astype("float64")
    frame["volume_t"] = frame["volume_t"].astype("float64")
    return frame.sort_values("date", kind="stable").reset_index(drop=True)


def daily_carbon_prices(auctions: pd.DataFrame) -> pd.Series:
    """One EUA price per auction date from :func:`parse_auction_report` rows.

    Keeps general allowances (contract ``T3PA``) only. Several general auctions
    on one date are combined by their volume-weighted mean clearing price, or by
    a plain mean when any of that date's volumes is missing.
    """
    general = auctions[auctions["contract"] == GENERAL_CONTRACT]
    prices: dict[date, float] = {}
    for day, group in general.groupby("date", sort=True):
        price = group["price_eur_t"].to_numpy(dtype="float64")
        volume = group["volume_t"].to_numpy(dtype="float64")
        if np.isnan(volume).any() or volume.sum() <= 0:
            prices[_to_date(day)] = float(price.mean())
        else:
            prices[_to_date(day)] = float(np.average(price, weights=volume))
    index = pd.Index(list(prices), name=INDEX_NAME, dtype="object")
    return pd.Series(list(prices.values()), index=index, name=CARBON_COLUMN).astype(
        "float64"
    )


def _to_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise FuelsError(f"expected a date, got {value!r}")


def _cached_fetch(
    fetch_bytes: FetchBytes, url: str, path: Path, *, reuse: bool
) -> bytes | None:
    if reuse and path.exists():
        return path.read_bytes()
    content = fetch_bytes(url)
    if content is not None:
        _write_bytes_atomic(path, content)
    return content


def _archive_workbooks(content: bytes, url: str) -> dict[int, bytes]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile as exc:
        raise FuelsError(f"{url}: not a zip file") from exc
    books: dict[int, bytes] = {}
    with archive:
        for member in archive.namelist():
            match = _YEAR_FILE.search(member)
            if match:
                books[int(match.group(1))] = archive.read(member)
    if not books:
        raise FuelsError(f"{url}: zip holds no yearly auction reports")
    return books


def _load_carbon(
    settings: Settings, fetch_bytes: FetchBytes, end: date | None, raw_dir: Path
) -> pd.Series:
    fuels = settings.fuels
    first_year = settings.data.start.year
    last_year = None if end is None else (end - timedelta(days=1)).year
    cache = raw_dir / "eex"
    frames: list[pd.DataFrame] = []

    if first_year <= fuels.eua_archive_last_year:
        url = fuels.eua_archive_url
        # The archive covers closed years only, so a cached copy stays valid.
        content = _cached_fetch(
            fetch_bytes, url, cache / url.rsplit("/", 1)[-1], reuse=True
        )
        if content is None:
            raise FuelsError(f"{url}: not found")
        books = _archive_workbooks(content, url)
        top = fuels.eua_archive_last_year
        if last_year is not None:
            top = min(top, last_year)
        for year in range(first_year, top + 1):
            if year not in books:
                raise FuelsError(f"{url}: no report for {year}")
            frames.append(parse_auction_report(books[year], f"EEX {year}"))

    year = max(first_year, fuels.eua_archive_last_year + 1)
    while last_year is None or year <= last_year:
        url = fuels.eua_year_url.format(year=year)
        # A year file is complete once the next year has begun; the current and
        # previous year are downloaded again in case late auctions were added.
        closed = last_year is not None and year < last_year - 1
        content = _cached_fetch(
            fetch_bytes, url, cache / url.rsplit("/", 1)[-1], reuse=closed
        )
        if content is None:
            if last_year is None or year == last_year:
                break  # the newest year may have no report yet
            raise FuelsError(f"{url}: not found although {last_year} is requested")
        frames.append(parse_auction_report(content, f"EEX {year}"))
        year += 1

    if not frames:
        return pd.Series(
            [], index=pd.Index([], name=INDEX_NAME), name=CARBON_COLUMN, dtype="float64"
        )
    daily = daily_carbon_prices(pd.concat(frames, ignore_index=True))
    keep = [end is None or d < end for d in daily.index]
    kept: pd.Series = daily[keep]
    return kept


# ------------------------------------------------------------------------ public


def load_daily_fuel_prices(
    settings: Settings,
    *,
    fetch_ttf: FetchTtf | None = None,
    fetch_bytes: FetchBytes | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Gas settlements and EUA auction prices per observation date.

    ``end`` is exclusive: observations dated ``end`` or later are dropped, so a
    live intraday bar is never mistaken for a settlement. ``None`` keeps
    everything published. Gas starts at ``data.start``; carbon covers whole
    years from ``data.start``'s year. Raw downloads are cached under
    ``data.raw_path / "fuels"``.
    """
    raw_dir = settings.data.raw_path / "fuels"
    gas = _load_gas(settings, fetch_ttf or yahoo_fetch_ttf, end, raw_dir)
    carbon = _load_carbon(
        settings,
        fetch_bytes or http_fetch_bytes(settings.fuels.timeout_s),
        end,
        raw_dir,
    )
    frame = pd.concat([gas, carbon], axis=1).sort_index()
    frame = frame[[GAS_COLUMN, CARBON_COLUMN]].astype("float64")
    frame.index = pd.Index(list(frame.index), name=INDEX_NAME, dtype="object")
    return frame


def last_known_before(
    daily: pd.DataFrame, delivery_days: Sequence[date]
) -> pd.DataFrame:
    """For each delivery day, each column's newest value observed strictly before it.

    ``daily`` is indexed by sorted, unique ``datetime.date`` values. Columns are
    filled independently, so a carbon auction on a gas holiday does not hide the
    last gas settlement. A value observed on the delivery day itself is never
    used, and days with nothing earlier get NaN. The row for local day D then
    holds prices from D-1 or earlier, which were published before 11:40 on D.
    """
    if not daily.index.is_unique or not daily.index.is_monotonic_increasing:
        raise ValueError("daily must be indexed by sorted, unique dates")
    wanted = np.array([np.datetime64(_to_date(d), "D") for d in delivery_days])
    result: dict[str, np.ndarray[Any, np.dtype[np.float64]]] = {}
    for column in daily.columns:
        observed = daily[column].dropna()
        stamps = np.array(
            [np.datetime64(_to_date(d), "D") for d in observed.index],
            dtype="datetime64[D]",
        )
        values = observed.to_numpy(dtype="float64")
        position = np.searchsorted(stamps, wanted, side="left") - 1
        filled = np.full(len(wanted), np.nan)
        found = position >= 0
        filled[found] = values[position[found]]
        result[str(column)] = filled
    index = pd.Index(list(delivery_days), name="delivery_day", dtype="object")
    return pd.DataFrame(result, index=index, columns=list(daily.columns))


# --------------------------------------------------------------------------- CLI


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_bytes(content)
    tmp.replace(path)


def _longest_business_day_gap(days: Sequence[date]) -> tuple[int, date | None]:
    """Most business days strictly between consecutive observations, and the
    observation that ended that gap."""
    if len(days) < 2:
        return 0, None
    stamps = np.array([np.datetime64(d, "D") for d in days])
    between = np.busday_count(stamps[:-1] + 1, stamps[1:])
    worst = int(np.argmax(between))
    return int(between[worst]), days[worst + 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download daily fuel prices.")
    parser.add_argument("--config", type=Path, default=None, help="settings YAML path")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    # Today's bars may be live prices rather than settlements; end is exclusive.
    frame = load_daily_fuel_prices(settings, end=date.today())
    path = settings.data.processed_path / FUELS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    frame.to_parquet(tmp)
    tmp.replace(path)

    print(f"rows            {len(frame)}")
    for column in (GAS_COLUMN, CARBON_COLUMN):
        days = [_to_date(d) for d in frame[column].dropna().index]
        gap, gap_end = _longest_business_day_gap(days)
        first = days[0] if days else None
        last = days[-1] if days else None
        print(f"{column}")
        print(f"  observations  {len(days)}")
        print(f"  first .. last {first} .. {last}")
        print(f"  longest gap   {gap} business days, ending {gap_end}")
    print(f"wrote           {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
