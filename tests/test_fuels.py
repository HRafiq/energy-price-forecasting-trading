from __future__ import annotations

import io
import math
import zipfile
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook  # type: ignore[import-untyped]

from src.config import CARBON_COLUMN, GAS_COLUMN, Settings
from src.ingest.fuels import (
    FuelsError,
    daily_carbon_prices,
    last_known_before,
    load_daily_fuel_prices,
    parse_auction_report,
    ttf_settlements,
)

HEADER = [
    None,
    "Date",
    "Time",
    "Auction Name",
    "Contract",
    "Status",
    "Auction Price €/tCO2",
    "Minimum Bid €/tCO2",
    "Auction Volume tCO2",
]

Row = tuple[object, str, str, str, object, object]


def workbook(rows: list[Row], *, with_status: bool = True) -> bytes:
    """An EEX-style report: header on row 6 (index 5), data newest first.

    Each row is (date, auction name, contract, status, price, volume).
    """
    book = Workbook()
    sheet = book.active
    sheet.title = "Primary Market Auction"
    sheet.append([])
    sheet.append([None, "Public"])
    sheet.append([None, "More information"])
    sheet.append([None, "EEX Emissions market / Primary Market Auction"])
    sheet.append([None, "References", None, None, None, "Prices", "Volumes"])
    header = HEADER if with_status else [h for h in HEADER if h != "Status"]
    sheet.append(header)
    for day, name, contract, status, price, volume in rows:
        values: list[object] = [None, day, None, name, contract]
        if with_status:
            values.append(status)
        values += [price, 1.0, volume]
        sheet.append(values)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_parse_keeps_successful_priced_rows_oldest_first() -> None:
    content = workbook(
        [
            (
                datetime(2024, 3, 5),
                "Auction 4. Period DE",
                "T3PA",
                "successful",
                60.5,
                1000,
            ),
            (
                datetime(2024, 3, 4),
                "Auction 4. Period CAP3 EU",
                "T3PA",
                "cancelled",
                None,
                900,
            ),
            (
                datetime(2024, 3, 4),
                "Auction 4. Period CAP3 PL",
                "T3PA",
                "unsuccessful",
                59.0,
                800,
            ),
            (
                datetime(2024, 3, 1),
                "Auction 4. Period CAP3 EU",
                "T3PA",
                "successful",
                58.25,
                700,
            ),
            ("Total", "", "", "", "n/a", None),
        ]
    )

    auctions = parse_auction_report(content)

    assert auctions["date"].tolist() == [date(2024, 3, 1), date(2024, 3, 5)]
    assert auctions["price_eur_t"].tolist() == [58.25, 60.5]
    assert auctions["volume_t"].tolist() == [700.0, 1000.0]


def test_parse_accepts_the_layout_without_a_status_column() -> None:
    content = workbook(
        [(datetime(2019, 1, 7), "Auction 3. Period CAP2-EU", "T3PA", "", 22.0, 5)],
        with_status=False,
    )

    auctions = parse_auction_report(content)

    assert auctions["date"].tolist() == [date(2019, 1, 7)]
    assert auctions["price_eur_t"].tolist() == [22.0]


def test_parse_rejects_a_workbook_without_a_price_header() -> None:
    book = Workbook()
    book.active.append(["Date", "Something else"])
    buffer = io.BytesIO()
    book.save(buffer)
    with pytest.raises(FuelsError):
        parse_auction_report(buffer.getvalue())


def test_daily_carbon_drops_aviation_and_weights_same_day_auctions() -> None:
    content = workbook(
        [
            (
                datetime(2024, 3, 6),
                "EUAA Auction CAP3 EU",
                "EAA3",
                "successful",
                70.0,
                500,
            ),
            (
                datetime(2024, 3, 5),
                "Auction 4. Period DE",
                "T3PA",
                "successful",
                61.0,
                3000,
            ),
            (
                datetime(2024, 3, 5),
                "Auction 4. Period CAP3 PL",
                "T3PA",
                "successful",
                65.0,
                1000,
            ),
            (
                datetime(2024, 3, 5),
                "EUAA Auction CAP3 EU",
                "EAA3",
                "successful",
                99.0,
                500,
            ),
        ]
    )

    daily = daily_carbon_prices(parse_auction_report(content))

    assert daily.name == CARBON_COLUMN
    assert daily.index.tolist() == [date(2024, 3, 5)]
    assert daily.iloc[0] == pytest.approx((61.0 * 3000 + 65.0 * 1000) / 4000)


def test_daily_carbon_uses_a_plain_mean_when_a_volume_is_missing() -> None:
    auctions = pd.DataFrame(
        {
            "date": [date(2024, 3, 5), date(2024, 3, 5)],
            "auction_name": ["a", "b"],
            "contract": ["T3PA", "T3PA"],
            "price_eur_t": [60.0, 64.0],
            "volume_t": [100.0, np.nan],
        }
    )
    assert daily_carbon_prices(auctions).iloc[0] == pytest.approx(62.0)


def ttf_bars(days: list[str], closes: list[float]) -> pd.DataFrame:
    index = pd.DatetimeIndex(pd.to_datetime(days)).tz_localize("America/New_York")
    return pd.DataFrame({"Open": closes, "Close": closes}, index=index)


def test_ttf_settlements_use_the_exchange_date_and_drop_missing_closes() -> None:
    bars = ttf_bars(["2025-01-02", "2025-01-03", "2025-01-06"], [50.0, math.nan, 52.0])

    series = ttf_settlements(bars, "TTF=F")

    assert series.index.tolist() == [date(2025, 1, 2), date(2025, 1, 6)]
    assert series.tolist() == [50.0, 52.0]


def test_ttf_settlements_reject_conflicting_duplicate_dates() -> None:
    with pytest.raises(FuelsError):
        ttf_settlements(ttf_bars(["2025-01-02", "2025-01-02"], [1.0, 2.0]), "TTF=F")


class FakeSources:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ttf_calls: list[tuple[str, date, date | None]] = []
        self.url_calls: list[str] = []
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(
                "reports/emission-spot-primary-market-auction-report-2025-data.xlsx",
                workbook(
                    [
                        (
                            datetime(2025, 12, 15),
                            "Auction 4. Period CAP3 EU",
                            "T3PA",
                            "successful",
                            80.0,
                            10,
                        ),
                    ]
                ),
            )
        self.bodies: dict[str, bytes] = {
            settings.fuels.eua_archive_url: archive.getvalue(),
            settings.fuels.eua_year_url.format(year=2026): workbook(
                [
                    (
                        datetime(2026, 1, 9),
                        "Auction 4. Period DE",
                        "T3PA",
                        "successful",
                        90.0,
                        10,
                    ),
                    (
                        datetime(2026, 1, 7),
                        "Auction 4. Period CAP3 EU",
                        "T3PA",
                        "successful",
                        85.0,
                        10,
                    ),
                ]
            ),
        }

    def fetch_ttf(self, ticker: str, start: date, end: date | None) -> pd.DataFrame:
        self.ttf_calls.append((ticker, start, end))
        return ttf_bars(
            ["2025-12-31", "2026-01-07", "2026-01-08", "2026-01-09"],
            [30.0, 31.0, 32.0, 33.0],
        )

    def fetch_bytes(self, url: str) -> bytes | None:
        self.url_calls.append(url)
        return self.bodies.get(url)


@pytest.fixture
def fuel_settings(settings: Settings, tmp_path: Path) -> Settings:
    data = settings.data.model_copy(
        update={"raw_dir": tmp_path / "raw", "start": date(2025, 1, 1)}
    )
    fuels = settings.fuels.model_copy(update={"eua_archive_last_year": 2025})
    return settings.model_copy(update={"data": data, "fuels": fuels})


def test_load_daily_fuel_prices_joins_both_series_with_nan_gaps(
    fuel_settings: Settings,
) -> None:
    fake = FakeSources(fuel_settings)

    frame = load_daily_fuel_prices(
        fuel_settings,
        fetch_ttf=fake.fetch_ttf,
        fetch_bytes=fake.fetch_bytes,
        end=date(2026, 1, 9),
    )

    assert frame.index.name == "observation_date"
    assert list(frame.columns) == [GAS_COLUMN, CARBON_COLUMN]
    assert frame.index.tolist() == [
        date(2025, 12, 15),
        date(2025, 12, 31),
        date(2026, 1, 7),
        date(2026, 1, 8),
    ]
    assert all(type(d) is date for d in frame.index)
    assert frame[GAS_COLUMN].tolist()[1:] == [30.0, 31.0, 32.0]
    assert np.isnan(frame[GAS_COLUMN].iloc[0])
    assert frame[CARBON_COLUMN].dropna().tolist() == [80.0, 85.0]
    assert frame.dtypes.tolist() == [np.dtype("float64")] * 2
    assert fake.ttf_calls == [("TTF=F", date(2025, 1, 1), date(2026, 1, 9))]
    raw = fuel_settings.data.raw_path / "fuels"
    assert (raw / "ttf_TTF_F_daily.csv").exists()
    # The end date is exclusive, so the next year's report is never requested.
    assert fuel_settings.fuels.eua_year_url.format(year=2027) not in fake.url_calls


def test_archive_is_cached_but_the_current_year_is_downloaded_again(
    fuel_settings: Settings,
) -> None:
    fake = FakeSources(fuel_settings)

    def load() -> None:
        load_daily_fuel_prices(
            fuel_settings,
            fetch_ttf=fake.fetch_ttf,
            fetch_bytes=fake.fetch_bytes,
            end=date(2026, 2, 1),
        )

    load()
    fake.url_calls.clear()

    load()

    assert fake.url_calls == [fuel_settings.fuels.eua_year_url.format(year=2026)]


def test_without_end_year_reports_are_read_until_one_is_missing(
    fuel_settings: Settings,
) -> None:
    fake = FakeSources(fuel_settings)

    frame = load_daily_fuel_prices(
        fuel_settings, fetch_ttf=fake.fetch_ttf, fetch_bytes=fake.fetch_bytes
    )

    assert frame[CARBON_COLUMN].dropna().tolist() == [80.0, 85.0, 90.0]
    assert fake.url_calls[-1] == fuel_settings.fuels.eua_year_url.format(year=2027)


def test_a_missing_past_year_report_is_an_error(fuel_settings: Settings) -> None:
    fake = FakeSources(fuel_settings)
    del fake.bodies[fuel_settings.fuels.eua_year_url.format(year=2026)]

    with pytest.raises(FuelsError):
        load_daily_fuel_prices(
            fuel_settings,
            fetch_ttf=fake.fetch_ttf,
            fetch_bytes=fake.fetch_bytes,
            end=date(2027, 3, 1),
        )


def daily_frame() -> pd.DataFrame:
    index = pd.Index(
        [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 5)],
        name="observation_date",
    )
    return pd.DataFrame(
        {GAS_COLUMN: [30.0, np.nan, 32.0], CARBON_COLUMN: [np.nan, 70.0, np.nan]},
        index=index,
    )


def test_last_known_before_excludes_same_day_observations() -> None:
    known = last_known_before(daily_frame(), [date(2024, 1, 3), date(2024, 1, 5)])

    # The auction on the 3rd does not count for the 3rd, and the gas settlement
    # on the 5th does not count for the 5th.
    assert known[GAS_COLUMN].tolist() == [30.0, 30.0]
    assert np.isnan(known[CARBON_COLUMN].iloc[0])
    assert known[CARBON_COLUMN].iloc[1] == 70.0


def test_last_known_before_fills_each_column_independently() -> None:
    known = last_known_before(daily_frame(), [date(2024, 1, 4), date(2024, 1, 8)])

    assert known[GAS_COLUMN].tolist() == [30.0, 32.0]
    assert known[CARBON_COLUMN].tolist() == [70.0, 70.0]
    assert known.index.name == "delivery_day"
    assert list(known.columns) == [GAS_COLUMN, CARBON_COLUMN]


def test_last_known_before_is_nan_before_the_first_observation() -> None:
    known = last_known_before(daily_frame(), [date(2023, 12, 31), date(2024, 1, 2)])

    assert known.isna().all().all()


def test_last_known_before_rejects_unsorted_input() -> None:
    with pytest.raises(ValueError):
        last_known_before(daily_frame().iloc[::-1], [date(2024, 1, 4)])
