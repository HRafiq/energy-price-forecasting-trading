from __future__ import annotations

import io
import traceback
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests

from src.ingest.entsoe import (
    TOKEN_ENV,
    EntsoeClient,
    EntsoeError,
    EntsoeHttpError,
    EntsoeOptions,
    NoMatchingDataError,
    RateLimiter,
    build_price_check_report,
    cache_path,
    compare_prices,
    day_ahead_table,
    imbalance_table,
    load_token,
    month_chunks,
    parse_day_ahead_document,
    parse_env_text,
    parse_imbalance_document,
    quarter_hour_grid,
    redact,
    retry_delay,
)

FAKE_TOKEN = "fake-token-0123456789abcdef"
PUB_NS = "urn:iec62325.351:tc57wg16:451-3:publicationdocument:7:3"
BAL_NS = "urn:iec62325.351:tc57wg16:451-6:balancingdocument:4:4"
ACK_NS = "urn:iec62325.351:tc57wg16:451-1:acknowledgementdocument:7:0"


@dataclass(frozen=True)
class Series:
    position: int
    resolution: str
    start: str
    end: str
    points: dict[int, float]
    curve: str = "A03"


def publication_xml(series: list[Series]) -> bytes:
    blocks = []
    for s in series:
        points = "".join(
            f"<Point><position>{p}</position><price.amount>{v}</price.amount></Point>"
            for p, v in sorted(s.points.items())
        )
        blocks.append(
            "<TimeSeries><mRID>1</mRID><businessType>A62</businessType>"
            "<currency_Unit.name>EUR</currency_Unit.name>"
            "<price_Measure_Unit.name>MWH</price_Measure_Unit.name>"
            "<classificationSequence_AttributeInstanceComponent.position>"
            f"{s.position}</classificationSequence_AttributeInstanceComponent.position>"
            f"<curveType>{s.curve}</curveType><Period><timeInterval>"
            f"<start>{s.start}</start><end>{s.end}</end></timeInterval>"
            f"<resolution>{s.resolution}</resolution>{points}</Period></TimeSeries>"
        )
    return (
        f'<?xml version="1.0" encoding="utf-8"?><Publication_MarketDocument '
        f'xmlns="{PUB_NS}"><type>A44</type>{"".join(blocks)}'
        "</Publication_MarketDocument>"
    ).encode()


def imbalance_xml(start: str, end: str, points: dict[str, dict[int, float]]) -> bytes:
    blocks = []
    for category, values in points.items():
        body = "".join(
            f"<Point><position>{p}</position>"
            f"<imbalance_Price.amount>{v}</imbalance_Price.amount>"
            f"<imbalance_Price.category>{category}</imbalance_Price.category></Point>"
            for p, v in sorted(values.items())
        )
        blocks.append(
            "<TimeSeries><businessType>A19</businessType>"
            "<currency_Unit.name>EUR</currency_Unit.name>"
            "<price_Measure_Unit.name>MWH</price_Measure_Unit.name>"
            "<curveType>A03</curveType><Period><timeInterval>"
            f"<start>{start}</start><end>{end}</end></timeInterval>"
            f"<resolution>PT15M</resolution>{body}</Period></TimeSeries>"
        )
    return (
        f'<Balancing_MarketDocument xmlns="{BAL_NS}"><type>A85</type>'
        f"{''.join(blocks)}</Balancing_MarketDocument>"
    ).encode()


def acknowledgement_xml(text: str, code: str = "999") -> bytes:
    return (
        f'<Acknowledgement_MarketDocument xmlns="{ACK_NS}"><Reason><code>{code}</code>'
        f"<text>{text}</text></Reason></Acknowledgement_MarketDocument>"
    ).encode()


@dataclass
class FakeResponse:
    status_code: int
    content: bytes
    headers: dict[str, str] = field(default_factory=dict)


class FakeTransport:
    """Answers from a callable and records every parameter set it was sent."""

    def __init__(
        self, answer: Callable[[Mapping[str, str]], FakeResponse | Exception]
    ) -> None:
        self.answer = answer
        self.calls: list[dict[str, str]] = []

    def __call__(
        self, url: str, params: Mapping[str, str], timeout_s: float
    ) -> FakeResponse:
        self.calls.append(dict(params))
        result = self.answer(params)
        if isinstance(result, Exception):
            raise result
        return result


class Sequence:
    def __init__(self, items: list[FakeResponse | Exception]) -> None:
        self.items = items

    def __call__(self, params: Mapping[str, str]) -> FakeResponse | Exception:
        return self.items.pop(0)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(
    tmp_path: Path,
    answer: Callable[[Mapping[str, str]], FakeResponse | Exception],
    now: str = "2026-09-14 12:00",
    **options: float,
) -> tuple[EntsoeClient, FakeTransport, FakeClock]:
    transport = FakeTransport(answer)
    clock = FakeClock()
    client = EntsoeClient(
        FAKE_TOKEN,
        tmp_path / "cache",
        options=EntsoeOptions(**options),  # type: ignore[arg-type]
        transport=transport,
        clock=clock,
        sleep=clock.sleep,
        now_utc=lambda: pd.Timestamp(now, tz="UTC"),
    )
    return client, transport, clock


# --- parsing ---------------------------------------------------------------


def test_a03_forward_fills_omitted_points_up_to_the_period_end() -> None:
    doc = publication_xml(
        [Series(1, "PT15M", "2026-05-30T22:00Z", "2026-05-30T23:15Z", {1: 10, 3: 30})]
    )
    frame = parse_day_ahead_document(doc)
    assert frame["price_eur_mwh"].tolist() == [10.0, 10.0, 30.0, 30.0, 30.0]
    assert frame["timestamp_utc"].iloc[-1] == pd.Timestamp("2026-05-30 23:00", tz="UTC")
    assert str(frame["timestamp_utc"].dt.tz) == "UTC"


def test_a01_does_not_fill_missing_points() -> None:
    doc = publication_xml(
        [
            Series(
                1,
                "PT15M",
                "2026-05-30T22:00Z",
                "2026-05-30T23:00Z",
                {1: 1, 3: 3},
                "A01",
            )
        ]
    )
    assert parse_day_ahead_document(doc)["price_eur_mwh"].tolist() == [1.0, 3.0]


def test_two_positions_and_resolutions_become_quarter_hour_columns() -> None:
    start, end = "2024-05-31T22:00Z", "2024-06-01T22:00Z"
    doc = publication_xml(
        [
            Series(2, "PT15M", start, end, {i: 100.0 + i for i in range(1, 97)}),
            Series(1, "PT60M", start, end, {1: 50.0, 2: 51.0, 24: 73.0}),
        ]
    )
    long = parse_day_ahead_document(doc)
    assert set(long["position"]) == {1, 2}
    grid = quarter_hour_grid(date(2024, 6, 1), date(2024, 6, 1))
    table = day_ahead_table(long, grid)

    assert len(table) == 96
    assert table["price_position_1"].iloc[:8].tolist() == [50.0] * 4 + [51.0] * 4
    assert table["price_position_1"].iloc[-1] == 73.0
    assert table["price_position_1"].iloc[12] == 51.0  # A03 carried forward
    assert table["price_position_2"].iloc[5] == 106.0
    assert set(table["resolution_minutes_position_1"]) == {60}
    assert set(table["resolution_minutes_position_2"]) == {15}


@pytest.mark.parametrize(
    ("day", "start", "end", "resolution", "periods", "quarter_hours"),
    [
        (date(2024, 3, 31), "2024-03-30T23:00Z", "2024-03-31T22:00Z", "PT60M", 23, 92),
        (
            date(2024, 10, 27),
            "2024-10-26T22:00Z",
            "2024-10-27T23:00Z",
            "PT15M",
            100,
            100,
        ),
    ],
)
def test_dst_days_keep_their_true_length(
    day: date, start: str, end: str, resolution: str, periods: int, quarter_hours: int
) -> None:
    doc = publication_xml([Series(1, resolution, start, end, {1: 5.0, periods: 9.0})])
    long = parse_day_ahead_document(doc)
    assert len(long) == periods
    table = day_ahead_table(long, quarter_hour_grid(day, day))
    assert len(table) == quarter_hours
    assert table["price_position_1"].notna().all()
    assert table["price_position_1"].iloc[-1] == 9.0
    assert table.index[-1] == pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=15)


def test_point_outside_the_period_is_rejected() -> None:
    doc = publication_xml(
        [Series(1, "PT60M", "2026-05-30T22:00Z", "2026-05-30T23:00Z", {1: 1, 2: 2})]
    )
    with pytest.raises(EntsoeError, match="outside"):
        parse_day_ahead_document(doc)


def test_acknowledgement_raises_a_typed_no_data_error() -> None:
    doc = acknowledgement_xml("No matching data found for Data item X.")
    with pytest.raises(NoMatchingDataError) as info:
        parse_day_ahead_document(doc)
    assert info.value.acknowledgement.code == "999"
    assert info.value.acknowledgement.no_matching_data


def test_other_acknowledgements_are_not_no_data() -> None:
    doc = acknowledgement_xml("The amount of requested data exceeds allowed limit.")
    with pytest.raises(EntsoeError) as info:
        parse_day_ahead_document(doc)
    assert not isinstance(info.value, NoMatchingDataError)


def test_imbalance_categories_are_parsed_and_pivoted() -> None:
    start, end = "2026-05-30T22:00Z", "2026-05-30T23:00Z"
    doc = imbalance_xml(start, end, {"A04": {1: 10.0, 3: 30.0}, "A05": {1: 11.0}})
    long = parse_imbalance_document(doc)
    assert set(long["category"]) == {"A04", "A05"}
    grid = pd.date_range(
        start, end, freq="15min", inclusive="left", name="timestamp_utc"
    )
    table = imbalance_table(long, grid)
    assert table["imbalance_price_excess_eur_mwh"].tolist() == [10.0, 10.0, 30.0, 30.0]
    assert table["imbalance_price_insufficient_eur_mwh"].tolist() == [11.0] * 4


def test_conflicting_duplicates_raise() -> None:
    start, end = "2026-05-30T22:00Z", "2026-05-30T23:00Z"
    a = parse_day_ahead_document(
        publication_xml([Series(1, "PT15M", start, end, {1: 1})])
    )
    b = parse_day_ahead_document(
        publication_xml([Series(1, "PT15M", start, end, {1: 2})])
    )
    grid = pd.date_range(start, end, freq="15min", inclusive="left")
    with pytest.raises(EntsoeError, match="conflicting"):
        day_ahead_table(pd.concat([a, b], ignore_index=True), grid)


# --- token handling ----------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        f"{TOKEN_ENV}={FAKE_TOKEN}\n",
        f'  {TOKEN_ENV} = "{FAKE_TOKEN}"  \n',
        f"# comment\nOTHER=1\nexport {TOKEN_ENV}='{FAKE_TOKEN}'\n",
        f"{TOKEN_ENV}={FAKE_TOKEN} # trailing comment\n",
    ],
)
def test_env_file_parsing_handles_quotes_and_whitespace(
    tmp_path: Path, text: str
) -> None:
    assert parse_env_text(text)[TOKEN_ENV] == FAKE_TOKEN
    path = tmp_path / ".env"
    path.write_text(text)
    assert load_token(environ={}, env_path=path) == FAKE_TOKEN


def test_environment_variable_wins_over_env_file(tmp_path: Path) -> None:
    path = tmp_path / ".env"
    path.write_text(f"{TOKEN_ENV}=from-file\n")
    assert load_token(environ={TOKEN_ENV: " from-env "}, env_path=path) == "from-env"


def test_missing_token_raises(tmp_path: Path) -> None:
    with pytest.raises(EntsoeError, match=TOKEN_ENV):
        load_token(environ={}, env_path=tmp_path / "missing.env")


def test_redact_handles_url_encoding() -> None:
    secret = "a b/c+d"
    text = "url?securityToken=a%20b%2Fc%2Bd and a b/c+d"
    assert secret not in redact(text, secret)
    assert "a%20b%2Fc%2Bd" not in redact(text, secret)


def _exception_text(exc: BaseException) -> str:
    return str(exc) + repr(exc) + "".join(traceback.format_exception(exc))


def test_connection_errors_never_show_the_token(tmp_path: Path) -> None:
    url = f"https://example.invalid/api?securityToken={FAKE_TOKEN}&documentType=A44"
    client, transport, _ = make_client(
        tmp_path,
        lambda _p: requests.ConnectionError(f"Max retries exceeded with url: {url}"),
        attempts=2,
    )
    with pytest.raises(EntsoeHttpError) as info:
        client.day_ahead_prices(date(2026, 5, 31), date(2026, 5, 31))
    assert FAKE_TOKEN not in _exception_text(info.value)
    assert "<token>" in str(info.value)
    assert info.value.__cause__ is None and info.value.__context__ is None
    assert len(transport.calls) == 2
    assert all(call["securityToken"] == FAKE_TOKEN for call in transport.calls)


def test_http_error_bodies_are_redacted(tmp_path: Path) -> None:
    body = f"<html>Unauthorized token {FAKE_TOKEN}</html>".encode()
    client, _, _ = make_client(tmp_path, lambda _p: FakeResponse(401, body))
    with pytest.raises(EntsoeHttpError) as info:
        client.day_ahead_prices(date(2026, 5, 31), date(2026, 5, 31))
    assert info.value.status == 401
    assert FAKE_TOKEN not in _exception_text(info.value)
    assert FAKE_TOKEN not in repr(client)


def test_acknowledgement_with_http_400_is_typed_and_redacted(tmp_path: Path) -> None:
    body = acknowledgement_xml(f"Invalid parameter securityToken={FAKE_TOKEN}", "999")
    client, transport, _ = make_client(tmp_path, lambda _p: FakeResponse(400, body))
    with pytest.raises(EntsoeError) as info:
        client.day_ahead_prices(date(2026, 5, 31), date(2026, 5, 31))
    assert FAKE_TOKEN not in _exception_text(info.value)
    assert len(transport.calls) == 1


# --- cache -----------------------------------------------------------------


def _day_doc(params: Mapping[str, str]) -> FakeResponse:
    start = pd.Timestamp(params["periodStart"], tz="UTC")
    end = pd.Timestamp(params["periodEnd"], tz="UTC")
    hours = int((end - start) / pd.Timedelta(hours=1))
    doc = publication_xml(
        [
            Series(
                1,
                "PT60M",
                start.strftime("%Y-%m-%dT%H:%MZ"),
                end.strftime("%Y-%m-%dT%H:%MZ"),
                {1: 42.0, hours: 43.0},
            )
        ]
    )
    return FakeResponse(200, doc)


def test_cache_paths_and_files_never_contain_the_token(tmp_path: Path) -> None:
    params = {
        "documentType": "A44",
        "in_Domain": "10Y1001A1001A82H",
        "out_Domain": "10Y1001A1001A82H",
        "periodStart": "202605302200",
        "periodEnd": "202605312200",
        "securityToken": FAKE_TOKEN,
    }
    path = cache_path(tmp_path, params)
    assert FAKE_TOKEN not in str(path)
    assert path.parts[-3:] == (
        "A44",
        "10Y1001A1001A82H_10Y1001A1001A82H",
        "202605302200_202605312200.raw",
    )

    client, _, _ = make_client(tmp_path, _day_doc)
    client.day_ahead_prices(date(2026, 4, 1), date(2026, 5, 31))
    files = [p for p in (tmp_path / "cache").rglob("*") if p.is_file()]
    assert len(files) == 4  # two months, payload plus metadata each
    for file in files:
        assert FAKE_TOKEN not in str(file)
        assert FAKE_TOKEN.encode() not in file.read_bytes()


def test_only_settled_months_are_served_from_cache(tmp_path: Path) -> None:
    client, transport, _ = make_client(tmp_path, _day_doc, now="2026-09-08 12:00")
    first = client.day_ahead_prices(date(2026, 8, 1), date(2026, 9, 2))
    assert len(transport.calls) == 2

    again, transport2, _ = make_client(tmp_path, _day_doc, now="2026-09-08 13:00")
    second = again.day_ahead_prices(date(2026, 8, 1), date(2026, 9, 2))
    # August ended more than seven days before the first download; September did not.
    assert [c["periodStart"] for c in transport2.calls] == ["202608312200"]
    assert again.cache_hits == 1
    pd.testing.assert_frame_equal(first, second)


def test_months_without_data_are_skipped(tmp_path: Path) -> None:
    def answer(params: Mapping[str, str]) -> FakeResponse:
        # April in local time starts at 2026-03-31 22:00 UTC.
        if params["periodStart"].startswith("202603"):
            return _day_doc(params)
        return FakeResponse(200, acknowledgement_xml("No matching data found for X"))

    client, _, _ = make_client(tmp_path, answer)
    long = client.day_ahead_prices(date(2026, 4, 1), date(2026, 5, 31))
    assert long["timestamp_utc"].min() == pd.Timestamp("2026-03-31 22:00", tz="UTC")
    assert long["timestamp_utc"].max() < pd.Timestamp("2026-04-30 22:00", tz="UTC")
    assert len(client.empty_chunks) == 1


def test_zipped_imbalance_payload_is_unpacked(tmp_path: Path) -> None:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(
            "a.xml",
            imbalance_xml("2026-05-30T22:00Z", "2026-05-30T22:30Z", {"A04": {1: 1}}),
        )
        archive.writestr(
            "b.xml",
            imbalance_xml("2026-05-30T22:30Z", "2026-05-30T23:00Z", {"A04": {1: 2}}),
        )
    client, transport, _ = make_client(
        tmp_path, lambda _p: FakeResponse(200, buffer.getvalue())
    )
    long = client.imbalance_prices(date(2026, 5, 31), date(2026, 5, 31))
    assert long["price_eur_mwh"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert transport.calls[0]["controlArea_Domain"] == "10Y1001A1001A83F"


# --- rate limiting and retries ------------------------------------------------


def test_rate_limiter_spaces_request_starts() -> None:
    clock = FakeClock()
    limiter = RateLimiter(0.5, clock, clock.sleep)
    assert limiter.wait() == 0.0
    clock.now += 0.2
    assert limiter.wait() == pytest.approx(0.3)
    clock.now += 1.0
    assert limiter.wait() == 0.0
    assert clock.sleeps == [pytest.approx(0.3)]


def test_retry_delay_backs_off_and_honours_retry_after() -> None:
    assert [retry_delay(a, 2.0, 60.0) for a in (1, 2, 3, 6)] == [2.0, 4.0, 8.0, 60.0]
    assert retry_delay(1, 2.0, 60.0, "7") == 7.0
    assert retry_delay(1, 2.0, 60.0, "999") == 60.0
    assert retry_delay(2, 2.0, 60.0, "Wed, 21 Oct 2015 07:28:00 GMT") == 4.0


def test_client_retries_429_and_5xx_then_succeeds(tmp_path: Path) -> None:
    ok = _day_doc({"periodStart": "202605302200", "periodEnd": "202605312200"})
    answers = Sequence(
        [FakeResponse(503, b""), FakeResponse(429, b"", {"Retry-After": "7"}), ok]
    )
    client, transport, clock = make_client(tmp_path, answers)
    long = client.day_ahead_prices(date(2026, 5, 31), date(2026, 5, 31))
    assert len(long) == 24
    assert client.requests_made == 3
    # Backoff sleeps of 2 s then 7 s; no extra rate-limit sleep once 0.5 s passed.
    assert clock.sleeps == [2.0, 7.0]


def test_client_gives_up_after_the_configured_attempts(tmp_path: Path) -> None:
    client, transport, clock = make_client(
        tmp_path, lambda _p: FakeResponse(500, b"oops"), attempts=3
    )
    with pytest.raises(EntsoeHttpError, match="after 3 attempts"):
        client.day_ahead_prices(date(2026, 5, 31), date(2026, 5, 31))
    assert len(transport.calls) == 3
    assert clock.sleeps == [2.0, 4.0]


def test_client_errors_are_not_retried(tmp_path: Path) -> None:
    client, transport, _ = make_client(tmp_path, lambda _p: FakeResponse(404, b"no"))
    with pytest.raises(EntsoeHttpError):
        client.day_ahead_prices(date(2026, 5, 31), date(2026, 5, 31))
    assert len(transport.calls) == 1


def test_consecutive_requests_respect_the_rate_limit(tmp_path: Path) -> None:
    client, transport, clock = make_client(tmp_path, _day_doc)
    client.day_ahead_prices(date(2026, 3, 1), date(2026, 5, 31))
    assert len(transport.calls) == 3
    assert clock.sleeps == [0.5, 0.5]


# --- chunks and report -----------------------------------------------------------


def test_month_chunks_follow_local_midnights() -> None:
    chunks = month_chunks(date(2024, 10, 15), date(2024, 11, 2))
    assert chunks == [
        (
            pd.Timestamp("2024-10-14 22:00", tz="UTC"),
            pd.Timestamp("2024-10-31 23:00", tz="UTC"),
        ),
        (
            pd.Timestamp("2024-10-31 23:00", tz="UTC"),
            pd.Timestamp("2024-11-02 23:00", tz="UTC"),
        ),
    ]
    with pytest.raises(ValueError):
        month_chunks(date(2024, 1, 2), date(2024, 1, 1))


def test_compare_prices_counts_mismatch_days() -> None:
    grid = quarter_hour_grid(date(2025, 9, 30), date(2025, 10, 1))
    smard = pd.Series(np.arange(len(grid), dtype="float64"), index=grid)
    entsoe = smard.copy()
    entsoe.iloc[-1] += 1.0
    result = compare_prices(entsoe, smard)
    assert result.periods == len(grid)
    assert result.mismatch_days == [date(2025, 10, 1)]
    assert result.max_abs_diff == 1.0


def test_report_lists_eras_and_holds_no_token() -> None:
    first, last = date(2025, 9, 30), date(2025, 10, 1)
    grid = quarter_hour_grid(first, last)
    smard = pd.Series(50.0, index=grid)
    day_ahead = pd.DataFrame(
        {"price_position_1": 50.0, "price_position_2": 51.0}, index=grid
    )
    imbalance = pd.DataFrame(
        {
            "imbalance_price_excess_eur_mwh": 60.0,
            "imbalance_price_insufficient_eur_mwh": 60.0,
        },
        index=grid,
    )
    report = build_price_check_report(day_ahead, imbalance, smard, first, last)
    assert "Hourly products (before 2025-10-01)" in report
    assert "15-minute products (from 2025-10-01)" in report
    assert "100.00%" in report
    assert FAKE_TOKEN not in report


# --- what may be cached as settled -------------------------------------------------


_ACK = (
    '<Acknowledgement_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-1:'
    'acknowledgementdocument:7:0"><Reason><code>999</code><text>{text}</text>'
    "</Reason></Acknowledgement_MarketDocument>"
)


def test_only_data_or_no_data_answers_count_as_final() -> None:
    from src.ingest.entsoe import payload_is_final

    no_data = _ACK.format(text="No matching data found for Data item").encode()
    limit = _ACK.format(
        text="The amount of requested data exceeds allowed limit"
    ).encode()
    market = (
        b'<Publication_MarketDocument xmlns="urn:iec62325.351:tc57wg16:451-3:'
        b'publicationdocument:7:3"><type>A44</type></Publication_MarketDocument>'
    )
    assert payload_is_final(market)
    assert payload_is_final(no_data)
    assert not payload_is_final(limit)
    assert not payload_is_final(b"<html><body>maintenance</body></html>")
    assert not payload_is_final(b"not xml at all")


def test_imbalance_prices_settle_later_than_day_ahead_prices() -> None:
    from src.ingest.entsoe import EntsoeOptions

    options = EntsoeOptions()
    assert options.imbalance_settle_days > options.settle_days
