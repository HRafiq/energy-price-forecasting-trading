from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pandas as pd
import pytest
import requests
import yaml

from src.config import DEFAULT_SETTINGS_PATH, Settings, WeatherConfig
from src.ingest import open_meteo
from src.ingest.open_meteo import (
    ARCHIVE_LAG,
    QUARTER_HOUR,
    OpenMeteoError,
    align_to_period_start,
    build_request_url,
    chunk_ranges,
    download_weather_forecasts,
    http_fetch_json,
    is_settled,
    parse_response,
)

POINTS: dict[str, tuple[float, float]] = {"north": (54.4, 6.8), "south": (48.8, 11.8)}
VARIABLES = ("wind_speed_100m", "shortwave_radiation")
START = date(2024, 3, 1)
T0 = pd.Timestamp("2024-03-01", tz="UTC")

ValueFn = Callable[[int, str, pd.Timestamp], float | None]


def code(point: int, variable: str, stamp: pd.Timestamp) -> float:
    """Unique value per point, variable and API stamp."""
    return 1e6 * point + 1e5 * VARIABLES.index(variable) + (stamp - T0) / QUARTER_HOUR


class FakeOpenMeteo:
    """Answers any Previous Runs URL from a value function; records every URL."""

    def __init__(self, value: ValueFn = code) -> None:
        self.value = value
        self.calls: list[str] = []
        self.drop: pd.Timestamp | None = None

    def __call__(self, url: str) -> Any:
        self.calls.append(url)
        query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
        assert query["timezone"] == "UTC"
        start = pd.Timestamp(query["start_date"], tz="UTC")
        end = pd.Timestamp(query["end_date"], tz="UTC") + pd.Timedelta(days=1)
        stamps = pd.date_range(start, end, freq=QUARTER_HOUR, inclusive="left")
        if self.drop is not None:
            stamps = stamps[stamps != self.drop]
        names = query["minutely_15"].split(",")
        lats, lons = query["latitude"].split(","), query["longitude"].split(",")
        locations = []
        for i, (lat, lon) in enumerate(zip(lats, lons, strict=True)):
            block: dict[str, list[Any]] = {
                "time": [s.strftime("%Y-%m-%dT%H:%M") for s in stamps]
            }
            for name in names:
                variable = name.rsplit("_previous_day", 1)[0]
                block[name] = [self.value(i, variable, s) for s in stamps]
            locations.append(
                {
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "utc_offset_seconds": 0,
                    "minutely_15": block,
                }
            )
        return locations

    def ranges(self) -> list[tuple[str, str]]:
        out = []
        for url in self.calls:
            query = parse_qs(urlsplit(url).query)
            out.append((query["start_date"][0], query["end_date"][0]))
        return out


@pytest.fixture
def wx_settings(settings: Settings, tmp_path: Path) -> Settings:
    weather = settings.weather.model_copy(
        update={
            "points": POINTS,
            "variables": VARIABLES,
            "start": START,
            "request_days": 3,
        }
    )
    data = settings.data.model_copy(update={"raw_dir": tmp_path / "raw"})
    return settings.model_copy(update={"weather": weather, "data": data})


def _download(
    s: Settings, end: date, fake: FakeOpenMeteo, **kwargs: Any
) -> pd.DataFrame:
    return download_weather_forecasts(s, end, fetch_json=fake, pause_s=0, **kwargs)


# --- request URLs ------------------------------------------------------------


def test_request_url_has_previous_day_suffix_and_location_lists(
    wx_settings: Settings,
) -> None:
    url = build_request_url(wx_settings.weather, date(2024, 3, 1), date(2024, 3, 31))
    query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}

    assert url.startswith("https://previous-runs-api.open-meteo.com/v1/forecast?")
    assert "latitude=54.4,48.8" in url
    assert query["latitude"] == "54.4,48.8"
    assert query["longitude"] == "6.8,11.8"
    assert query["minutely_15"] == (
        "wind_speed_100m_previous_day2,shortwave_radiation_previous_day2"
    )
    assert query["start_date"] == "2024-03-01"
    assert query["end_date"] == "2024-03-31"
    assert query["timezone"] == "UTC"
    assert "hourly" not in query


def test_request_url_follows_lead_days(wx_settings: Settings) -> None:
    config = wx_settings.weather.model_copy(update={"lead_days": 3})
    url = build_request_url(config, START, START)
    assert "wind_speed_100m_previous_day3" in url
    assert "previous_day2" not in url


def test_chunk_ranges_are_anchored_at_start_and_clipped() -> None:
    ranges = chunk_ranges(date(2024, 3, 1), date(2024, 3, 8), 3)
    assert ranges == [
        (date(2024, 3, 1), date(2024, 3, 3)),
        (date(2024, 3, 4), date(2024, 3, 6)),
        (date(2024, 3, 7), date(2024, 3, 8)),
    ]


# --- parsing ------------------------------------------------------------------


def test_parse_response_builds_wide_columns_in_config_order(
    wx_settings: Settings,
) -> None:
    config = wx_settings.weather
    payload = FakeOpenMeteo()(build_request_url(config, START, START))
    frame = parse_response(payload, config)

    assert (
        list(frame.columns)
        == config.columns
        == [
            "wx_north_wind_speed_100m",
            "wx_north_shortwave_radiation",
            "wx_south_wind_speed_100m",
            "wx_south_shortwave_radiation",
        ]
    )
    assert len(frame) == 96
    assert str(pd.DatetimeIndex(frame.index).tz) == "UTC"
    assert (frame.dtypes == "float64").all()
    stamp = T0 + 5 * QUARTER_HOUR
    assert frame.loc[stamp, "wx_south_wind_speed_100m"] == 1e6 + 5
    assert frame.loc[stamp, "wx_north_shortwave_radiation"] == 1e5 + 5


def test_parse_response_maps_null_to_nan(wx_settings: Settings) -> None:
    config = wx_settings.weather
    fake = FakeOpenMeteo(lambda p, v, s: None if p == 1 and s.hour == 3 else 1.0)
    frame = parse_response(fake(build_request_url(config, START, START)), config)

    assert frame["wx_south_wind_speed_100m"].isna().sum() == 4
    assert frame["wx_north_wind_speed_100m"].notna().all()


def _one_day_payload(config: WeatherConfig) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = FakeOpenMeteo()(
        build_request_url(config, START, START)
    )
    return payload


def test_parse_response_rejects_wrong_location_count(wx_settings: Settings) -> None:
    config = wx_settings.weather
    with pytest.raises(OpenMeteoError, match="expected 2 locations"):
        parse_response(_one_day_payload(config)[:1], config)


def test_parse_response_rejects_swapped_locations(wx_settings: Settings) -> None:
    config = wx_settings.weather
    with pytest.raises(OpenMeteoError, match="out of order"):
        parse_response(_one_day_payload(config)[::-1], config)


def test_parse_response_rejects_missing_variable_and_non_utc(
    wx_settings: Settings,
) -> None:
    config = wx_settings.weather
    payload = _one_day_payload(config)
    del payload[1]["minutely_15"]["shortwave_radiation_previous_day2"]
    with pytest.raises(OpenMeteoError, match="shortwave_radiation_previous_day2"):
        parse_response(payload, config)

    payload = _one_day_payload(config)
    payload[0]["utc_offset_seconds"] = 3600
    with pytest.raises(OpenMeteoError, match="not in UTC"):
        parse_response(payload, config)


def test_parse_response_surfaces_api_error(wx_settings: Settings) -> None:
    with pytest.raises(OpenMeteoError, match="Parameter 'x'"):
        parse_response(
            {"error": True, "reason": "Parameter 'x' invalid"}, wx_settings.weather
        )


# --- alignment ----------------------------------------------------------------


def test_averaged_values_move_to_the_start_of_their_period(
    wx_settings: Settings,
) -> None:
    """Radiation stamped 06:15 is the mean over 06:00-06:15, so it labels 06:00."""
    config = wx_settings.weather.model_copy(update={"points": {"north": (54.4, 6.8)}})
    stamps = pd.date_range(
        "2024-07-15 06:00", periods=5, freq=QUARTER_HOUR, tz="UTC", name="t"
    )
    raw = pd.DataFrame(
        {
            "wx_north_wind_speed_100m": [10.0, 11.0, 12.0, 13.0, 14.0],
            "wx_north_shortwave_radiation": [0.0, 0.0, 8.0, 40.0, 90.0],
        },
        index=stamps,
    )
    aligned = align_to_period_start(raw, config)

    assert aligned.index.name == "timestamp_utc"
    at = pd.Timestamp("2024-07-15 06:15", tz="UTC")
    assert aligned.loc[at, "wx_north_wind_speed_100m"] == 11.0
    assert aligned.loc[at, "wx_north_shortwave_radiation"] == 8.0
    # The value stamped 06:00 belongs to 05:45-06:00; nothing closes 07:00 yet.
    assert aligned.loc[at - 2 * QUARTER_HOUR, "wx_north_shortwave_radiation"] == 0.0
    assert pd.isna(aligned.loc[stamps[-1], "wx_north_shortwave_radiation"])
    assert pd.isna(aligned.loc[stamps[0] - QUARTER_HOUR, "wx_north_wind_speed_100m"])


def test_download_aligns_every_period_including_the_last(
    wx_settings: Settings,
) -> None:
    end = date(2024, 3, 7)
    frame = _download(wx_settings, end, FakeOpenMeteo())

    grid = pd.date_range(
        T0, pd.Timestamp("2024-03-08", tz="UTC"), freq=QUARTER_HOUR, inclusive="left"
    )
    assert frame.index.equals(grid)
    assert frame.index.name == "timestamp_utc"
    assert list(frame.columns) == wx_settings.weather.columns
    assert frame.notna().all().all()
    assert (frame.dtypes == "float64").all()
    for t in (grid[0], grid[291], grid[-1]):  # 291: a chunk's last quarter-hour
        assert frame.loc[t, "wx_south_wind_speed_100m"] == code(1, VARIABLES[0], t)
        closing = t + QUARTER_HOUR
        assert frame.loc[t, "wx_south_shortwave_radiation"] == code(
            1, VARIABLES[1], closing
        )


def test_download_keeps_api_nulls_as_nan(wx_settings: Settings) -> None:
    gap = pd.Timestamp("2024-03-02 12:00", tz="UTC")
    fake = FakeOpenMeteo(lambda p, v, s: None if s == gap else 5.0)
    frame = _download(wx_settings, date(2024, 3, 3), fake)

    nulls = frame.isna()
    assert nulls.to_numpy().sum() == 4
    assert nulls.loc[gap, "wx_north_wind_speed_100m"]
    assert nulls.loc[gap - QUARTER_HOUR, "wx_north_shortwave_radiation"]


def test_download_rejects_a_gap_in_the_time_axis(wx_settings: Settings) -> None:
    fake = FakeOpenMeteo()
    fake.drop = pd.Timestamp("2024-03-02 10:00", tz="UTC")
    with pytest.raises(OpenMeteoError, match="missing"):
        _download(wx_settings, date(2024, 3, 3), fake)


def test_download_validates_arguments(wx_settings: Settings) -> None:
    with pytest.raises(ValueError, match="before weather.start"):
        _download(wx_settings, date(2024, 2, 1), FakeOpenMeteo())
    with pytest.raises(ValueError, match="timezone-aware"):
        _download(
            wx_settings,
            date(2024, 3, 3),
            FakeOpenMeteo(),
            as_of=pd.Timestamp("2024-03-01 12:00"),
        )


def test_as_of_masks_values_newer_runs_filled_in(wx_settings: Settings) -> None:
    as_of = pd.Timestamp("2024-03-05 12:00", tz="UTC")
    frame = _download(wx_settings, date(2024, 3, 7), FakeOpenMeteo(), as_of=as_of)
    cutoff = as_of + pd.Timedelta(days=2) - ARCHIVE_LAG
    assert cutoff == pd.Timestamp("2024-03-07 02:00", tz="UTC")

    wind = frame["wx_north_wind_speed_100m"]
    radiation = frame["wx_north_shortwave_radiation"]
    assert wind.loc[:cutoff].notna().all()
    assert wind.loc[cutoff + QUARTER_HOUR :].isna().all()
    assert radiation.loc[: cutoff - QUARTER_HOUR].notna().all()
    assert radiation.loc[cutoff:].isna().all()


# --- caching ------------------------------------------------------------------


def test_settle_rule_uses_lead_days_plus_seven() -> None:
    end = date(2024, 3, 15)
    assert is_settled(date(2024, 3, 6), end, lead_days=2)
    assert not is_settled(date(2024, 3, 7), end, lead_days=2)
    assert not is_settled(date(2024, 3, 6), end, lead_days=3)


#: A download time well after the test dates: chunks can settle, nothing is masked.
LATE_DOWNLOAD = pd.Timestamp("2024-06-01", tz="UTC")


def test_settled_chunks_are_served_from_cache(wx_settings: Settings) -> None:
    end = date(2024, 3, 15)
    fake = FakeOpenMeteo()
    first = _download(wx_settings, end, fake, as_of=LATE_DOWNLOAD)
    assert fake.ranges() == [
        ("2024-03-01", "2024-03-03"),
        ("2024-03-04", "2024-03-06"),
        ("2024-03-07", "2024-03-09"),
        ("2024-03-10", "2024-03-12"),
        ("2024-03-13", "2024-03-15"),
        ("2024-03-16", "2024-03-16"),
    ]
    cache = wx_settings.data.raw_path / "open_meteo"
    assert (cache / "2024-03-01_2024-03-03.json").exists()

    fake.calls.clear()
    second = _download(wx_settings, end, fake, as_of=LATE_DOWNLOAD)

    assert fake.ranges() == [
        ("2024-03-07", "2024-03-09"),
        ("2024-03-10", "2024-03-12"),
        ("2024-03-13", "2024-03-15"),
        ("2024-03-16", "2024-03-16"),
    ]
    pd.testing.assert_frame_equal(first, second)


def test_chunk_cached_before_it_settled_is_downloaded_again(
    wx_settings: Settings,
) -> None:
    early = FakeOpenMeteo(lambda p, v, s: None if s.day == 5 else 1.0)
    _download(
        wx_settings, date(2024, 3, 8), early, as_of=LATE_DOWNLOAD
    )  # nothing settled yet

    later = FakeOpenMeteo(lambda p, v, s: 2.0)
    frame = _download(wx_settings, date(2024, 3, 15), later, as_of=LATE_DOWNLOAD)
    assert ("2024-03-04", "2024-03-06") in later.ranges()
    assert frame.notna().all().all()
    assert (frame == 2.0).all().all()

    later.calls.clear()
    _download(wx_settings, date(2024, 3, 15), later, as_of=LATE_DOWNLOAD)
    assert ("2024-03-04", "2024-03-06") not in later.ranges()
    assert ("2024-03-01", "2024-03-03") not in later.ranges()


def test_changed_request_ignores_the_cache(wx_settings: Settings) -> None:
    end = date(2024, 3, 15)
    _download(wx_settings, end, FakeOpenMeteo())

    weather = wx_settings.weather.model_copy(update={"variables": VARIABLES[:1]})
    changed = wx_settings.model_copy(update={"weather": weather})
    fake = FakeOpenMeteo()
    frame = _download(changed, end, fake)

    assert ("2024-03-01", "2024-03-03") in fake.ranges()
    assert list(frame.columns) == [
        "wx_north_wind_speed_100m",
        "wx_south_wind_speed_100m",
    ]


def test_pauses_only_between_network_calls(wx_settings: Settings) -> None:
    end = date(2024, 3, 15)
    fake = FakeOpenMeteo()
    pauses: list[float] = []
    download_weather_forecasts(
        wx_settings,
        end,
        fetch_json=fake,
        pause_s=1.5,
        sleep=pauses.append,
        as_of=LATE_DOWNLOAD,
    )
    assert pauses == [1.5] * 5

    pauses.clear()
    download_weather_forecasts(
        wx_settings,
        end,
        fetch_json=fake,
        pause_s=1.5,
        sleep=pauses.append,
        as_of=LATE_DOWNLOAD,
    )
    assert pauses == [1.5] * 3  # two settled chunks from cache, four downloads


# --- HTTP ---------------------------------------------------------------------


class _Response:
    def __init__(self, status_code: int, body: Any) -> None:
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self) -> Any:
        return self._body


def test_http_fetch_retries_server_errors_with_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        _Response(503, {"reason": "busy"}),
        _Response(429, {"reason": "slow down"}),
        _Response(200, [{"ok": 1}]),
    ]
    monkeypatch.setattr(requests, "get", lambda *a, **k: responses.pop(0))
    sleeps: list[float] = []
    fetch = http_fetch_json(10, attempts=4, backoff_s=2.0, sleep=sleeps.append)

    assert fetch("https://example.test") == [{"ok": 1}]
    assert sleeps == [2.0, 4.0]


def test_http_fetch_does_not_retry_bad_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def get(url: str, **kwargs: Any) -> _Response:
        calls.append(url)
        return _Response(400, {"error": True, "reason": "Cannot initialize"})

    monkeypatch.setattr(requests, "get", get)
    fetch = http_fetch_json(10, sleep=lambda s: None)
    with pytest.raises(OpenMeteoError, match="HTTP 400: Cannot initialize"):
        fetch("https://example.test")
    assert len(calls) == 1


def test_http_fetch_gives_up_after_all_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def get(url: str, **kwargs: Any) -> _Response:
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", get)
    fetch = http_fetch_json(10, attempts=3, backoff_s=1.0, sleep=lambda s: None)
    with pytest.raises(OpenMeteoError, match="after 3 attempts"):
        fetch("https://example.test")


# --- CLI ----------------------------------------------------------------------


def test_main_writes_parquet_and_reports(
    tmp_path: Path,
    wx_settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    seen: dict[str, Any] = {}

    def fake_download(
        settings: Settings, end: date, *, as_of: pd.Timestamp
    ) -> pd.DataFrame:
        seen.update(end=end, as_of=as_of)
        frame = _download(wx_settings, date(2024, 3, 3), FakeOpenMeteo())
        frame.iloc[:3, 0] = np.nan
        return frame

    monkeypatch.setattr(open_meteo, "download_weather_forecasts", fake_download)
    assert open_meteo.main(["--config", str(config_path), "--end", "2024-03-03"]) == 0

    assert seen["end"] == date(2024, 3, 3)
    assert seen["as_of"].tzinfo is not None
    written = pd.read_parquet(tmp_path / "processed" / open_meteo.OUTPUT_FILE)
    assert len(written) == 3 * 96
    out = capsys.readouterr().out
    assert "rows            288" in out
    assert "first complete  2024-03-01 00:45:00+00:00" in out
    assert "wx_north_wind_speed_100m" in out


def test_main_defaults_end_to_two_days_after_today(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = yaml.safe_load(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["data"]["processed_dir"] = str(tmp_path / "processed")
    config_path = tmp_path / "settings.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    seen: dict[str, Any] = {}

    def fake_download(
        settings: Settings, end: date, *, as_of: pd.Timestamp
    ) -> pd.DataFrame:
        seen.update(end=end, as_of=as_of)
        index = pd.date_range(T0, periods=4, freq=QUARTER_HOUR, name="timestamp_utc")
        return pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]}, index=index)

    monkeypatch.setattr(open_meteo, "download_weather_forecasts", fake_download)
    open_meteo.main(["--config", str(config_path)])
    assert seen["end"] == seen["as_of"].date() + timedelta(days=2)


def test_downloads_without_a_download_time_cache_nothing_as_settled(
    wx_settings: Settings,
) -> None:
    end = date(2024, 3, 15)
    fake = FakeOpenMeteo()
    _download(wx_settings, end, fake)
    metas = list((wx_settings.data.raw_path / "open_meteo").glob("*.meta.json"))
    assert metas
    assert all(json.loads(m.read_text())["settled_at_fetch"] is False for m in metas)

    fake.calls.clear()
    _download(wx_settings, end, fake)
    assert len(fake.ranges()) == 6
