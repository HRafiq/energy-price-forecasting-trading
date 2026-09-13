"""Join the SMARD dataset with weather forecasts and fuel prices for the models.

    uv run python -m src.ingest.build_inputs

Reads ``data/processed/`` ``smard_quarterhour.parquet``,
``weather_forecast_quarterhour.parquet`` and ``fuels_daily.parquet``, and writes
the model-input file named by ``data.inputs_file``. Its columns are exactly the
columns with a publication rule in ``config/settings.yaml``, in that order, so
the information set can serve every one of them to models.

Weather forecasts are aligned by UTC period start and are missing before
``weather.start``. Fuel columns on the rows of local day x hold the last price
published before day x, so a rule of "known before the target day" is safe.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import FUEL_COLUMNS, Settings, load_settings
from src.features.clock import local_dates
from src.timegrid import ensure_utc_index

__all__ = ["FUELS_FILE", "WEATHER_FILE", "join_inputs", "main"]

WEATHER_FILE = "weather_forecast_quarterhour.parquet"
FUELS_FILE = "fuels_daily.parquet"


def join_inputs(
    market: pd.DataFrame,
    weather: pd.DataFrame,
    fuels_by_delivery_day: pd.DataFrame,
    settings: Settings,
) -> pd.DataFrame:
    """Market rows with weather and fuel columns added, in publication-rule order.

    ``fuels_by_delivery_day`` is indexed by local delivery date and holds, for
    each day, the last fuel prices published before it.
    """
    idx = ensure_utc_index(market.index)
    ensure_utc_index(weather.index)
    weather_columns = settings.weather.columns
    missing_weather = [c for c in weather_columns if c not in weather.columns]
    if missing_weather:
        raise ValueError(f"weather data lacks columns {missing_weather}")
    missing_fuels = [c for c in FUEL_COLUMNS if c not in fuels_by_delivery_day.columns]
    if missing_fuels:
        raise ValueError(f"fuel data lacks columns {missing_fuels}")

    frame = market.copy()
    aligned = weather[weather_columns].reindex(idx)
    for column in weather_columns:
        frame[column] = aligned[column].to_numpy(dtype="float64")
    delivery_days = local_dates(idx, settings.market.timezone)
    for column in FUEL_COLUMNS:
        by_day = fuels_by_delivery_day[column]
        frame[column] = by_day.reindex(list(delivery_days)).to_numpy(dtype="float64")

    declared = list(settings.availability.columns)
    unexpected = [c for c in frame.columns if c not in declared]
    absent = [c for c in declared if c not in frame.columns]
    if unexpected or absent:
        raise ValueError(
            f"inputs do not match availability rules; unexpected {unexpected}, "
            f"missing {absent}"
        )
    return frame[declared]


def main(argv: list[str] | None = None) -> int:
    from src.ingest.fuels import last_known_before

    parser = argparse.ArgumentParser(description="Join market, weather and fuel data.")
    parser.add_argument("--config", type=Path, default=None, help="settings YAML path")
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    processed = settings.data.processed_path
    market = pd.read_parquet(settings.data.dataset_path)
    weather = pd.read_parquet(processed / WEATHER_FILE)
    daily_fuels = pd.read_parquet(processed / FUELS_FILE)

    delivery_days = sorted(
        set(local_dates(pd.DatetimeIndex(market.index), settings.market.timezone))
    )
    fuels_by_day = last_known_before(daily_fuels, delivery_days)
    frame = join_inputs(market, weather, fuels_by_day, settings)

    out = settings.data.inputs_path
    tmp = out.with_name(f"{out.name}.tmp")
    frame.to_parquet(tmp)
    tmp.replace(out)

    print(f"rows {len(frame):,}, columns {frame.shape[1]}")
    for group, columns in (
        ("weather", settings.weather.columns),
        ("fuels", list(FUEL_COLUMNS)),
    ):
        complete = frame[columns].notna().all(axis=1)
        first = frame.index[complete.to_numpy()].min() if complete.any() else None
        missing = float(frame[columns].isna().to_numpy().mean())
        print(f"{group}: first complete period {first}, missing share {missing:.1%}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
