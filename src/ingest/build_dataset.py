"""Build the DE-LU market dataset from SMARD and check its quality.

    uv run python -m src.ingest.build_dataset

Writes the dataset named in ``config/settings.yaml``, by default
``data/processed/smard_quarterhour.parquet``, with a ``*_quality.json`` report
beside it. A build that fails a structural check never replaces the last good
dataset, because the dashboard reads that file: it writes
``*_quality_rejected.json`` instead and exits non-zero.

Rows are delivery periods at ``data.modeling_resolution``. SMARD publishes
volumes as MWh per interval; columns named ``*_mw`` are stored as average MW so
values at any resolution compare directly. Columns named ``*_eur_mwh`` are kept
as published. Before ``market.quarter_hour_products_from`` the market traded
hourly products: those quarter-hours repeat their hour's price and
``price_product_minutes`` is 60 instead of 15.

The dataset is a historical record holding both actuals and the grid operators'
day-ahead forecasts. Which columns may be used as features at the 12:00 gate is
decided in ``src/features`` in Phase 2, not here. See docs/data_guide.md §9.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    PRICE_SERIES,
    PRODUCT_COLUMN,
    RESOLUTION_STEP,
    Settings,
    load_settings,
)
from src.ingest.quality import QualityReport, assert_resolution, build_quality_report
from src.ingest.smard import SmardClient
from src.timegrid import HOUR

__all__ = ["PRODUCT_COLUMN", "build_dataset", "main", "publish"]

LOAD_ACTUAL = "load_actual_mw"
LOAD_FORECAST = "load_forecast_mw"
RENEWABLES_ACTUAL = (
    "wind_onshore_actual_mw",
    "wind_offshore_actual_mw",
    "solar_actual_mw",
)
RENEWABLES_FORECAST = (
    "wind_onshore_forecast_mw",
    "wind_offshore_forecast_mw",
    "solar_forecast_mw",
)
REQUIRED_SERIES = (
    PRICE_SERIES,
    LOAD_ACTUAL,
    LOAD_FORECAST,
    *RENEWABLES_ACTUAL,
    *RENEWABLES_FORECAST,
)
VOLUME_SUFFIX = "_mw"
PRICE_SUFFIX = "_eur_mwh"


def _volume_columns(names: Iterable[str]) -> list[str]:
    """Columns to convert to MW. Every series must declare its unit by suffix."""
    volumes: list[str] = []
    for name in names:
        if "_eur" in name and not name.endswith(PRICE_SUFFIX):
            raise ValueError(
                f"series {name!r} looks like a price but does not end in "
                f"{PRICE_SUFFIX!r}; refusing to guess its unit"
            )
        if name.endswith(VOLUME_SUFFIX):
            volumes.append(name)
        elif not name.endswith(PRICE_SUFFIX):
            raise ValueError(
                f"series {name!r} must end in {VOLUME_SUFFIX!r}, converted to "
                f"average MW, or {PRICE_SUFFIX!r}, kept as published"
            )
    return volumes


def _residual(frame: pd.DataFrame, load: str, renewables: tuple[str, ...]) -> pd.Series:
    # min_count makes the residual NaN when any component is missing, instead of
    # silently treating a missing wind value as zero wind.
    return frame[load] - frame[list(renewables)].sum(axis=1, min_count=len(renewables))


def _extend_through(
    frame: pd.DataFrame,
    published_rows: pd.DataFrame,
    settings: Settings,
    through: date,
    step: pd.Timedelta,
) -> pd.DataFrame:
    """Rows up to the end of local day ``through``, past the last published price.

    A backtest stops at the last price, because a row without one has nothing to
    score. A live run forecasts a day whose prices do not exist yet, while that day's
    load forecast was published hours before the gate. Trimming at the last price
    threw those rows away, so the daily run found no load forecast and no weather for
    the day it had to forecast. The rows added here keep whatever SMARD has already
    published and leave the price blank; the availability rules hide a target day's
    price from the models regardless.
    """
    end = settings.market.local_midnight_utc(through + timedelta(days=1)) - step
    last = pd.DatetimeIndex(frame.index)[-1]
    if end <= last:
        return frame
    tail_index = pd.date_range(last + step, end, freq=step, name=frame.index.name)
    return pd.concat([frame, published_rows.reindex(tail_index)])


def build_dataset(
    settings: Settings, client: SmardClient, through: date | None = None
) -> pd.DataFrame:
    """The SMARD dataset, trimmed at the last published price.

    ``through`` is for a live run: the rows of that local delivery day are kept even
    though its prices are not published yet (see ``_extend_through``).
    """
    missing = [name for name in REQUIRED_SERIES if name not in settings.smard.series]
    if missing:
        raise ValueError(f"smard.series is missing required columns: {missing}")
    volumes = _volume_columns(settings.smard.series)

    start = settings.market.local_midnight_utc(settings.data.start)
    columns = {
        name: client.series(filter_id, name, start=start)
        for name, filter_id in settings.smard.series.items()
    }
    frame = pd.concat(columns, axis=1).sort_index()
    frame.index.name = "timestamp_utc"

    published = np.flatnonzero(frame[PRICE_SERIES].notna().to_numpy())
    if published.size == 0:
        raise ValueError("SMARD returned no price values")
    index = pd.DatetimeIndex(frame.index)
    last_price = index[int(published[-1])]
    published_rows = frame
    frame = frame.loc[(index >= start) & (index <= last_price)].copy()

    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    if through is not None:
        frame = _extend_through(frame, published_rows, settings, through, step)
    index = pd.DatetimeIndex(frame.index)
    assert_resolution(index, step)

    # MWh per interval -> average MW over the interval.
    frame[volumes] = frame[volumes] * (HOUR / step)

    frame["residual_load_actual_mw"] = _residual(frame, LOAD_ACTUAL, RENEWABLES_ACTUAL)
    frame["residual_load_forecast_mw"] = _residual(
        frame, LOAD_FORECAST, RENEWABLES_FORECAST
    )
    switch = settings.market.local_midnight_utc(
        settings.market.quarter_hour_products_from
    )
    frame[PRODUCT_COLUMN] = np.where(index < switch, 60, 15).astype("int64")
    return frame


def publish(frame: pd.DataFrame, report: QualityReport, dataset_path: Path) -> bool:
    """Write the dataset and its report only if the build passed.

    A failed build writes ``*_quality_rejected.json`` and leaves the last good
    dataset and its report untouched. A passing build removes any stale
    rejection. Both files are written in full to temporary paths first and then
    swapped in back to back; the report's ``n_rows`` and ``end_utc`` let a
    reader confirm the pair belongs together if a crash ever split the swap.
    """
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = dataset_path.with_name(f"{dataset_path.stem}_quality.json")
    rejected_path = dataset_path.with_name(f"{dataset_path.stem}_quality_rejected.json")
    report_json = report.model_dump_json(indent=2)
    if not report.structural_ok:
        rejected_path.write_text(report_json, encoding="utf-8")
        return False

    dataset_tmp = dataset_path.with_name(f"{dataset_path.name}.tmp")
    report_tmp = report_path.with_name(f"{report_path.name}.tmp")
    frame.to_parquet(dataset_tmp)
    report_tmp.write_text(report_json, encoding="utf-8")
    dataset_tmp.replace(dataset_path)
    report_tmp.replace(report_path)
    rejected_path.unlink(missing_ok=True)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the SMARD market dataset.")
    parser.add_argument("--config", type=Path, default=None, help="settings YAML path")
    parser.add_argument(
        "--through",
        type=date.fromisoformat,
        default=None,
        help="live run: keep the rows of this local delivery day, prices unpublished",
    )
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    client = SmardClient(settings.smard, settings.data.raw_path / "smard")
    frame = build_dataset(settings, client, through=args.through)
    step = RESOLUTION_STEP[settings.data.modeling_resolution]
    switch = settings.market.local_midnight_utc(
        settings.market.quarter_hour_products_from
    )
    report = build_quality_report(
        frame, settings.market.timezone, step, quarter_hour_products_from=switch
    )
    published = publish(frame, report, settings.data.dataset_path)

    print(f"rows            {report.n_rows} (expected {report.expected_rows})")
    print(f"resolution      {report.resolution}")
    print(f"range (UTC)     {report.start_utc} .. {report.end_utc}")
    print(f"missing ts      {report.missing_timestamps}")
    print(f"duplicate ts    {report.duplicate_timestamps}")
    print(f"DST day issues  {len(report.day_length_issues)}")
    print(f"hourly products {report.hourly_product_mismatches} mismatched hours")
    print(f"structural ok   {report.structural_ok}")
    if published:
        print(f"wrote           {settings.data.dataset_path}")
    else:
        print("REJECTED        last good dataset kept; see *_quality_rejected.json")
    return 0 if published else 1


if __name__ == "__main__":
    raise SystemExit(main())
