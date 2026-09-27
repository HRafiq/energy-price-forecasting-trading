"""T9: what kind of day the forecast earns its keep on.

    uv run python -m src.health.experiments.t9_forecast_edge

The plan is ``docs/plans/forecast_edge_plan.md``, committed before this file.
T8 must have run: this reads its saved daily table and re-solves nothing.

T8 showed the production forecast is worth €11.75 a day over a rule that
forecasts nothing, and that 5% of days carry 41% of it. Concentration by itself
proves nothing, because a heavy-tailed difference between two similar strategies
concentrates whether or not the big days have anything in common. This asks
whether those days *do* have something in common: are they the days that
departed from their own seasonal pattern?

Every characteristic describes the day from realised prices and the fixed shape
alone. None of them uses the forecast or its errors, so none can be circular.
Each is reported in euros and as a share of the day's perfect-foresight profit,
because the edge is bounded by what was on the table and a pattern in euros
could otherwise be nothing but "big days are big".
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import spearmanr

from src.config import PRICE_SERIES, REPO_ROOT, Settings, load_settings
from src.health.experiments.t1_mechanism import mean_interval
from src.health.experiments.t8_fixed_shape import (
    OUT as T8_OUT,
)
from src.health.experiments.t8_fixed_shape import (
    shape_for,
    trailing_window,
)
from src.trading.strategies import MEDIAN_FORECAST

__all__ = [
    "CHARACTERISTICS",
    "add_edge",
    "capture_by_agreement",
    "characterise",
    "describe_days",
    "quintile_table",
    "results_markdown",
]

PRODUCTION = "lightgbm_conformal"
#: The fixed arm the edge is measured against, the best in T8.
FIXED = "fixed_shape_seasonal"
#: Characteristics of a day, and whether the prediction is that the edge rises
#: with the value (+1) or falls with it (-1).
CHARACTERISTICS = {
    "shape_agreement": -1,
    "spread": +1,
    "peak_shift": +1,
}
QUINTILES = 5
OUT = "t9_forecast_edge"


def characterise(
    realised: NDArray[np.float64],
    shape: NDArray[np.float64],
    hours: float,
    threshold: float,
) -> dict[str, Any]:
    """What kind of day this was, from its prices and its seasonal shape alone.

    Nothing here touches the forecast or its errors, so none of these can be
    circular: they describe the day, not the model's opinion of it.
    """
    return {
        # How well the day's ranking of periods held. The optimiser needs the
        # order, not the prices, so this is what the fixed rule is betting on.
        "shape_agreement": float(spearmanr(realised, shape).statistic),
        "spread": float(realised.max() - realised.min()),
        "peak_shift": abs(float(np.argmax(realised)) - float(np.argmax(shape))) * hours,
        "spike": bool(realised.max() >= threshold),
    }


def add_edge(joined: pd.DataFrame) -> pd.DataFrame:
    """The forecast's edge over the fixed arm, in euros and normalised.

    The sign is the whole experiment: positive means the production forecast
    earned more than the rule that forecasts nothing. The normalised column
    divides by what was on the table that day, so a pattern that is only "big
    days are big" shows up as a pattern that vanishes there.
    """
    joined = joined.copy()
    joined["edge_eur"] = joined[MEDIAN_FORECAST.name] - joined[FIXED]
    joined["edge_share"] = joined["edge_eur"] / joined["perfect_foresight"]
    return joined


def describe_days(settings: Settings, daily: pd.DataFrame) -> pd.DataFrame:
    """One row per delivery day: its edge, and what kind of day it was."""
    tz = settings.market.timezone
    threshold = settings.evaluation.spike_threshold_eur_mwh
    dispatch = pd.read_parquet(
        settings.data.processed_path / "trading" / PRODUCTION / "dispatch.parquet",
        columns=["target_day", "strategy", "realised_price"],
    )
    control = dispatch[dispatch["strategy"] == MEDIAN_FORECAST.name]
    prices = pd.read_parquet(settings.data.inputs_path, columns=[PRICE_SERIES])[
        PRICE_SERIES
    ].dropna()

    rows: list[dict[str, Any]] = []
    for day in cast(list[date], list(daily.index)):
        group = control[control["target_day"] == day]
        index = pd.DatetimeIndex(group.index)
        realised = group["realised_price"].to_numpy(dtype="float64")
        window = trailing_window(prices, settings.market.local_midnight_utc(day))
        if window.empty:
            continue
        shape = shape_for(window, index, tz, day.month)
        hours = (index[1] - index[0]).total_seconds() / 3600.0
        rows.append(
            {"target_day": day} | characterise(realised, shape, hours, threshold)
        )
    described = pd.DataFrame(rows).set_index("target_day")
    return add_edge(daily.join(described, how="inner"))


def _groups(frame: pd.DataFrame, name: str) -> pd.Series:
    """Quintile labels for a characteristic, lowest fifth first.

    A characteristic with many tied values cannot be cut into five equal groups.
    ``peak_shift`` is one: on a large share of days the fixed shape already puts
    the peak in the right period, so its lowest bins collapse onto zero. The
    duplicate edges are dropped, so such a characteristic simply reports fewer
    groups than five rather than five that would not be equal anyway.
    """
    codes = pd.qcut(frame[name], QUINTILES, duplicates="drop").cat.codes
    labels = pd.Index([f"Q{code + 1}" for code in sorted(codes.unique())])
    return pd.Series(
        pd.Categorical.from_codes(list(codes), categories=labels, ordered=True),
        index=frame.index,
    )


def quintile_table(frame: pd.DataFrame, name: str, value: str) -> list[dict[str, Any]]:
    """Mean edge per quintile of one characteristic, each with its interval."""
    labels = _groups(frame, name)
    total = float(frame[value].sum())
    out: list[dict[str, Any]] = []
    for label in labels.cat.categories:
        scope = cast(pd.Series, labels == label)
        interval = mean_interval(frame[value], scope)
        out.append(
            {
                "quintile": str(label),
                "low_edge": float(frame.loc[scope, name].min()),
                "high_edge": float(frame.loc[scope, name].max()),
                "share_of_total": float(frame.loc[scope, value].sum() / total)
                if total
                else float("nan"),
                **interval,
            }
        )
    return out


def _verdict(table: list[dict[str, Any]], direction: int) -> bool:
    """Whether the predicted end beats the other, on non-overlapping intervals."""
    high, low = table[-1], table[0]
    strong, weak = (low, high) if direction < 0 else (high, low)
    return bool(strong["low"] > weak["high"])


def run(settings: Settings) -> dict[str, Any]:
    """Every characteristic, in euros and normalised, with its verdict."""
    saved = settings.data.processed_path / "experiments" / f"{T8_OUT}_daily.parquet"
    if not saved.exists():
        raise RuntimeError(f"run T8 first: {saved} is missing")
    frame = describe_days(settings, pd.read_parquet(saved))

    values = {"edge_eur": "euros", "edge_share": "normalised"}
    summary: dict[str, Any] = {
        "days": len(frame),
        "first_day": str(frame.index.min()),
        "last_day": str(frame.index.max()),
        "fixed_arm": FIXED,
        "spike_threshold_eur_mwh": settings.evaluation.spike_threshold_eur_mwh,
        "total_edge_eur": float(frame["edge_eur"].sum()),
        "mean_edge_eur": float(frame["edge_eur"].mean()),
        "correlations": {
            name: float(spearmanr(frame[name], frame["edge_eur"]).statistic)
            for name in CHARACTERISTICS
        },
        "quintiles": {
            value: {
                name: quintile_table(frame, name, value) for name in CHARACTERISTICS
            }
            for value in values
        },
        "spike": {
            value: {
                "spike_days": mean_interval(frame[value], frame["spike"]),
                "other_days": mean_interval(frame[value], ~frame["spike"]),
                "count": int(frame["spike"].sum()),
            }
            for value in values
        },
    }
    summary["capture_by_agreement"] = capture_by_agreement(frame)
    summary["verdicts"] = {
        name: {
            value: _verdict(summary["quintiles"][value][name], direction)
            for value in values
        }
        for name, direction in CHARACTERISTICS.items()
    }
    summary["verdicts"]["spike"] = {
        value: bool(
            summary["spike"][value]["spike_days"]["low"]
            > summary["spike"][value]["other_days"]["high"]
        )
        for value in values
    }
    return summary | {"daily": frame}


def capture_by_agreement(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Each arm's capture of the day's own ceiling, by quintile of agreement.

    Not pre-registered: added after the run, because P1 on its own is partly
    mechanical. That the fixed rule does worse on days which departed from the
    fixed shape is close to definitional. What is not definitional is how the
    *forecast* behaves on those days, and the answer changes the reading: it
    degrades too, only more slowly. The edge is resilience, not brilliance.
    """
    labels = _groups(frame, "shape_agreement")
    rows: list[dict[str, Any]] = []
    for label in labels.cat.categories:
        part = frame[labels == label]
        ceiling = float(part["perfect_foresight"].sum())
        forecast = float(part[MEDIAN_FORECAST.name].sum()) / ceiling
        fixed = float(part[FIXED].sum()) / ceiling
        rows.append(
            {
                "quintile": str(label),
                "days": len(part),
                "low_agreement": float(part["shape_agreement"].min()),
                "high_agreement": float(part["shape_agreement"].max()),
                "forecast_capture": forecast,
                "fixed_capture": fixed,
                "gap": forecast - fixed,
            }
        )
    return rows


def _cell(item: dict[str, Any], places: int = 2) -> str:
    return (
        f"{item['mean']:+.{places}f} ({item['low']:+.{places}f} to "
        f"{item['high']:+.{places}f})"
    )


def results_markdown(summary: dict[str, Any]) -> str:
    """The results page, and the verdict on each prediction."""
    v = summary["verdicts"]
    lines = [
        "# T9: what kind of day the forecast earns its keep on",
        "",
        f"Generated by `python -m src.health.experiments.{OUT}`. "
        f"{summary['days']} validation days, {summary['first_day']} to "
        f"{summary['last_day']}. The edge is the production forecast's profit "
        f"over `{summary['fixed_arm']}`, which forecasts nothing: "
        f"€{summary['total_edge_eur']:,.0f} in total, "
        f"€{summary['mean_edge_eur']:.2f} a day. Plan: "
        "`docs/plans/forecast_edge_plan.md`, committed before the code.",
        "",
        "## Verdicts on the predictions, fixed before the run",
        "",
        "A prediction holds only where the two ends' 95% intervals do not overlap.",
        "",
        "| prediction | in euros | normalised by the day's ceiling |",
        "|---|---|---|",
        f"| **P1** the edge is larger when the day's ranking departs from its "
        f"seasonal norm | {'HOLDS' if v['shape_agreement']['edge_eur'] else 'fails'} "
        f"| {'HOLDS' if v['shape_agreement']['edge_share'] else 'fails'} |",
        f"| **P2** the edge is larger when the spread is wider | "
        f"{'HOLDS' if v['spread']['edge_eur'] else 'fails'} | "
        f"{'HOLDS' if v['spread']['edge_share'] else 'fails'} |",
        f"| **P3** the edge is larger on spike days | "
        f"{'HOLDS' if v['spike']['edge_eur'] else 'fails'} | "
        f"{'HOLDS' if v['spike']['edge_share'] else 'fails'} |",
        f"| **P4** the edge is larger when the peak moved | "
        f"{'HOLDS' if v['peak_shift']['edge_eur'] else 'fails'} | "
        f"{'HOLDS' if v['peak_shift']['edge_share'] else 'fails'} |",
        "",
        "Spearman correlation of each characteristic with the daily edge in euros: "
        + ", ".join(f"`{k}` {c:+.3f}" for k, c in summary["correlations"].items())
        + ".",
    ]
    titles = {
        "shape_agreement": "P1: how well the day's ranking of periods held",
        "spread": "P2: the day's spread, highest realised price minus lowest",
        "peak_shift": "P4: hours the day's peak moved from the seasonal peak",
    }
    for name, title in titles.items():
        lines += [
            "",
            f"## {title}",
            "",
            "| quintile | range | edge, €/day | share of total | normalised |",
            "|---|---|---|---|---|",
        ]
        euros = summary["quintiles"]["edge_eur"][name]
        shares = summary["quintiles"]["edge_share"][name]
        for row, share in zip(euros, shares, strict=True):
            lines.append(
                f"| {row['quintile']} | {row['low_edge']:.2f} to "
                f"{row['high_edge']:.2f} | {_cell(row)} | "
                f"{row['share_of_total']:.1%} | {_cell(share, 4)} |"
            )
    lines += [
        "",
        "## Why P1 holds: both arms degrade, one twice as fast",
        "",
        "Added after the run, not pre-registered. P1 on its own is partly "
        "mechanical, because a day that departed from the fixed shape is by "
        "construction a day the fixed rule had trouble with. What is not "
        "mechanical is what the forecast does on those same days.",
        "",
        "| quintile | agreement | days | forecast capture | fixed capture | gap |",
        "|---|---|---|---|---|---|",
    ]
    for row in summary["capture_by_agreement"]:
        lines.append(
            f"| {row['quintile']} | {row['low_agreement']:.2f} to "
            f"{row['high_agreement']:.2f} | {row['days']} | "
            f"{row['forecast_capture']:.1%} | {row['fixed_capture']:.1%} | "
            f"{row['gap'] * 100:+.1f} pts |"
        )
    spike = summary["spike"]
    lines += [
        "",
        f"## P3: spike days, a maximum at or above "
        f"€{summary['spike_threshold_eur_mwh']:.0f}/MWh",
        "",
        f"{spike['edge_eur']['spike_days']['days']} of {summary['days']} days.",
        "",
        "| days | edge, €/day | normalised |",
        "|---|---|---|",
        f"| spike | {_cell(spike['edge_eur']['spike_days'])} | "
        f"{_cell(spike['edge_share']['spike_days'], 4)} |",
        f"| other | {_cell(spike['edge_eur']['other_days'])} | "
        f"{_cell(spike['edge_share']['other_days'], 4)} |",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="What the forecast's edge is made of.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    summary = run(settings)
    daily = summary.pop("daily")
    out = settings.data.processed_path / "experiments"
    out.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(out / f"{OUT}_daily.parquet")
    (out / f"{OUT}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )
    (REPO_ROOT / "docs" / "results" / f"{OUT}.md").write_text(
        results_markdown(summary), encoding="utf-8"
    )
    print(results_markdown(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
