"""Write docs/results/phase4_backtest.md from the Phase 4 backtest outputs.

    uv run python -m src.trading.phase4_report

Reads the summaries, notes and daily tables that ``python -m src.trading.backtest``
wrote to ``data/processed/backtest/<suite>/`` for every suite, and the saved
hold-out forecasts for their accuracy. Every number in the report is computed here
from those files, so the report cannot drift from the runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import Settings, load_settings
from src.forecasting.evaluate import score
from src.trading.attribution import DIRECTIONS, HOUR_BLOCKS
from src.trading.backtest import capture_in_months, pnl_differences

__all__ = ["build_report", "main", "markdown_table"]

RESULTS_PATH = (
    Path(__file__).resolve().parents[2] / "docs" / "results" / "phase4_backtest.md"
)
LABELS = {
    "perfect_foresight": "Perfect foresight",
    "median_forecast": "Median forecast",
    "mean_forecast": "Mean forecast",
    "quantile_q25": "Quantile-aware q25",
    "quantile_q10": "Quantile-aware q10",
}
STRATEGIES = tuple(LABELS)
FORECAST_STRATEGIES = STRATEGIES[1:]
VARIANTS = {
    "level_shift": "Every price too high by the same amount",
    "noise_everywhere": "Random error in every quarter-hour",
    "noise_idle_periods": "Random error only where perfect foresight idles",
    "noise_traded_periods": "Random error only where perfect foresight trades",
    "peak_one_hour_early": "Afternoon and evening prices one hour early",
}
Column = tuple[str, str, str]


def markdown_table(frame: pd.DataFrame, columns: list[Column]) -> list[str]:
    """Markdown table lines; each column is (field, header, format string)."""
    lines = [
        "| " + " | ".join(header for _, header, _ in columns) + " |",
        "|" + "---|" * len(columns),
    ]
    for _, row in frame.iterrows():
        cells = [fmt.format(row[field]) for field, _, fmt in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _suite(folder: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = pd.read_csv(folder / "summary.csv")
    notes = json.loads((folder / "notes.json").read_text(encoding="utf-8"))
    return summary, notes


def _row(summary: pd.DataFrame, **match: str) -> pd.Series:
    mask = pd.Series(True, index=summary.index)
    for column, value in match.items():
        mask &= summary[column] == value
    rows = summary[mask]
    if len(rows) != 1:
        raise ValueError(f"expected one summary row for {match}, found {len(rows)}")
    return rows.iloc[0]


def _attribution_rows(row: pd.Series) -> pd.DataFrame:
    records = []
    for block, _ in HOUR_BLOCKS:
        cells = {d: float(row[f"cost_{block}_{d}_eur"]) for d in DIRECTIONS}
        records.append(
            {
                "block": f"{block} h",
                "under": cells["under"],
                "over": cells["over"],
                "total": cells["under"] + cells["over"],
                "share": (cells["under"] + cells["over"]) / float(row["gap_eur"]),
            }
        )
    return pd.DataFrame(records)


def forecast_accuracy(
    settings: Settings, models: list[str], days: list[date]
) -> pd.DataFrame:
    """Hold-out accuracy of each model's saved forecasts on the traded days."""
    rows = []
    folder = settings.data.processed_path / "forecasts" / "holdout"
    for model in models:
        frame = pd.read_parquet(folder / f"{model}.parquet")
        frame = frame[frame["target_day"].isin(days)]
        scores = score(frame, settings.forecasting.quantiles)
        rows.append(
            {
                "model": model,
                "mean_pinball": scores["mean pinball"],
                "mae_of_median": scores["MAE of median"],
                "coverage_90": scores["coverage 90%"],
            }
        )
    return pd.DataFrame(rows)


def build_report(settings: Settings) -> str:
    """The full Phase 4 results report as markdown."""
    backtest = settings.data.processed_path / "backtest"
    production = settings.forecasting.production_model
    battery = settings.battery
    validation, validation_notes = _suite(backtest / "validation")
    value, _ = _suite(backtest / "decision-value")
    synthetic, synthetic_notes = _suite(backtest / "synthetic")
    sweep, _ = _suite(backtest / "degradation")
    attribution, _ = _suite(backtest / "attribution")
    holdout, _ = _suite(backtest / "holdout")
    value_pnl = pd.read_parquet(backtest / "decision-value" / "pnl_daily.parquet")
    holdout_pnl = pd.read_parquet(backtest / "holdout" / "pnl_daily.parquet")
    holdout_attribution = pd.read_parquet(
        backtest / "holdout" / "attribution_summary.parquet"
    )
    holdout_days = sorted(holdout_pnl["target_day"].unique())
    holdout_models = list(dict.fromkeys(holdout["model"]))
    holdout_months = {pd.Timestamp(day).month for day in holdout_days}
    same_months = capture_in_months(value_pnl, holdout_months).set_index(
        ["model", "strategy"]
    )
    month_names = ", ".join(
        pd.Timestamp(2000, month, 1).strftime("%B") for month in sorted(holdout_months)
    )
    production_notes = validation_notes[production]

    lines = [
        "# Phase 4 backtest: profit, decision value and error economics",
        "",
        "Generated by `python -m src.trading.phase4_report` from the outputs of "
        "`python -m src.trading.backtest`. Every schedule is optimized on the "
        "forecast issued at 11:40 the day before, committed before the 12:00 gate and "
        "settled at the realised day-ahead price, with the battery as a price taker.",
        "",
        f"Battery: {battery.power_mw:g} MW / {battery.capacity_mwh:g} MWh, "
        f"{battery.round_trip_efficiency:.0%} round trip, "
        f"€{battery.degradation_eur_per_mwh:g} wear per MWh discharged, start and end "
        f"at {battery.initial_soc_fraction:.0%} state of charge, at most "
        f"{battery.max_cycles_per_day:g} cycles a day.",
        "",
        f"Validation window: {production_notes['first']} to "
        f"{production_notes['last']}, "
        f"{production_notes['days']} days. Hold-out: {holdout_days[0]} to "
        f"{holdout_days[-1]}, {len(holdout_days)} days, run once with the setup frozen "
        "on the validation window.",
        "",
        "Capture ratio is a strategy's profit divided by perfect foresight's.",
        "",
        "## Hold-out against validation, production model",
        "",
        f"The hold-out covers {month_names} only, and those months trade differently "
        "from the year's average, so validation is shown both over the same calendar "
        "months of 2024 and 2025 and over the whole window.",
        "",
    ]
    rows = []
    for strategy in STRATEGIES:
        v = _row(validation, strategy=strategy)
        h = _row(holdout, model=production, strategy=strategy)
        rows.append(
            {
                "strategy": LABELS[strategy],
                "v_capture": v["capture_ratio"],
                "s_capture": same_months.loc[(production, strategy), "capture_ratio"],
                "h_capture": h["capture_ratio"],
                "v_day": v["pnl_eur"] / v["days"],
                "h_day": h["pnl_eur"] / h["days"],
                "h_losing": h["losing_days"],
                "h_drawdown": h["max_drawdown_eur"],
                "h_cycles": h["cycles_per_day"],
            }
        )
    lines += markdown_table(
        pd.DataFrame(rows),
        [
            ("strategy", "strategy", "{}"),
            ("v_capture", "validation capture", "{:.1%}"),
            ("s_capture", "validation capture, same months", "{:.1%}"),
            ("h_capture", "hold-out capture", "{:.1%}"),
            ("v_day", "validation € a day", "{:,.1f}"),
            ("h_day", "hold-out € a day", "{:,.1f}"),
            ("h_losing", "hold-out losing days", "{:.0f}"),
            ("h_drawdown", "hold-out max drawdown, €", "{:,.1f}"),
            ("h_cycles", "hold-out cycles a day", "{:.2f}"),
        ],
    )

    lines += ["", "## Hold-out by forecasting model", ""]
    accuracy = forecast_accuracy(settings, holdout_models, holdout_days).set_index(
        "model"
    )
    rows = []
    for model in holdout_models:
        v = _row(value, model=model, strategy="median_forecast")
        rows.append(
            {
                "model": model,
                "v_pinball": v["mean_pinball"],
                "h_pinball": accuracy.loc[model, "mean_pinball"],
                "h_coverage": accuracy.loc[model, "coverage_90"],
                **{
                    f"h_{s}": _row(holdout, model=model, strategy=s)["capture_ratio"]
                    for s in ("median_forecast", "mean_forecast", "quantile_q25")
                },
                "v_median": v["capture_ratio"],
                "s_median": same_months.loc[
                    (model, "median_forecast"), "capture_ratio"
                ],
            }
        )
    lines += markdown_table(
        pd.DataFrame(rows),
        [
            ("model", "model", "{}"),
            ("v_pinball", "validation pinball", "{:.2f}"),
            ("h_pinball", "hold-out pinball", "{:.2f}"),
            ("h_coverage", "hold-out 90% coverage", "{:.1%}"),
            ("v_median", "validation capture, median", "{:.1%}"),
            ("s_median", "same months, median", "{:.1%}"),
            ("h_median_forecast", "hold-out capture, median", "{:.1%}"),
            ("h_mean_forecast", "hold-out capture, mean", "{:.1%}"),
            ("h_quantile_q25", "hold-out capture, q25", "{:.1%}"),
        ],
    )
    difference_columns: list[Column] = [
        ("model", "model", "{}"),
        ("strategy", "strategy", "{}"),
        ("days", "days", "{:.0f}"),
        ("mean_daily_difference_eur", "€ a day vs production", "{:+.2f}"),
        ("ci_low_eur", "95% low", "{:+.2f}"),
        ("ci_high_eur", "95% high", "{:+.2f}"),
        ("total_difference_eur", "total, €", "{:+,.0f}"),
    ]
    holdout_differences = pd.concat(
        [
            pnl_differences(holdout_pnl, base_model=production, strategy=s)
            for s in ("median_forecast", "mean_forecast")
        ],
        ignore_index=True,
    )
    lines += [
        "",
        "Daily profit minus the production model's on the hold-out, with a 95% "
        "interval from a 7-day block bootstrap. Evidence only: the production model "
        "was frozen before the hold-out.",
        "",
        *markdown_table(holdout_differences, difference_columns),
    ]

    lines += [
        "",
        "## Validation window, production model",
        "",
        *markdown_table(
            validation.assign(label=validation["strategy"].map(LABELS)),
            [
                ("label", "strategy", "{}"),
                ("pnl_eur", "P&L, €", "{:,.0f}"),
                ("capture_ratio", "capture", "{:.1%}"),
                ("eur_per_mwh_discharged", "€ per MWh sold", "{:.1f}"),
                ("cycles_per_day", "cycles a day", "{:.2f}"),
                ("losing_days", "losing days", "{:.0f}"),
                ("max_drawdown_eur", "max drawdown, €", "{:,.1f}"),
                ("longest_losing_streak_days", "longest losing streak", "{:.0f}"),
                ("top_decile_day_share", "profit from best 10% of days", "{:.1%}"),
            ],
        ),
    ]

    lines += ["", "## Decision value: accuracy against profit (T2)", ""]
    rows = []
    for model in dict.fromkeys(value["model"]):
        row: dict[str, Any] = {"model": model}
        first = _row(value, model=model, strategy="median_forecast")
        row["pinball"] = first["mean_pinball"]
        row["coverage"] = first["coverage_90"]
        for strategy in FORECAST_STRATEGIES:
            row[strategy] = _row(value, model=model, strategy=strategy)["capture_ratio"]
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("pinball")
    lines += markdown_table(
        table,
        [
            ("model", "model", "{}"),
            ("pinball", "validation pinball", "{:.2f}"),
            ("coverage", "90% coverage", "{:.1%}"),
            *[
                (s, f"capture, {LABELS[s].lower()}", "{:.1%}")
                for s in FORECAST_STRATEGIES
            ],
        ],
    )
    value_differences = pd.concat(
        [
            pnl_differences(value_pnl, base_model=production, strategy=s)
            for s in ("median_forecast", "mean_forecast", "quantile_q25")
        ],
        ignore_index=True,
    )
    lines += [
        "",
        "Daily profit minus the production model's on the validation window:",
        "",
        *markdown_table(value_differences, difference_columns),
        "",
        "Synthetic forecasts built from the realised prices, traded with median "
        f"dispatch. The target error of €{synthetic_notes['target_mae_eur_mwh']:.2f}"
        "/MWh is the production model's mean absolute error.",
        "",
        *markdown_table(
            synthetic.assign(label=synthetic["variant"].map(VARIANTS)),
            [
                ("label", "forecast", "{}"),
                ("mean_mae_eur_mwh", "mean absolute error, €/MWh", "{:.2f}"),
                ("capture_ratio", "capture", "{:.1%}"),
                ("lost_eur", "lost against perfect foresight, €", "{:,.0f}"),
                ("cycles_per_day", "cycles a day", "{:.2f}"),
            ],
        ),
    ]

    lines += [
        "",
        "## Wear price in the optimizer (T3)",
        "",
        "Profit is always charged the true wear of "
        f"€{battery.degradation_eur_per_mwh:g} per MWh discharged.",
        "",
    ]
    shown = sweep[sweep["strategy"].isin(["median_forecast", "perfect_foresight"])]
    shown = shown.assign(
        label=shown["strategy"].map(LABELS),
        cap=shown["cycle_cap"].map(lambda c: "none" if pd.isna(c) else f"{c:g} a day"),
    )
    lines += markdown_table(
        shown,
        [
            ("label", "strategy", "{}"),
            ("cap", "cycle cap", "{}"),
            ("optimizer_wear_eur_per_mwh", "wear price, €/MWh", "{:g}"),
            ("cycles_per_day", "cycles a day", "{:.2f}"),
            ("revenue_eur", "revenue, €", "{:,.0f}"),
            ("pnl_true_wear_eur", "P&L at true wear, €", "{:,.0f}"),
            ("losing_days_true_wear", "losing days", "{:.0f}"),
        ],
    )

    lines += ["", "## Where the gap is lost, by local hour and direction (T1)", ""]
    lines += [
        "Shapley shares of the gap to perfect foresight. *Too low* means the median "
        "forecast was below the realised price, *too high* above it, for every "
        "strategy; before October 2025 an hourly product takes the direction of its "
        "mean error. Shares add up to the gap. A negative share means that, averaged "
        "over correction orders, correcting that group lowers profit while other "
        "errors remain.",
    ]
    sections = [
        (f"Validation, {LABELS[s].lower()}", _row(attribution, strategy=s))
        for s in attribution["strategy"]
    ]
    sections.append(
        (
            "Hold-out, median forecast",
            holdout_attribution[
                holdout_attribution["strategy"] == "median_forecast"
            ].iloc[0],
        )
    )
    for title, shares in sections:
        gap = float(shares["gap_eur"])
        under, over = float(shares["cost_under_eur"]), float(shares["cost_over_eur"])
        lines += [
            "",
            f"**{title}:** gap €{gap:,.0f} over {int(shares['days'])} days; "
            f"median too low €{under:,.0f} ({under / gap:.0%}), "
            f"too high €{over:,.0f} ({over / gap:.0%}).",
            "",
            *markdown_table(
                _attribution_rows(shares),
                [
                    ("block", "local hours", "{}"),
                    ("under", "median too low, €", "{:+,.0f}"),
                    ("over", "median too high, €", "{:+,.0f}"),
                    ("total", "total, €", "{:+,.0f}"),
                    ("share", "share of gap", "{:.0%}"),
                ],
            ),
        ]

    lines += [
        "",
        "## Limitations",
        "",
        "- The battery is a price taker and its orders carry no limit price; a desk "
        "would bid price-quantity curves.",
        "- Trading is day-ahead only, with no intraday re-trading after the auction.",
        "- Each day starts and ends half full, so no energy is carried across "
        "midnight.",
        "- Grid fees, taxes and balancing-market revenue are left out.",
        "- Imbalance risk from outages is measured separately, in "
        "`docs/results/t4_imbalance.md`.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the Phase 4 results report.")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    RESULTS_PATH.write_text(build_report(settings), encoding="utf-8")
    print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
