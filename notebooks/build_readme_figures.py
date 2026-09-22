"""Draw the README figures in docs/img/ from saved forecasts and backtest outputs.

    uv run --extra eda python notebooks/build_readme_figures.py

Writes example_day.png (one backtest day: the forecast fan and the committed
schedule), calibration.png (reliability of the forecasters) and decision_value.png
(forecast accuracy against trading profit). Titles and subtitles are computed from
the data.
"""

from __future__ import annotations

import json
import textwrap
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter

from src.config import REPO_ROOT, load_settings
from src.forecasting.base import quantile_column

OUT = REPO_ROOT / "docs" / "img"
EXAMPLE_DAY = date(2025, 11, 21)
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
MODEL_LABELS = {
    "lightgbm_conformal": "LightGBM + conformal",
    "lightgbm_quantile": "LightGBM quantile",
    "quantile_forest": "Quantile forest",
    "qra": "QRA",
    "mstl": "MSTL",
    "lear": "LEAR",
    "naive_previous_day": "Naive, yesterday",
    "seasonal_naive_previous_week": "Seasonal naive, last week",
}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "axes.axisbelow": True,
        "font.size": 9,
        "legend.frameon": False,
    }
)


def titled(fig: plt.Figure, title: str, subtitle: str, top: float) -> None:
    fig.subplots_adjust(top=top)
    fig.text(0.02, 0.985, title, ha="left", va="top", fontsize=12, weight="bold")
    fig.text(0.02, 0.935, textwrap.fill(subtitle, 120), ha="left", va="top", color=INK2)


def example_day() -> None:
    settings = load_settings()
    processed = settings.data.processed_path
    forecast = pd.read_parquet(
        processed / "forecasts" / "comparison" / "lightgbm_conformal.parquet"
    )
    forecast = forecast[forecast["target_day"] == EXAMPLE_DAY].sort_index()
    dispatch = pd.read_parquet(
        processed / "backtest" / "validation" / "dispatch.parquet"
    )
    dispatch = dispatch[dispatch["target_day"] == EXAMPLE_DAY]
    pnl = pd.read_parquet(processed / "backtest" / "validation" / "pnl_daily.parquet")
    pnl = pnl[pnl["target_day"] == EXAMPLE_DAY].set_index("strategy")["pnl_eur"]

    local = pd.DatetimeIndex(forecast.index).tz_convert(settings.market.timezone)
    hours = (local - local[0]) / pd.Timedelta(hours=1)
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [3, 2]}
    )
    top.fill_between(
        hours,
        forecast["q05"],
        forecast["q95"],
        color=BLUE,
        alpha=0.15,
        linewidth=0,
        label="90% range",
    )
    top.fill_between(
        hours,
        forecast["q25"],
        forecast["q75"],
        color=BLUE,
        alpha=0.35,
        linewidth=0,
        label="50% range",
    )
    top.plot(hours, forecast["q50"], color=BLUE, linewidth=2, label="median forecast")
    top.plot(
        hours,
        forecast["actual"],
        color=INK,
        linewidth=1.3,
        linestyle=(0, (4, 3)),
        label="realised price",
    )
    top.set_ylabel("€/MWh")
    top.legend(loc="upper left", ncols=4)

    width = 0.25 / 2.2
    for offset, strategy, color, label in (
        (-width / 2, "median_forecast", BLUE, "median-forecast dispatch"),
        (width / 2, "perfect_foresight", MUTED, "perfect foresight"),
    ):
        net = dispatch[dispatch["strategy"] == strategy].sort_index()["net_mw"]
        bottom.bar(
            hours + 0.125 + offset,
            net.to_numpy(),
            width=width,
            color=color,
            label=label,
        )
    bottom.axhline(0, color=MUTED, linewidth=0.8)
    bottom.set_ylabel("MW, sell above 0, buy below")
    bottom.set_xlabel("Hour of the delivery day, local time")
    bottom.set_xticks(range(0, 25, 3))
    bottom.legend(loc="lower left", bbox_to_anchor=(0.0, 1.0), ncols=2)
    titled(
        fig,
        f"One day in the backtest: {EXAMPLE_DAY:%d %B %Y}",
        "The forecast was issued at 11:40 the day before and the schedule committed "
        "before the 12:00 gate. Median-forecast dispatch earned "
        f"€{pnl['median_forecast']:,.0f} at the realised prices; perfect foresight "
        f"€{pnl['perfect_foresight']:,.0f}.",
        top=0.86,
    )
    fig.savefig(OUT / "example_day.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def calibration() -> None:
    settings = load_settings()
    folder = settings.data.processed_path / "forecasts" / "comparison"
    levels = np.array(settings.forecasting.quantiles)
    fig, ax = plt.subplots(figsize=(6.4, 5.6))
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="perfectly calibrated")
    gaps = []
    for model, color in (
        ("lightgbm_conformal", BLUE),
        ("lightgbm_quantile", ORANGE),
        ("quantile_forest", AQUA),
        ("naive_previous_day", YELLOW),
    ):
        frame = pd.read_parquet(folder / f"{model}.parquet")
        start = settings.evaluation.validation_start
        frame = frame[(frame["target_day"] >= start) & frame["actual"].notna()]
        shares = np.array(
            [(frame["actual"] <= frame[quantile_column(q)]).mean() for q in levels]
        )
        ax.plot(
            levels,
            shares,
            marker="o",
            markersize=5,
            color=color,
            linewidth=2.4 if model == "lightgbm_conformal" else 1.5,
            label=MODEL_LABELS[model],
        )
        gaps.append(f"{MODEL_LABELS[model]} {100 * np.abs(shares - levels).mean():.1f}")
    ax.set_xlabel("Forecast quantile")
    ax.set_ylabel("Share of realised prices at or below it")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.legend(loc="upper left")
    titled(
        fig,
        "Calibration on the validation window",
        "Mean gap from the diagonal, percentage points: " + "; ".join(gaps) + ".",
        top=0.84,
    )
    fig.savefig(OUT / "calibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def decision_value() -> None:
    settings = load_settings()
    summary = pd.read_csv(
        settings.data.processed_path / "backtest" / "decision-value" / "summary.csv"
    )
    median = summary[summary["strategy"] == "median_forecast"].set_index("model")
    synthetic = pd.read_csv(
        settings.data.processed_path / "backtest" / "synthetic" / "summary.csv"
    ).set_index("variant")
    notes = json.loads(
        (
            settings.data.processed_path / "backtest" / "synthetic" / "notes.json"
        ).read_text()
    )
    rho = median["mean_pinball"].corr(median["capture_ratio"], method="spearman")

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    span = median["capture_ratio"].max() - median["capture_ratio"].min()
    label_y = None
    for model, row in median.sort_values("capture_ratio", ascending=False).iterrows():
        color = BLUE if model == "lightgbm_conformal" else MUTED
        ax.scatter(
            row["mean_pinball"],
            row["capture_ratio"],
            s=70,
            color=color,
            edgecolor=SURFACE,
            linewidth=1.5,
            zorder=3,
        )
        y = row["capture_ratio"]
        label_y = y if label_y is None else min(y, label_y - 0.08 * span)
        ax.annotate(
            MODEL_LABELS[str(model)],
            (row["mean_pinball"], y),
            xytext=(row["mean_pinball"] + 0.5, label_y),
            textcoords="data",
            va="center",
            color=INK2,
            arrowprops={"arrowstyle": "-", "color": GRID, "linewidth": 0.8},
        )
    ax.set_xlim(median["mean_pinball"].min() - 0.5, median["mean_pinball"].max() + 3.5)
    ax.set_xlabel("Mean pinball loss on the validation window, €/MWh (lower is better)")
    ax.set_ylabel("Share of perfect-foresight profit captured")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    lost = synthetic["lost_eur"]
    titled(
        fig,
        "Does a better forecast make the battery more money?",
        f"Eight forecasters, median-forecast dispatch, June 2024 to May 2026. Rank "
        f"correlation {rho:.2f}. Synthetic forecasts with the production model's "
        f"€{notes['target_mae_eur_mwh']:.0f}/MWh average error: a level shift loses "
        f"€{lost['level_shift']:,.0f}, random noise €{lost['noise_everywhere']:,.0f}, "
        "the evening one hour early "
        f"€{lost['peak_one_hour_early']:,.0f} with a "
        f"€{synthetic.loc['peak_one_hour_early', 'mean_mae_eur_mwh']:.0f}/MWh error.",
        top=0.84,
    )
    fig.savefig(OUT / "decision_value.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def next_euro() -> None:
    """What each lever is worth to the battery that actually trades.

    The forecast and the asset are usually argued about separately, in different
    units, so they never get compared. Every bar here is the same quantity: how
    much more the median-forecast strategy would have earned over the same 730
    days, as a share of the perfect-foresight ceiling. Measuring the cap against
    the ceiling instead would answer a different question, what a trader with
    perfect knowledge would gain from it, and would not belong on the same axis.
    """
    processed = load_settings().data.processed_path
    sweep = pd.read_csv(processed / "backtest" / "degradation" / "summary.csv")
    true_wear = sweep["optimizer_wear_eur_per_mwh"] == sweep["true_wear_eur_per_mwh"]
    capped = sweep["cycle_cap"] == 2.0
    uncapped = sweep["cycle_cap"].isna()

    def pnl(rows: pd.Series, strategy: str) -> float:
        row = sweep[rows & (sweep["strategy"] == strategy)]
        return float(row["pnl_true_wear_eur"].iloc[0])

    ceiling = pnl(capped & true_wear, "perfect_foresight")
    traded = pnl(capped & true_wear, "median_forecast")
    # Every bar is a gain to the strategy that actually trades, over the same
    # ceiling. Taking the cap off the perfect-foresight run instead would raise
    # the ceiling by 0.7%, which is a different question from what this desk
    # would earn, and putting the two side by side would not be a comparison.
    no_cap = pnl(uncapped & true_wear, "median_forecast")
    free_wear = pnl(capped & (sweep["optimizer_wear_eur_per_mwh"] == 0.0),
                    "median_forecast")

    levers = [
        ("A perfect forecast", (ceiling - traded) / ceiling, BLUE),
        (
            "Pricing wear correctly\n(already done)",
            (traded - free_wear) / ceiling,
            MUTED,
        ),
        ("Lifting the two-cycle cap", (no_cap - traded) / ceiling, MUTED),
    ]
    fig, ax = plt.subplots(figsize=(7.6, 3.1))
    names = [name for name, _, _ in levers]
    values = [value * 100 for _, value, _ in levers]
    ax.barh(names, values, color=[c for _, _, c in levers], height=0.52)
    for y, value in enumerate(values):
        ax.text(value + 0.18, y, f"{value:.1f}%", va="center", color=INK, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.22)
    ax.set_xlabel("Share of the perfect-foresight profit, 730 days")
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    titled(
        fig,
        "Where the next euro is",
        f"A 1 MW / 2 MWh battery on DE-LU day-ahead, June 2024 to May 2026. Each "
        f"bar is what this battery, trading on this forecast, would gain. The "
        f"forecast is worth two orders of magnitude more than the warranty cap: "
        f"EUR {ceiling - traded:,.0f} over the two years against EUR "
        f"{no_cap - traded:,.0f}.",
        0.72,
    )
    fig.savefig(OUT / "next_euro.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    example_day()
    calibration()
    decision_value()
    next_euro()
    print(f"wrote {', '.join(p.name for p in sorted(OUT.glob('*.png')))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
