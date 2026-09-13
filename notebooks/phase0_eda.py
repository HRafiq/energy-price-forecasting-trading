# %% [markdown]
# # Phase 0 EDA: DE-LU day-ahead prices
#
# Run from the repository root after building the dataset:
#
#     uv run --extra eda python notebooks/phase0_eda.py
#
# Writes figures and a summary table to docs/figures/phase0/. The file uses the
# percent cell format, so it opens as a notebook in VS Code or Jupyter, but it
# runs as plain Python and its outputs are reproducible from the command line.
#
# The figures cover the hourly-product era only: up to the switch to 15-minute
# products or the hold-out start, whichever comes first. The dataset is
# quarter-hourly, so quarter-hours are averaged back to hours here. The hold-out
# is never explored: what we see would shape modeling choices (T6).

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.figure import Figure

from src.config import REPO_ROOT, load_settings

plt.switch_backend("Agg")

settings = load_settings()
TZ = settings.market.timezone
OUT = REPO_ROOT / "docs" / "figures" / "phase0"
OUT.mkdir(parents=True, exist_ok=True)

PRICE = "price_eur_mwh"
SPIKE_EUR_MWH = 200.0  # descriptive EDA threshold, not a model parameter
MINUS = "\N{MINUS SIGN}"  # typographic minus for chart text

raw = pd.read_parquet(settings.data.dataset_path).drop(columns="price_product_minutes")
# An hour with any quarter-hour missing stays missing instead of averaging three.
full = raw.resample("h").mean().where(raw.resample("h").count() == 4)
eda_end_day = min(
    settings.evaluation.holdout_start, settings.market.quarter_hour_products_from
)
df = full.loc[full.index < settings.market.local_midnight_utc(eda_end_day)].copy()
local = pd.DatetimeIndex(df.index).tz_convert(TZ)
df["local_date"] = local.date
df["local_hour"] = local.hour
df["month"] = local.month
df["year"] = local.year
first_day, last_day = local[0].date(), local[-1].date()

# Descriptive regime labels for plots only; boundaries are approximate.
REGIMES = [
    ("Before the gas crisis", "2018-10-01", "2021-07-01"),
    ("Gas crisis", "2021-07-01", "2023-07-01"),
    ("After the crisis", "2023-07-01", str(eda_end_day)),
]

# %%
# Chart styling: reference palette, light surface, recessive chrome.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "legend.frameon": False,
        "legend.labelcolor": INK_2,
    }
)


def header(fig: Figure, title: str, subtitle: str, top: float = 0.80) -> None:
    fig.subplots_adjust(top=top)
    fig.text(0.07, 0.965, title, fontsize=12, color=INK, va="top", weight="bold")
    fig.text(0.07, 0.905, subtitle, fontsize=9, color=INK_2, va="top")


def save(fig: Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def local_ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day).tz_localize(TZ).tz_convert("UTC")


period_note = f"{first_day:%d %b %Y} to {last_day:%d %b %Y}, hourly-product era"

# %% [markdown]
# ## 1. Daily average price

# %%
daily = df[PRICE].groupby(df["local_date"]).mean()
daily.index = pd.to_datetime(daily.index)
fig, ax = plt.subplots(figsize=(10, 4))
ax.axhline(0, color=AXIS, linewidth=0.8)
ax.plot(daily.index, daily.to_numpy(), color=SERIES[0], linewidth=1.0)
peak_day = daily.idxmax()
ax.annotate(
    f"Highest day {peak_day:%d %b %Y}: €{daily.max():.0f}/MWh",
    xy=(peak_day, daily.max()),
    xytext=(12, -2),
    textcoords="offset points",
    color=INK_2,
    va="top",
)
ax.set_ylabel("€/MWh")
header(fig, "Daily average day-ahead price, DE-LU", period_note)
save(fig, "01_daily_price.png")

# %% [markdown]
# ## 2. Distribution of hourly prices

# %%
price = df[PRICE].dropna()
lo, hi = -150, 600
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(
    price[(price >= lo) & (price <= hi)],
    bins=np.arange(lo, hi + 5, 5),
    color=SERIES[0],
    rwidth=0.8,
)
ax.axvline(0, color=INK_2, linewidth=0.8)
ax.set_xlabel("€/MWh, €5 bins")
ax.set_ylabel("Hours")
header(
    fig,
    "Distribution of hourly prices",
    f"{(price < 0).mean():.1%} of hours are negative. "
    f"{int((price < lo).sum())} hours below {MINUS}€{abs(lo)} and "
    f"{int((price > hi).sum())} above €{hi} fall outside the chart. {period_note}.",
)
save(fig, "02_price_distribution.png")

# %% [markdown]
# ## 3. The two tails by year

# %%
yearly_tails = df.groupby("year")[PRICE].agg(
    negative=lambda s: int((s < 0).sum()),
    spike=lambda s: int((s > SPIKE_EUR_MWH).sum()),
)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
panels = [
    ("negative", "Hours with a negative price"),
    ("spike", f"Hours above €{SPIKE_EUR_MWH:.0f}/MWh"),
]
for ax, (column, label) in zip(axes, panels, strict=True):
    ax.bar(
        yearly_tails.index.astype(str), yearly_tails[column], color=SERIES[0], width=0.7
    )
    ax.set_title(label, loc="left", fontsize=10)
header(
    fig,
    "The two tails, by year",
    f"{first_day.year} starts {first_day:%d %b} and {last_day.year} ends "
    f"{last_day:%d %b}, so both are partial years.",
    top=0.75,
)
save(fig, "03_tails_by_year.png")

# %% [markdown]
# ## 4. Daily shape by season, after the crisis

# %%
recent_from = local_ts(REGIMES[2][1])
recent = df.loc[df.index >= recent_from]
season_of = {
    12: "Winter",
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
}
profile = (
    recent.groupby([recent["month"].map(season_of), "local_hour"])[PRICE]
    .median()
    .unstack(0)
)
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.axhline(0, color=AXIS, linewidth=0.8)
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]
for i, season in enumerate(SEASONS):
    ax.plot(profile.index, profile[season], color=SERIES[i], linewidth=2, label=season)
# Direct labels at the line ends, nudged apart so they never overlap.
MIN_GAP = 6.0
label_y = sorted((float(profile[s].iloc[-1]), s) for s in SEASONS)
for k in range(1, len(label_y)):
    if label_y[k][0] - label_y[k - 1][0] < MIN_GAP:
        label_y[k] = (label_y[k - 1][0] + MIN_GAP, label_y[k][1])
for y, season in label_y:
    ax.annotate(
        season,
        xy=(23, y),
        xytext=(6, 0),
        textcoords="offset points",
        color=INK_2,
        va="center",
    )
ax.set_xticks(range(0, 24, 3))
ax.set_xlim(0, 25.5)
ax.set_xlabel("Local delivery hour")
ax.set_ylabel("Median €/MWh")
ax.legend(loc="upper left", ncols=4)
header(
    fig,
    "Median price by hour of day and season",
    f"{recent_from.tz_convert(TZ):%b %Y} to {last_day:%b %Y}, after the gas crisis",
)
save(fig, "04_daily_shape_by_season.png")

# %% [markdown]
# ## 5. Price against residual load, by regime

# %%
residual_gw = df["residual_load_actual_mw"] / 1000.0
valid = residual_gw.notna() & df[PRICE].notna()
x_lo, x_hi = residual_gw[valid].quantile([0.001, 0.999])
cmap = LinearSegmentedColormap.from_list("blues", BLUES)
fig, axes = plt.subplots(1, 3, figsize=(12, 4.4), sharex=True, sharey=True)
fig.subplots_adjust(right=0.88, wspace=0.12)
hexes = []
for ax, (label, a, b) in zip(axes, REGIMES, strict=True):
    mask = valid & (df.index >= local_ts(a)) & (df.index < local_ts(b))
    hb = ax.hexbin(
        residual_gw[mask],
        df.loc[mask, PRICE],
        gridsize=40,
        extent=(x_lo, x_hi, lo, hi),
        norm=LogNorm(vmin=1),
        mincnt=1,
        cmap=cmap,
        linewidths=0,
    )
    hexes.append(hb)
    ax.grid(False)
    last_month = pd.Timestamp(b) - pd.Timedelta(days=1)
    ax.set_title(
        f"{label}\n{pd.Timestamp(a):%b %Y} to {last_month:%b %Y}",
        loc="left",
        fontsize=10,
    )
    ax.set_xlabel("Residual load, GW")
vmax = max(float(h.get_array().max()) for h in hexes)
for h in hexes:
    h.set_norm(LogNorm(vmin=1, vmax=vmax))
axes[0].set_ylabel("€/MWh")
cax = fig.add_axes((0.905, 0.14, 0.012, 0.56))
fig.colorbar(hexes[-1], cax=cax, label="Hours per cell, log scale")
header(
    fig,
    "Price against residual load",
    "Residual load is load minus wind and solar output. Actual values, used here "
    "to explain price formation, never as a forecasting feature. "
    f"View limited to {MINUS}€{abs(lo)} to €{hi}.",
    top=0.74,
)
save(fig, "05_price_vs_residual_load.png")

# %% [markdown]
# ## 6. Summary tables


# %%
def md_table(frame: pd.DataFrame) -> str:
    cols = [str(frame.index.name or "")] + [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in frame.astype(object).iterrows():
        lines.append("| " + " | ".join([str(idx), *map(str, row.tolist())]) + " |")
    return "\n".join(lines)


def yearly_stats(s: pd.Series) -> pd.Series:
    return pd.Series(
        {
            "hours": int(s.count()),
            "mean": round(s.mean(), 1),
            "median": round(s.median(), 1),
            "std": round(s.std(), 1),
            "p01": round(s.quantile(0.01), 1),
            "p99": round(s.quantile(0.99), 1),
            "min": round(s.min(), 1),
            "max": round(s.max(), 1),
            "negative hours": int((s < 0).sum()),
            f"hours > €{SPIKE_EUR_MWH:.0f}": int((s > SPIKE_EUR_MWH).sum()),
        },
        dtype=object,
    )


price_table = df.groupby("year")[PRICE].apply(yearly_stats).unstack()
price_table.index.name = "year"

pairs = [
    ("Wind onshore", "wind_onshore_actual_mw", "wind_onshore_forecast_mw"),
    ("Wind offshore", "wind_offshore_actual_mw", "wind_offshore_forecast_mw"),
    ("Solar", "solar_actual_mw", "solar_forecast_mw"),
    ("Load", "load_actual_mw", "load_forecast_mw"),
    ("Residual load", "residual_load_actual_mw", "residual_load_forecast_mw"),
]
rows = {}
for label, actual_col, forecast_col in pairs:
    both = df[[actual_col, forecast_col]].dropna() / 1000.0
    err = both[forecast_col] - both[actual_col]
    rows[label] = {
        "mean actual GW": round(both[actual_col].mean(), 2),
        "MAE GW": round(err.abs().mean(), 2),
        "bias GW": round(err.mean(), 2),
        "correlation": round(both[actual_col].corr(both[forecast_col]), 3),
    }
forecast_table = pd.DataFrame(rows).T
forecast_table.index.name = "series"

corr_rows = {}
for label, a, b in REGIMES:
    m = valid & (df.index >= local_ts(a)) & (df.index < local_ts(b))
    corr_rows[label] = {
        "hours": int(m.sum()),
        "mean price": round(df.loc[m, PRICE].mean(), 1),
        "Spearman price vs residual load": round(
            # Spearman = Pearson on ranks; avoids a scipy dependency.
            df.loc[m, PRICE].rank().corr(residual_gw[m].rank()),
            3,
        ),
    }
regime_table = pd.DataFrame(corr_rows).T.astype({"hours": int})
regime_table.index.name = "regime"

summary = f"""# Phase 0 EDA summary

Generated by `notebooks/phase0_eda.py`. Data: Bundesnetzagentur | SMARD.de,
CC BY 4.0. Period: {period_note}. Prices in €/MWh.

## Hourly price by year

{md_table(price_table)}

## Grid-operator day-ahead forecasts against actuals

A sanity check that the configured SMARD series are what we think they are.
Bias is forecast minus actual.

{md_table(forecast_table)}

## Price and residual load by regime

{md_table(regime_table)}
"""
(OUT / "summary.md").write_text(summary, encoding="utf-8")
print(summary)

# %% [markdown]
# ## 7. Tables for the data guide (docs/data_guide.md)

# %%
DATA_COLUMNS = list(full.columns)


def column_stats(frame: pd.DataFrame) -> pd.DataFrame:
    rows = {}
    for col in DATA_COLUMNS:
        is_price = col == PRICE
        values = frame[col].dropna() / (1.0 if is_price else 1000.0)
        rows[col] = {
            "unit": "€/MWh" if is_price else "GW",
            "missing hours": int(frame[col].isna().sum()),
            "mean": round(values.mean(), 1),
            "p01": round(values.quantile(0.01), 1),
            "median": round(values.median(), 1),
            "p99": round(values.quantile(0.99), 1),
            "min": round(values.min(), 1),
            "max": round(values.max(), 1),
        }
    table = pd.DataFrame(rows).T
    table.index.name = "column"
    return table


def daily_spread(prices: pd.Series) -> pd.Series:
    ordered = np.sort(prices.dropna().to_numpy())
    return pd.Series(
        {
            "max_minus_min": ordered[-1] - ordered[0],
            "top2_minus_bottom2": ordered[-2:].mean() - ordered[:2].mean(),
        }
    )


spreads = df.groupby("local_date")[PRICE].apply(daily_spread).unstack()
spreads["year"] = [d.year for d in spreads.index]
spread_table = (
    spreads.groupby("year")
    .agg(
        **{
            "days": ("max_minus_min", "size"),
            "mean max-min": ("max_minus_min", "mean"),
            "mean top-2 minus bottom-2": ("top2_minus_bottom2", "mean"),
            "p90 top-2 minus bottom-2": (
                "top2_minus_bottom2",
                lambda s: s.quantile(0.9),
            ),
        }
    )
    .round(1)
)

BANDS = [0, 6, 10, 16, 20, 24]
BAND_LABELS = ["00-05", "06-09", "10-15", "16-19", "20-23"]
band = pd.cut(df["local_hour"], bins=BANDS, right=False, labels=BAND_LABELS)
before_crisis = df.index < local_ts(REGIMES[0][2])
after_crisis = df.index >= recent_from


def band_shares(mask: pd.Series) -> pd.Series:
    return band[mask].value_counts(normalize=True).reindex(BAND_LABELS)


negative = df[PRICE] < 0
spike = df[PRICE] > SPIKE_EUR_MWH
timing_table = pd.DataFrame(
    {
        "negative hours, before crisis": band_shares(negative & before_crisis),
        "negative hours, after crisis": band_shares(negative & after_crisis),
        f"hours above €{SPIKE_EUR_MWH:.0f}, after crisis": band_shares(
            spike & after_crisis
        ),
    }
).map(lambda x: f"{x:.0%}")
timing_table.index.name = "local hours"

season_rows = {}
for season in SEASONS:
    curve = profile[season]
    season_rows[season] = {
        "cheapest hour": f"{int(curve.idxmin()):02d}:00",
        "cheapest median": round(float(curve.min()), 1),
        "dearest hour": f"{int(curve.idxmax()):02d}:00",
        "dearest median": round(float(curve.max()), 1),
        "dearest minus cheapest": round(float(curve.max() - curve.min()), 1),
    }
season_table = pd.DataFrame(season_rows).T
season_table.index.name = "season"

RL_BINS = [-20, 0, 10, 20, 30, 40, 50, 60, 80]
rl_band = pd.cut(residual_gw, RL_BINS)
merit_columns: dict[str, pd.Series] = {}
for label, a, b in REGIMES:
    in_regime = valid & (df.index >= local_ts(a)) & (df.index < local_ts(b))
    grouped = df.loc[in_regime, PRICE].groupby(rl_band[in_regime], observed=False)
    merit_columns[f"{label}: median"] = grouped.median().round(1)
    if label == REGIMES[2][0]:
        merit_columns[f"{label}: p90"] = grouped.quantile(0.9).round(1)
        merit_columns[f"{label}: hours"] = grouped.size()
merit_table = pd.DataFrame(merit_columns)
merit_table = merit_table.astype(object).where(merit_table.notna(), "no hours")
merit_table.index = pd.Index(
    [f"{int(i.left)} to {int(i.right)}" for i in merit_table.index],
    name="residual load GW",
)

night = df["local_hour"] < 4
night_solar_note = (
    f"Measured solar between local 00:00 and 03:59 averages "
    f"{df.loc[night, 'solar_actual_mw'].mean():.1f} MW, maximum "
    f"{df.loc[night, 'solar_actual_mw'].max():.0f} MW. The forecast in the same "
    f"hours averages {df.loc[night, 'solar_forecast_mw'].mean():.1f} MW."
)
negative_residual_hours = int((residual_gw < 0).sum())

EXAMPLE_DAY = "2024-05-12"
example = df.loc[df["local_date"] == pd.Timestamp(EXAMPLE_DAY).date()]
example_local = pd.DatetimeIndex(example.index).tz_convert(TZ)
example_table = pd.DataFrame(
    {
        "price €/MWh": example[PRICE].round(1).to_numpy(),
        "load GW": (example["load_actual_mw"] / 1000).round(1).to_numpy(),
        "wind GW": (
            (example["wind_onshore_actual_mw"] + example["wind_offshore_actual_mw"])
            / 1000
        )
        .round(1)
        .to_numpy(),
        "solar GW": (example["solar_actual_mw"] / 1000).round(1).to_numpy(),
        "residual load GW": (example["residual_load_actual_mw"] / 1000)
        .round(1)
        .to_numpy(),
        "residual load forecast GW": (example["residual_load_forecast_mw"] / 1000)
        .round(1)
        .to_numpy(),
    },
    index=pd.Index([f"{t:%H:%M}" for t in example_local], name="local time"),
).iloc[::2]

guide_tables = f"""# Tables for the data guide

Generated by `notebooks/phase0_eda.py`. Period: {period_note}.

## Column statistics

{md_table(column_stats(df))}

## Daily price spread by year, €/MWh

{md_table(spread_table)}

## When the tails happen

{md_table(timing_table)}

## Daily shape by season, after the crisis, median €/MWh

{md_table(season_table)}

## Median price by residual load band, €/MWh

Bands include their upper edge. Actual residual load.

{md_table(merit_table)}

## Other facts

- Hours with negative actual residual load: {negative_residual_hours}
- {night_solar_note}

## Example day: {EXAMPLE_DAY}, every second hour

{md_table(example_table)}
"""
(OUT / "data_guide_tables.md").write_text(guide_tables, encoding="utf-8")
print(guide_tables)
