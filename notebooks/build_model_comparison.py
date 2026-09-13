"""Build and execute notebooks/model_comparison.ipynb from the comparison run.

    uv run --extra eda python notebooks/build_model_comparison.py

The notebook reads the walk-forward forecasts in
data/processed/forecasts/comparison/, scores the validation window, and
records why the production model was chosen. The decision text lives in
``DECISION`` below so the reasoning is versioned with the code that produced
the evidence; every number in it appears in the notebook's own output.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "notebooks" / "model_comparison.ipynb"

INTRO = """# Price forecast model comparison

Which model should produce the day-ahead price forecasts on the dashboard, and why?

**Setup.** Every model forecasts each quarter-hour of the next local day at 11:40 on
the day before, using only information published by then (see
`src/forecasting/information.py`). Forecasts are walk-forward: a model never sees
the day it forecasts. Scores cover the validation window, 1 June 2024 to 31 May
2026. The hold-out from 1 June 2026 is not touched here.

**Candidates.**

| Model | Idea | Reference |
|---|---|---|
| `lear` | Sparse linear model on arcsinh-scaled prices, conformal ranges | LEAR, Lago et al. 2021 |
| `lightgbm_quantile` | One gradient-boosted model per quantile, pinball loss | Widely used quantile GBM |
| `lightgbm_conformal` | Gradient-boosted median model, conformal ranges per hour | Split conformal prediction |
| `quantile_forest` | Random forest keeping leaf prices, any quantile | Meinshausen 2006 |
| `mstl` | Daily and weekly seasonal decomposition of recent prices | statsforecast MSTL |
| `qra` | Regularized quantile regression on the other models' medians | Nowotarski and Weron 2015; Uniejewski and Weron 2021 |
| `naive_previous_day`, `seasonal_naive_previous_week` | Baselines to beat | Phase 1 |

**Selection rules** (docs/decisions.md): lowest mean pinball loss on the validation
window; honest coverage of the 50%, 80% and 90% ranges; performance on the
15-minute months, negative-price days and spike days; a Diebold-Mariano test so a
win is not noise; run time that fits the 11:40 deadline; a near tie goes to the
simpler, better calibrated model. Phase 4 re-checks the top models on trading profit.
"""

SETUP = '''import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import REPO_ROOT, load_settings
from src.forecasting.base import quantile_column
from src.forecasting.evaluate import daily_pinball, diebold_mariano, pinball, score, segment_scores
from src.forecasting.walkforward import day_range

warnings.filterwarnings("ignore")
settings = load_settings()
QUANTILES = settings.forecasting.quantiles
FORECASTS = settings.data.processed_path / "forecasts" / "comparison"
ORDER = ["lear", "lightgbm_quantile", "lightgbm_conformal", "quantile_forest", "mstl", "qra",
         "naive_previous_day", "seasonal_naive_previous_week"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
                           "#898781", "#c3c2b7"], strict=True))
plt.rcParams.update({"figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e1e0d9", "font.size": 9})

runs = {}
for name in ORDER:
    path = FORECASTS / f"{name}.parquet"
    if path.exists():
        table = pd.read_parquet(path)
        runs[name] = table[[d >= settings.evaluation.validation_start for d in table["target_day"]]]
print({name: f"{t['target_day'].nunique()} days" for name, t in runs.items()})
'''

OVERALL = '''rows = {}
naive = score(runs["naive_previous_day"], QUANTILES)["mean pinball"]
for name, table in runs.items():
    s = score(table, QUANTILES)
    valid = table[table["actual"].notna()]
    crossings = int(table["crossings_repaired"].groupby(table["target_day"]).first().sum())
    rows[name] = {
        "mean pinball": round(s["mean pinball"], 2),
        "vs naive": f"{s['mean pinball'] / naive - 1:+.1%}",
        "MAE of median": round(s["MAE of median"], 2),
        "RMSE of median": round(s["RMSE of median"], 2),
        "coverage 50%": f"{s['coverage 50%']:.1%}",
        "coverage 80%": f"{s['coverage 80%']:.1%}",
        "coverage 90%": f"{s['coverage 90%']:.1%}",
        "median width 90%": round(float((valid["q95"] - valid["q05"]).median()), 1),
        "periods with crossed quantiles": crossings,
    }
overall = pd.DataFrame(rows).T.sort_values("mean pinball")
overall
'''

SEGMENTS = '''segments = pd.DataFrame({
    name: segment_scores(table, settings)["mean pinball"].round(2) for name, table in runs.items()
}).T
segments = segments.drop(columns=["Validation window"]).loc[overall.index]
segments
'''

MONTHLY = '''fig, ax = plt.subplots(figsize=(10, 4))
for name, table in runs.items():
    daily = daily_pinball(table, QUANTILES)
    monthly = daily.groupby([pd.Timestamp(d).to_period("M") for d in daily.index]).mean()
    ax.plot(monthly.index.to_timestamp(), monthly.to_numpy(), label=name, color=COLORS[name],
            linewidth=2 if not name.startswith(("naive", "seasonal")) else 1.2)
ax.set_title("Mean pinball loss by month, validation window", loc="left")
ax.set_ylabel("EUR/MWh")
ax.legend(ncols=4, frameon=False, fontsize=8)
plt.show()
'''

RELIABILITY = '''fig, ax = plt.subplots(figsize=(6, 5))
ax.plot([0, 1], [0, 1], color="#c3c2b7", linewidth=1)
for name, table in runs.items():
    valid = table[table["actual"].notna()]
    empirical = [float(np.mean(valid["actual"] <= valid[quantile_column(q)])) for q in QUANTILES]
    ax.plot(QUANTILES, empirical, marker="o", color=COLORS[name], label=name, linewidth=1.5)
ax.set_xlabel("Nominal quantile")
ax.set_ylabel("Share of prices at or below the forecast")
ax.set_title("Reliability: on the diagonal is calibrated", loc="left")
ax.legend(frameon=False, fontsize=8)
plt.show()
'''

RELIABILITY_TABLE = '''shares = {}
for name, table in runs.items():
    valid = table[table["actual"].notna()]
    shares[name] = {
        f"q{round(q * 100):02d}": f"{np.mean(valid['actual'] <= valid[quantile_column(q)]):.1%}"
        for q in QUANTILES
    }
pd.DataFrame(shares).T.loc[overall.index]
'''

BY_HOUR = '''fig, ax = plt.subplots(figsize=(10, 4))
for name, table in runs.items():
    valid = table[table["actual"].notna()]
    actual = valid["actual"].to_numpy()
    loss = np.mean([pinball(actual, valid[quantile_column(q)].to_numpy(), q) for q in QUANTILES], axis=0)
    hours = pd.DatetimeIndex(valid.index).tz_convert(settings.market.timezone).hour
    ax.plot(range(24), pd.Series(loss).groupby(hours).mean().to_numpy(), color=COLORS[name], label=name)
ax.set_xticks(range(0, 24, 3))
ax.set_xlabel("Local delivery hour")
ax.set_ylabel("Mean pinball loss, EUR/MWh")
ax.set_title("Where in the day the errors are", loc="left")
ax.legend(ncols=4, frameon=False, fontsize=8)
plt.show()
'''

SIGNIFICANCE = '''def dm_table(reference):
    """Diebold-Mariano test of every model against ``reference`` on daily pinball loss."""
    ref = daily_pinball(runs[reference], QUANTILES)
    calendar = day_range(min(ref.index), max(ref.index))
    rows = {}
    for name, table in runs.items():
        if name == reference:
            continue
        other = daily_pinball(table, QUANTILES).reindex(calendar).to_numpy()
        statistic, p_value = diebold_mariano(other, ref.reindex(calendar).to_numpy())
        rows[name] = {"DM statistic, positive means worse": round(statistic, 2),
                      "p-value": f"{p_value:.4f}"}
    return pd.DataFrame(rows).T


best = overall.index[0]
chosen = settings.forecasting.production_model
print(f"Lowest mean pinball loss: {best}. Production model: {chosen}.")
dm_table(best)
'''

SIGNIFICANCE_CHOSEN = '''dm_table(chosen)
'''

RUNTIME = '''import mlflow

mlflow.set_tracking_uri(f"sqlite:///{REPO_ROOT / 'mlflow.db'}")
found = mlflow.search_runs(experiment_names=["price-forecast-comparison"], order_by=["start_time DESC"])
columns = {"tags.model": "model", "metrics.run_seconds": "run seconds",
           "params.refit_every_days": "refit every days"}
if found.empty:
    runtime = pd.DataFrame(columns=list(columns.values()))
else:
    runtime = found.drop_duplicates("tags.model")[list(columns)].rename(columns=columns).set_index("model")
runtime
'''

DECISION = """## Decision: LightGBM conformal goes to production

| model | mean pinball | vs naive | 90% range coverage | median 90% width, EUR/MWh | crossed periods |
|---|---|---|---|---|---|
| lightgbm_quantile | 5.00 | -47.9% | 73.3% | 40.7 | 47,184 |
| **lightgbm_conformal** | **5.11** | **-46.8%** | **85.6%** | **58.7** | **0** |
| quantile_forest | 5.18 | -46.1% | 94.6% | 82.0 | 0 |
| qra | 5.21 | -45.8% | 88.9% | 72.5 | 3,522 |
| mstl | 7.74 | -19.4% | 94.4% | 128.2 | 0 |
| lear | 8.06 | -16.1% | 84.9% | 82.5 | 0 |
| naive_previous_day | 9.61 | +0.0% | 86.4% | 128.6 | 0 |
| seasonal_naive_previous_week | 11.49 | +19.6% | 85.4% | 136.4 | 0 |

All figures are from the tables above.

**Why LightGBM conformal.**

1. **It ties for the best accuracy.** Tested against it, LightGBM quantile gives p = 0.11 and the
   quantile forest p = 0.37: neither difference is significant. It is 46.8% better than naive.
2. **Its ranges are close to honest.** Prices fall below its q05 to q95 forecasts 7.0%, 11.5%, 25.4%,
   50.3%, 74.0%, 87.8% and 92.7% of the time, against targets of 5% to 95%.
3. **Its ranges are sharp.** The median 90% range is 58.7 EUR/MWh wide, against 82.0 for the quantile
   forest, which covers more than it should, and 72.5 for QRA.
4. **It never crosses quantiles and is a single model**, simple to run before the 11:40 deadline.

**Why not the others.**

- **LightGBM quantile** has the lowest loss, but its ranges are overconfident: its 90% range covers only
  73.3% of prices and its quantiles crossed in two thirds of all periods. On the dashboard that would
  understate spike and negative-price risk, the most expensive error for a battery.
- **QRA** is well calibrated, but it is not more accurate than LightGBM conformal (p = 0.07), it is
  worse on spike days, 8.74 against 8.20, and it needs all five member models running every day.
- **The quantile forest** is the conservative alternative: accurate, never crossing, but wider than needed.

**Weaknesses to watch.** The tails are slightly narrow, 85.6% coverage for a 90% range. On spike days
LightGBM quantile is better, 7.78 against 8.20. Hyperparameters are defaults, not tuned. Phase 4
re-checks the top three models on trading profit, and Phase 6 monitors coverage drift.
"""


def build() -> nbformat.NotebookNode:
    nb = nbformat.v4.new_notebook()
    md, code = nbformat.v4.new_markdown_cell, nbformat.v4.new_code_cell
    nb.cells = [
        md(INTRO),
        code(SETUP),
        md("## Overall scores on the validation window\n\nSorted by mean pinball loss, "
           "lower is better. Coverage should be close to the nominal level."),
        code(OVERALL),
        md("## Mean pinball loss by segment\n\nThe 15-minute months, negative-price days "
           "and spike days are where trading value concentrates."),
        code(SEGMENTS),
        md("## Stability over time"),
        code(MONTHLY),
        md("## Calibration"),
        code(RELIABILITY),
        md("Share of prices at or below each forecast quantile. A calibrated model matches "
           "the quantile level."),
        code(RELIABILITY_TABLE),
        md("## Error by hour of day"),
        code(BY_HOUR),
        md("## Is a win significant?\n\nDiebold-Mariano test on daily mean pinball loss, "
           "Newey-West variance over seven days. First against the lowest-loss model:"),
        code(SIGNIFICANCE),
        md("Then against the production model:"),
        code(SIGNIFICANCE_CHOSEN),
        md("## Run time"),
        code(RUNTIME),
        md(DECISION),
    ]
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    return nb


def main() -> int:
    NOTEBOOK.write_text(nbformat.writes(build()), encoding="utf-8")
    return subprocess.call(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--inplace", "--ExecutePreprocessor.timeout=1200", str(NOTEBOOK)],
        cwd=ROOT,
    )


if __name__ == "__main__":
    raise SystemExit(main())
