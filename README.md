# energy-price-forecasting-trading

Probabilistic electricity price forecasting (DE-LU day-ahead, 15-minute products) and battery arbitrage trading. Six forecasting models compared, with LightGBM and conformal ranges in production; MILP battery dispatch with perfect-foresight, median, mean and quantile-aware strategies; a walk-forward backtest with error attribution and a one-time hold-out; a React dashboard with model-health monitoring to come.

## Status

**Phase 4 of 9: backtest, decision value and error economics.** Phases 0 to 3 are complete: the 15-minute dataset, baselines and leak-free walk-forward evaluation, the forecasting model comparison, and battery dispatch. The project is built in reviewed phases.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync --extra dev --extra eda
uv run python -m src.ingest.build_dataset     # download SMARD data, write data/processed/
uv run python notebooks/phase0_eda.py         # figures and tables in docs/figures/phase0/
uv run python -m src.forecasting.run_baselines # baseline results in docs/results/
uv run python -m src.ingest.fuels             # gas and carbon prices
uv run python -m src.ingest.open_meteo        # archived weather forecasts
uv run python -m src.ingest.build_inputs      # model-input dataset
uv run python -m src.forecasting.run_comparison  # six models, tracked in mlflow.db
uv run python notebooks/build_model_comparison.py  # evaluation notebook
uv run python -m src.forecasting.production --day 2026-05-31  # one production forecast
uv run python -m src.trading.run_strategies   # battery strategies over the validation window
uv run python -m src.ingest.entsoe --what day-ahead  # ENTSO-E prices; needs ENTSOE_API_KEY in .env
uv run python -m src.ingest.entsoe --what imbalance  # German imbalance prices
uv run python -m src.trading.backtest --suite validation  # also decision-value, synthetic, degradation, attribution
uv run python -m src.health.experiments.t4_imbalance  # outage and imbalance-price stress test
uv run python -m src.forecasting.run_holdout --confirm-holdout  # hold-out forecasts, once, after freezing
uv run python -m src.trading.backtest --suite holdout --confirm-holdout
uv run python -m src.trading.phase4_report      # docs/results/phase4_backtest.md
uv run --extra eda python notebooks/build_backtest_report.py  # backtest notebook
uv run pytest
```

## Data

| Source | Used for | Licence |
|---|---|---|
| [SMARD.de](https://www.smard.de), Bundesnetzagentur | DE-LU day-ahead prices; load, wind and solar actuals and day-ahead forecasts | CC BY 4.0 |
| [Open-Meteo](https://open-meteo.com) Previous Runs API | Weather forecasts issued two days ahead, 12 German points, from March 2024 | CC BY 4.0; free API for non-commercial use |
| Yahoo Finance, ticker TTF=F | Daily Dutch TTF gas futures settlement | Yahoo terms, personal use; not redistributed |
| [EEX](https://www.eex.com) EU ETS primary auctions | Daily EU carbon allowance auction price | EEX public reports; not redistributed |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | Day-ahead price cross-check; German imbalance prices | Platform terms of use; free API token; not redistributed |

Data: Bundesnetzagentur | SMARD.de, and weather data by Open-Meteo.com, both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Downloaded data is not committed; the ingest command rebuilds it.

## Layout

```
config/settings.yaml   zone, resolution, SMARD series, hold-out, battery, quantiles
src/config.py          validated settings loader
src/timegrid.py        UTC and local delivery-day grid (DST-safe)
src/ingest/            SMARD client, dataset builder, data-quality checks
src/forecasting/       information set, walk-forward harness, models, production forecaster
src/trading/           battery model, MILP dispatch, settlement, strategies, backtest, attribution
src/health/            failure experiments: M3 leakage, T4 imbalance risk
src/...                export, narration (later phases)
notebooks/             exploration scripts; logic lives in src/
tests/                 network-free tests, including DST and granularity checks
docs/                  decisions, production notes, learning notes, figures, mockup
```

## Documentation

- [docs/data_guide.md](docs/data_guide.md): the data column by column, what it shows, and how it feeds features, forecasts and trading
- [notebooks/model_comparison.ipynb](notebooks/model_comparison.ipynb): six forecasting models compared, and why LightGBM with conformal ranges was chosen
- [docs/results/phase1_baselines.md](docs/results/phase1_baselines.md): baseline forecast results, the numbers to beat
- [docs/results/m3_leakage.md](docs/results/m3_leakage.md): how much post-gate data would flatter a backtest
- [docs/results/phase3_strategies.md](docs/results/phase3_strategies.md): battery strategies over the validation window, against the perfect-foresight ceiling
- [notebooks/backtest_report.ipynb](notebooks/backtest_report.ipynb): the backtest in charts, from validation to the hold-out
- [docs/results/phase4_backtest.md](docs/results/phase4_backtest.md): hold-out and validation profit, decision value, wear pricing and where the gap is lost
- [docs/results/t4_imbalance.md](docs/results/t4_imbalance.md): what an outage costs when the shortfall is settled at the imbalance price
- [docs/results/entsoe_price_check.md](docs/results/entsoe_price_check.md): ENTSO-E day-ahead prices against SMARD, and German imbalance prices
- [docs/production_notes.md](docs/production_notes.md): the failure catalogue with measured results
- [docs/mockup/](docs/mockup/): the approved dashboard design
