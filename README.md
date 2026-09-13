# energy-price-forecasting-trading

Probabilistic electricity price forecasting (DE-LU day-ahead) and battery arbitrage trading. LightGBM quantile models, MILP dispatch, walk-forward backtesting, React dashboard with model-health monitoring.

## Status

**Phase 0 of 9: data and exploratory analysis.** The project is built in reviewed phases. The plan is in [the handoff doc](docs/Price_Forecasting_Trading_Project_Handoff_v2.md) and the background in [the concepts primer](docs/Price_Forecasting_Trading_Concepts_Primer.md).

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

```bash
uv sync --extra dev --extra eda
uv run python -m src.ingest.build_dataset     # download SMARD data, write data/processed/
uv run python notebooks/phase0_eda.py         # figures and tables in docs/figures/phase0/
uv run pytest
```

## Data

| Source | Used for | Licence |
|---|---|---|
| [SMARD.de](https://www.smard.de), Bundesnetzagentur | DE-LU day-ahead prices; load, wind and solar actuals and day-ahead forecasts | CC BY 4.0 |

Data: Bundesnetzagentur | SMARD.de, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Downloaded data is not committed; the ingest command rebuilds it. Why SMARD rather than OPSD or ENTSO-E is recorded in [docs/decisions.md](docs/decisions.md).

## Layout

```
config/settings.yaml   zone, resolution, SMARD series, hold-out, battery, quantiles
src/config.py          validated settings loader
src/timegrid.py        UTC and local delivery-day grid (DST-safe)
src/ingest/            SMARD client, dataset builder, data-quality checks
src/...                forecasting, trading, health, export, narration (later phases)
notebooks/             exploration scripts; logic lives in src/
tests/                 network-free tests, including DST and granularity checks
docs/                  decisions, production notes, learning notes, figures, mockup
```

## Documentation

- [docs/data_guide.md](docs/data_guide.md): the data column by column, what it shows, and how it feeds features, forecasts and trading
- [docs/decisions.md](docs/decisions.md): dated decisions and the alternatives considered
- [docs/production_notes.md](docs/production_notes.md): the failure catalogue with measured results
- [docs/learning_notes.md](docs/learning_notes.md): concept explanations, phase by phase
- [docs/mockup/](docs/mockup/): the approved dashboard design
