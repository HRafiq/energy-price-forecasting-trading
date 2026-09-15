# Short names for the pipeline commands in the README. `make help` lists them.
UV := uv run
MLFLOW_PORT ?= 5001
SUITES := validation decision-value synthetic degradation attribution

.PHONY: help setup data entsoe forecast backtest holdout report figures export api web dashboard health experiments mlflow test

help:
	@echo "setup     install dependencies with uv"
	@echo "data      download SMARD prices, weather, gas and carbon; build model inputs"
	@echo "entsoe    ENTSO-E price cross-check and German imbalance prices (needs .env token)"
	@echo "forecast  baselines and the eight-model walk-forward comparison"
	@echo "backtest  battery strategies, backtest suites and the outage stress test"
	@echo "holdout   hold-out forecasts and backtest, once, after freezing every choice"
	@echo "report    results report and notebooks"
	@echo "figures   README figures in docs/img/"
	@echo "export    dashboard artifacts from the backtest outputs"
	@echo "api       dashboard API on port 8000"
	@echo "web       React dev server on port 5173, proxying the API"
	@echo "dashboard build the React app and serve it with the API on port 8000"
	@echo "health    drift monitor (M2) and incident log from saved forecasts"
	@echo "experiments Phase 6 failure experiments: M1, D1, D5, then M2"
	@echo "mlflow    MLflow UI for the experiment runs at http://127.0.0.1:$(MLFLOW_PORT)"
	@echo "test      lint, type checks and tests"

setup:
	uv sync --extra dev --extra eda

data:
	$(UV) python -m src.ingest.build_dataset
	$(UV) python -m src.ingest.fuels
	$(UV) python -m src.ingest.open_meteo
	$(UV) python -m src.ingest.build_inputs

entsoe:
	$(UV) python -m src.ingest.entsoe --what day-ahead
	$(UV) python -m src.ingest.entsoe --what imbalance
	$(UV) python -m src.ingest.entsoe --what report

forecast:
	$(UV) python -m src.forecasting.run_baselines
	$(UV) python -m src.forecasting.run_comparison

backtest:
	$(UV) python -m src.trading.run_strategies
	for suite in $(SUITES); do $(UV) python -m src.trading.backtest --suite $$suite || exit 1; done
	$(UV) python -m src.health.experiments.t4_imbalance

holdout:
	$(UV) python -m src.forecasting.run_holdout --confirm-holdout
	$(UV) python -m src.trading.backtest --suite holdout --confirm-holdout

report:
	$(UV) python -m src.trading.phase4_report
	$(UV) --extra eda python notebooks/build_model_comparison.py
	$(UV) --extra eda python notebooks/build_backtest_report.py

figures:
	$(UV) --extra eda python notebooks/build_readme_figures.py

export:
	$(UV) python -m src.export.artifacts

api:
	$(UV) uvicorn api.main:app --port 8000

web:
	cd frontend && npm install && npm run dev

dashboard:
	cd frontend && npm install && npm run build
	$(UV) uvicorn api.main:app --port 8000

health:
	$(UV) python -m src.health.experiments.m2_drift

experiments:
	$(UV) python -m src.health.experiments.m1_regime_shift
	$(UV) python -m src.health.experiments.d1_missing_weather
	$(UV) python -m src.health.experiments.d5_deadline
	$(MAKE) health

mlflow:
	$(UV) mlflow ui --backend-store-uri sqlite:///mlflow.db --port $(MLFLOW_PORT)

test:
	$(UV) ruff check .
	$(UV) mypy
	$(UV) pytest
