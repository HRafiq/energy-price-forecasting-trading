# Short names for the pipeline commands in the README. `make help` lists them.
UV := uv run
MLFLOW_PORT ?= 5001
AIRFLOW_PORT ?= 8080
AIRFLOW_VERSION ?= 3.3.1
# Delivery day for a one-off pipeline run; tomorrow by default.
DAY ?= $(shell date -v+1d +%F 2>/dev/null || date -d tomorrow +%F)
# Airflow keeps its database and logs here; the DAGs live in the repo.
AIRFLOW_ENV := AIRFLOW_HOME=$(CURDIR)/airflow_home \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False
AIRFLOW := $(AIRFLOW_ENV) $(CURDIR)/.venv-airflow/bin/airflow
SUITES := validation decision-value synthetic degradation attribution

.PHONY: help setup data entsoe forecast backtest holdout report figures export api web dashboard health experiments mlflow airflow airflow-setup pipeline settle test

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
	@echo "airflow-setup  create the Airflow environment in .venv-airflow"
	@echo "airflow   Airflow UI and scheduler at http://127.0.0.1:$(AIRFLOW_PORT)"
	@echo "pipeline  one live run without Airflow: make pipeline DAY=2026-09-17"
	@echo "settle    value the schedule committed for a day: make settle DAY=2026-09-16"
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

airflow-setup:
	uv venv --python 3.11 .venv-airflow
	.venv-airflow/bin/python -m ensurepip --upgrade
	.venv-airflow/bin/python -m pip install "apache-airflow==$(AIRFLOW_VERSION)" \
		--constraint "https://raw.githubusercontent.com/apache/airflow/constraints-$(AIRFLOW_VERSION)/constraints-3.11.txt"
	$(AIRFLOW) db migrate

airflow:
	$(AIRFLOW) standalone

pipeline:
	$(UV) python -m src.ingest.build_dataset
	$(UV) python -m src.ingest.open_meteo
	$(UV) python -m src.ingest.fuels
	$(UV) python -m src.ingest.build_inputs
	$(UV) python -m src.pipeline.daily_run --day $(DAY)

settle:
	$(UV) python -m src.pipeline.daily_run --day $(DAY) --settle

test:
	$(UV) ruff check .
	$(UV) mypy
	$(UV) pytest
