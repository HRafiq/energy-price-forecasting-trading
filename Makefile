# Short names for the pipeline commands in the README. `make help` lists them.
UV := uv run
MLFLOW_PORT ?= 5001
AIRFLOW_PORT ?= 8080
AIRFLOW_VERSION ?= 3.3.1
# Delivery day for a one-off pipeline run; tomorrow by default.
DAY ?= $(shell date -v+1d +%F 2>/dev/null || date -d tomorrow +%F)
# Airflow keeps its database and logs here; the DAGs live in the repo.
# standalone starts its own airflow processes by name, so its venv goes on PATH.
AIRFLOW_ENV := AIRFLOW_HOME=$(CURDIR)/airflow_home \
	AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags \
	AIRFLOW__CORE__LOAD_EXAMPLES=False \
	PATH="$(CURDIR)/.venv-airflow/bin:$$PATH"
AIRFLOW := $(AIRFLOW_ENV) $(CURDIR)/.venv-airflow/bin/airflow
SUITES := validation decision-value synthetic degradation attribution

.PHONY: help setup data entsoe forecast backtest holdout report figures export api web dashboard health experiments mlflow airflow airflow-setup pipeline settle test launchd-install launchd-uninstall

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
	@echo "launchd-install    macOS: keep Airflow running from login and the Mac awake for the daily run"
	@echo "launchd-uninstall  remove those two launchd agents"
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
	$(UV) python -m src.ingest.build_dataset --through $(DAY)
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

# Unattended daily runs on macOS. One launchd agent keeps Airflow running from login;
# the other holds the Mac awake from 08:20 UTC for 4 h 15 min, which covers the 10:30
# Berlin run, the 12:00 gate and the export in both summer and winter time. The start
# is converted to this machine's local time at install, so reinstall after the
# machine's own clocks change (never, in a zone without daylight saving such as
# Asia/Dubai). launchd cannot wake a sleeping Mac; `sudo pmset repeat wakeorpoweron
# MTWRFSU <local time>` can, and is a system setting left to the owner of the machine.
# Installing registers and switches on the daily DAG. Airflow then runs the latest
# slot it missed: a day that already has a committed schedule is left alone, and a
# day without one gets a late schedule, recorded as late.
LAUNCHD_DIR := $(HOME)/Library/LaunchAgents
LAUNCHD_LABEL := local.energy-price-forecasting
AWAKE_SECONDS := 15300
DAG_ID := daily_pipeline

launchd-install:
	@test -x .venv-airflow/bin/airflow || { echo "Airflow is not installed: run make airflow-setup first"; exit 1; }
	@mkdir -p "$(LAUNCHD_DIR)" airflow_home/logs
	@set -e; \
	uid=$$(id -u); \
	for job in airflow awake; do \
		launchctl bootout "gui/$$uid/$(LAUNCHD_LABEL).$$job" 2>/dev/null || true; \
		for _ in $$(seq 1 20); do launchctl print "gui/$$uid/$(LAUNCHD_LABEL).$$job" >/dev/null 2>&1 || break; sleep 0.5; done; \
	done; \
	for _ in $$(seq 1 40); do lsof -nP -iTCP:$(AIRFLOW_PORT) -sTCP:LISTEN >/dev/null 2>&1 || break; sleep 0.5; done; \
	if lsof -nP -iTCP:$(AIRFLOW_PORT) -sTCP:LISTEN >/dev/null 2>&1; then \
		echo "port $(AIRFLOW_PORT) is still in use, by another Airflow or one still shutting down: wait and rerun make launchd-install"; exit 1; \
	fi; \
	window=$$($(UV) python -c "from datetime import datetime, timezone; t = datetime.now(timezone.utc).replace(hour=8, minute=20, second=0, microsecond=0).astimezone(); print(t.hour, t.minute)"); \
	hour=$${window% *}; minute=$${window#* }; \
	uv_dir=$$(dirname "$$(command -v uv)"); \
	sed -e "s#@REPO@#$(CURDIR)#g" -e "s#@PATH@#$$uv_dir:/usr/bin:/bin:/usr/sbin:/sbin#g" \
		ops/launchd/airflow.plist > "$(LAUNCHD_DIR)/$(LAUNCHD_LABEL).airflow.plist"; \
	sed -e "s#@HOUR@#$$hour#g" -e "s#@MINUTE@#$$minute#g" -e "s#@SECONDS@#$(AWAKE_SECONDS)#g" \
		ops/launchd/awake.plist > "$(LAUNCHD_DIR)/$(LAUNCHD_LABEL).awake.plist"; \
	$(AIRFLOW) dags reserialize >/dev/null; \
	$(AIRFLOW) dags unpause $(DAG_ID) >/dev/null; \
	paused=$$($(AIRFLOW) dags list -o json 2>/dev/null | $(UV) python -c "import json, sys; print(next((str(r['is_paused']) for r in json.load(sys.stdin) if r['dag_id'] == '$(DAG_ID)'), 'missing'))"); \
	if [ "$$paused" != "False" ]; then echo "$(DAG_ID) is not switched on (is_paused: $$paused); check that dags/ imports cleanly with airflow dags list-import-errors"; exit 1; fi; \
	for job in airflow awake; do \
		plist="$(LAUNCHD_DIR)/$(LAUNCHD_LABEL).$$job.plist"; \
		launchctl bootstrap "gui/$$uid" "$$plist" 2>/dev/null || { sleep 3; launchctl bootstrap "gui/$$uid" "$$plist"; }; \
	done; \
	echo "Airflow runs from login with $(DAG_ID) switched on; the Mac is held awake daily from $$hour:$$minute local time for $$(( $(AWAKE_SECONDS) / 60 )) minutes"

launchd-uninstall:
	@uid=$$(id -u); \
	for job in airflow awake; do \
		launchctl bootout "gui/$$uid/$(LAUNCHD_LABEL).$$job" 2>/dev/null || true; \
		rm -f "$(LAUNCHD_DIR)/$(LAUNCHD_LABEL).$$job.plist"; \
	done; \
	if test -x .venv-airflow/bin/airflow; then $(AIRFLOW) dags pause $(DAG_ID) >/dev/null 2>&1 || true; fi; \
	echo "launchd agents removed and $(DAG_ID) paused"
