"""The daily live pipeline, as Airflow runs it.

The DAG is deliberately thin: every task shells into ``src/``, so the same run
works without Airflow (``make pipeline DAY=2026-09-17``). Airflow lives in its own
environment, ``.venv-airflow``, and the project keeps its pinned one, so the tasks
call ``uv run`` in the repository rather than importing project code here.

Timing, for delivery day D+1 forecast on day D:

* the run starts at 10:30 local, an hour before the 11:40 issue time;
* ingest refreshes prices, weather and fuels, retrying on a flaky download;
* ``wait_for_inputs`` rebuilds the dataset and inputs and runs the readiness check
  on every poke, until every feed has arrived or the deadline passes, so a feed
  published after the 10:30 ingest is still seen. It exits non-zero while anything
  is missing;
* if the feeds arrive, ``forecast`` runs; if the sensor times out, ``forecast_late``
  runs instead. Both call the same command, which walks the fallback chain and
  writes an incident when it has to step down, so a late feed still produces a
  committed schedule before the gate;
* ``export_dashboard`` refreshes the dashboard, and ``check_deadline`` compares
  the time the forecast went out with the 12:00 gate. Airflow 3 dropped
  task-level SLAs, and its DAG-level deadline alerts are not usable from a test
  run, so the check is a task like any other: it writes a critical incident when
  the schedule was committed late, and leaves the run green so the missed gate is
  reported once, in the health log;
* ``settle_yesterday`` values the schedule committed a day earlier, once the
  auction has published its prices.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pendulum
from airflow.providers.standard.operators.bash import BashOperator
from airflow.providers.standard.sensors.bash import BashSensor
from airflow.sdk import DAG
from airflow.utils.trigger_rule import TriggerRule

REPO = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))
TZ = pendulum.timezone("Europe/Berlin")
#: The delivery day is the day after the run's logical date.
DAY = "{{ macros.ds_add(ds, 1) }}"
YESTERDAY = "{{ ds }}"
RUN = f"cd {REPO} && uv run python -m"
#: The sensor gives up ten minutes before the forecast is issued at 11:40. Both
#: values are read from the environment so an end-to-end test can force a timeout
#: in seconds instead of waiting out the real deadline.
SENSOR_TIMEOUT_S = int(os.environ.get("PIPELINE_SENSOR_TIMEOUT_S", 60 * 60))
SENSOR_POKE_S = int(os.environ.get("PIPELINE_SENSOR_POKE_S", 300))

with DAG(
    dag_id="daily_pipeline",
    description="Forecast tomorrow, commit a schedule before the 12:00 gate",
    schedule="30 10 * * *",
    start_date=datetime(2026, 9, 15, tzinfo=TZ),
    catchup=False,
    max_active_runs=1,
    default_args={
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
        "retry_exponential_backoff": True,
    },
    tags=["energy", "live"],
) as dag:
    prices = BashOperator(
        task_id="ingest_prices",
        # The day being forecast has no prices yet; keep its rows anyway.
        bash_command=f"{RUN} src.ingest.build_dataset --through {DAY}",
    )
    weather = BashOperator(
        task_id="ingest_weather",
        bash_command=f"{RUN} src.ingest.open_meteo",
    )
    fuels = BashOperator(
        task_id="ingest_fuels",
        bash_command=f"{RUN} src.ingest.fuels",
    )
    inputs = BashOperator(
        task_id="build_inputs",
        bash_command=f"{RUN} src.ingest.build_inputs",
    )

    wait = BashSensor(
        task_id="wait_for_inputs",
        # Rebuild on every poke: checking the file built at 10:30 would never see a
        # load forecast that SMARD publishes a few minutes later. A failed rebuild
        # keeps the last good files, so readiness always runs after it.
        bash_command=(
            f"{RUN} src.ingest.build_dataset --through {DAY}; "
            f"{RUN} src.ingest.build_inputs; "
            f"{RUN} src.pipeline.readiness --day {DAY}"
        ),
        poke_interval=SENSOR_POKE_S,
        timeout=SENSOR_TIMEOUT_S,
        mode="reschedule",
        retries=0,
    )

    forecast = BashOperator(
        task_id="forecast",
        bash_command=f"{RUN} src.pipeline.daily_run --day {DAY}",
    )
    # The gate closes whether or not the feeds arrived: the same command runs, and
    # the fallback chain inside it degrades to whatever it can still forecast.
    forecast_late = BashOperator(
        task_id="forecast_late",
        bash_command=f"{RUN} src.pipeline.daily_run --day {DAY}",
        trigger_rule=TriggerRule.ONE_FAILED,
        retries=0,
    )

    export = BashOperator(
        task_id="export_dashboard",
        bash_command=(
            # No --issued-utc: the export stamps the time it actually ran, not
            # the run's logical date, which is midnight.
            f"{RUN} src.export.artifacts --steps core health "
            f"--run-kind live --data-through {YESTERDAY}"
        ),
        trigger_rule=TriggerRule.ONE_SUCCESS,
    )

    check = BashOperator(
        task_id="check_deadline",
        bash_command=f"{RUN} src.pipeline.daily_run --day {DAY} --check-deadline",
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )

    settle = BashOperator(
        task_id="settle_yesterday",
        bash_command=f"{RUN} src.pipeline.daily_run --day {YESTERDAY} --settle",
        trigger_rule=TriggerRule.ALL_DONE,
        retries=1,
    )

    [prices, weather, fuels] >> inputs >> wait
    wait >> [forecast, forecast_late] >> export >> check >> settle
