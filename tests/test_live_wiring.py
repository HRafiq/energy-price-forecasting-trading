"""The live run asks for the day being forecast everywhere it is started."""

from __future__ import annotations

from src.config import REPO_ROOT

DAG = (REPO_ROOT / "dags" / "daily_pipeline.py").read_text(encoding="utf-8")
MAKEFILE = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")


def _between(text: str, start: str, end: str) -> str:
    return text[text.index(start) : text.index(end)]


def test_the_dag_builds_the_dataset_through_the_delivery_day() -> None:
    # Once at the 10:30 ingest, and again on every poke of the readiness sensor.
    assert DAG.count("src.ingest.build_dataset --through {DAY}") == 2


def test_the_sensor_rebuilds_its_inputs_before_checking_readiness() -> None:
    sensor = _between(DAG, 'task_id="wait_for_inputs"', 'task_id="forecast"')
    assert (
        sensor.index("build_dataset")
        < sensor.index("build_inputs")
        < sensor.index("src.pipeline.readiness")
    )


def test_make_pipeline_builds_through_the_delivery_day() -> None:
    pipeline = _between(MAKEFILE, "\npipeline:", "\nsettle:")
    assert "src.ingest.build_dataset --through $(DAY)" in pipeline


def test_installing_the_agents_switches_the_dag_on_and_removing_them_off() -> None:
    install = _between(MAKEFILE, "\nlaunchd-install:", "\nlaunchd-uninstall:")
    remove = MAKEFILE[MAKEFILE.index("\nlaunchd-uninstall:") :]
    assert "DAG_ID := daily_pipeline" in MAKEFILE
    # A DAG Airflow has not registered yet cannot be unpaused, and the command says
    # so without failing, so it is registered first and the result is checked.
    assert install.index("dags reserialize") < install.index("dags unpause $(DAG_ID)")
    assert '[ "$$paused" != "False" ]' in install
    assert "dags pause $(DAG_ID)" in remove


def test_both_forecast_tasks_have_a_backstop_timeout() -> None:
    for task, end in (
        ('task_id="forecast"', 'task_id="forecast_late"'),
        ('task_id="forecast_late"', 'task_id="export_dashboard"'),
    ):
        assert "execution_timeout=FORECAST_TIMEOUT" in _between(DAG, task, end)
    assert "FORECAST_TIMEOUT = timedelta(minutes=15)" in DAG
