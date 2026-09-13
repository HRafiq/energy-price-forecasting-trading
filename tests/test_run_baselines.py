from __future__ import annotations

import pytest

from src.forecasting.run_baselines import main


@pytest.mark.parametrize("value", ["0", "-3"])
def test_quick_runs_need_at_least_one_day(value: str) -> None:
    with pytest.raises(SystemExit):
        main(["--last-days", value])
