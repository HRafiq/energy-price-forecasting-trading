"""How much market history a run needs, and why it is not the model's ten days."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from src.config import Settings
from src.forecasting.models.common import FEATURE_HISTORY_DAYS
from src.pipeline import history
from src.pipeline.drift_check import LOOKBACK_DAYS

DAY = date(2026, 9, 23)


def test_an_ordinary_day_is_sized_by_the_drift_monitor_not_the_model(
    settings: Settings,
) -> None:
    """The monitor needs the realised price of every day it reads back to."""
    days = history.days_needed(settings, DAY, refitting=False)

    assert days == LOOKBACK_DAYS + history.MARGIN_DAYS
    assert days > FEATURE_HISTORY_DAYS + history.MARGIN_DAYS


def test_the_window_covers_every_rung_the_chain_could_bid_with(
    settings: Settings,
) -> None:
    """A fallback rung may be the one that bids, and it reads back further."""
    chain = history._chain_lookback(settings)

    assert chain > FEATURE_HISTORY_DAYS
    assert history.days_needed(settings, DAY, refitting=False) >= chain


def test_the_window_covers_what_the_drift_monitor_reads(settings: Settings) -> None:
    """Otherwise the monitor reports its own missing prices as unpublished."""
    assert history.days_needed(settings, DAY, refitting=False) > LOOKBACK_DAYS


def test_a_refit_day_needs_the_whole_training_window(settings: Settings) -> None:
    ordinary = history.days_needed(settings, DAY, refitting=False)

    refitting = history.days_needed(settings, DAY, refitting=True)

    assert refitting > ordinary and refitting > 700


def test_a_registry_it_cannot_read_is_treated_as_due(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Too much history wastes bandwidth; too little makes the refit impossible."""

    def _broken(settings: Settings, **kwargs: object) -> None:
        raise RuntimeError("no registry here")

    monkeypatch.setattr(history, "served_version", _broken)
    assert history.refit_is_due(settings, DAY) is True

    monkeypatch.setattr(history, "served_version", lambda s, **k: None)
    assert history.refit_is_due(settings, DAY) is True


def test_a_model_serving_since_yesterday_is_not_due(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fresh = DAY - timedelta(days=1)
    monkeypatch.setattr(history, "served_version", lambda s, **k: ("7", fresh))

    assert history.refit_is_due(settings, DAY) is False

    stale = DAY - timedelta(days=settings.pipeline.refit_every_days)
    monkeypatch.setattr(history, "served_version", lambda s, **k: ("7", stale))
    assert history.refit_is_due(settings, DAY) is True


def test_the_cli_prints_the_number_first_so_a_shell_can_read_it(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(history, "load_settings", lambda path: settings)
    monkeypatch.setattr(history, "served_version", lambda s, **k: ("7", DAY))

    assert history.main(["--day", str(DAY)]) == 0

    first, second = capsys.readouterr().out.splitlines()[:2]
    assert first.isdigit() and int(first) == LOOKBACK_DAYS + history.MARGIN_DAYS
    assert "keeps serving" in second
