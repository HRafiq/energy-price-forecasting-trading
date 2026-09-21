"""How much market history one live run needs, so a run fetches only that.

The dataset build downloads from ``data.start`` by default, which is years of
prices, generation and weather. On a laptop that is paid once and cached; on a
scheduled runner, which is a fresh machine every time, it is paid on every run,
for a bid that needs a fortnight.

Several things in a run reach back, and the longest of them sets the window:

* the production model's features need ``FEATURE_HISTORY_DAYS`` days;
* so do the rungs below it, and by more: the seasonal naive baseline measures its
  own errors over a window before the lag it copies from, so it reaches back
  further than the model it stands in for. The window is taken over every rung
  the chain can use, because any of them may be the one that bids;
* the drift monitor reads saved live forecasts ``LOOKBACK_DAYS`` back and needs
  the realised price of each, which on an ordinary day is what binds;
* a refit day needs the whole training window, which is two years. That is the
  expensive case and it comes round once every ``pipeline.refit_every_days``.

``MARGIN_DAYS`` is added to whichever applies, to absorb publication lags, the
hour a DST change moves, and a run picking up after some days missed. It is
slack on top of a window that is already computed, not a substitute for
computing it: a requirement the code does not know about is not covered by it.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

from src.config import Settings, load_settings
from src.forecasting.models.common import FEATURE_HISTORY_DAYS
from src.forecasting.production import build_production_model
from src.pipeline.chain import STEPS
from src.pipeline.drift_check import LOOKBACK_DAYS
from src.pipeline.model_source import refit_due, served_version

__all__ = ["MARGIN_DAYS", "days_needed", "main", "refit_is_due"]

#: Slack over the longest window a run needs, in days.
MARGIN_DAYS = 30


def refit_is_due(settings: Settings, target_day: date) -> bool:
    """Whether the served model is due a refit before ``target_day``.

    A registry that cannot be read is treated as due: fetching more history than
    a run needs wastes bandwidth, fetching less would make the refit impossible.
    """
    try:
        served = served_version(settings)
    except Exception:  # a registry this run cannot read, for any reason
        return True
    if served is None:
        return True
    _, forecasts_from = served
    return refit_due(forecasts_from, target_day, settings.pipeline.refit_every_days)


def days_needed(settings: Settings, target_day: date, *, refitting: bool) -> int:
    """Days of history before ``target_day`` that this run has to have."""
    if refitting:
        # The model itself says how far a fit reaches back: its training window
        # plus the history its features need on top.
        fit_lookback = build_production_model(settings).fit_lookback_days
        if fit_lookback is None:
            raise ValueError(
                f"{settings.forecasting.production_model} does not declare a fit "
                "lookback, so the history a refit needs is unknown"
            )
        return int(fit_lookback) + MARGIN_DAYS
    # Not the model's ten days: the drift monitor reads further back, and so do
    # the fallback rungs, any of which may be the one that ends up bidding.
    return max(FEATURE_HISTORY_DAYS, LOOKBACK_DAYS, _chain_lookback(settings)) + (
        MARGIN_DAYS
    )


def _chain_lookback(settings: Settings) -> int:
    """The furthest back any rung of the fallback chain reads to forecast."""
    reaches = []
    for step in STEPS:
        try:
            lookback = step.build(settings).lookback_days
        except Exception:  # a rung this configuration cannot build cannot bid
            continue
        if lookback is not None:
            reaches.append(int(lookback))
    return max(reaches, default=FEATURE_HISTORY_DAYS)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Days of market history the run for a delivery day needs."
    )
    parser.add_argument("--day", type=date.fromisoformat, required=True)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args(argv)

    settings = load_settings(args.config)
    refitting = refit_is_due(settings, args.day)
    days = days_needed(settings, args.day, refitting=refitting)
    why = "a refit is due" if refitting else "the served model keeps serving"
    print(days)
    print(f"{args.day}: {days} days of history, because {why}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
