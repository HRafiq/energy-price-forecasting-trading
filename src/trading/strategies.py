"""Three ways to turn tomorrow's price fan into a day-ahead schedule.

- **Perfect foresight** optimizes on the realised prices. Nobody can trade it; it
  is the ceiling that forecast-driven strategies are measured against.
- **Median forecast** optimizes on q50 for both legs of every trade.
- **Quantile-aware** dispatch at a level below one half values selling at that
  quantile and buying at the mirrored one: q25 and q75 at level 0.25. That is the
  unfavourable side of the fan on both legs. A trade happens only if it still pays
  when prices land low where the battery sells and high where it buys, so thin or
  uncertain spreads are left alone. At level 0.5 it would equal median dispatch.

Every strategy settles at the realised prices. Only perfect foresight reads them
before settlement: the optimizer of a forecast-driven strategy receives forecast
columns alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from src.config import Settings
from src.forecasting.base import quantile_column
from src.trading.battery import Battery
from src.trading.optimizer import optimize_dispatch, product_blocks
from src.trading.settlement import Settlement, settle

__all__ = [
    "MEAN",
    "MEAN_FORECAST",
    "MEDIAN_FORECAST",
    "PERFECT_FORESIGHT",
    "PRODUCT",
    "REALISED",
    "DayResult",
    "Strategy",
    "add_mean_forecast",
    "build_strategies",
    "dispatch_day",
    "quantile_aware",
]

#: Column with the realised day-ahead price.
REALISED = "actual"
#: Column with the length of the day-ahead product, 60 or 15 minutes.
PRODUCT = "price_product_minutes"


@dataclass(frozen=True)
class Strategy:
    """Which price curve values each leg of a trade."""

    name: str
    sell_column: str
    buy_column: str

    @property
    def uses_realised_prices(self) -> bool:
        return REALISED in (self.sell_column, self.buy_column)


PERFECT_FORESIGHT = Strategy("perfect_foresight", REALISED, REALISED)
MEDIAN_FORECAST = Strategy("median_forecast", "q50", "q50")
#: Column with the mean of the forecast distribution, see ``add_mean_forecast``.
MEAN = "mean"
MEAN_FORECAST = Strategy("mean_forecast", MEAN, MEAN)


def add_mean_forecast(
    frame: pd.DataFrame, quantiles: tuple[float, ...]
) -> pd.DataFrame:
    """Add a ``mean`` column: the average of each period's forecast distribution.

    A price taker whose profit is linear in the price earns most on average by
    optimizing on the expected price, not the median. The quantile forecasts pin
    the distribution down at a few levels only, so the quantile function is taken
    as straight lines between them, with the outer slopes extended to levels 0
    and 1. The mean is the area under that function. Tails heavier than the
    extended lines, such as rare spikes beyond q95, are underestimated.
    """
    levels = np.asarray(sorted(quantiles), dtype=float)
    if len(levels) < 2:
        raise ValueError("at least two quantile levels are needed for a mean")
    columns = [quantile_column(float(level)) for level in levels]
    values = np.sort(frame[columns].to_numpy(dtype=float), axis=1)
    widths = np.diff(levels)
    inner = ((values[:, 1:] + values[:, :-1]) / 2 * widths).sum(axis=1)
    low_slope = (values[:, 1] - values[:, 0]) / widths[0]
    high_slope = (values[:, -1] - values[:, -2]) / widths[-1]
    low_tail = levels[0] * (values[:, 0] - low_slope * levels[0] / 2)
    high_tail = (1 - levels[-1]) * (values[:, -1] + high_slope * (1 - levels[-1]) / 2)
    return frame.assign(**{MEAN: inner + low_tail + high_tail})


def quantile_aware(level: float) -> Strategy:
    """Sell valued at quantile ``level``, buy at ``1 - level``."""
    if not 0 < level < 0.5:
        raise ValueError(
            f"dispatch quantile {level} must lie strictly between 0 and 0.5"
        )
    sell = quantile_column(level)
    return Strategy(f"quantile_{sell}", sell, quantile_column(1 - level))


def build_strategies(settings: Settings) -> tuple[Strategy, ...]:
    """Perfect foresight, median forecast and each configured quantile level."""
    return (
        PERFECT_FORESIGHT,
        MEDIAN_FORECAST,
        *(quantile_aware(level) for level in settings.trading.dispatch_quantiles),
    )


@dataclass(frozen=True)
class DayResult:
    """One strategy's schedule for one delivery day, settled.

    ``schedule`` holds the optimizer columns plus ``sell_price`` and
    ``buy_price`` (the curves optimized against) and ``realised_price``.
    ``planned_value_eur`` is the schedule's value on those curves, net of
    degradation; ``settlement`` values it at the realised prices.
    """

    strategy: str
    target_day: date
    product_minutes: int
    schedule: pd.DataFrame
    settlement: Settlement
    planned_value_eur: float
    solve_seconds: float


def dispatch_day(
    day: pd.DataFrame,
    strategy: Strategy,
    battery: Battery,
    *,
    time_limit_s: float = 60.0,
) -> DayResult:
    """Optimize and settle one delivery day.

    ``day`` holds every period of one target day: forecast quantile columns,
    ``actual``, ``price_product_minutes`` and ``target_day``.
    """
    needed = {
        strategy.sell_column,
        strategy.buy_column,
        REALISED,
        PRODUCT,
        "target_day",
    }
    missing = needed - set(day.columns)
    if missing:
        raise ValueError(f"day frame lacks columns {sorted(missing)}")
    target_days = day["target_day"].unique()
    if len(target_days) != 1:
        raise ValueError("day frame must hold exactly one target day")
    products = day[PRODUCT]
    if products.nunique() != 1:
        raise ValueError("one delivery day cannot mix product lengths")

    sell = day[strategy.sell_column]
    buy = day[strategy.buy_column]
    result = optimize_dispatch(
        sell,
        battery,
        buy_prices=buy,
        products=product_blocks(pd.DatetimeIndex(day.index), products),
        time_limit_s=time_limit_s,
    )
    realised = day[REALISED]
    schedule = result.schedule.assign(
        sell_price=sell.to_numpy(dtype=float),
        buy_price=buy.to_numpy(dtype=float),
        realised_price=realised.to_numpy(dtype=float),
    )
    return DayResult(
        strategy=strategy.name,
        target_day=target_days[0],
        product_minutes=int(products.iloc[0]),
        schedule=schedule,
        settlement=settle(result.schedule, realised, battery),
        planned_value_eur=result.objective_eur,
        solve_seconds=result.solve_seconds,
    )
