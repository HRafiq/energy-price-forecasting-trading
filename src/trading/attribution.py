"""What forecast errors cost a trading strategy, by local hour block and direction.

A forecast-driven strategy earns less than perfect foresight on most days. The gap
is the price of not knowing tomorrow's prices. This module splits that gap by where
the forecast was wrong (six blocks of local hours) and which way it was wrong:

- **over**: the forecast was above the realised price in that period;
- **under**: the forecast was below it.

The direction of a period is the sign of ``q50 - actual`` by default, for every
strategy: quantile-aware strategies are grouped by the median forecast's error too.
Before 1 October 2025 the auction traded hours, so the optimizer only sees an
hour's average forecast; the four quarter-hours of an hourly product take the
direction of the product's mean error. Periods whose error is smaller than
``ERROR_TOLERANCE_EUR_MWH`` belong to neither direction.
Each (block, direction) pair with at least one period on the day is a *group*.

Coalition values
----------------
For any set S of groups, v(S) is the settled P&L the strategy would have earned
had its forecast been exact in the periods of those groups and unchanged
elsewhere. To build that day, the realised price is written into the columns the
strategy optimizes against (its sell column and its buy column) in those periods;
the day is dispatched again with the same strategy and battery and settled at the
realised prices. v(no groups) is the strategy's own P&L, and v(all groups) is the
perfect-foresight P&L (the ceiling), which the module checks to within a cent.

Why a Shapley split
-------------------
Fixing errors one group at a time gives answers that depend on which group is
fixed first. The battery plans the whole day at once: a correct evening price
decides whether to sell then, which decides whether a cheap midday hour is worth
charging in. So the value of fixing midday is small if the evening is still wrong
and large once it is right, or the other way round. Measured one at a time, the
group values do not add up to the gap, and the leftover can be most of it.

The Shapley share of a group settles this by averaging over fix orders. Take an
ordering of all groups and fix them one after another; each group is credited
with how much P&L rose when it was added to the groups before it. Along any
ordering those credits add up to v(all) - v(none), which is the gap. Averaging a
group's credit over orderings gives its share, so the shares always add up to the
gap too. A share can be negative: fixing a group can, on average, lead the
optimizer to a worse schedule while other errors remain.

With at most ``exact_max_players`` groups (default 4, so at most 24 orderings)
every ordering is used and the result is the exact Shapley value. With more,
``permutations`` random orderings are drawn and each is also used reversed
(antithetic sampling: a group that comes early in one ordering comes late in its
twin, which cancels much of the sampling noise). The random generator is seeded
from the target day and the strategy name, so a result does not depend on the
run or the number of worker processes. Coalitions shared by several orderings are
solved once and cached.

Local hours come from converting each period to the market time zone. On the
autumn DST day the repeated 02:00 hour stays in block 00-05, which then holds
seven hours (28 quarter-hours); on the spring day it holds five.
"""

from __future__ import annotations

import hashlib
import itertools
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.forecasting.walkforward import HoldoutAccessError
from src.trading.battery import Battery
from src.trading.optimizer import DispatchError, product_blocks
from src.trading.run_strategies import CEILING_TOLERANCE_EUR
from src.trading.strategies import (
    PERFECT_FORESIGHT,
    PRODUCT,
    REALISED,
    Strategy,
    dispatch_day,
)

__all__ = [
    "COST_COLUMNS",
    "DAY_COLUMNS",
    "DIRECTIONS",
    "ERROR_TOLERANCE_EUR_MWH",
    "HOUR_BLOCKS",
    "Attribution",
    "DayAttribution",
    "attribute_day",
    "attribute_days",
    "counterfactual_frame",
    "counterfactual_pnl",
    "error_directions",
    "local_hour_blocks",
    "ordering_seed",
    "product_directions",
    "shapley_orderings",
]

#: Blocks of local hours, in display order. Every hour 0-23 is in exactly one.
HOUR_BLOCKS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("00-05", (0, 1, 2, 3, 4, 5)),
    ("06-10", (6, 7, 8, 9, 10)),
    ("11-14", (11, 12, 13, 14)),
    ("15-17", (15, 16, 17)),
    ("18-20", (18, 19, 20)),
    ("21-23", (21, 22, 23)),
)
#: Over-forecast: forecast above the realised price. Under-forecast: below it.
DIRECTIONS: tuple[str, str] = ("over", "under")
#: Errors smaller than this, in €/MWh, count as exact and have no direction.
ERROR_TOLERANCE_EUR_MWH = 1e-9
#: Columns of the per-group cost table of ``attribute_days``.
COST_COLUMNS = (
    "target_day",
    "strategy",
    "block",
    "direction",
    "periods",
    "mean_abs_error_eur_mwh",
    "cost_eur",
)
#: Columns of the per-day table of ``attribute_days``.
DAY_COLUMNS = (
    "target_day",
    "strategy",
    "strategy_pnl_eur",
    "ceiling_pnl_eur",
    "gap_eur",
    "attributed_eur",
    "residual_eur",
    "players",
    "orderings",
    "exact",
    "solves",
    "seconds",
)

_HOUR_TO_BLOCK = {hour: name for name, hours in HOUR_BLOCKS for hour in hours}
_NO_DIRECTION = ""


@dataclass(frozen=True)
class DayAttribution:
    """The gap to perfect foresight of one strategy on one day, split by errors.

    ``costs`` has one row per (block, direction) in ``HOUR_BLOCKS`` x
    ``DIRECTIONS`` order, with columns ``block``, ``direction``, ``periods`` (how
    many quarter-hours had that error direction in that block),
    ``mean_abs_error_eur_mwh`` (NaN when there are none) and ``cost_eur``, the
    Shapley share. A group without periods is not a player and costs exactly 0.

    ``residual_eur`` is gap minus the sum of shares: within a cent of 0, unless
    some period with no error direction still had a wrong sell or buy price (a
    quantile strategy on a period where q50 was exact), whose effect no group
    owns. ``players`` counts the groups, ``orderings`` the fix orders averaged,
    ``exact`` says whether all orderings were used, ``solves`` counts optimizer
    runs for this strategy (baseline, the ceiling when not supplied, and every
    distinct coalition).
    """

    target_day: date
    strategy: str
    strategy_pnl_eur: float
    ceiling_pnl_eur: float
    gap_eur: float
    residual_eur: float
    players: int
    orderings: int
    exact: bool
    solves: int
    seconds: float
    costs: pd.DataFrame

    @property
    def attributed_eur(self) -> float:
        """Sum of the group shares; gap = attributed + residual."""
        return float(self.costs["cost_eur"].sum())


@dataclass(frozen=True)
class Attribution:
    """Attribution over many days.

    ``costs`` is long format with ``COST_COLUMNS``; ``days`` has one row per day
    and strategy with ``DAY_COLUMNS``; ``failed`` lists days whose solve failed.
    """

    costs: pd.DataFrame
    days: pd.DataFrame
    failed: dict[date, str] = field(default_factory=dict)


def local_hour_blocks(index: pd.DatetimeIndex, timezone: str) -> NDArray[np.str_]:
    """The hour-block label of every period, by its local start hour."""
    if index.tz is None:
        raise ValueError("the index must be timezone-aware")
    hours = index.tz_convert(timezone).hour
    return np.array([_HOUR_TO_BLOCK[int(hour)] for hour in hours], dtype=np.str_)


def error_directions(forecast: pd.Series, realised: pd.Series) -> NDArray[np.str_]:
    """``"over"``, ``"under"`` or ``""`` (exact) for every period."""
    error = forecast.to_numpy(dtype=float) - realised.to_numpy(dtype=float)
    return np.where(
        error > ERROR_TOLERANCE_EUR_MWH,
        DIRECTIONS[0],
        np.where(error < -ERROR_TOLERANCE_EUR_MWH, DIRECTIONS[1], _NO_DIRECTION),
    ).astype(np.str_)


def product_directions(
    forecast: pd.Series, realised: pd.Series, products: NDArray[np.int64]
) -> NDArray[np.str_]:
    """Error direction of each day-ahead product, repeated on its periods.

    The four quarter-hours of an hourly product share one schedule and one realised
    price, so errors inside the hour cancel for the optimizer; the product's mean
    error decides the direction. When every period is its own product this equals
    ``error_directions``.
    """
    labels = pd.Series(np.asarray(products))
    error = pd.Series(forecast.to_numpy(dtype=float) - realised.to_numpy(dtype=float))
    mean_error = error.groupby(labels).transform("mean").to_numpy(dtype=float)
    return np.where(
        mean_error > ERROR_TOLERANCE_EUR_MWH,
        DIRECTIONS[0],
        np.where(mean_error < -ERROR_TOLERANCE_EUR_MWH, DIRECTIONS[1], _NO_DIRECTION),
    ).astype(np.str_)


def counterfactual_frame(
    day: pd.DataFrame, strategy: Strategy, mask: NDArray[np.bool_]
) -> pd.DataFrame:
    """A copy of ``day`` whose strategy columns hold the realised price in ``mask``.

    Both the sell column and the buy column are replaced, so both legs of a trade
    see exact prices in the masked periods; a column used for both legs is
    replaced once. Other columns and unmasked periods are untouched.
    """
    _require_forecast_strategy(strategy)
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != (len(day),):
        raise ValueError("mask must have one value per period")
    frame = day.copy()
    realised = day[REALISED].to_numpy(dtype=float)
    for column in dict.fromkeys((strategy.sell_column, strategy.buy_column)):
        values = frame[column].to_numpy(dtype=float, copy=True)
        values[selected] = realised[selected]
        frame[column] = values
    return frame


def counterfactual_pnl(
    day: pd.DataFrame,
    strategy: Strategy,
    battery: Battery,
    mask: NDArray[np.bool_],
    *,
    time_limit_s: float = 60.0,
) -> float:
    """Settled P&L of ``strategy`` had its forecast been exact in ``mask``."""
    frame = counterfactual_frame(day, strategy, mask)
    result = dispatch_day(frame, strategy, battery, time_limit_s=time_limit_s)
    return result.settlement.pnl_eur


def ordering_seed(target_day: date, strategy_name: str) -> int:
    """A reproducible seed for one day and strategy, the same in every process.

    Python's built-in ``hash`` of a string changes between processes, so the seed
    comes from a SHA-256 digest instead.
    """
    key = f"{pd.Timestamp(target_day).date().isoformat()}|{strategy_name}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def shapley_orderings(
    players: int, *, permutations: int, exact_max_players: int, seed: int
) -> tuple[list[tuple[int, ...]], bool]:
    """The fix orders to average over, and whether they are all of them.

    With at most ``exact_max_players`` players every ordering is returned. With
    more, ``permutations`` random orderings are drawn and each is followed by its
    reverse, giving ``2 * permutations`` orderings.
    """
    if players < 0:
        raise ValueError("players cannot be negative")
    if permutations < 1:
        raise ValueError("permutations must be at least 1")
    if players <= exact_max_players:
        return list(itertools.permutations(range(players))), True
    rng = np.random.default_rng(seed)
    orderings: list[tuple[int, ...]] = []
    for _ in range(permutations):
        order = tuple(int(player) for player in rng.permutation(players))
        orderings += [order, order[::-1]]
    return orderings, False


def attribute_day(
    day: pd.DataFrame,
    strategy: Strategy,
    battery: Battery,
    *,
    timezone: str,
    error_column: str = "q50",
    time_limit_s: float = 60.0,
    ceiling_pnl_eur: float | None = None,
    permutations: int = 4,
    exact_max_players: int = 4,
) -> DayAttribution:
    """Split one day's gap to perfect foresight into Shapley shares per group.

    ``ceiling_pnl_eur`` may carry perfect foresight's settled P&L for this same
    day, already solved, so several strategies share one ceiling solve. Raises
    ``ValueError`` for perfect foresight itself, and ``RuntimeError`` if exact
    prices in every group miss the ceiling by more than a cent, which would mean
    the replacement left some forecast price in place.
    """
    started = time.perf_counter()
    _require_forecast_strategy(strategy)
    if error_column not in day.columns:
        raise ValueError(f"day frame lacks the error column {error_column!r}")
    day = day.sort_index()
    baseline = dispatch_day(day, strategy, battery, time_limit_s=time_limit_s)
    strategy_pnl = baseline.settlement.pnl_eur
    fixed_solves = 1
    if ceiling_pnl_eur is None:
        ceiling = dispatch_day(
            day, PERFECT_FORESIGHT, battery, time_limit_s=time_limit_s
        )
        ceiling_pnl_eur = ceiling.settlement.pnl_eur
        fixed_solves += 1

    realised = day[REALISED].to_numpy(dtype=float)
    blocks = local_hour_blocks(pd.DatetimeIndex(day.index), timezone)
    products = product_blocks(pd.DatetimeIndex(day.index), day[PRODUCT])
    directions = product_directions(day[error_column], day[REALISED], products)
    abs_error = np.abs(day[error_column].to_numpy(dtype=float) - realised)
    groups = [
        (block, direction, (blocks == block) & (directions == direction))
        for block, _ in HOUR_BLOCKS
        for direction in DIRECTIONS
    ]
    player_groups = [i for i, (_, _, mask) in enumerate(groups) if mask.any()]
    player_masks = [groups[i][2] for i in player_groups]

    # v(S) per coalition, keyed by the set of player positions; the empty
    # coalition is the strategy's own schedule, already solved.
    cache: dict[frozenset[int], float] = {frozenset(): strategy_pnl}

    def value(coalition: frozenset[int]) -> float:
        if coalition not in cache:
            mask = np.zeros(len(day), dtype=bool)
            for player in coalition:
                mask |= player_masks[player]
            cache[coalition] = counterfactual_pnl(
                day, strategy, battery, mask, time_limit_s=time_limit_s
            )
        return cache[coalition]

    everyone = frozenset(range(len(player_groups)))
    grand = value(everyone)
    unassigned = directions == _NO_DIRECTION
    stale = np.zeros(len(day), dtype=bool)
    for column in (strategy.sell_column, strategy.buy_column):
        error = np.abs(day[column].to_numpy(dtype=float) - realised)
        stale |= unassigned & (error > ERROR_TOLERANCE_EUR_MWH)
    if not stale.any() and abs(grand - ceiling_pnl_eur) > CEILING_TOLERANCE_EUR:
        raise RuntimeError(
            f"{baseline.target_day} {strategy.name}: exact prices in every group "
            f"settle at €{grand:.4f}, not the perfect-foresight "
            f"€{ceiling_pnl_eur:.4f}"
        )

    orderings, exact = shapley_orderings(
        len(player_groups),
        permutations=permutations,
        exact_max_players=exact_max_players,
        seed=ordering_seed(baseline.target_day, strategy.name),
    )
    shares = np.zeros(len(player_groups), dtype=np.float64)
    for order in orderings:
        members: frozenset[int] = frozenset()
        previous = strategy_pnl
        for player in order:
            members = members | {player}
            current = value(members)
            shares[player] += current - previous
            previous = current
    shares /= len(orderings)

    share_of = dict(zip(player_groups, shares, strict=True))
    rows: list[dict[str, object]] = []
    for i, (block, direction, mask) in enumerate(groups):
        periods = int(mask.sum())
        rows.append(
            {
                "block": block,
                "direction": direction,
                "periods": periods,
                "mean_abs_error_eur_mwh": (
                    float(abs_error[mask].mean()) if periods else float("nan")
                ),
                "cost_eur": float(share_of.get(i, 0.0)),
            }
        )
    costs = pd.DataFrame(rows)
    gap = ceiling_pnl_eur - strategy_pnl
    residual = gap - float(costs["cost_eur"].sum())
    if not stale.any() and abs(residual) > CEILING_TOLERANCE_EUR:
        raise RuntimeError(
            f"{baseline.target_day} {strategy.name}: shares miss the gap by "
            f"€{residual:.4f}"
        )
    return DayAttribution(
        target_day=baseline.target_day,
        strategy=strategy.name,
        strategy_pnl_eur=strategy_pnl,
        ceiling_pnl_eur=ceiling_pnl_eur,
        gap_eur=gap,
        residual_eur=residual,
        players=len(player_groups),
        orderings=len(orderings),
        exact=exact,
        solves=fixed_solves + len(cache) - 1,
        seconds=time.perf_counter() - started,
        costs=costs,
    )


def _require_forecast_strategy(strategy: Strategy) -> None:
    if strategy.uses_realised_prices:
        raise ValueError(
            f"{strategy.name} reads realised prices, so it has no forecast error "
            "to attribute"
        )


@dataclass(frozen=True)
class _DayJob:
    """Everything one worker process needs for one day."""

    frame: pd.DataFrame
    strategies: tuple[Strategy, ...]
    battery: Battery
    timezone: str
    error_column: str
    time_limit_s: float
    permutations: int
    exact_max_players: int


def _attribute_day_job(job: _DayJob) -> tuple[list[DayAttribution], str | None]:
    """Every strategy on one day, sharing one perfect-foresight solve."""
    try:
        ceiling = dispatch_day(
            job.frame, PERFECT_FORESIGHT, job.battery, time_limit_s=job.time_limit_s
        )
        results = [
            attribute_day(
                job.frame,
                strategy,
                job.battery,
                timezone=job.timezone,
                error_column=job.error_column,
                time_limit_s=job.time_limit_s,
                ceiling_pnl_eur=ceiling.settlement.pnl_eur,
                permutations=job.permutations,
                exact_max_players=job.exact_max_players,
            )
            for strategy in job.strategies
        ]
    except DispatchError as exc:
        return [], f"solver failed: {exc}"
    return results, None


def attribute_days(
    forecasts: pd.DataFrame,
    days: list[date],
    strategies: tuple[Strategy, ...] | list[Strategy],
    battery: Battery,
    *,
    timezone: str,
    holdout_start: date,
    error_column: str = "q50",
    workers: int = 1,
    time_limit_s: float = 60.0,
    permutations: int = 4,
    exact_max_players: int = 4,
    allow_holdout: bool = False,
) -> Attribution:
    """Attribute every forecast strategy's gap on every day in ``days``.

    ``forecasts`` holds many target days in the day-frame format; days listed but
    absent from it are ignored. Perfect foresight in ``strategies`` is skipped,
    since it has no gap. Days are independent, so with ``workers > 1`` they run in
    a process pool; seeded orderings make the result the same for any worker
    count. Raises ``HoldoutAccessError`` if any day is in the hold-out, unless
    ``allow_holdout`` is set by the frozen final evaluation.
    """
    inside = [day for day in days if day >= holdout_start]
    if inside and not allow_holdout:
        raise HoldoutAccessError(
            f"{len(inside)} days fall in the hold-out starting {holdout_start}"
        )
    forecast_strategies = tuple(s for s in strategies if not s.uses_realised_prices)
    if not forecast_strategies:
        raise ValueError("no forecast-driven strategy to attribute")
    if workers < 1:
        raise ValueError("workers must be at least 1")
    wanted = set(days)
    jobs = [
        _DayJob(
            frame=frame.sort_index(),
            strategies=forecast_strategies,
            battery=battery,
            timezone=timezone,
            error_column=error_column,
            time_limit_s=time_limit_s,
            permutations=permutations,
            exact_max_players=exact_max_players,
        )
        for day, frame in forecasts.groupby("target_day")
        if day in wanted
    ]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            outputs = list(pool.map(_attribute_day_job, jobs, chunksize=1))
    else:
        outputs = [_attribute_day_job(job) for job in jobs]

    cost_parts: list[pd.DataFrame] = []
    day_rows: list[dict[str, object]] = []
    failed: dict[date, str] = {}
    for job, (results, failure) in zip(jobs, outputs, strict=True):
        if failure is not None:
            failed[job.frame["target_day"].iloc[0]] = failure
            continue
        for result in results:
            part = result.costs.copy()
            part.insert(0, "strategy", result.strategy)
            part.insert(0, "target_day", result.target_day)
            cost_parts.append(part)
            day_rows.append(
                {
                    "target_day": result.target_day,
                    "strategy": result.strategy,
                    "strategy_pnl_eur": result.strategy_pnl_eur,
                    "ceiling_pnl_eur": result.ceiling_pnl_eur,
                    "gap_eur": result.gap_eur,
                    "attributed_eur": result.attributed_eur,
                    "residual_eur": result.residual_eur,
                    "players": result.players,
                    "orderings": result.orderings,
                    "exact": result.exact,
                    "solves": result.solves,
                    "seconds": result.seconds,
                }
            )
    costs = (
        pd.concat(cost_parts, ignore_index=True)
        if cost_parts
        else pd.DataFrame(columns=list(COST_COLUMNS))
    )
    return Attribution(
        costs=costs[list(COST_COLUMNS)],
        days=pd.DataFrame(day_rows, columns=list(DAY_COLUMNS)),
        failed=failed,
    )
