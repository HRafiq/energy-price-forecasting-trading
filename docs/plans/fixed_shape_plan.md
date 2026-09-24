# A benchmark that does not forecast at all: a fixed plan

Written 2026-09-24, before any code for it existed. Committed on its own, before
the code.

## The question, in plain words

Every baseline in this repository is a forecaster. The lowest entry in the
decision-value comparison, "yesterday's price for this quarter-hour", captures
77.6% of perfect foresight, and it is still a forecast fed through the same
optimiser. Nothing here measures what the battery earns knowing nothing about
tomorrow at all.

That matters because the German day-ahead shape is stable. Solar pushes the
middle of the day down, the peak sits in the evening, and a battery can exploit
that with a wall clock and no model whatsoever. If a clock captures most of the
available profit, then the forecasting work is worth the difference and not the
whole, and the comparison this repository reports is flattering by omission.

So: how much of the profit comes from the shape of an ordinary day, and how much
from knowing what tomorrow specifically will do?

## What the arms are

All on the same 730 validation days, the same 1 MW / 2 MWh battery, the same €8
wear, the same two-cycle cap, settled at realised prices exactly as the backtest
settles.

* **Ceiling, `perfect_foresight`:** the saved backtest value. Unchanged.
* **Control, `median_forecast`:** the production model's saved dispatch. It must
  reproduce the saved validation profit to the cent.
* **Reference, `naive_previous_day`:** the existing naive forecaster, for
  continuity with the published comparison.
* **`fixed_shape`:** no forecast. For each delivery day, the mean price of each
  local time of day over the **trailing 730 days ending before that day**, and
  the production optimiser solved once on that average profile. The schedule
  uses nothing about the day it trades.
* **`fixed_shape_seasonal`:** the same, with the average taken over the same
  calendar month only, so a June day is dispatched on what past Junes looked
  like. This is the fair version, because summer and winter shapes differ
  enormously, and it is the one to headline.

Both fixed arms are climatology, not forecasts: they use only what a day of that
kind usually looks like, and nothing about tomorrow. The trailing window matches
the length of the production model's training window, so neither side has more
history than the other. Averages are taken by local time of day and mapped onto
the target day's own index, so the 23 and 25 hour days are handled without
special-casing.

## What is reported

This is a measurement, not an experiment: there is nothing to adopt or reject,
so there is no pass mark. Reported:

* capture for every arm, and profit in euros over the 730 days;
* the decomposition the question asks for: what the shape alone earns, what a
  naive forecast adds on top, what the production model adds on top of that, and
  what is left unreachable;
* paired daily differences with 95% moving-block bootstrap intervals (7-day
  blocks, 5,000 draws, seed 7) for production against each fixed arm, so the
  gap the forecaster buys has an interval rather than a point;
* the June to September and October to May split, because the seasonal arm
  should gain most where the shape is most stable;
* cycles a day for each arm, since a fixed schedule cannot respond to a day that
  does not deserve two cycles.

## What the outcome means, decided before seeing it

There is no result here that counts as failure, and the number is reported
whichever way it falls.

* **A low fixed-shape capture** means the shape is worth little and the forecast
  is carrying the result. The existing headline stands as it is.
* **A high fixed-shape capture** means most of the money is in the clock, and
  the article's framing has to change: the forecaster's contribution is the gap
  above the clock, not the gap above another forecast. That is the more likely
  outcome given how regular the German shape is, and it is the reason for running
  this at all.
