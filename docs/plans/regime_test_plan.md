# Does the forecast-free rule survive a different regime: a fixed plan

Written 2026-09-27, before any code for it existed, and not changed after the run
starts. Committed on its own before the code.

## The question, in plain words

T8 found that a rule which forecasts nothing captures 84.9% of the
perfect-foresight ceiling, against the production forecast's 90.1%. T9 found why:
four delivery days in five are ordinary, and on an ordinary day the rule is the
equal of the forecast. The forecast is paid for entirely on the fifth day, where
it holds 81% capture while the rule drops to 67%.

Both results come from one window, June 2024 to May 2026. The rule averages the
trailing 730 days of the same calendar month, so it is a **lagging estimator**: it
should fail when the shape of a day is moving, not when prices are merely large.
This tests whether it does, on the most disorderly two years in the dataset.

An earlier version of this plan justified the test by price volatility. That was
wrong and is recorded here as wrong: the battery trades spreads, not levels, and
2022's shock was mostly a level shock. The mechanism under test is drift in the
**shape**, not the size of prices.

## The windows

Both 730 delivery days, both with a full 730 days of price history behind them.

* **`crisis`, 2021-06-01 to 2023-05-31.** Gas crisis and its unwind. Day-ahead was
  entirely 60-minute products.
* **`recent`, 2024-06-01 to 2026-05-31.** The T8 validation window exactly, so the
  numbers join up with what is already published.

**A confounder named in advance.** The two windows do not trade the same product:
quarter-hourly day-ahead began on 2025-10-01, part-way through `recent`. Finer
products give the battery more freedom and could move capture on their own. The
`recent` window is therefore also reported split at that date, so the hourly part
of `recent` can be compared with the wholly hourly `crisis` window.

## The arms

All solved with the production optimiser, the 1 MW / 2 MWh battery, €8 wear,
two-cycle cap, tied to the day-ahead products actually traded that day, and
settled at realised prices.

* **`perfect_foresight`:** solved on the day's realised prices. The ceiling. In
  the `recent` window it must reproduce the saved backtest ceiling to the cent.
* **`fixed_shape`** and **`fixed_shape_seasonal`:** as T8 defines them.
* **`yesterday`:** the previous delivery day's realised prices, mapped by local
  time of day, used as the forecast. The most trivial forecast that exists, and
  the only one available in the `crisis` window, where no trained model exists.
  It is not the repository's `naive_previous_day` model, which carries the full
  quantile pipeline, and is named differently to keep them apart.
* **`median_forecast`:** the production forecast, reported for `recent` only,
  where it exists. There is no trained model for the `crisis` window and building
  one is out of scope here.

## The predictions, fixed before the run

* **R1, primary.** `fixed_shape_seasonal` captures at least 5 percentage points
  less in `crisis` than in `recent`.
* **R2, the crossover.** `yesterday` beats `fixed_shape_seasonal` in `crisis`. It
  loses to it in `recent`, so a reversal is the cleanest evidence that responding
  to current conditions is what pays when the shape moves.
* **R3, the mechanism.** The share of ordinary days, T9's shape agreement at or
  above 0.90, is lower in `crisis` than in `recent`. Reported per calendar year
  across everything the data allows, from October 2020 on, since it needs only
  realised prices and no dispatch at all.

## Reported alongside, not part of the decision

Capture and profit in euros for every arm in both windows, because capture ratios
are not comparable across regimes when the ceiling itself moves by a factor;
cycles per day; losing days; the hourly and quarter-hourly split of `recent`; and
the per-year series of ordinary-day share.

## What each outcome means

* **R1, R2 and R3 hold.** The forecast-free rule is a fair-weather approximation,
  T8's 84.9% is conditional on a settled regime, and the article must say so.
* **R3 holds, R1 and R2 fail.** The days did change and the rule coped anyway. The
  rule is then tougher than its construction suggests, and T8's finding gets
  harder to argue with, not easier.
* **Nothing holds.** The rule is regime-robust across a gas crisis. T8 stands
  unqualified and the article has to report that a rule with no forecast in it
  survived the worst two years in the record.

## Limits, stated in advance

One episode is one episode; this cannot establish what fixed rules do in general.
Capture ratios across windows are shares of very different ceilings. And the
production forecast's own number for the `crisis` window is not obtainable here,
so R2 compares the rule against a trivial forecast, not against the real one.
