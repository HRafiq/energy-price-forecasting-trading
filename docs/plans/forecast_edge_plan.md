# What kind of day does the forecast earn its keep on: a fixed plan

Written 2026-09-27, before any code for it existed, and not changed after the run
starts. Committed on its own before the code.

## The question, in plain words

T8 measured a rule that forecasts nothing: average the price of each local time of
day over the trailing 730 days of the same calendar month, and dispatch on that.
Over the 730 validation days it captured 84.9% of the perfect-foresight ceiling
against the production forecast's 90.1%. The forecast is worth €11.75 a day on a
1 MW / 2 MWh battery, and 5% of days carry 41% of that total.

Concentration alone does not say the forecast is doing anything intelligent. A
heavy-tailed difference between two similar strategies will concentrate whether or
not the big days have anything in common. The claim I want to publish is stronger
than that: **the forecast earns its keep on the days that go wrong.** This
experiment tests it, and is designed so that it can fail.

## The edge

For each delivery day, the euros the production forecast earned over the seasonal
fixed rule: `median_forecast - fixed_shape_seasonal`, both settled at realised
prices, taken from T8's saved daily table. Nothing is re-solved.

## What describes a day

Each is computed from realised prices and the fixed shape alone. None uses the
forecast or its errors, so none can be circular: they describe the day, not the
model's opinion of it.

* **`shape_agreement`:** Spearman rank correlation between the day's realised
  price vector and its seasonal fixed shape. This is the mechanism under test. The
  optimiser needs the *ranking* of periods, not their prices, so a day whose
  ranking held is a day the fixed rule had no reason to get wrong.
* **`spread`:** the day's highest realised price minus its lowest. How much money
  was on the table.
* **`peak_shift`:** hours between the period of the day's realised maximum and
  the period of the fixed shape's maximum. A direct timing measure, and the same
  quantity the existing finding "timing matters more than size" is about.
* **`spike`:** whether the day's maximum reached the evaluation spike threshold,
  €200/MWh. The repository's existing vocabulary, so the answer joins up with T1
  and T5.

## The analysis

For each continuous characteristic: split the 730 days into quintiles, and report
the mean daily edge in each with a 95% moving-block bootstrap interval, 7-day
blocks, 5,000 draws, seed 7, exactly as every other experiment here. `spike` is a
two-group split. Report each characteristic's Spearman correlation with the daily
edge, and the share of the total edge the top quintile carries.

**The trap this must not fall into.** The edge is bounded above by what perfect
foresight could earn that day, which itself grows with the spread. A pattern in
euros could therefore be nothing but "big days are big". Every characteristic is
therefore reported twice: in euros, and as a share of that day's perfect-foresight
profit. A finding that survives normalisation is about the day's character; one
that does not is about its size.

## The predictions, fixed before the run

* **P1, primary.** Mean edge is higher in the lowest `shape_agreement` quintile
  than in the highest, with 95% intervals that do not overlap, in euros **and**
  normalised.
* **P2.** Mean edge rises with `spread`. Expected to hold in euros; the honest
  test is whether it survives normalisation.
* **P3.** Mean edge is higher on spike days than on other days.
* **P4.** Mean edge rises with `peak_shift`.

## What each outcome means

* **P1 holds.** The sentence is earned and publishable with the quintile table
  behind it: the forecast pays on the days that depart from their own seasonal
  pattern, which is exactly what a forecast should be for.
* **P1 fails, P2 holds in euros only.** The edge is about magnitude, not
  abnormality. The sentence becomes the weaker and duller "the forecast is worth
  more when there is more money on the table", and the article says that instead.
* **Nothing holds.** The concentration is heavy-tailed noise between two similar
  strategies. The sentence is deleted, and all that can be claimed is a mean of
  €11.75 a day with its interval.

No arm is adopted or rejected here: this is a measurement, not a decision, and it
changes what the article may claim rather than what the desk does.
