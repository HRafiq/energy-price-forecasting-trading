# Train the forecaster on the money: a fixed plan

Written 2026-09-18, before any code for it existed, and not changed after the run
started, apart from this postscript.

*Postscript, 2026-09-19:* the run passed the criterion; the result is in
docs/results/t2_money_loss.md and the T2 production note, with the checks run
afterwards (a rival, other seeds, where the gain sits) marked as such.

## The question, in plain words

Today the forecaster is trained to be accurate: it is rewarded for guessing every
quarter-hour's price closely, whether or not the battery trades in that quarter-hour.
The money then comes second: the optimiser takes the guess and decides when to buy
and sell. Two of our own findings say accuracy and money are not the same thing: a
weekly-refitted model was measurably more accurate and earned no more, and an
evening peak shifted by one hour cost as much as doubling the error everywhere.

So: if the forecaster is instead rewarded for the profit its guesses lead to, does
the battery earn more?

## The two arms

* **Control, `median_forecast`:** the production model as it is. Its point forecast
  (q50) values both legs of every trade; the battery is 1 MW / 2 MWh with €8 wear per
  MWh discharged and a two-cycle cap, refit every 28 days, settled at real prices.
  The control must reproduce the saved validation profit to within €0.01, which
  proves both arms read the same inputs.
* **Candidate, `money_point`:** the same model, then trained a little further on the
  money. Starting from the production model's own q50 as a fixed base, it adds K
  small correction trees whose only aim is the battery's profit. The corrected point
  values both legs of every trade. The forecast ranges (q05 to q95) are not touched;
  only what the optimiser is handed changes. Same refit days, same battery, same
  settlement.

## How "trained on the money" works

The battery's profit for a day is a straight sum over quarter-hours of price times
energy traded, minus wear. Given a price guess p_hat, the optimiser picks a
schedule; profit is then measured at the real prices p. That profit is not smooth in
p_hat (a small change in the guess can flip a whole schedule), so it cannot be used
directly as a training loss. The standard fix is the SPO+ loss (Elmachtoub and
Grigas, "Smart Predict, then Optimize", Management Science 2022): a convex stand-in
for the profit regret, with a known subgradient. For one day it needs two schedules:

* the schedule the optimiser would pick at the real prices p, and
* the schedule it would pick at the prices 2 p_hat - p.

The gradient of the loss with respect to the guess at quarter-hour t is
-2 dt (net_real_t - net_shifted_t), where net is discharge minus charge power in the
two schedules. In words: where the real-price schedule sells but the shifted one
does not, the guess is pushed up; where it buys but the shifted one does not, the
guess is pushed down; where the two agree, nothing moves. LightGBM takes this as a
custom objective with a constant second derivative, so each new tree is a step
along that gradient.

The schedules come from the same battery problem the optimiser solves, as a linear
program (the two-mode binary dropped) solved with HiGHS, because a training run
solves it about 70,000 times. Before anything is trained, that linear program must
reproduce the production optimiser's profit on 60 real validation days to within
€0.01, except on days with negative prices, where the relaxation may differ and the
count of such days is reported. Hourly products before 1 October 2025 keep their
constraint, so the gradient never asks for a schedule the auction could not trade.

## What is fixed before the run

* **Data:** the saved comparison forecasts of the production model, 2024-03-30 to
  2026-05-31, and the inputs file. The production model is refit on the same days
  the comparison run refit it (every 28 days from 2024-03-30); each refit's q50 for
  the next day must match the saved comparison forecast, which proves the base is
  the same model.
* **Tuning window:** 2024-03-30 to 2024-05-31, the 63 forecast days before the
  validation window. The candidate's settings are chosen there and then frozen:
  the learning rate from {1, 5, 20} (with the second derivative fixed at 1, so a
  tree's step is the learning rate times the mean gradient in its leaf) and the
  number of extra trees K from {25, 100}, read off the same training run. The
  setting with the highest total profit on the tuning days is used on validation,
  even if none beats the control there; that tuning result is reported.
* **Validation window:** the 730 days from 2024-06-01 to 2026-05-31, forecast
  walk-forward. Nothing on or after 2026-06-01 is read: the hold-out stays frozen.
* **Everything else** as in the production run: 730 training days per fit, the
  same features, the same LightGBM base parameters.

## The criterion, fixed before the run

Adopt `money_point` only if the mean daily profit difference against
`median_forecast` over the 730 validation days has a 95% moving-block bootstrap
interval (7-day blocks, 5,000 draws) entirely above zero.

## Reported alongside, not part of the decision

Total profit and capture of perfect foresight for both arms; the difference June to
September and October to May; the point forecast's pinball loss and MAE, which are
expected to get worse; cycles a day; planned value minus settled profit (a model
trained on the money may plan optimistically); cash in the 15:00 to 21:00 window and
by clock block; and the tuning-window result.

## Risks named in advance

1. **Weak signal.** The SPO+ gradient is zero on every quarter-hour where the two
   schedules agree, and on many days they agree everywhere. Learning may be too weak
   to move the forecast, giving a result within noise.
2. **Optimistic plans.** A guess pushed up where the battery should sell makes the
   plan's value overstate what settles; the planned-minus-settled figure will show
   it, as it did for the q75 test.
3. **Relaxation mismatch.** The training schedules come from a linear program, the
   trading from the mixed-integer one; on negative-price days they can differ.
4. **Overfitting to spiky training days.** Corrections learned on a volatile month
   could cost money on quiet months; the seasonal split will show it.
5. **Runtime.** About 70,000 linear programs per fit; 27 fits on validation plus
   tuning. Fits run in parallel across refit days.

## What each outcome means

* Interval above zero: adopt, push the branch, add a production note, and this
  becomes the headline finding.
* Interval containing zero: the idea is not shown to pay at this battery and
  horizon; report it as tested and rejected, like the sunset and q75 tests.
* Interval below zero: training on the money lost money; report it the same way.
