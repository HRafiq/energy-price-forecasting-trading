# A model of the chance of an evening spike, feeding the dispatch: a fixed plan

Written 2026-09-19, before any code for it existed, and not changed after the run
starts.

## The question, in plain words

Two experiments have now said the same thing from opposite sides. Telling the
production model why evenings spike did not help: on spiky evenings its guess ran
about €30 too low before and €28 too low after. Training it on the money helped a
little, because the reward changed rather than the inputs. The forecaster is asked
for one price per quarter-hour and, on a day that might spike, a model trained to be
accurate on average hedges towards the ordinary evening. It is the wrong question.

This experiment asks a different one: how likely is it that this evening spikes?
That answer is a probability, which a model is allowed to state without hedging.
The dispatch then bets on it, two ways, and the test is whether either earns more.

## What is already known, and drives the design

On the 730 validation days, 131 evenings (17:00 to 20:59) reach €200. On those
evenings the real price averages €196, the production q50 €162, and the production
q90 €194. So the forecast already carries a good spike-level price in its upper
range; what it lacks is a reason to use it. The tuning window before validation
(63 days) holds only 2 spike evenings, so nothing here is tuned there: every
setting is fixed by rule.

## The pieces

**The spike model.** One LightGBM binary classifier per refit, on the production
refit calendar (every 28 days over the saved comparison run's days), trained on the
730 days before each refit day. One row per day. Label: the day's evening maximum
price reached `evaluation.spike_threshold_eur_mwh` (€200). Inputs, all known at
11:40 on the day before: the nine `spike_drivers` columns, and day-level means of
the standard features (load forecast, onshore and offshore wind power, radiation,
temperature, the residual-load estimate, the gas plant's marginal cost, the
previous day's mean and max price, the seven-day same-clock mean) with weekday,
holiday and day of year. Parameters fixed now: 300 trees, learning rate 0.05, 15
leaves, at least 20 rows per leaf, 80% row and column sampling, seed 7. It is not
calibrated further; its Brier score, log loss and reliability by probability bin are
reported.

**The dispatch arms,** all on the production forecast, the 1 MW / 2 MWh battery, €8
wear, two-cycle cap, settled at real prices:

* **Control, `median`:** median dispatch as today.
* **`blend`:** for the evening quarter-hours the price the optimiser values is
  `(1 - p) q50 + p q90`, where `p` is the day's spike probability; every other
  quarter-hour stays at q50, and both legs use the same curve. Confidence moves the
  evening price from the median towards the forecast's own upper range.
* **`hold_back`:** median dispatch, plus a rule: when `p` is at least 0.5, the
  battery must be full at 17:00, so the whole store is there for the evening. The
  optimiser gets this as a floor on the state of charge; prices are unchanged. It
  may still sell and refill earlier in the day if that pays, which the plan first
  said it could not; the floor is the rule.
  The threshold of one half is fixed in advance, as "more likely than not"; the
  results at 0.3 and 0.7 are reported, not chosen from.

## The criterion, fixed before the run

For each arm separately: adopt it only if its mean daily profit difference against
`median` over the 730 validation days has a 95% moving-block bootstrap interval
(7-day blocks, 5,000 draws) entirely above zero. If both pass, the one with the
higher mean is preferred and the difference between them is reported with its
interval. No other arm is added after the run.

## Reported alongside, not part of the decision

The classifier's Brier score against the base rate, its log loss, area under the
ROC curve and reliability table; each arm's capture, cycles a day and planned
value minus settled profit; the difference June to September and October to May;
cash by clock block; the difference on spike evenings against the rest; and how
often `hold_back` fired.

## Risks named in advance

1. **A weak classifier.** Spikes are rare and their drivers are the ones the
   features experiment just found unhelpful; the probabilities may be barely better
   than the base rate, in which case both arms trade noise.
2. **Blend optimism.** Raising the evening valuation on days that do not spike
   repeats the q75 test's loss, which shifted energy into the evening at a cost.
   The blend limits this by scaling with `p`, but a poorly calibrated `p` does
   not.
3. **Hold-back costs afternoons.** On flagged days that do not spike, the charge
   held for the evening misses a real afternoon sale; the false-positive rate
   decides this.
4. **The threshold.** One half is a convention, not a tuned value; the reported
   0.3 and 0.7 results say how sensitive the arm is, but cannot be adopted.

## What each outcome means

* An arm's interval above zero: adopt it, push, production note, and it becomes
  the next candidate for the live pipeline (a registered spike model beside the
  forecaster, and the dispatch rule in the plan step).
* Neither: reported as tested and rejected, and the sequence of three experiments
  (money loss, spike features, spike probability) is written up together as what
  the evening loss does and does not respond to.
