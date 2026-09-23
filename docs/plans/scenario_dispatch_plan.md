# Dispatching on price paths instead of point forecasts: a fixed plan

Written 2026-09-23, before any code for it existed, and not changed after the run
starts. Committed on its own, before the code, so the order is checkable here.

## The question, in plain words

The battery solves its day against one number per quarter-hour, the median of the
forecast. The forecast also carries ranges, but they are calibrated one
quarter-hour at a time: ninety-six honest marginal distributions that say nothing
about how the periods move together. What the battery actually trades is the
ordering across periods, whether 19:00 beats 18:00 and by how much, and that
ordering is exactly what marginals do not contain.

The T1 mechanism run measured the cost of getting it wrong. The forecast put the
evening window's peak in the right hour on 510 of 730 days, against 478 for the
naive rule of copying yesterday's peak hour, and the days that were an hour or
more out carry 41.5% of the window's cost (29.2 to 55.1). When it was wrong it was
early on 112 days and late on 108: symmetric, which is why selling the window at
q75 failed. A bias correction cannot fix a variance problem.

So: does dispatching against a set of coherent whole-evening price paths, instead
of one path, earn more?

## Why the obvious version of this cannot work, and what that leaves

The optimiser maximises

    sum over t of dt * ((sell_t - wear) * discharge_t - buy_t * charge_t)

which is linear in the prices for a fixed schedule, and every constraint it is
subject to (power, capacity, state of charge, the cycle cap, ending where the day
started) is price-independent. So for any schedule x and any distribution of
prices P,

    E[ profit(x, P) ]  =  profit( x, E[P] )

and therefore the schedule that maximises expected profit over a scenario set is
the same schedule that maximises profit at the scenario mean. Generating a
thousand coherent paths and dispatching on their average changes nothing that
dispatching on the mean forecast would not already change.

That is not a hypothesis. It is already measured: on the same 730 days, median
dispatch made €149,498.16 and mean dispatch €149,523.09, a difference of €25 over
two years. Whatever dependence structure the paths carry, a risk-neutral objective
integrates it away.

The consequence for this experiment is the whole of its design. Scenarios can only
change the schedule through an objective that is **not** linear in the realised
profit. So the arms below are risk-sensitive, and the experiment is really asking
a sharper question than the one it started with: is it worth giving up expected
value to avoid being an hour wrong?

## The scenarios

Built from the production model's saved validation forecasts, with no information
from on or after the delivery day:

* For each past delivery day, the realised price of each quarter-hour is placed
  within that day's forecast quantiles by linear interpolation, giving a rank in
  [0, 1] per period: a 96-vector describing how that day actually landed inside
  its own forecast.
* A scenario for the target day is one such historical vector, drawn from days
  strictly before it, applied back through the target day's own quantiles. The
  marginals are the model's; the dependence across periods is one observed day's.
* 200 scenarios per day, drawn without replacement from the most recent 200
  eligible days, so the sample is the recent dependence structure rather than a
  parametric guess. Seasonal mismatch is a named risk below, not a fix.

This is an empirical copula in everything but name, and it is deliberately
non-parametric: fitting a Gaussian copula would impose a dependence shape the
evening is unlikely to have.

## The arms

All on the same 730 validation days, the 1 MW / 2 MWh battery, €8 wear, two-cycle
cap, settled at realised prices exactly as the backtest settles:

* **Control, `median`:** the production schedule. It must reproduce the saved
  validation profit to the cent.
* **Reference, `mean`:** dispatch at the scenario mean. By the argument above this
  should reproduce mean-forecast dispatch, €149,523, and it is included as a
  check on the scenario machinery rather than as a candidate. If it does not come
  out within a euro or two of that, the scenario generator is wrong and the run
  stops.
* **`cvar_25`:** maximise `0.75 * E[profit] + 0.25 * CVaR_20%[profit]`, the mean
  blended with the average profit of the worst fifth of scenarios.
* **`cvar_50`:** the same with equal weight on the two.

CVaR enters the linear program the standard way, with one auxiliary variable for
the value at risk and one shortfall variable per scenario, so the problem stays
linear and solves with the same HiGHS relaxation the T5 work measured at about
7 ms a day.

## The criterion, fixed before the run

For each CVaR arm separately: adopt only if its paired daily profit difference
against `median`, settled at realised prices over the 730 validation days, has a
95% moving-block bootstrap interval (7-day blocks, 5,000 draws, seed 7) entirely
above zero. If both pass, the one with the higher mean is preferred and the
difference between them is reported.

## Reported alongside, not part of the decision

Profit on the 220 days the forecast misplaced the evening peak against the 510 it
did not, which is the subset this is aimed at; the share of days where the
schedule differs from the control at all, and by how many MWh; profit on spike
days against other days; the June to September and October to May split; cycles a
day; and the expected-profit sacrifice each arm makes at its own scenario mean,
which is the price being paid for the hedge.

## Risks named in advance

1. **Hedging costs expected value by construction.** Under a linear objective the
   risk-neutral schedule is optimal in expectation, so any CVaR arm gives up
   expected profit at the scenario mean. It can only win on realised prices if the
   forecast's own mean is biased where it matters. The evening under-forecast (49%
   of the hold-out shortfall came from forecasts too low between 18:00 and 21:00)
   says it probably is. That makes this partly a test of whether hedging
   compensates for bias, which is a confound and is reported rather than hidden.
2. **The scenarios inherit the marginals' bias.** The copula fixes dependence, not
   level. If the evening quantiles are low, every scenario's evening is low.
3. **Seasonal mismatch.** A December dependence vector applied to a June day may
   put its ramp in the wrong place. The recency window is the only mitigation and
   it is a weak one.
4. **Spreading may simply lose.** With peak errors symmetric at 112 early and 108
   late, a hedge that splits discharge across two hours earns the average of a good
   hour and a bad one on every day, including the 510 the forecast got right.

## What each outcome means

* An arm's interval above zero: adopt, push, production note, and scenario
  dispatch becomes a candidate for the live plan step.
* Neither: reported as tested and rejected, and the reason is itself the finding.
  It would mean the evening problem is not reachable from the dispatch at all under
  a linear profit, and that the remaining attack is on the forecast's conditional
  mean in the evening rather than on what the optimiser does with it.
