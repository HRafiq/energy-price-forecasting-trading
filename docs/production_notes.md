# Production notes

The failure catalogue for the forecasting and trading pipeline. Each entry records
what can break, what was measured, and the mitigation. Measured facts are filled in
as phases land.

---

## D1 · Missing or late upstream data (Phase 0: observed, Phase 7: fallback chain built)

**Observed:** the build of 13 Sep 2026 found SMARD's prices for local day
13 September missing, 96 quarter-hours, while prices for 14 September were
already published. Energy-Charts had the missing day.

**What exists now:** the quality report counts gaps per column, and the SMARD
client re-downloads unsettled weeks on every run, so a late day is picked up on
a later refresh.

**Since Phase 1:** both baselines fill a missing source day from a second lag,
then from the last published price, and record how many periods needed it.

**Built in Phase 7:** the live pipeline walks a fallback chain from the production
model down to the two baselines (`src/pipeline/chain.py`), and every step down writes
an incident to the health log (`src/health/incidents.py`). D5 describes it.

## D2 · Data revisions (Phase 0: partial)

**What breaks:** actuals and forecasts are revised after publication. A model
trained on today's revised history saw cleaner inputs than it would have had
at the time.

**What exists now:** the SMARD client caches raw weekly chunks. A chunk counts
as settled once four newer chunks exist, set by `smard.refresh_recent_chunks`.
Each cached chunk records how many newer chunks existed when it was downloaded,
and only a chunk that had already settled then is reused.

**Bug found in review, fixed:** the first version re-downloaded only the newest
four chunks on each run. A week cached while still incomplete kept its gaps
forever once runs were more than four weeks apart, and the quality report still
passed because value gaps are reported, not failed. Caught by an independent
verification pass; the regression test is
`test_chunk_cached_while_incomplete_is_refetched_after_it_settles`.

**Known limit:** revisions published after a chunk has settled are still missed.

**Not done yet:** detecting revisions by diffing a re-fetched chunk against its
cached copy, and snapshotting training data as-of a date.

## D3 · DST days (Phase 0: test passing)

**What breaks:** a delivery day in Europe/Berlin has 23 hourly periods in March
and 25 in October. Code that assumes 24 local hours per day either crashes or,
worse, silently duplicates or drops an hour.

**Measured:**
- SMARD timestamps are true UTC instants. The weekly chunks containing the 2024
  DST changes hold 167 and 169 hourly points.
- Localizing a naive 24-hour grid raises on both DST days. With
  `nonexistent="shift_forward"` it instead produces a duplicated hour with no
  error, which the quality report catches as a 24-row day expected to have 23.

**Mitigation:** all timestamps in `src/` are UTC. Periods within a delivery day
are identified by ordinal, never by local hour (`src/timegrid.py`). The quality
report checks every day's row count against the DST calendar. Tests:
`tests/test_timegrid.py`, `tests/test_quality.py`.

**Observed gap:** SMARD has no load forecast for local day 2020-01-31, all
24 hours. The quality report lists it and the forecast residual load is NaN
there. No fill was added: the day stays NaN in the dataset. LightGBM reads missing
values natively, the quantile forest replaces them with a sentinel and LEAR with
training medians, and the day lies outside every training window of the validation
and hold-out runs.

## D4 · Granularity change (Phase 0: policy set)

**What breaks:** day-ahead products became 15-minute for delivery from
2025-10-01. Mixing hourly and quarter-hour rows in one table corrupts lags,
daily aggregates and the optimizer's time step.

**Measured on SMARD:**
- Before the switch, the quarter-hour price series repeats each hourly value
  four times. Distinct quarter-hour prices start in the chunk beginning
  2025-09-28 22:00 UTC.
- After the switch, the hourly price equals the mean of its four quarter-hour
  prices.
- Hourly volumes equal the sum of the four quarter-hour volumes, so every
  volume is energy per interval.

**Measured on SMARD before the switch:** load, wind and solar are genuinely
quarter-hourly, both actuals and forecasts. Only the price repeats within each
hour.

**Mitigation:** one modeling resolution, quarter-hourly, set in
`config/settings.yaml`. Config validation refuses a SMARD resolution that
differs from it, and ingestion raises `GranularityError` if the data's step
differs. Pre-switch rows carry `price_product_minutes = 60`, and the quality
report fails if any pre-switch hour has differing quarter-hour prices.

## M3 · Leakage in disguise (Phase 0: found, Phase 1: enforced, Phase 2: measured)

**What breaks:** a feature that is not known at 12:00 on day D, when bids for
day D+1 close. A backtest using it looks better than any live system can be.

**Found:**
- SMARD's day-ahead wind and solar forecasts are submitted at 18:00 the day
  before delivery, six hours after the gate. EU Regulation 543/2013 puts the
  same deadline on every TSO: Article 14(1)(d) names the data item, a forecast
  of wind and solar generation per bidding zone for each market time unit of the
  following day, and Article 14(2)(d) sets it to be published no later than
  18:00 Brussels time one day before delivery, so switching source does not
  help. ENTSO-E's own duty to publish it follows from Article 3.
- The day-ahead load forecast is due two hours before gate closure, Article
  6(1)(b), but may be updated afterwards. No vintage is published.
- These forecasts track actuals closely: correlation 0.995 for solar and 0.987
  for onshore wind over the tuning period. A model fed the post-gate forecasts
  gets nearly the advantage it would get from actuals.

**Enforced since Phase 1:** models never see the raw dataset. The information
set in `src/forecasting/information.py` removes every value not published at
11:40 on day D, column by column, and refuses columns without a rule.
`tests/test_information.py` replaces all unpublished data with garbage and
requires identical forecasts from a deliberately greedy model and from both
baselines.

**Measured in Phase 2** (`python -m src.health.experiments.m3_leakage`, results in
`docs/results/m3_leakage.md`): the same LightGBM quantile model, walk-forward
over 1 June 2025 to 31 May 2026, on three feature sets.

| feature set | mean pinball | change vs honest | MAE of median, EUR/MWh | spike days, mean pinball |
|---|---|---|---|---|
| honest, gate-available only | 4.57 | 0% | 14.22 | 6.85 |
| adds grid-operator D+1 forecasts, published 18:00 | 3.95 | -13.6% | 12.57 | 6.44 |
| adds measured D+1 wind, solar and load | 4.00 | -12.5% | 12.72 | 6.24 |

The post-gate forecasts flatter the backtest almost exactly as much as true
actuals do, because they track actuals so closely. Every reported model uses
only the honest set; a dashboard built on either leaky set would promise errors
about 13% smaller than live trading could deliver.

## M3 · Leakage through a weather archive (Phase 2: found and blocked)

**What breaks:** Open-Meteo's Previous Runs API labels values as forecasts
issued 48 hours before valid time. For valid times less than about 45 hours
ahead it silently returns newer forecasts under that label. A pipeline that
downloads recent data and trusts the label trains and forecasts on fresher
weather than was available at 11:40.

**Found by:** the weather ingestion work on 2026-09-13, while verifying the lead
time on recent dates.

**Mitigation:** ingestion masks every value stamped later than the download time
plus the lead time minus a 10-hour margin. Every period of the next local day
stays visible at 11:40. A cached chunk only counts as settled when it ended well
before the download day; the first version compared with the requested end date
instead, which an independent review showed could cache fresher values for good.

## D1 · Gaps in fuel prices (Phase 2: observed)

**Observed:** the TTF gas ticker follows the US exchange calendar and misses 76
European trading days. EU carbon auctions pause for about three weeks each
January.

**Not a gap:** flat, zero-volume TTF bars look like stale prints but are mostly
real settlement-only days, 183 of 251 bars in 2022. They are kept; dropping them
erased months of the gas crisis.

**Mitigation:** fuel columns carry the last price published before each day, so
a missing day repeats the previous price. Gas prices move slowly day to day, so
a one-day repeat costs little; a three-week carbon pause is visible in the data.

---

## M4 · Negative prices (Phase 3: verified in the optimizer)

**What breaks:** code that assumes a positive price. MAPE divides by it, and a rule
such as "charge below €30" never expects to be paid to charge. A linear program
without a binary goes further: at a negative price it charges and discharges at
once and books profit for burning energy.

**Measured:** a test charges through quarter-hours at -€30 and settles exactly the
€22.80 the hand calculation gives. Another shows the relaxation booking €3.00 on a
day whose real optimum is €1.50. Forecast evaluation uses pinball loss and MAE,
never MAPE.

**Mitigation:** one binary per period forbids charging and discharging at once.
Settlement multiplies the price by net power, so buying at a negative price earns
money.

---

## T4 · Imbalance risk (Phase 4: measured)

**What breaks:** a battery that cannot deliver its committed schedule pays the
imbalance price on the shortfall.

**Simplification:** the backtest trades day-ahead only, and every schedule is
feasible by construction. The optimizer respects power, capacity and efficiency,
and each day starts and ends at the same state of charge, so no day's position
depends on a forecast being right. Imbalance comes from an outage or from
intraday trading changing the plan.

**Measured,** validation window, median dispatch: the battery delivers nothing
during an outage and follows its committed schedule as far as its state of charge
allows afterwards, with every deviation settled at the German imbalance price,
reBAP, from ENTSO-E. An average day earns €204.79.

| Outage | Mean cost, € | 95th percentile, € | Worst day, € | Days the cost exceeds the day's profit |
|---|---|---|---|---|
| Two hours at a random time | 43.53 | 208 | 5,314 | 7.3% |
| The worst two-hour window of each day | 277.11 | 751 | 9,772 | 50.8% |
| Four hours at a random time | 76.31 | 296 | 9,641 | 13.6% |

The damage usually lands after the outage, not in it. On 26 August 2024 a two-hour
outage from 12:00 missed the midday charge; the committed evening sale then failed
at 19:45, when reBAP was €13,194, and the day cost €9,772, 48 days of profit.
Nothing is re-traded on the intraday market once the outage is known, which would
usually reduce the cost. The figures are not an upper bound either: on 22% of
random-outage days the outage gains, because a missed charge is paid at reBAP.
Details: `docs/results/t4_imbalance.md`.

**Mitigation, not yet built:** check availability before bidding, close an outage's
position on the intraday market instead of leaving it to reBAP, and hold back
energy on days when the evening carries spike risk.

---

## T5 · Price-taker assumption (Phase 3: stated)

**Assumption:** the battery's orders do not move the clearing price, and every
committed MW settles at it. For 1 to 5 MW in a market that clears tens of GW this
holds; for a portfolio of hundreds of MW it would overstate profit, because selling
into the evening peak lowers that peak.

**Also left out:** grid fees and taxes, and the auction's volume increments.
Orders are volumes without limit prices, so a forecast strategy can buy into a
surprise spike; a desk would bid price-quantity curves. The P&L is trading margin
net of wear. The follow-up below tests that last point and rejects it.

---

## T5 · Bid curves from the quantile fan (validation: tested, rejected)

**Question:** the note above names the price-taker assumption as a simplification
and price-quantity curves as what a desk would bid instead. The forecast already
carries the natural limits: its lower quantiles say what a bad price for a sale
looks like, its upper quantiles what a bad price for a purchase looks like. Does
bidding with limits from the fan earn more than bidding volumes?

**Design, fixed before the run** (`docs/plans/bid_curves_plan.md`, committed before
the code): the production schedule's volumes on the 730 validation days, bid three
ways. `volumes` is the backtest as it stands, fixed volumes at the clearing price,
and it must reproduce the saved validation profit to the cent. `limits_q25` puts a
limit on every order from the fan, sales at q25 and purchases at q75; `limits_q10`
uses q10 and q90. An order executes only if the realised price reaches its limit.
The day is then replayed period by period through the store: a sale the store
cannot cover is a delivery failure settled at the German imbalance price, as T4
settles deviations, and the end-of-day state-of-charge gap is valued at the day's
terminal prices. Criterion for each arm: adopt only if its paired daily profit
difference against `volumes` has a 95% moving-block bootstrap interval (7-day
blocks, 5,000 draws) entirely above zero.

**Measured** (`docs/results/t5_bid_curves.md`):

| arm | profit | capture | withheld sales | withheld purchases | undelivered | imbalance cash |
|---|---|---|---|---|---|---|
| volumes | €149,498 | 90.10% | 0 | 0 | 0.0 MWh | €0 |
| limits_q25 | €92,818 | 55.94% | 2,841 | 4,137 | 702.3 MWh | -€44,271 |
| limits_q10 | €119,158 | 71.82% | 1,332 | 1,864 | 386.7 MWh | -€29,563 |

Against `volumes`, per day: `limits_q25` -€77.64 (-100.65 to -56.87), `limits_q10`
-€41.56 (-60.29 to -25.77). Neither interval lies above zero, so neither is adopted.
Both arms lose more on spike days (-€129.52 and -€76.76) than on other days, which is
the opposite of what limits are meant to protect against.

**Why, and this is the useful part:** with imbalance and the end-of-day gap left
out, the difference is -€11.94 (-30.00 to +7.96) for `limits_q25` and -€1.15
(-12.64 to +11.24) for `limits_q10`. Both intervals cross zero. The withheld orders
themselves cost almost nothing; the entire loss is the consequence of withholding
them. Buy and sell orders come in pairs. A purchase withheld at midday because the
price ran above its limit is the energy an evening sale was counting on, and the
battery then cannot deliver what it committed. Withholding the purchase saved a few
euros; failing to deliver the sale cost tens of thousands.

**Reading:** limits are not a free option here because the optimiser was never told
about them. It plans a day whose sales depend on its purchases, and then the limits
break that chain one order at a time. Bidding curves properly means planning for
them, with linked or block orders or a stochastic plan over the fan, which is a
different and larger piece of work than this experiment. What is rejected is the
cheap version, and it is rejected for a reason that would apply to any strategy that
withholds one leg of a paired trade.

**Not tested here:** linked orders and block orders. An optimiser that plans
against its own uncertainty is the T7 note below.

---

## T7 · Dispatching on price paths instead of point forecasts (validation: tested, rejected)

**Question:** the battery plans its day against one number per quarter-hour, the
median. The forecast's ranges are calibrated one quarter-hour at a time, so
ninety-six honest marginal distributions say nothing about how the periods move
together, and what the battery trades is exactly that: whether 19:00 beats 18:00
and by how much. The T1 mechanism run measured the cost of getting it wrong. The
forecast put the evening window's peak in the right hour on 510 of 730 days
against 478 for copying yesterday's peak hour, and the days an hour or more out
carry 41.5% of the window's cost (29.2 to 55.1). When it was wrong it was early on
112 days and late on 108, which is why selling the window at q75 failed: a bias
correction cannot fix a variance problem. Does dispatching against a set of
coherent whole-evening price paths earn more than dispatching against one?

**The part worth keeping, before any result.** It cannot, under a risk-neutral
objective, and this is provable rather than measured. The optimiser maximises

    sum over t of dt * ((sell_t - wear) * discharge_t - buy_t * charge_t)

which is linear in the prices for a fixed schedule, and every constraint it is
subject to (power, capacity, state of charge, the cycle cap, ending where the day
started) is price-independent. So for any schedule x and any price distribution P,
`E[profit(x, P)] = profit(x, E[P])`, and the schedule maximising expected profit
over a scenario set is the schedule at the scenario mean. Generating a thousand
coherent paths and dispatching on their average cannot differ from dispatching on
that average. Checked in code and not merely argued: the risk-neutral schedule
over 200 paths and the schedule at their mean differ by 0.00 MW. Anyone holding a
quantile forecast and a linear dispatch can stop here rather than build the
machinery.

That leaves risk-sensitive objectives as the only way a scenario set can act, so
the question the experiment actually answers is whether it is worth giving up
expected value to avoid being an hour wrong.

**Design, fixed before the run** (`docs/plans/scenario_dispatch_plan.md`, committed
on its own before the code, so the order is checkable): scenarios come from an
empirical copula. For every past delivery day the realised price of each
quarter-hour is located inside that day's own forecast quantiles, giving a rank
per period: a vector of how that day landed within its forecast. A scenario for
the target day is one such vector, drawn from days strictly before it, read back
through the target day's quantiles. The marginals are the model's; the dependence
across periods is one real day's. 200 scenarios a day, and because a Berlin day
has 92, 96 or 100 quarter-hours a day only draws on days of its own shape. Arms:
`median` is the production control, `mean` dispatches at the scenario mean,
`cvar_25` and `cvar_50` maximise the mean blended with the average profit of the
worst fifth of scenarios. Criterion: adopt only if the paired daily profit
difference against `median` has a 95% moving-block bootstrap interval (7-day
blocks, 5,000 draws) entirely above zero.

**Tails, because the first run did not earn the word rejection.** 14.4% of
validation periods land outside q05 to q95, and on the 7.3% above it the realised
price averaged 138.7 EUR/MWh against a fan top of 116.9. Clamping those to the
edge asks a hedge to insure a calmer world than the one that exists, which would
have rejected the generator rather than the idea. Each tail now has an exponential
shape whose scale is a multiple of the day's own spread, fitted on past days only
and refitted as the pool moves: 0.58 below and 0.75 above. A 900 EUR/MWh price on a
fan topping at 136 comes back as 519 rather than 136. Values far beyond that are
still compressed by the rank clip.

**Measured** (`docs/results/t7_scenario_dispatch.md`), 526 validation days,
2024-12-19 to 2026-05-31:

| arm | profit | what its own scenarios promised |
|---|---|---|
| median (control) | €106,662 | |
| mean | €105,187 | €118,338 |
| cvar_25 | €105,103 | €117,931 |
| cvar_50 | €103,845 | €115,917 |

Against the control, per day: `mean` -€2.80 (-4.97 to -0.97), `cvar_25` -€2.96
(-4.42 to -1.60), `cvar_50` -€5.36 (-7.25 to -3.74). No interval lies above zero,
so nothing is adopted. Every arm is worse with honest tails than it was with
clamped ones.

**Why, and this is the useful part:** the scenario set promised €118,338 and the
days delivered €105,187, an optimism of €13,151 over 526 days against €3,179 when
the tails were clamped. Imagining bigger spikes makes the battery trade harder to
catch them, and they do not arrive where it imagined. More dispersion is not
better-placed dispersion. The loss also sits almost entirely in the point estimate
rather than the hedging: of `cvar_25`'s -€2.96 a day, the scenario mean carries
-€2.80 and the CVaR term only -€0.16. Hedging is nearly free here; it is the
scenario average that is a worse guide than the plain median.

**Reading:** the ordering across periods, which was the thing marginals were
missing, cannot be delivered through the dispatch, because a linear objective
integrates it straight back out. What remains is the forecast's own numbers in the
evening. That is a different problem from the one this experiment tested, and it
is where the next attack belongs.

**Not tested:** a non-linear objective arising from the market rather than from
risk preference, such as recourse against an intraday leg, which would give
scenarios something to act on that risk aversion does not.

---

## T1 · Error-direction asymmetry (Phase 4: measured)

**What breaks:** an average error hides which errors cost money. The thesis to test:
a forecast that is too low in the evening leaves the battery empty when prices
spike, while errors in flat midday hours cost little.

**Method:** each day's gap to perfect foresight is split into Shapley shares by
local hour block and by whether the median forecast was too low or too high. A
group's share is the profit gained by correcting its errors, averaged over many
orders of correcting the groups, so the shares add up exactly to the gap. Before
October 2025 an hourly product takes the direction of its mean error.

**Measured,** production model, median dispatch:

| Local hours | Validation: too low, € | Validation: too high, € | Hold-out: too low, € | Hold-out: too high, € |
|---|---|---|---|---|
| 00-05 | 693 | 1,177 | 150 | 223 |
| 06-10 | 2,782 | 892 | 717 | 190 |
| 11-14 | -30 | 3,540 | -300 | 452 |
| 15-17 | 3,557 | -271 | 29 | 74 |
| 18-20 | 2,280 | 963 | 1,561 | 213 |
| 21-23 | -178 | 1,018 | -255 | 145 |
| **Gap** | **€16,424 over 730 days** | | **€3,197 over 105 days** | |

- **Too low costs more than too high:** 55% of the gap on validation, 59% on the
  hold-out.
- **The evening is where it bites:** forecasts too low from 15:00 to 20:59 cost 36%
  of the validation gap; on the hold-out, too low from 18:00 to 20:59 alone cost 49%.
- **Midday is not free:** forecasts too high from 11:00 to 14:59 cost 22% of the
  validation gap. When the forecast puts the solar valley too high, the battery
  buys too little or in the wrong quarter-hours.

**Mitigation, not yet built:** the upper quantiles from 15:00 to 21:00 matter most,
as the two follow-up notes below show. No spike model is built, and the drift monitor
(M2) pools coverage over all periods rather than watching the evening.

---

## T1 · Late afternoon, not just the evening (validation: measured)

**Question:** the T1 table shows the 18:00 to 20:59 block taking 35% of the validation
gap from June to September but 16% from October to May, while 15:00 to 17:59 moves the
other way, 8% against 23%. Does the loss follow sunset, which comes more than four hours
earlier in December than in June, rather than the clock?

**Measured:** the same Shapley split of the production model's median-dispatch gap,
€16,424 over 730 validation days, repeated with windows anchored to each day's sunrise
and sunset at the centre of Germany, boundaries rounded to the whole hour so an hourly
product is never split. Both splits give identical per-day gaps. Two criteria were fixed
before the run and both passed on the estimates: the sunset window's share of the gap
differs between June to September and October to May by 7.0 points, under the limit of
10, and it takes 41.3% of the whole gap against 39.8% for the six hours of the 15-17 and
18-20 clock blocks. On 70 June and July days the window after sunset is cut at the end of
the delivery day, so the sunset window holds five hours, which lowers its summer share.
(docs/results/t1_sunset.md)

**Added after the run:** the criteria compared point estimates, and the intervals show
they cannot decide the question. The sunset window's 1.5-point lead over the clock
window has a 95% interval of -4.3 to +6.5 points, and the seasonal differences of both
windows have intervals about 25 to 40 points wide that include zero (moving-block
bootstrap over weeks). The data cannot tell sunset from the clock, and the claim that
the losses follow sunset is not made.

**What is clear:** about 40% of the gap is lost between 15:00 and 21:00, 43% from June
to September and 39% from October to May, though the difference between those two is not
pinned down. Within that window the loss shifts with the season beyond noise. On the
clock, 18:00 to 20:59 takes 19.6 points more of the gap in summer (95% interval +5.2 to
+39.9) and 15:00 to 17:59 takes 15.7 points less (-28.2 to -2.9). Against sunset the
shift runs the other way: the three hours before sunset take 20.8 points more in summer
and the three hours after it 27.8 points less.

**What it changes:** a fix aimed at the 18:00 to 21:00 evening peak alone would target a
block that carries 16% of the winter gap. The window to target is 15:00 to 21:00.

---

## T1 · How 15:00 to 21:00 loses money (validation: measured)

**Question:** the Shapley split puts €6,530 of the €16,424 validation gap on forecast
errors between 15:00 and 21:00, 39.8% (95% interval 32.0 to 47.2). Before choosing a
fix, what do the two schedules actually do differently there: sell at the wrong
quarter-hours, arrive with too little stored, keep too much back, or skip cycles?

**Measured:** the saved validation schedules of median dispatch and perfect
foresight, which reproduce the saved daily profit and the attribution's gaps to
within €0.01, read three ways. The window's cash is split exactly into discharge
volume, discharge price, charge volume, charge price and wear, and the cash gap is
also given for every clock block of the day. Each day's window cost is attached to
four flags whose thresholds were fixed before the data was read. And the forecast's
peak hour in the window is compared with the realised one. Moving-block bootstrap
intervals over weeks throughout. A read-only first pass of the flags came before the
module, so the reading below is of the evidence, not a test fixed in advance.
(docs/results/t1_mechanism.md)

**What is clear:**

- **The window's cost sits where the forecast came in below the outcome.**
  Forecasts below the realised price carry 89.4% of it (79.4 to 101.0). Across the
  day the pattern turns: on point estimates, forecasts below the outcome carry more
  of the cost from 06:00 to 10:59 and 15:00 to 20:59, and forecasts above it at night,
  from 11:00 to 14:59 and from 21:00. That is consistent with a forecast whose daily
  shape is too flat, missing peaks from below and troughs from above, as the T1 note
  already found at midday.
- **The cash traded in the window cannot be told apart.** Perfect foresight's window
  cash is -€3.55 a day against the median schedule's (-10.02 to +2.87), and even the
  upper bound is below the €8.94 a day the Shapley split puts on the window. Inside
  it, perfect foresight sells at better prices, +€4.08 a day (+2.59 to +6.09), and
  buys more energy, -€5.01 a day (-7.01 to -3.00), mostly October to May. Over the
  whole day its cash comes out ahead from 11:00 to 14:59, from 21:00 and at night.
  Block cash and Shapley cost are two accountings of the same gap, so that does not
  say where the window's forecast errors cost.
- **The battery is not short of energy going in.** The median schedule reaches 15:00
  with 0.08 MWh more stored than perfect foresight (0.04 to 0.12) and leaves at 21:00
  with 0.07 MWh less (0.03 to 0.11), about a third of a quarter-hour at full power.
- **No single mechanism.** Days where both schedules sell similar energy but perfect
  foresight gets the better price carry the largest share of the window cost, 44.9%
  (31.0 to 58.9), but the interval overlaps skipped cycles, 20.4% (6.7 to 35.5), and
  arriving emptier, 16.4% (4.3 to 34.0).
- **The peak hour is right more often than not, but not by much.** The forecast put
  the window's highest-priced hour in the right place on 510 of 730 days, against 478
  of 729 for simply taking the day before's realised peak hour. Days with the peak an
  hour or more out carry 41.5% of the window cost (29.2 to 55.1). The comparison is by
  whole hour, so it says nothing about timing within the hour.

**What it changes:** the first pass read the flags alone and pointed at timing inside
the window. That stays open: the one clearly non-zero part of window cash is a price
effect, and the peak-hour measure cannot see timing below the hour. What the evidence
does count against, on average, is a charge reserve before 15:00. The cost sits in
forecasts below the outcome, so the cheapest test of whether correcting that pays is
on the dispatch side: value selling in the window at the forecast's q75 instead of the
median, keep buying at the median, and judge it on validation profit against a bar
fixed before the run. Two risks are named in advance: the median schedule already
sells slightly more in the window than perfect foresight, and a change in the window
alone leaves the midday over-forecast cost untouched.

---

## T1 · Selling the window at q75 (validation: tested, rejected)

**Question:** the mechanism note above proposed the cheapest correction for forecasts
that come in below the outcome from 15:00 to 21:00: value selling in that window at
the forecast's q75 instead of the median, keep buying at the median everywhere, and
change nothing else. Does it earn more?

**Criterion, fixed before the run:** adopt it only if the mean daily profit difference
against median dispatch over the 730 validation days has a 95% moving-block bootstrap
interval (7-day blocks, 5,000 draws) entirely above zero.

**Measured:** both arms dispatched on the production model's saved validation
forecasts for the same 1 MW / 2 MWh battery with €8 wear, settled at the realised
prices; the median arm reproduces the saved validation profit to within €0.01. Median
dispatch made €149,498, 90.10% of perfect foresight; selling the window at q75 made
€147,605, 88.96%. The difference is -€2.59 a day (-3.95 to -0.89), €1,893 less over
the two years, so the criterion fails and the interval lies below zero. It loses in
June to September, -€3.46 a day (-5.46 to -1.38), and October to May cannot be told
from zero, -€2.16 (-4.00 to +0.05). (docs/results/t1_window_q75.md)

**Why it loses:** the schedule sold 1.83 MWh a day in the window against 1.56, but
only 3.41 MWh over the whole day against 3.28, so about half the extra window sales
came out of other hours and the rest from cycling more, 1.80 times a day against 1.73.
Cash from 18:00 to 20:59 rose by €20.24 a day (+14.87 to +24.83), while cash after
21:00 fell by most of that, -€17.11 a day (-21.13 to -12.15), and cash in the morning
and at midday fell too, -€2.37 (-3.60 to -1.36) and -€3.01 (-5.29 to -1.23); those
last two turn a small gain into the loss. The two arms differ only in the window's sell
valuation, so these shifts are the change's doing. That is the first of the two risks
named before the run: the median schedule already sold slightly more in the window than
perfect foresight, and a higher sell valuation there pushed it further the same way.
The plan also turned optimistic: planned value ran €18.56 a day above what the schedule
settled at, against €13.25 below for median dispatch.

**What it changes:** the dispatch rule stays as it is. A higher valuation in one window
shifted sales and cycling across the day and cost profit, which is consistent with the
mechanism note's reading that the forecast's daily shape is too flat rather than too
low in one block, though the shifts themselves follow from the battery's limits and do
not single out one reading. This one test does not rule out a correction over the whole
day's shape, in the forecast or in dispatch, but a window-only valuation is not one to
use.

---

## T1 · What the evening loss does not respond to (validation: two experiments, both rejected)

**The question in one line.** Of the money the battery misses against a trader who knows
tomorrow's prices, about 40% is lost between 15:00 and 21:00, mostly on evenings when
the real price jumps far above the forecast. Two experiments tried to fix that from the
forecasting side. Each had its pass mark written down before it ran (the plans are
`docs/plans/spike_features_plan.md` and `docs/plans/spike_probability_plan.md`): the
change is adopted only if its mean daily profit gain over today's schedule, across the
730 validation days, has a 95% range lying entirely above zero. The range is where the
daily gain would land if the two years were redrawn in week-long pieces, so a gain
counts only when it is too large to be a lucky run of days. Both experiments failed that
mark, and the way they failed is the useful part, so they are reported together.

Two words used throughout: the *dispatch* is the battery's plan for the day, when to
charge and when to sell, which the optimiser builds from the price forecast; a
*quarter-hour* is the market's 15-minute trading slot, 96 in a day.

**Experiment 1: tell the model why evenings spike.** An evening spikes when solar has
gone, demand is at its peak and the wind is low, so the price is set by gas plants and,
on the worst days, by imports or reserve plants. Nine new columns were added that say
this directly for the day ahead: the evening load forecast and how steeply it climbs
from the afternoon, how much afternoon solar there is to lose, how still the evening
wind will be, the evening residual load (demand minus wind and solar, the part the power
stations must cover) and how it compares with the last week, and whether recent evenings
spiked. The production model was refit with those columns, on exactly the same days and
settings as before, and compared on the 730 validation days.

*Result:* the battery made €148,944 with the columns against €149,498 without: €0.76 a
day less, with a 95% range of -€1.97 to +€0.52, so no measurable change. The model
barely used the columns (each took under 1% to 3% of its splitting gain), and the thing
they were meant to fix hardly moved: on spike evenings the forecast ran €30/MWh too low
before and €28/MWh too low after. The reason is plain once seen. The model already had
the hour, the load forecast, the wind and solar forecasts and the gas price, so it could
already tell that an evening would be tight; the next experiment's classifier, built
from the same inputs, shows the signal is there, and the forecast's own upper range
(q90, the price it puts a 90% chance of staying under) already averages €194 on spike
evenings against a real €196. It still guessed low, because a model trained to be
accurate on the typical evening hedges towards it. Missing day-level signals about
whether the evening will be tight were not the problem.
(docs/results/t1_spike_features.md)

**Experiment 2: predict the chance of a spike, and let the dispatch bet on it.** A
separate model was asked a question that allows an honest answer: how likely is it that
this evening's price reaches €200? It answers with a probability, which it can state
without hedging. It was trained on the same information, refit every 28 days like the
forecaster, and turned out well: on the 730 validation days it put the spiky evening
above the calm one 91% of the time (AUC 0.906), and its probabilities meant what they
said: of the 61 days it put above 70%, 84% spiked; of the 607 days it put below 10%, 8%
spiked. Two ways of using it were then tried in the dispatch, both fixed in advance.
*Blend* raises the evening price the optimiser plans against from the median towards q90
in proportion to the probability. *Hold back* requires the battery to be full at 17:00
whenever the probability is at least one half, so the whole store is there for the
evening.

*Result:* blend made €0.65 a day less than today's dispatch (-€2.14 to +€0.49), hold
back €0.27 a day more (-€0.27 to +€1.01). Neither range clears zero, so neither is
adopted. Hold back fired on 74 days and on 62 of them changed nothing: when a spike is
visible in the forecast, the optimiser already arrives at 17:00 full. It gained on 5
days and lost on 7, and a single day (+€232) is larger than its whole +€199 total. Blend
repeated what an earlier test had shown (T1 · Selling the window at q75 above): raising
the evening valuation moves sales into 18:00 to 21:00 (+€5.09 a day) and out of the
afternoon and the late evening (-€2.05 and -€2.47 a day), for no net gain.
(docs/results/t1_spike_probability.md)

**What the two say together, with the money-trained forecaster below (T2):**

- The evening loss is not a shortage of day-level signals. The model can already see a
  tight evening coming, and a good spike classifier built from the same inputs confirms
  that the signal is there.
- It is not a confidence problem either. Knowing that a spike is likely, with
  probabilities that mean what they say, does not say which quarter-hour it lands in.
  Holding charge for the evening is something the dispatch already does when the peak is
  visible.
- The one change that earned anything, the forecaster trained on the battery's profit,
  worked by sharpening the shape of the evening by a euro or two, and its gain came from
  a handful of dark, still winter evenings.
- All of this is consistent with the earlier timing result (T2 · Error cost is not error
  size: an evening shifted one hour early costs as much as random noise everywhere):
  what is left is timing within the evening on spike days. Moving it would take a
  forecast of the peak's shape, which quarter-hour and how high, on the days that spike.
  Nothing tried here does that, and the data at hand gives no cheap way to; that is a
  real information gap, at the quarter-hour rather than the day.

**Why the failures are reported.** Each had its pass mark fixed before it ran, and a
test that can only pass is not a test. The optimiser gained an optional floor on the
state of charge for the hold-back rule, and the nine columns stay as an opt-in feature
group; the production model and the live pipeline are unchanged.

---

## T2 · Error cost is not error size (Phase 4: measured)

**What breaks:** judging a forecaster by its average error. A battery's profit
depends on which hours are cheap and dear and by how much, so two forecasts with
the same error can earn very different amounts.

**Measured,** validation window, median dispatch, capture of perfect foresight:

| Synthetic forecast | Mean absolute error, €/MWh | Capture |
|---|---|---|
| Every price too high by the same amount | 16.0 | 100.0% |
| Random error in every quarter-hour | 16.0 | 85.1% |
| Random error only where perfect foresight idles | 16.0 | 79.3% |
| Random error only where perfect foresight trades | 16.0 | 77.7% |
| Afternoon and evening prices one hour early | 7.1 | 85.1% |

The error of €16.0 is the production model's own. A level shift leaves every spread
intact and costs €44 over two years. Shifting the evening one hour early has less
than half the error and costs as much as noise everywhere. Errors in idle hours are
not free either: they invent spreads the battery then trades.

Across the eight forecasting models the ranking by accuracy mostly holds, with
median dispatch capturing 77.6% for naive to 90.9% for QRA. The spread is far wider
for quantile-aware dispatch at q25: 87.8% for LightGBM quantile but 72.8% for the
quantile forest, whose wide ranges make it hold back too often.

**Mitigation:** judge forecasts by the profit they produce as well as their error,
and look at timing and spreads in the spread-defining hours.

---

## T2 · A forecaster trained on the money (validation: tested, adopted)

**Question:** the production model is trained to be accurate: rewarded for guessing
every quarter-hour's price closely, whether or not the battery trades then. The money
comes second, from the optimiser. T2 above and the weekly-refit note both say accuracy
and money are not the same thing. If the forecaster is instead rewarded for the profit
its guesses lead to, does the battery earn more?

**Design, fixed before the run:** the production model's q50 stays as a fixed base, and
100 extra LightGBM trees are boosted on the SPO+ loss (Elmachtoub and Grigas, 2022), a
convex stand-in for the profit shortfall whose gradient at a quarter-hour is the
difference between the battery's schedule at the real prices and its schedule at the
guessed prices, mirrored through the real ones: where the real-price schedule sells and
the guessed one does not, the guess is pushed up. The schedules come from the production
optimiser's linear relaxation solved with HiGHS, which reproduces the integer optimum to
the cent on real days except where a negative price lets the relaxation charge and
discharge at once (the check is on the results page). The corrected point values both
legs of dispatch; the forecast ranges are not touched. Refits follow the production
calendar, every 28 days, and each refit's base must reproduce the saved comparison q50
exactly. The learning rate and tree count were chosen on the 63 forecast days before the
validation window, by highest profit there, and frozen: learning rate 5 with 100 trees
made €136 more than median dispatch on those days, the three other gentle settings €90
to €121, and the two aggressive ones lost, so the pick among the gentle four is within
noise. Criterion: adopt only if the mean daily profit difference against median dispatch
over the 730 validation days has a 95% moving-block bootstrap interval above zero. The
hold-out was not read.

**Measured:** median dispatch made €149,498, 90.10% of perfect foresight; the money
point made €151,636, 91.39%. The difference is +€2.93 a day (+1.53 to +4.74), €2,138
over two years, so the criterion holds. The point's accuracy did not worsen as the plan
expected; MAE moved from 16.03 to 15.96 €/MWh: the corrections average +€1 in the
morning and evening and -€1 overnight, sharpening the daily shape where the battery
trades rather than moving the level. Cycles rose from 1.73 to 1.76 a day, and planned
value moved closer to settlement, -€7.64 a day against -€13.25, not further away.
(docs/results/t2_money_loss.md)

**What the gain rests on:** the money arm wins on 365 days and loses on 269, but the ten
best days carry 58% of the €2,138, and two dark, still winter days with evening spikes,
6 November and 12 December 2024, carry a third. Without October to December 2024 the
gain is +€1.62 a day (+0.77 to +2.56); in 2025 alone it is +€1.34 (+0.14 to +2.52), only
just above zero. An independent review found no leakage: the correction trees see only
the rows, features and past prices the base model sees. Two checks were run afterwards,
outside the pre-registered test. A cheap rival, q50 plus its mean residual per local
hour over the last 42 training days, lost money, -€2.43 a day (-3.80 to -1.07) with a
worse MAE of 17.39, so the gain is not reducible to a simple hour-of-day bias
correction. Refitting three blocks with two other seeds for the correction trees gave
the same sign and size (results page). The two best days were still, and 12 December
also dark: onshore wind was at its lowest for the season on both, and both peaked at
17:00, €820 and €936.

**What it changes:** a forecaster can earn more without becoming more accurate, which is
the decision-value finding from the other side. The gain is real by the rule fixed in
advance and small, about 1.4% of profit, and it comes from rare spike days. The live
pipeline still trades the median: serving the money point needs the correction trees
registered beside the base, and the frozen hold-out, scored once, is the only remaining
out-of-sample test. The next experiments go after the spike days themselves: features
that describe why an evening spikes, and a model of the chance of a spike that the
dispatch can bet on.

---

## T3 · Wear against revenue (Phase 4: measured)

**What breaks:** an optimizer that ignores wear cycles on every small spread. The
revenue looks higher, but the asset pays for it.

**Measured,** validation window, median dispatch, profit always charged the true
wear of €8 per MWh discharged:

| Wear price given to the optimizer | Cycle cap | Cycles a day | Revenue, € | P&L at true wear, € | Losing days |
|---|---|---|---|---|---|
| €0 | 2 a day | 1.90 | 169,322 | 148,223 | 26 |
| €8 (true) | 2 a day | 1.73 | 168,638 | 149,498 | 15 |
| €25 | 2 a day | 1.35 | 159,877 | 144,931 | 8 |
| €0 | none | 2.17 | 170,451 | 146,418 | 28 |
| €8 (true) | none | 1.76 | 169,269 | 149,721 | 15 |

Ignoring wear raises revenue by €684 but lowers profit by €1,275 with the cap, and
by €3,303 without it. Perfect foresight shows the same shape: profit at the true
wear peaks when the optimizer is given the true wear.

**Mitigation:** price wear at its true cost in the optimizer, and keep a cycle cap
as the warranty limit.

---

## D1 · A delivery day without prices (Phase 4: observed)

**Observed:** on 14 September 2026 neither SMARD nor ENTSO-E had day-ahead prices
for delivery day 13 September 2026, while the days either side were complete.

**Mitigation:** the backtest checks every day for complete prices and forecasts
before trading it, skips an incomplete day and lists it with the reason in the run
notes, so a gap cannot turn into a silent zero-profit day.

---

## T6 · Backtest overfitting and the hold-out (Phase 4: measured)

**What breaks:** a strategy tuned on the same days that report its result looks
better than it will trade. Every look at a result is a chance to fit noise.

**Discipline:** every choice, from the forecasting model to the battery limits and
the headline strategy, was made on the validation window, 1 June 2024 to 31 May
2026, and recorded before any hold-out forecast existed. The hold-out, 1 June to
14 September 2026, was then forecast walk-forward once and traded once. Both
commands refuse to run without an explicit confirmation, and refuse a second run
unless told to overwrite.

**Measured,** production model, median dispatch:

| Window | Capture | € a day | Pinball loss | 90% range coverage |
|---|---|---|---|---|
| Validation, whole window | 90.1% | 204.8 | 5.11 | 85.6% |
| Validation, June to September only | 94.3% | 237.8 | 4.61 | 85.0% |
| Hold-out, June to mid-September 2026 | 91.0% | 306.2 | 6.95 | 77.8% |

Against the whole window the hold-out looks as good as validation. Against the
same summer months it is 3.4 points lower, and that is the fair comparison. The
forecast itself was worse in summer 2026: pinball loss 6.95 against 4.61, and 90%
ranges covering 77.8% of prices against 85.0%. Profit per day was still higher
because prices swung more: perfect foresight earned €337 a day against €252 in the
same months of 2024 and 2025. The naive baseline captured 84.0%, up from 80.8%, so
the production model's lead over naive narrowed from 13.6 to 6.9 points. LightGBM
quantile again traded better than the production model, as on validation; the
production model was not changed on hold-out evidence.

**Mitigation:** the drift monitor (M2) was built in Phase 6. On the hold-out its
coverage signal alerted only on 11 September 2026, while the pinball ratio alerted from
2 July, so both signals are needed.

---

Phase 6 adds M2, M1, D1 and D5 below, and Phase 7 extends D5 with the live chain.
Phase 8 and the briefing work after it add two M5 entries, the live runs add a D1
entry on the day being forecast, and a follow-up to M1 asks whether weekly refits pay.
The T1 follow-ups sit with T1 above.

## M2 · Drift monitoring (Phase 6: measured, live: running daily)

**What breaks:** a model loses calibration gradually. Its ranges stop holding the
realised price, and nobody notices until profit falls.

**Monitor:** two signals over the last 28 traded days: 90% interval coverage, and
mean pinball loss divided by its validation median. Both thresholds were fixed on
validation days only, 1 June 2024 to 31 May 2026, before the hold-out was read:
coverage takes the highest threshold on a 0.5-point grid with at most 5% of
validation days in alert, and the pinball ratio the lowest threshold on a 0.05 grid
with at most 5% of days in alert.

**Measured:** the coverage threshold is 74.0%, with 4.8% of validation days in
alert; the pinball ratio threshold is 1.50, with 4.2% of days in alert, against a
validation median of 4.91 €/MWh. On the hold-out the pinball signal first alerted
on 2 July 2026, 31 days after the start, and was in alert on 6.7% of 105 traded
days. The coverage signal first alerted only on 11 September 2026, 102 days after
the start, on 2.9% of days. Coverage over the whole hold-out was 77.8%, but its
28-day rolling value stayed above 74% until September: a validation run in spring
2026 that fell to 62.6% had already set a low bar. Each alert run writes one drift
incident for retraining review, 9 in total; 5 of them fall in the validation window
whose days set the thresholds, so they are in-sample.

**Mitigation:** run both signals. On the hold-out the pinball ratio alerted first, but
that is one observation: on validation the longest coverage alert, 28 days from 20 March
2026, came with no pinball alert at all. A coverage threshold chosen on a window that
contains its own deep dip reacts late. The live pipeline now runs the monitor every day
after settling, with these thresholds unchanged, on settled days the production model
forecast; fallback days are skipped because the thresholds describe the production
model's ranges. It needs 28 such days before it can alert, and the window is not filled
with hold-out forecasts. An alert writes a drift incident for retraining review and does
not trigger an early refit, a rule that was never backtested. The dashboard's drift
panel still shows the saved validation and hold-out result; live alerts reach it only as
incidents. Which signal leads should still be decided on validation episodes, not on the
hold-out.

---

## M1 · Regime shift (Phase 6: measured)

**What breaks:** a model trained in one price regime keeps forecasting that regime.
A tree ensemble predicts from values learned on its training prices and does not
extrapolate beyond them, so when the level moves, the median and the ranges stay
behind and the battery trades on stale spreads.

**Measured:** LightGBM with conformal ranges, 730 training days, no weather features,
walked through 1 January 2021 to 31 December 2023. Fitted once on 2019 to 2020 prices
(mean 34 €/MWh, 99th percentile 73 €/MWh), its median peaked at 100.35 €/MWh while
prices reached 871 €/MWh. Its 90% coverage was 25.7% over the three years, 6.2% in
2022 and 2.4% in 2022Q3; its lowest rolling 28-day coverage was 0.1%. It captured
28.9% of perfect foresight, €61,880 against €214,142. Refitting every 91 days gave
77.1% coverage and 73.5% capture (€157,475); every 28 days, the production cadence,
80.6% and 80.7% (€172,807). Refits still lag fast moves: quarterly fell to 47.7%
coverage in 2022Q3, and monthly to 44.4% over 28 days in autumn 2021.

**Mitigation:** refit at least every 28 days, as the backtests do, and alert on rolling
coverage and the pinball ratio (M2) so a jump in price level triggers an early refit.
The live pipeline refits on that schedule: before a day is forecast, a served version
that first forecast 28 or more days earlier is refit for that day, checked for finite
quantiles and registered, and a refit is postponed while any feed is missing. No drift
alert triggers an early refit yet. Keep the previous-day baseline running as a
fallback: it re-anchors on yesterday's prices every day and covered 86.2% over the three
years, though its rolling 28-day coverage fell to 58.8% in autumn 2021 and it captured
only 69.8%.

---

## D1 · Weather forecast missing at issue time (Phase 6: measured)

**What breaks:** at 11:40 the production model reads archived weather forecasts for
the delivery day. If the feed has not arrived, the model still runs, but its
weather inputs are empty and it forecasts from the rest.

**Measured:** on the 730 validation days, with the Phase 2 refit schedule. The
production model recomputed in the same run matched the saved Phase 2 forecasts
exactly, so the comparison is like for like. With the delivery day's weather
blanked and no mitigation, pinball loss rose 50.8%, from 5.11 to 7.70 €/MWh, and 90%
coverage fell from 85.6% to 74.5%; median dispatch captured 87.8% instead of 90.1%,
€3,742 less over the two years. A fallback model trained without weather features
lost less: pinball loss 6.44 €/MWh, coverage 85.2%, capture 88.9%, €1,973 below the
full model. It recovers about half of the loss and keeps the ranges honest. The
baselines lose far more: naive previous day captured 77.6% (€20,662 less) and
seasonal naive previous week 79.7% (€17,206 less).

**Mitigation, built in Phase 7:** the live chain's first fallback is the production
model without weather features, fitted when a run needs it, rather than the full model
on empty inputs. Among the baselines, naive previous day has the lower pinball loss,
9.61 against 11.49, but seasonal naive earned more here; the D5 simulation keeps the
order fixed before this run, by pinball loss, and the live chain puts seasonal naive
first, on validation profit.

---

## D5 · The 12:00 deadline (Phase 6: simulated, Phase 7: built)

**What breaks:** the gate closes at 12:00 whether or not the pipeline worked. A day
without a forecast by then is a day without a position.

**Simulated:** failures injected on the 730 validation days with a fixed seed, as
scenario assumptions rather than estimates: weather feed late on 10% of days, the
model step failing on 3%, yesterday's prices late on 1%; the seed drew 9.0%, 2.6%
and 1.6%. The chain runs the full
model, then a model without weather, then naive previous day, then seasonal naive
previous week; late data is retried once after 5 minutes and, in this scenario,
never arrives. Measured step runtimes are 0.10 s for either model and 0.01 s for a
baseline, so every forecast was submitted before the gate, the latest 10.0 minutes
after the 11:40 issue time. That share follows from the assumptions: a hung step or
a longer wait is not simulated.

**Simulated result:** 95 days needed a fallback: 64 used the model without weather, 19
naive previous day and 12 seasonal naive. With the chain, median dispatch earned
€148,660, 89.6% of perfect foresight, €838 less over the two years than running the full
model on every day. Without it, those 95 days have no position: 87.0% of days are traded
and profit falls to €132,762, 80.0% capture. Against that counterfactual of no position
on failed days, the chain keeps €15,898 over the two years. Most of it comes from
trading at all: on the 64 days when only the weather feed was late, running the full
model on empty weather inputs would have earned €11,253 against €11,311 for the model
without weather, €59 less. Each fallback day writes an incident marked as simulated: 76
late data, 19 pipeline.

**Mitigation:** the chain itself. Phase 7 builds it as the Airflow sensor and branch.
Every model fit and every rung of the chain now runs under a time limit,
`pipeline.step_time_limit_s`, 120 s against the 8 s a production fit takes: a step still
running is abandoned and the chain moves on. The registry calls and file reads around
those steps have no limit of their own; only the forecast task's 15-minute Airflow
timeout covers them, and a kill late in the morning could still miss the gate. One
change was decided on validation profit: the live chain puts seasonal naive ahead of
naive previous day, because it earned €132,292 against €128,836 on the same 730 days,
79.7% of perfect foresight against 77.6%, although its pinball loss is worse, 11.49
against 9.61. The experiment above keeps the order it was frozen with, so its numbers
stand as reported.

**Built in Phase 7:** the chain is no longer only a simulation. The daily pipeline runs
it for real: a readiness check reports each feed separately, the chain steps down when
one is late, and every fallback writes an incident with the step used and the time the
forecast went out. A first live run for 17 September 2026 reported the load forecast and
weather unpublished, skipped the production and no-weather rungs, and committed a
seasonal naive schedule planned at €783 that settled at €405; it was started by hand at
13:21 Berlin on 16 September, 81 minutes after the gate. The run for 18 September later
hit the same 0 of 96 while SMARD held the full load forecast, so the pipeline rather
than the feeds was at fault, as the D1 entry on the day being forecast explains. A run
for 16 September, whose feeds were complete, used the production model served from the
MLflow registry; started by hand a day late, its schedule was planned at €289 and
settled at €308. Airflow 3 removed task-level SLAs, and its DAG-level deadline alerts
crashed the end-to-end test run, so the deadline is a task of its own: after the export,
it compares the time the forecast went out with the 12:00 gate from the run record and
writes a critical pipeline incident when the schedule was committed late.

---

## M5 · Narration that invents a number (Phase 8: enforced)

**What breaks:** a language model asked to describe a trading day will write a figure
that reads plausibly and is in no dataset. On a desk that figure is worse than no
briefing at all, because it arrives in the same sentence as the true ones and carries
their authority. The rule this project was built on is that the deterministic core
produces every number and the model only narrates, so the question is not whether to
trust the model but how the rule is enforced when it does not hold.

**How it is enforced:** the model never sees the run. It is given one payload per tab,
assembled from the same API functions the page itself calls, and is told that every
number it writes must appear there. The prose that comes back is then read for
numbers and each one is looked up in that payload, allowing for thousands separators,
rounding, and a share written as a percentage. A briefing carrying a figure that is
not there is refused and sent back once, with the figures that failed; a rewrite
that passes is shown, and one that fails again is discarded. The API then answers
with a deterministic writer over the same payload, names the rejected figures in
`rejected`, and sets `fell_back`. The same happens, without a rewrite, when the
model cannot be reached, so a briefing always arrives and nothing that fails the
check is returned.

**Measured:** the four payloads are 954, 1,154, 647 and 2,783 bytes, holding
16, 23, 11 and 28 distinct numbers. The briefings written from them state
7, 9, 6 and 9 figures, and every one is found. A payload of that size is small
enough to send whole and to check a sentence against, which is the reason the
briefing is scoped to one tab and one day rather than to the run.

**Measured against a real model:** each run is twenty calls to gpt-4o-mini over
the backtest's last day: four briefings per tab and four follow-up questions, two on
the overview and one each on the forecast and trading tabs. Under the first
instructions the model's briefing was shown 3 times in 20. Every figure it was
refused for was one the payload holds in another form: a date written with a month
name, such as "September 14, 2026", which reads as the numbers 14 and 2026, or an
hour block such as "11-14" written by its digits. An earlier run of five calls also
dropped a minus sign and wrote a difference the model had worked out itself.

The instructions now say how each of those is written, and a refused draft gets one
rewrite. A second run found a fault the check cannot see because it carries no
number: told to say a day is not settled "if the payload says prices are not
published", the model said so in 6 of 20 briefings, all on tabs whose payload has no
such flag. With the rule tied to the key itself, a third run showed the model's
briefing 20 times in 20, with no mention of settlement: 19 first drafts passed, and
one rewrite passed after its first draft named two blocks by their digits. To test
the rewrite on every tab, a refused first draft was written by hand for each and the
model was asked to repair it: three rewrites passed, and the fourth named a block by
its digits again, so the template answered.

**What the check cannot catch:** it verifies the prose against the payload, not the
payload against the page. The capture ratio first arrived rounded to two places, so
0.8957 reached the model as 0.9 and the briefing wrote 90% under a Trading tab
showing 89.6%. The check passed it, correctly: the prose matched what it was given.
Shares are now carried to four places, judged by the value rather than the name of
the key, so a share added later cannot bring the fault back. Rounding itself stays
allowed, which is a deliberate limit: 0.8957 supports 89.57%, 89.6% and 90%, though
not 89%. Grounding moves the trust from the model to the payload assembly, where it
can be tested, rather than removing it.

**What review caught, twice:** the first version of the check allowed any whole
number within one of a payload figure and mined digits out of strings. Against the
real export it accepted "charged over 20 periods" where the data says 19, "965 EUR"
where it says 964.46, and a loss of -14 EUR that existed only as the tail of the date
2026-09-14. The repair opened two more holes, which a second review found: labels
were blanked by plain substring replacement, so "11-14%" and "200:000" swallowed
their own digits and left nothing to check, and a percentage was matched against any
payload number, so "29%" passed because 29 was the traded-day count.

What holds now is narrower. Rounding, half up or half to even, is the only latitude.
Only a clock time or a date the payload holds may be quoted, and only as a whole
token. A percentage must come from a share. A figure that cannot be valued, spelled
out in words or written "½", counts as unsupported rather than being passed over.
Both rounds of exploits are pinned by tests, so a repair that reopens either fails
the suite rather than the dashboard.

**The limit worth stating plainly:** the check asks whether a figure is in the
payload, not whether it is in the right place. The battery runs 2 hours and cycles
1.71 times a day, and a briefing claiming it "cycled 2 times a day" passes, because 2
is in the payload as the duration. The real model does this unprompted: grounded
briefings from gpt-4o-mini called the evening block's 1028.78 EUR share of the gap
to perfect foresight "an earning", and a pinball loss of 9.49 EUR/MWh "earnings".
Catching that would mean tying each sentence to the key it describes, which this
does not attempt. What it does catch is the figure
that exists nowhere in the data, which is the one that cannot be argued with.

**Without a model:** the deterministic writer answers every request unless
`NARRATION_PROVIDER` switches a model on, which is what the test suite runs against,
so the grounding path and the fallback are exercised on every run with no network
and no spend. Why the model is off by default is the next note.

---

## M5 · A grounded briefing that misreads the page (after Phase 8: measured, left out by choice)

**What breaks:** the grounding check proves that every figure in a briefing is on
the page. It cannot prove that the sentence gives the figure its right meaning, and
a briefing that states a true number in the wrong role reads exactly as confidently
as one that does not.

**Measured:** the third run in the note above, twenty gpt-4o-mini briefings, all
shown, nineteen as first drafts and one after a rewrite, was then read sentence by
sentence against its payload. Seven misstated what a figure means.

- **Forecast, all five:** each read the tab's gap to perfect foresight over 29
  traded days, split by hour block into over-forecast and under-forecast shares, as
  the battery's own performance. The evening block's 1028.78 EUR under-forecast
  share became "an underperformance of 1028.78 EUR" in three of them, and its -5.55
  EUR over-forecast share an overperformance in three. One put the -5.55 EUR in "the
  preceding block"; one called the 1028.78 EUR a value at stake, the name of a
  different figure, and described the afternoon as underperforming, which the
  afternoon blocks' negative under-forecast shares contradict.
- **Trading, one of five:** the window's 8755.24 EUR over 29 traded days was given
  as what the battery made on 2026-09-14.
- **Overview, one of six:** the widest forecast band of the day, 101.07 EUR/MWh from
  q05 to q95, was presented as the result of the day's highest and lowest prices,
  which are 588.01 EUR/MWh apart.
- **Model health, four:** none.

Two of the forecast briefings also told the operator the dispatch strategy should
be reviewed, against the rule to say what the numbers show and not what to do.

**The decision:** the template writes the briefing by default, and a model writes
it only when `NARRATION_PROVIDER` names one and its key is set; a key alone is not
enough. The template says less: on the forecast tab it gives coverage and the
worst hour but not the gap to perfect foresight by block, and it answers only one of
the three follow-up questions. Everything it does say is grounded by construction
and in its right role. What the model adds is breadth and fluency, and at a cost of
7 briefings in 20 misstating a figure, every forecast briefing among them, that is
not worth showing an operator unread.

**Not added, on purpose: a reflection workflow.** The fix is known. A LangGraph
workflow would have a generator write the briefing and a reflection step review
each sentence against what each payload key means, its unit, its window and its
direction, then send the generator specific feedback on what to change before the
grounding check runs. It needs carefully engineered generator and reflection
prompts, and it needs evals: a labelled set of briefings with known role errors, the
reviewer's recall on that set, and a pass bar fixed before looking. The eval is most
of the work, because the reviewer is a language model with the same blind spot as
the writer, and without it there is no evidence the loop catches what it is for.
Every briefing would also take two or three more calls. For a panel that restates
numbers already on the screen, that effort is better spent on the forecast and the
live record, so the workflow is intentionally not built. It becomes worth building
when the briefing is expected to explain rather than restate: answering the
follow-up questions properly, or a question the operator types.

---

## D1 · The day being forecast, trimmed away (live: found and fixed)

**What broke:** the dataset build trims every row after the last published price.
That is right for a backtest, where a row without a price has nothing to score, and
wrong for a live run. Before the 12:00 auction the last published price is 23:45 on
the issue day, so the trim removed every row of the day being forecast, including the
load forecast SMARD publishes for it that morning. The inputs are built on the
dataset's rows, so that day's weather went too, and the readiness check counted 0 of
96 periods for both.

**How it showed:** the runs for 17 and 18 September both reported the load forecast
and weather missing and fell back to seasonal naive. The second was on time, issued
at 11:34 Berlin, 26 minutes before the gate. Checked directly at 11:33 Berlin that day,
SMARD already held all 96 quarter-hours of the next day's load forecast, so the feed
was not late: the pipeline had thrown it away. A backtest cannot find this, because
every day it replays already has its prices.

**Fix:** a live build asks for the delivery day explicitly (`build_dataset --through`,
passed by the daily pipeline and the Airflow DAG). That day's rows are kept on a
complete quarter-hour grid with the price left blank, along with everything SMARD has
already published for them. The default build is unchanged, so no backtest reads
different data, and the availability rules still hide a target day's price from the
models. Tests pin both behaviours.

**Found in review of the fix:** two more ways a live day could go wrong. The readiness
sensor only reread the inputs built at 10:30, so a load forecast published minutes
later would never be seen and the run would fall back anyway; each poke now rebuilds
the dataset and inputs first. And nothing stopped a rerun from replacing a committed
schedule: Airflow runs the latest slot it missed when it starts, which would have
re-forecast a day already traded and rewritten its record as late. It had happened
once by hand, on 16 September, without changing the settled figure. A bid is final at
the gate, so the daily run now refuses to replace a committed schedule unless told to
with `--replace`.

---

## M1 · Weekly against 28-day refits (validation: measured)

**Question:** the regime-shift experiment showed that refitting matters, but it only
compared the 28-day production cadence against slower ones. Is faster worth it?

**Measured:** the production model walked forward over the 793 days of the Phase 2
comparison run, refitted every 28 days (29 fits) and every 7 days (114 fits), scored on
730 validation days and traded on the same 730. The 28-day arm reproduced the saved
Phase 2 forecasts exactly, so the arms differ only in cadence. On 182 of the 730 days
both arms were running the same fit, in the week after a shared refit, and forecast
identically, which dilutes every difference below.

Weekly refitting forecast better: mean pinball 5.00 against 5.11, 2.1% lower, a paired
daily difference of -0.108 €/MWh with a 95% interval of -0.196 to -0.018 (moving-block
bootstrap over 7-day blocks). The verdict survives 28-day blocks, which match the
control's refit cycle, though only just: -0.210 to -0.001. The gain was not confined to
quiet hours: pinball fell 2.5% between 17:00 and 21:00 and 2.0% in the other hours. 90%
coverage rose from 85.6% to 86.5%, a borderline gain whose interval crosses zero with
28-day blocks.

It brought no measurable gain in profit. Median dispatch made €149,825 against
€149,498, €327 more over two years and 0.2 capture points, but the paired daily
difference of +€0.45 has an interval from -€0.70 to +€2.03, which over 730 days allows
anything from about €510 less to €1,480 more. (docs/results/m1_refit_cadence.md)

**Reading:** a forecast measurably better on pinball, in the evening as much as
anywhere, did not earn measurably more. That is the decision-value finding again:
battery profit turns on the timing and shape of the price curve, which pinball averages
over, and a difference this small cannot be told apart from noise in two years. Compute
is not the obstacle, since the weekly arm's 793 days ran in 16 minutes on this machine.
The cost would be operational: the live pipeline refits every 28 days, and weekly
refits would mean training, checking and registering a version every week instead of
every four, for a live record that would no longer match the backtested cadence.

**Not tested:** a regime shift. Faster refits should matter most when prices move
quickly, as in 2021 to 2023, and this run covers only the calmer validation window. The
hold-out cannot be used to decide it.
