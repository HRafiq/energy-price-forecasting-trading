# Production notes

The failure catalogue for the forecasting and trading pipeline. Each entry records what can break,
what we measured, and the mitigation. Measured facts are filled in as phases
land; the "In my words" parts are for the project owner to write.

---

## D1 · Missing or late upstream data (Phase 0: observed)

**Observed:** the build of 13 Sep 2026 found SMARD's prices for local day
13 September missing, 96 quarter-hours, while prices for 14 September were
already published. Energy-Charts had the missing day.

**What exists now:** the quality report counts gaps per column, and the SMARD
client re-downloads unsettled weeks on every run, so a late day is picked up on
a later refresh.

**Since Phase 1:** both baselines fill a missing source day from a second lag,
then from the last published price, and record how many periods needed it.

**Still to build, Phase 6 and 7:** the full fallback chain from the chosen model
down to the baselines, and an incident record for each occurrence.

**In my words:** _to write_

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

**In my words:** _to write_

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
there. How to fill such gaps is a Phase 1 decision.

**In my words:** _to write_

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

**In my words:** _to write_

## M3 · Leakage in disguise (Phase 0: risk found in the data source)

**What breaks:** a feature that is not known at 12:00 on day D, when bids for
day D+1 close. A backtest using it looks better than any live system can be.

**Found:**
- SMARD's day-ahead wind and solar forecasts are submitted at 18:00 the day
  before delivery, six hours after the gate. EU Regulation 543/2013, Article
  14(1)(d), sets the same 18:00 deadline on ENTSO-E, so switching source does
  not help.
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
| adds grid-operator D+1 forecasts, published 18:00 | 3.95 | −13.6% | 12.57 | 6.44 |
| adds measured D+1 wind, solar and load | 4.00 | −12.5% | 12.72 | 6.24 |

The post-gate forecasts flatter the backtest almost exactly as much as true
actuals do, because they track actuals so closely. Every reported model uses
only the honest set; a dashboard built on either leaky set would promise errors
about 13% smaller than live trading could deliver.

**In my words:** _to write_

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

**In my words:** _to write_

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

**In my words:** _to write_

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

**In my words:** _to write_

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

**In my words:** _to write_

---

## T5 · Price-taker assumption (Phase 3: stated)

**Assumption:** the battery's orders do not move the clearing price, and every
committed MW settles at it. For 1 to 5 MW in a market that clears tens of GW this
holds; for a portfolio of hundreds of MW it would overstate profit, because selling
into the evening peak lowers that peak.

**Also left out:** grid fees and taxes, and the auction's volume increments.
Orders are volumes without limit prices, so a forecast strategy can buy into a
surprise spike; a desk would bid price-quantity curves. The P&L is trading margin
net of wear.

**In my words:** _to write_

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

**Mitigation:** the upper quantiles in the evening matter most; spike prediction
(M5) and drift monitoring of evening coverage target this directly.

**In my words:** _to write_

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

**In my words:** _to write_

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

**In my words:** _to write_

---

## D1 · A delivery day without prices (Phase 4: observed)

**Observed:** on 14 September 2026 neither SMARD nor ENTSO-E had day-ahead prices
for delivery day 13 September 2026, while the days either side were complete.

**Mitigation:** the backtest checks every day for complete prices and forecasts
before trading it, skips an incomplete day and lists it with the reason in the run
notes, so a gap cannot turn into a silent zero-profit day.

**In my words:** _to write_

---

## T6 · Backtest overfitting and the hold-out (Phase 4: measured)

**What breaks:** a strategy tuned on the same days that report its result looks
better than it will trade. Every look at a result is a chance to fit noise.

**Discipline:** every choice, from the forecasting model to the battery limits and
the headline strategy, was made on the validation window, 1 June 2024 to 31 May
2026, and recorded before any hold-out forecast existed. The hold-out, 1 June to
14 September 2026, was then forecast walk-forward once and traded once. Both
commands refuse to run without an explicit confirmation and refuse a second run.

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

**Mitigation:** the coverage drop is what drift monitoring (M2, Phase 6) is meant to
catch, with an alert when rolling 90% coverage falls below its threshold.

**In my words:** _to write_

---

Phase 6 adds M2, M1, D1 and D5 below; later phases add M5 and S1.

## M2 · Drift monitoring (Phase 6: measured)

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
2026 that fell to 62.6% had already set a low bar. Each alert run writes one drift incident for retraining review, 9 in total; 5 of
them fall in the validation window whose days set the thresholds, so they are
in-sample.

**Mitigation:** run both signals. On the hold-out the pinball ratio alerted first, but
that is one observation: on validation the longest coverage alert, 28 days from 20
March 2026, came with no pinball alert at all. A coverage threshold chosen on a window
that contains its own deep dip reacts late. Which signal leads in the live pipeline is
decided on validation episodes, not on the hold-out.

**In my words:** _to write_

---

## M1 · Regime shift (Phase 6: measured)

**What breaks:** a model trained in one price regime keeps forecasting that regime.
A tree ensemble predicts from values learned on its training prices and does not
extrapolate beyond them, so when the level moves, the median and the ranges stay
behind and the battery trades on stale spreads.

**Measured:** LightGBM with conformal ranges, 730 training days, no weather features,
walked through 1 January 2021 to 31 December 2023. Fitted once on 2019 to 2020 prices
(mean 34 €/MWh, 99th percentile 73 €/MWh), its median peaked at 100.35 €/MWh while prices reached 871 €/MWh. Its 90% coverage was 25.7% over the three years, 6.2% in
2022 and 2.4% in 2022Q3; its lowest rolling 28-day coverage was 0.1%. It captured
28.9% of perfect foresight, €61,880 against €214,142. Refitting every 91 days gave
77.1% coverage and 73.5% capture (€157,475); every 28 days, the production cadence,
80.6% and 80.7% (€172,807). Refits still lag fast moves: quarterly fell to 47.7%
coverage in 2022Q3, and monthly to 44.4% over 28 days in autumn 2021.

**Mitigation:** refit at least every 28 days, and alert on rolling coverage and the
pinball ratio (M2) so a jump in price level triggers an early refit instead of
waiting for the schedule. Keep the previous-day baseline running as a fallback: it re-anchors on yesterday's
prices every day and covered 86.2% over the three years, though its rolling 28-day
coverage fell to 58.8% in autumn 2021 and it captured only 69.8%.

**In my words:** _to write_

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

**Mitigation:** keep a model trained without the feed ready as the first fallback,
rather than running the full model on empty inputs. Among the baselines, naive
previous day has the lower pinball loss, 9.61 against 11.49, but seasonal naive
earned more here; the D5 chain keeps the order fixed before this run, by pinball
loss.

**In my words:** _to write_

---

## D5 · The 12:00 deadline (Phase 6: simulated)

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

**Simulated result:** 95 days needed a fallback: 64 used the model without weather, 19 naive
previous day and 12 seasonal naive. With the chain, median dispatch earned €148,660,
89.6% of perfect foresight, €838 below the full model every day. Without it, those
95 days have no position: 87.0% of days are traded and profit falls to €132,762,
80.0% capture. Against that counterfactual of no position on failed days, the chain keeps €15,898
over the two years. Most of it comes from trading at all: on the 64 days when only
the weather feed was late, running the full model on empty weather inputs would have
earned €11,253 against €11,311 for the model without weather, €59 less. Each fallback
day writes an incident marked as simulated: 76 late data, 19 pipeline.

**Mitigation:** the chain itself, with a hard stop on every step so a hung model
falls through to a baseline instead of waiting past the gate. Phase 7 implements it
as the Airflow sensor and branch, with one change decided on validation profit: the
live chain puts seasonal naive ahead of naive previous day, because it earned
€132,292 against €128,836 on the same 730 days, 79.7% of perfect foresight against
77.6%, although its pinball loss is worse, 11.49 against 9.61. The experiment above
keeps the order it was frozen with, so its numbers stand as reported.

**Built in Phase 7:** the chain is no longer only a simulation. The daily pipeline
runs it for real: a readiness check reports each feed separately, the chain steps
down when one is late, and every fallback writes an incident with the step used and
the time the forecast went out. A first live run for 17 September 2026 found the load
forecast and weather unpublished, skipped the production and no-weather rungs, and
committed a seasonal naive schedule worth €783. A run for 16 September, whose feeds
were complete, used the production model served from the MLflow registry and
committed €289. Airflow 3 removed task-level SLAs, and its DAG-level deadline alerts crashed the
end-to-end test run, so the deadline is a task of its own: after the export, it
compares the time the forecast went out with the 12:00 gate from the run record
and writes a critical pipeline incident when the schedule was committed late.

**In my words:** _to write_

---
