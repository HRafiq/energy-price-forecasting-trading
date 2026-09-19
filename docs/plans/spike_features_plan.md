# Features for why an evening spikes: a fixed plan

Written 2026-09-19, before any code for it existed, and not changed after the run
starts. One change before it: the scarcity column first compared with the previous
28 days, but every model is handed ten days of history, so the feature contract test
showed it could not be built the same way live; it compares with the previous 7 days
instead.

## The question, in plain words

The window from 15:00 to 21:00 carries about 40% of the money the battery misses
against a perfect trader, and nearly all of that sits where the forecast came in
below the real price: on evenings that spiked, the model guessed too low. The model
already knows the hour, the load forecast, yesterday's wind and solar forecasts, the
weather forecast and the gas price. What it is not handed directly is the thing a
trader looks at first: how tight the evening will be. Does giving it that, as a few
new columns, make the battery earn more?

## Why evenings spike, and what each new column stands for

A spike is the price the last plant needed sets when solar has gone, demand is at
its evening peak, and wind is low, so that gas turbines, and on the worst days
imports or reserve plants, set the price. The columns, all computable at 11:40 on the
day before from data already in the inputs file:

| column | what it is | why |
|---|---|---|
| `load_forecast_evening_mean_mw` | mean grid-operator load forecast for day X over 17:00 to 20:59 | evening demand |
| `load_forecast_evening_ramp_mw` | that mean minus the mean over 12:00 to 15:59 | how steeply demand climbs into the evening |
| `wx_radiation_afternoon_mean` | mean forecast radiation over 12:00 to 15:59 of day X | how much solar there is to lose |
| `wx_wind_power_evening_mean` | mean onshore wind-power proxy over 17:00 to 20:59 of day X | stillness in the evening |
| `residual_persistence_evening_max_mw` | max over 17:00 to 20:59 of the existing residual-load estimate (load forecast for X minus X-1's wind and solar forecasts) | scarcity: how much the plants must cover |
| `residual_persistence_evening_z_7d` | that max, minus its mean over the previous 7 days, divided by their standard deviation | scarcity against what was normal lately |
| `price_prev_day_evening_max` | max price over 17:00 to 20:59 of day X-1 | recent spike memory |
| `price_evening_max_7d` | max price over 17:00 to 20:59 of days X-7 to X-1 | recent spike memory, longer |
| `spike_days_last_7d` | count of days X-7 to X-1 whose max price reached `evaluation.spike_threshold_eur_mwh` (200) | spiky regime |

Each is one value per day, repeated on every row of that day, so the trees can pair
it with the clock. The evening hours are 17:00 to 20:59 local, the block where the
T1 split put the summer loss; the afternoon is 12:00 to 15:59. Nothing here uses the
day's own prices or anything published after the issue time; the feature contract
test (`tests/test_features.py`) proves it for every column.

The new columns form a feature group of their own, `spike_drivers`, that the
production model does not use unless asked, so the control arm keeps reproducing
the saved forecasts and the live model does not change silently.

## The two arms

* **Control, `production`:** the production model on the saved Phase 2 comparison
  run's days, refit every 28 days as that run was. It must reproduce the saved
  forecasts to within 0.01 €/MWh, which proves both arms read the same inputs.
* **Candidate, `spike_drivers`:** the same model, same parameters, same refit
  days, with the nine columns added to its features. Nothing is tuned.

Both are scored on the 730 validation days and traded with median dispatch, the
1 MW / 2 MWh battery, €8 wear and the two-cycle cap, settled at real prices. The
hold-out is never read.

## The criterion, fixed before the run

Adopt `spike_drivers` only if the mean daily profit difference against `production`
over the 730 validation days has a 95% moving-block bootstrap interval (7-day
blocks, 5,000 draws) entirely above zero.

## Reported alongside, not part of the decision

Pinball loss, 90% coverage and MAE of the median, overall and on spike days (a day
whose max price reaches €200); the mean signed error of q50 over 17:00 to 20:59 on
spike days, which is the under-forecast the columns are meant to reduce; capture;
the difference June to September and October to May; feature importance of the new
columns; and cash by clock block.

## Risks named in advance

1. **Already known.** Trees can build most of these from the columns they have;
   the gain may be nil.
2. **Rare events.** About 143 of 730 validation days spike. A few columns that
   fire on them can help there and cost a little everywhere else.
3. **Persistence of the wrong kind.** The recent-spike columns may make the model
   chase last week's evenings; the seasonal split will show it.

## What each outcome means

* Interval above zero: adopt, push, production note, and the columns go to the
  production model (a registered version with the new features, and the
  spike-probability model gets them as inputs).
* Interval containing or below zero: reported as tested and rejected; the columns
  still go to the spike-probability experiment as candidate inputs, since that
  model asks a different question.
