# Production notes

The failure catalogue from the handoff (§6). Each entry records what can break,
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
report fails if any pre-switch hour has differing quarter-hour prices. See
`docs/decisions.md`.

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

Later phases add D5, M1, M2, M4, M5, T1 to T6 and S1.
