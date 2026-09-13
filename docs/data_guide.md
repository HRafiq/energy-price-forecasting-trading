# Data guide

What the data behind this project is, column by column, what it looks like, what
can go wrong with it, and how it feeds the forecasting and trading code.

This is a living document. Each phase adds its part, so the guide always
describes the code as it is.

| Part | Status |
|---|---|
| 1 to 8: decision, source, dataset, columns, quality, what the data shows | Written in Phase 0 |
| 9: what is known at the gate | Phase 0 finding; feature rules fixed in Phase 2 |
| 10: feature engineering | Planned, Phase 2 |
| 11: how the data feeds forecasting | Planned, Phases 1 and 2 |
| 12: how the data feeds trading optimization | Planned, Phases 3 and 4 |
| 13: dashboard artifacts | Planned, Phase 5 |

Every number here is reproducible. `notebooks/phase0_eda.py` writes the figures,
[summary.md](figures/phase0/summary.md) and
[data_guide_tables.md](figures/phase0/data_guide_tables.md). Unless a table says
otherwise, statistics cover the hourly-product era, 1 October 2018 to
30 September 2025, with quarter-hours averaged to hours, in local time. The
hold-out period is never analysed (§7).

---

## 1. The decision this data serves

A battery trading in the German day-ahead market makes one decision a day. By
12:00 on day D it bids for every 15-minute delivery period of day D+1. The
exchange then publishes one clearing price per quarter-hour at about 12:45. Every column in the
dataset is judged by one question: **could we have known this at 11:40 on
day D?**

| When | What happens |
|---|---|
| Day D−1, 12:45 | Prices for day D are published |
| Day D−1, 18:00 | Grid operators submit day-ahead wind and solar forecasts for day D |
| Day D, by 10:00 | Day-ahead load forecast for D+1 is due, two hours before the gate |
| Day D, 11:40 | Our forecast and dispatch plan must be ready |
| Day D, 12:00 | Gate closes for day D+1 |
| Day D, 12:45 | Prices for day D+1 are published |
| Day D, 18:00 | Day-ahead wind and solar forecasts for D+1 are submitted, **after the gate** |

The 18:00 and two-hours-before-gate deadlines come from EU Regulation 543/2013,
Articles 14(1)(d) and 6(1)(b). SMARD's forecast-data page confirms that grid
operators submit the day-ahead generation forecasts at 18:00 for the following
day.

---

## 2. Source

| | |
|---|---|
| Provider | [SMARD.de](https://www.smard.de), run by Bundesnetzagentur, Germany's energy regulator |
| Licence | CC BY 4.0. Credit line: Bundesnetzagentur \| SMARD.de |
| Access | The JSON files behind the SMARD website. No key, but not a versioned API |
| Code | `src/ingest/smard.py` downloads; `src/ingest/build_dataset.py` assembles |
| Rebuild | `uv run python -m src.ingest.build_dataset` |

**How the download works.** Each series arrives as weekly files of quarter-hour
values, cached under `data/raw/smard/`. A week counts as settled once four newer
weeks exist. Only a week that had already settled when it was downloaded is
reused; anything else is downloaded again. A full download takes about eight
minutes, a refresh about ten seconds.

**Cross-check.** Our prices matched Energy-Charts, from Fraunhofer ISE, to within
€0.01 on six test days, including both DST days and one day after the switch to
15-minute products. Energy-Charts republishes SMARD data, so this confirms our
download and time alignment, not SMARD itself.

Why SMARD instead of OPSD or ENTSO-E is recorded in
[decisions.md](decisions.md).

---

## 3. The dataset

| Property | Value |
|---|---|
| File | `data/processed/smard_quarterhour.parquet`, not committed |
| Quality report | `data/processed/smard_quarterhour_quality.json` |
| Grain | One row per 15-minute delivery period |
| Index | `timestamp_utc`: start of the period, timezone-aware UTC |
| Columns | 12: nine downloaded, two derived, one product flag |
| Coverage, build of 13 Sep 2026 | 30 Sep 2018 22:00 UTC to 14 Sep 2026 21:45 UTC, 278,976 quarter-hours |
| Tuning period | 262,944 quarter-hours, of which 17,472 carry 15-minute products |
| Hold-out so far | 16,032 quarter-hours |

**Time conventions**

- **Everything is stored in UTC.** Local time, Europe/Berlin, is used only for
  display and calendar features.
- **A delivery day is a local calendar day.** It has 92 quarter-hours on the
  last Sunday of March, 100 on the last Sunday of October and 96 otherwise. The dataset starts at 22:00 UTC
  because that is local midnight on 1 October 2018, the first day of the DE-LU
  bidding zone.
- **Periods within a day are numbered by position**, from 0, never by local
  clock time. Local 02:00 to 02:59 is missing in March and happens twice in
  October. This lives in `src/timegrid.py`.
- **The series is quarter-hourly for its whole length.** Day-ahead products
  became 15-minute for delivery from 1 October 2025. Before that the market
  traded hourly products, so those quarter-hours repeat their hour's price and
  `price_product_minutes` is 60. Load, wind and solar are genuinely
  quarter-hourly throughout.
- **Volumes are average MW.** SMARD publishes MWh per quarter-hour; the builder
  multiplies by four. Tables in this guide show GW for readability.
- **The dataset ends at the last published price.** Other columns can end
  earlier. In the build of 13 Sep 2026, prices for 14 September were already
  published but the wind and solar forecasts for that day were not, consistent
  with their 18:00 submission time, and measured values ended 34 hours before
  the last price. The quality report lists these trailing gaps.

---

## 4. Column dictionary

| Column | SMARD filter | Meaning | Kind | Known at 11:40 on day D for day D+1? |
|---|---|---|---|---|
| `price_eur_mwh` | 4169 | Day-ahead clearing price, DE-LU, €/MWh. Before 1 Oct 2025 all quarter-hours of an hour share its price | Target | No, it is what we forecast. Prices up to the end of day D are known |
| `load_actual_mw` | 410 | Total grid load, measured | Actual | No. Earlier hours of day D only, after a publication delay not yet measured |
| `load_forecast_mw` | 411 | Grid operators' day-ahead load forecast | Forecast | Likely. Due two hours before the gate, may be updated later |
| `wind_onshore_actual_mw` | 4067 | Onshore wind generation, measured | Actual | No, as for load |
| `wind_offshore_actual_mw` | 1225 | Offshore wind generation, measured | Actual | No, as for load |
| `solar_actual_mw` | 4068 | Solar generation, measured | Actual | No, as for load |
| `wind_onshore_forecast_mw` | 123 | Grid operators' day-ahead onshore wind forecast | Forecast | **No. Submitted at 18:00 on day D** |
| `wind_offshore_forecast_mw` | 3791 | Grid operators' day-ahead offshore wind forecast | Forecast | **No**, as above |
| `solar_forecast_mw` | 125 | Grid operators' day-ahead solar forecast | Forecast | **No**, as above |
| `residual_load_actual_mw` | Derived | Load minus onshore wind, offshore wind and solar, all actuals | Derived | No |
| `residual_load_forecast_mw` | Derived | The same, from the forecast columns | Derived | **No**, it inherits the renewables timing |
| `price_product_minutes` | Derived | 60 for hourly products before 1 Oct 2025, 15 for quarter-hour products after | Flag | Yes, fixed by the calendar |

**Residual load** is the demand left for conventional power plants after wind
and solar. It is the best single explanation of the price (§8.5). A residual
value is missing whenever any input is missing; a gap is never treated as zero.

---

## 5. What the values look like

### Column statistics, hourly-product era, hourly values

| column | unit | missing hours | mean | p01 | median | p99 | min | max |
|---|---|---|---|---|---|---|---|---|
| price_eur_mwh | €/MWh | 0 | 93.3 | -10.0 | 69.4 | 483.5 | -500.0 | 936.3 |
| load_actual_mw | GW | 0 | 54.8 | 36.0 | 54.7 | 74.6 | 30.9 | 81.3 |
| load_forecast_mw | GW | 24 | 54.2 | 36.2 | 54.0 | 71.4 | 30.5 | 77.6 |
| wind_onshore_actual_mw | GW | 0 | 11.8 | 0.6 | 9.1 | 39.1 | 0.0 | 48.5 |
| wind_offshore_actual_mw | GW | 0 | 2.8 | 0.0 | 2.6 | 6.5 | 0.0 | 7.9 |
| solar_actual_mw | GW | 0 | 6.2 | 0.0 | 0.2 | 37.8 | 0.0 | 52.1 |
| wind_onshore_forecast_mw | GW | 0 | 11.7 | 0.7 | 8.9 | 38.9 | 0.2 | 46.6 |
| wind_offshore_forecast_mw | GW | 0 | 2.8 | 0.1 | 2.7 | 6.2 | 0.0 | 6.8 |
| solar_forecast_mw | GW | 0 | 6.2 | 0.0 | 0.2 | 37.4 | 0.0 | 50.4 |
| residual_load_actual_mw | GW | 0 | 34.0 | 2.3 | 34.6 | 63.8 | -8.4 | 74.6 |
| residual_load_forecast_mw | GW | 24 | 33.5 | 1.2 | 34.4 | 61.6 | -14.0 | 70.2 |

- **Price** runs from −€500, the exchange's lower price limit, reached once on
  2 July 2023 at 14:00, to €936.3. The median, €69.4, sits far below the mean,
  €93.3: the distribution has a long right tail.
- **Residual load goes negative in 311 hours.** In those hours wind and solar
  alone exceeded grid load.
- **Solar's median is 0.2 GW** because about half of all hours are dark.
- **The load forecast and forecast residual miss 24 hours**, all on
  31 January 2020 (§6).

### An example day: Sunday 12 May 2024

Every second hour, local time.

| local time | price €/MWh | load GW | wind GW | solar GW | residual load GW | residual load forecast GW |
|---|---|---|---|---|---|---|
| 00:00 | 65.5 | 36.7 | 14.4 | 0.0 | 22.3 | 24.6 |
| 02:00 | 44.5 | 33.9 | 14.0 | 0.0 | 19.9 | 23.2 |
| 04:00 | 46.3 | 33.9 | 13.4 | 0.0 | 20.6 | 23.2 |
| 06:00 | 17.6 | 35.0 | 12.2 | 2.6 | 20.2 | 21.8 |
| 08:00 | 2.4 | 41.1 | 7.3 | 20.7 | 13.1 | 13.3 |
| 10:00 | -25.0 | 44.3 | 4.4 | 37.2 | 2.7 | 0.2 |
| 12:00 | -100.1 | 44.3 | 3.0 | 41.4 | -0.1 | -4.3 |
| 14:00 | -132.8 | 41.1 | 2.8 | 38.7 | -0.4 | -3.8 |
| 16:00 | -30.0 | 40.5 | 8.1 | 30.8 | 1.7 | 2.7 |
| 18:00 | 22.5 | 44.6 | 16.8 | 13.3 | 14.5 | 18.1 |
| 20:00 | 75.7 | 45.3 | 19.7 | 0.8 | 24.8 | 25.7 |
| 22:00 | 44.6 | 45.1 | 25.4 | 0.0 | 19.7 | 17.9 |

A low Sunday load meets strong sun. Solar reaches 41.4 GW at 12:00, residual
load falls to about zero from 12:00 to 14:00, and the price drops to −€132.8 at
14:00. As the sun sets, wind picks up and residual load climbs back to 24.8 GW
at 20:00, where the price is €75.7. A battery was paid to charge at midday and
could sell into the evening. The forecast residual load was 4.2 GW below the
actual at 12:00.

---

## 6. Data quality

Checks run on every build by `src/ingest/quality.py`. Structural checks decide
pass or fail. Value checks are reported only, because negative and extreme prices
are real.

| Check | What it catches | Result, build of 13 Sep 2026 |
|---|---|---|
| Resolution | Rows at a step other than 15 minutes | 15-minute throughout |
| Missing timestamps | Periods absent from the index | 0 |
| Duplicate timestamps | The same period twice | 0 |
| DST day length | Days whose row count differs from 92, 96 or 100 as the calendar requires | 0 |
| Hourly products | A pre-switch hour whose quarter-hours carry different prices | 0 |
| Column gaps | Missing values, longest gap, unpublished tail | Load forecast gap below |
| Price summary | Negative and extreme prices, reported and never failed | Recorded in the quality report |

**Known issues**

- **Load forecast gap on 31 January 2020.** SMARD has no load forecast for that
  local day, 96 quarter-hours. The fill policy is a Phase 1 decision.
- **Solar is not exactly zero at night.** See the night-time solar figures in
  [data_guide_tables.md](figures/phase0/data_guide_tables.md). The forecast is
  zero there. Harmless for price modelling, but it matters for error ratios.
- **Recent days can be missing for a while.** The build of 13 Sep 2026 had
  prices for 14 September but none for 13 September, 96 quarter-hours, while
  Energy-Charts already had them. The quality report flags such gaps, and the
  newest weeks are re-downloaded on every run, so the gap closes on a later
  refresh. For the live dashboard this is the missing-data case, D1.
- **Revisions after a week has settled are not captured.** See D2 in
  [production_notes.md](production_notes.md).
- **Forecast vintages are unknown.** SMARD publishes one value per quarter-hour, not
  the history of revisions.

### Grid-operator forecasts against actuals

Bias is forecast minus actual.

| series | mean actual GW | MAE GW | bias GW | correlation |
|---|---|---|---|---|
| Wind onshore | 11.8 | 1.09 | -0.1 | 0.987 |
| Wind offshore | 2.82 | 0.48 | 0.01 | 0.932 |
| Solar | 6.18 | 0.44 | 0.0 | 0.995 |
| Load | 54.84 | 2.08 | -0.65 | 0.965 |
| Residual load | 34.05 | 2.55 | -0.56 | 0.971 |

The configured series are the right ones: each forecast tracks its own actual
closely. The same closeness is the warning: a solar forecast that correlates 0.995 with the actual
gives a model almost everything the actual would, so using the post-gate
forecasts would be a strong leak (§9).

---

## 7. Periods and splits

| Period | Local dates | Use |
|---|---|---|
| Tuning | 1 Oct 2018 to 31 Mar 2026 | Exploration, feature and model design, walk-forward validation |
| Hold-out | From 1 Apr 2026 | Final test in Phase 4 only. Not explored, not tuned on |

The hold-out start is a proposal awaiting review. It leaves six months of genuine
quarter-hour prices, October 2025 to March 2026, for training and tuning. The
hold-out covers spring and summer only, so winter evening spikes are not tested
until it grows. The Phase 0 figures describe the hourly-product era and do not
yet cover the 15-minute months.

**Regime labels** used in the figures are descriptive, with approximate
boundaries. They are not model inputs.

| regime | local dates | hours | mean price €/MWh | rank correlation, price with residual load |
|---|---|---|---|---|
| Before the gas crisis | Oct 2018 to Jun 2021 | 24,096 | 39.5 | 0.779 |
| Gas crisis | Jul 2021 to Jun 2023 | 17,520 | 178.3 | 0.656 |
| After the crisis | Jul 2023 to Sep 2025 | 19,752 | 83.5 | 0.874 |

---

## 8. What the data shows

Each figure answers one question about price formation. Together they explain
why the forecasting and trading design looks the way it does, and they are the
first place to look when a forecast or a trade goes wrong.

### 8.1 How has the price level changed?

![Daily average day-ahead price](figures/phase0/01_daily_price.png)

**What it shows:** the mean of each local day's hourly prices.

**What we see**

- 2019 and 2020 were calm, with yearly means of €37.7 and €30.5.
- Prices climbed through 2021: €55.0 in the first half, €138.0 in the second.
  2022 averaged €235.4, and the highest day was 26 August 2022 at €699.
- From 2023 the level settled, with yearly means of €95.2 in 2023, €78.5 in 2024
  and €88.0 for January to September 2025. Day-to-day swings stayed much larger
  than before 2021.

**Why it matters**

- The price is not stationary. A model trained on 2019 and 2020 saw a price
  above €200 in only one hour. This is the regime-shift experiment, M1.
- The length of the training window is a real modelling choice, tested in
  Phase 2 rather than assumed.

### 8.2 What does the distribution of hourly prices look like?

![Distribution of hourly prices](figures/phase0/02_price_distribution.png)

**What we see**

- 3.3% of hours are negative, and prices pile up around zero: 1,173 hours between
  −€5 and €0 and 1,457 between €0 and €5.
- The right tail is long: 168 hours above €600.
- There are two humps, around €35 to €45 and around €85 to €100. That is the calm
  years and the post-2021 years mixed in one histogram.

**Why it matters**

- MAPE is unusable: it divides by the price, which is often zero or negative.
  Forecasts are scored with pinball loss and MAE.
- The quantile model must be able to produce negative prices and a wide upper
  tail.
- A distribution pooled across regimes describes no single year well, so
  statistics are reported by year or regime.

### 8.3 How often do the two tails occur?

![The two tails by year](figures/phase0/03_tails_by_year.png)

| year | negative hours | hours above €200 |
|---|---|---|
| 2019 | 211 | 0 |
| 2020 | 298 | 1 |
| 2021 | 139 | 823 |
| 2022 | 69 | 4,642 |
| 2023 | 301 | 112 |
| 2024 | 457 | 129 |
| 2025, Jan to Sep | 525 | 110 |

**What we see**

- Negative hours fell during the gas crisis, then reached new highs each year
  from 2023.
- Hours above €200 were common only in 2021 and 2022. Since 2023 there have been
  about 110 to 130 a year, one to two percent of hours.

**Why it matters**

- Negative hours are a battery's best charging opportunities, and there are more
  of them every year.
- Spike hours carry a large share of arbitrage value but are rare. A backtest
  needs several years to say anything reliable about spikes, and a dedicated
  spike classifier, M5, may be needed.

### 8.4 When in the day is power cheap or expensive?

![Median price by hour of day and season](figures/phase0/04_daily_shape_by_season.png)

**Daily shape after the crisis, median €/MWh**

| season | cheapest hour | median | dearest hour | median | dearest minus cheapest |
|---|---|---|---|---|---|
| Winter | 03:00 | 69.3 | 17:00 | 112.9 | 43.7 |
| Spring | 13:00 | 3.7 | 20:00 | 125.2 | 121.5 |
| Summer | 13:00 | 17.0 | 20:00 | 130.8 | 113.8 |
| Autumn | 13:00 | 72.0 | 19:00 | 138.8 | 66.8 |

**When the tails happen, share of each kind of hour by local time band**

| local hours | negative, before crisis | negative, after crisis | above €200, after crisis |
|---|---|---|---|
| 00:00 to 05:59 | 34% | 11% | 0% |
| 06:00 to 09:59 | 14% | 6% | 19% |
| 10:00 to 15:59 | 40% | 71% | 11% |
| 16:00 to 19:59 | 8% | 11% | 48% |
| 20:00 to 23:59 | 4% | 1% | 22% |

**What we see**

- Spring and summer have a deep midday valley from solar and a peak at 20:00.
- Autumn has the highest evening peak; winter is the flattest, peaking at 17:00
  after an early sunset.
- Before the crisis a third of negative hours came at night, when wind is the
  only renewable running.
  After the crisis 71% fall between 10:00 and 15:59: negative prices have become
  a solar phenomenon.
- Hours above €200 cluster on the evening ramp, 16:00 to 19:59.

**Why it matters**

- This is the trade: charge in the midday valley, discharge on the evening ramp.
- Hour-of-day by season is a core feature interaction.
- Forecast errors on the evening ramp are the expensive ones; errors at 03:00
  barely matter. This is the error-asymmetry analysis, T1.

**Daily spread, €/MWh**

| year | mean highest minus lowest hour | mean top-2 minus bottom-2 hours |
|---|---|---|
| 2019 | 30.1 | 28.3 |
| 2020 | 32.5 | 30.0 |
| 2021 | 80.3 | 75.6 |
| 2022 | 187.0 | 177.2 |
| 2023 | 97.9 | 91.7 |
| 2024 | 111.2 | 103.8 |
| 2025, Jan to Sep | 137.3 | 128.3 |

The top-2 minus bottom-2 spread is the gross opportunity for a two-hour battery,
before efficiency losses and degradation. It is more than four times larger
than in 2019 and still rising after the crisis.

### 8.5 How does residual load set the price?

![Price against residual load](figures/phase0/05_price_vs_residual_load.png)

**Median price by residual load, €/MWh.** Bands include their upper edge.

| residual load GW | before crisis | gas crisis | after crisis | after crisis, p90 | after crisis, hours |
|---|---|---|---|---|---|
| -20 to 0 | no hours | -8.8 | -11.6 | -0.0 | 300 |
| 0 to 10 | -11.2 | 4.4 | 0.0 | 25.4 | 1,747 |
| 10 to 20 | 8.5 | 64.8 | 46.5 | 76.9 | 3,115 |
| 20 to 30 | 27.9 | 103.0 | 80.2 | 99.6 | 4,681 |
| 30 to 40 | 36.2 | 146.3 | 98.3 | 121.0 | 5,319 |
| 40 to 50 | 45.3 | 197.8 | 121.0 | 152.9 | 3,220 |
| 50 to 60 | 55.2 | 237.4 | 146.5 | 203.3 | 1,092 |
| 60 to 80 | 71.9 | 317.3 | 189.5 | 382.4 | 278 |

**What we see**

- This is the merit order in data. The more demand is left after wind and solar,
  the more expensive the last power plant needed.
- Before the crisis the curve was flat and tight: roughly €10 to €20 more per
  10 GW.
- During the crisis the same residual load cost about four times as much above
  20 GW, and the spread was wide. Gas set the price, and gas was expensive and volatile.
- After the crisis the curve is steeper than before and bends up sharply at the
  top. The 90th percentile nearly doubles from the 50 to 60 GW band to the 60 to
  80 GW band.
- Below 10 GW of residual load the median price is near or below zero in every
  regime.

**Why it matters**

- Residual load is the master feature, but only a gate-available estimate of it
  can be used (§9).
- The mapping from residual load to price moves with fuel costs. That is why gas
  and carbon prices are planned features and why retraining cadence matters.
- The bands where the 90th percentile jumps are where quantile forecasts must
  widen. Getting that right is what calibration means in practice.

---

## 9. What is known at the gate

Feature eligibility as currently established. Phase 2 turns this into enforced
rules with tests.

| Information | Known at 11:40 on day D for day D+1? | Basis |
|---|---|---|
| Prices up to the end of day D | Yes | Published 12:45 on day D−1 |
| Measured load, wind and solar for earlier hours of day D | Partly | Publication delay not yet measured |
| Grid operators' load forecast for D+1 | Likely | Due two hours before the gate; updates and vintages unknown |
| Grid operators' wind and solar forecasts for D+1 | **No** | Submitted 18:00 on day D |
| Grid operators' wind and solar forecasts for day D | Yes | Submitted 18:00 on day D−1 |
| Calendar: hour, weekday, holidays, DST | Yes | Deterministic |
| Gas and carbon closing prices of day D−1 | Yes | Not yet ingested |
| Weather model runs issued before 11:40 on day D | Yes, if the issue time is known | Candidate source to verify in Phase 2 |

**Consequence.** The dataset's renewable forecast columns, and the forecast
residual load built from them, describe day D+1 with information that arrives
six hours after the gate. They stay in the dataset for analysis and for the
leakage experiment M3, but no reported model may use them for day D+1. The
options are recorded in [decisions.md](decisions.md).

---

## 10. Feature engineering

*Planned, Phase 2.* For each feature this section will give its definition,
source columns, availability rule from §9, and the test that enforces it.
Candidate families from the plan: calendar and DST-aware hour position, price
lags such as the same hour one day and one week earlier, the load forecast, a
gate-available wind and solar estimate, gas and carbon prices, and an estimated
residual load built only from gate-available inputs.

## 11. How the data feeds forecasting

*Planned, Phases 1 and 2.* Target: `price_eur_mwh` for each 15-minute delivery period of
day D+1. Baselines: naive and seasonal-naive. Model: LightGBM quantile regression at
seven quantiles, trained walk-forward. This section will document training
windows, the fill policy for gaps, and the forecast output schema.

## 12. How the data feeds trading optimization

*Planned, Phases 3 and 4.* Inputs: the quantile forecasts for day D+1, realised
prices for settlement and the perfect-foresight benchmark, and battery parameters
from `config/settings.yaml`. This section will document how each input enters the
optimization and how profit and loss is attributed back to forecast errors.

## 13. Dashboard artifacts

*Planned, Phase 5.* The files the backtest writes for the dashboard, and which
panel reads each one.

---

## Changelog

| Date | Phase | Change |
|---|---|---|
| 2026-09-13 | 0 | First version: decision timeline, source, dataset, columns, quality, five figures, gate availability |
| 2026-09-13 | 0 | Dataset switched to 15-minute periods, volumes in MW, product flag added, hold-out proposal moved to 1 Apr 2026 |
