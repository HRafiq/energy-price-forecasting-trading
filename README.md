# Day-Ahead Electricity Price Forecasting and Battery Trading (DE-LU)

A probabilistic price forecaster feeding a battery dispatch optimiser, backtested on real German day-ahead market data. Built to answer one question: does a better price forecast actually make a battery more money?

**The finding.** Trading a 1 MW / 2 MWh battery on two years of DE-LU day-ahead prices, my quantile forecast captured 90.1% of the perfect-foresight profit, €74,700 per MW per year. The honest comparison is not the 77.6% the same optimiser reaches on yesterday's prices, because that is still a forecast, but a rule that forecasts nothing at all: dispatching on the average price of each time of day in that calendar month captures 84.9%. The forecast is worth the 5.2 points between them, about €4,300 per MW per year, and it earns nine tenths of that on the three fifths of days that depart most from their seasonal pattern; on the most ordinary fifth it is worth nothing measurable. Better forecasts earned more, a rank correlation of -0.83 across eight forecasters, but timing mattered more than size: an evening shifted one hour early lost as much as random noise with twice the average error. What binds is the forecast, not the battery: measured the same way, as what this battery trading on this forecast would gain, closing the forecast gap is worth 9.9% of the achievable profit and lifting the two-cycle warranty cap 0.1%. For an operator, the next euro belongs in evening timing and drift monitoring: on an untouched summer hold-out, capture came in 3.4 points below the same months of validation because the forecast degraded.

![Battery trading dashboard on the last hold-out day](docs/img/dashboard.png)

*My dashboard on 14 September 2026, the last hold-out day: the forecast issued at 11:40 the day before topped out at €198/MWh and missed the €740/MWh evening peak, yet the schedule solved for a 1 MW / 2 MWh battery earned €964, 93.2% of the €1,034 perfect foresight made. Every control reads backtest results for 835 days. The panel above the chart is the desk briefing, written from the same numbers the page shows.*

## What's in the system

| Component | What it does | Method |
|---|---|---|
| Data pipeline | DE-LU prices, load, wind and solar at 15-minute resolution; weather forecasts, gas, carbon, imbalance prices | SMARD, Open-Meteo, Yahoo TTF, EEX, ENTSO-E API |
| Price forecaster | Quantiles q05 to q95 for every quarter-hour of the next day, issued at 11:40 | LightGBM with conformal ranges, chosen from eight models |
| Battery optimiser | Charge and discharge schedule for one delivery day | MILP in PuLP with CBC, wear cost, two-cycle cap |
| Backtester | Walk-forward over two years, profit against perfect foresight, one frozen hold-out | Daily re-solve, Shapley attribution, block bootstrap |
| Model health | Failure experiments measured in euros: a price regime shift, drift, a missing weather feed and the 12:00 deadline; an incident log | Walk-forward reruns, rolling alerts with thresholds fixed on validation, a fallback chain |
| Live pipeline | A daily run that forecasts tomorrow, commits a schedule before the 12:00 gate and settles it the next day | GitHub Actions on a schedule, or the same steps as an Airflow DAG; readiness sensor with a deadline, fallback chain with a time limit on every step, MLflow model registry refit every 28 days, a daily drift check, state kept on its own branch |
| Dashboard | Forecast fan, calibration, error by hour, the day's schedule, cumulative profit for any battery from 0.5 to 5 MW and 1 to 4 hours, a Model health tab, and a Live tab with every day the pipeline traded | React and FastAPI over exported backtest results; one day's schedule solved on request |
| Desk briefing | Three to five sentences about the tab you are on, and three follow-up questions | A deterministic writer over that page's own numbers; a language model can be switched on, with every figure it writes checked against them before it is shown |

## Results

### Forecast quality

Validation window, June 2024 to May 2026, 70,080 quarter-hours.

| Model | Pinball loss | 90% interval coverage | MAE on median (€/MWh) |
|---|---|---|---|
| Seasonal naive (same quarter-hour, last week) | 11.49 | 85.4% | 35.34 |
| Naive (same quarter-hour, yesterday) | 9.61 | 86.4% | 29.05 |
| LEAR (regularised linear model) | 8.06 | 84.9% | 24.59 |
| Quantile forest | 5.18 | 94.6% | 16.80 |
| LightGBM quantile (raw) | 5.00 | 73.3% | 15.43 |
| **LightGBM + conformal (production)** | **5.11** | **85.6%** | **16.03** |
| LightGBM + conformal, hold-out June to September 2026 | 6.95 | 77.8% | 20.24 |

![Where the next euro is: a perfect forecast is worth 9.9% of the ceiling, lifting the cycle cap 0.1%](docs/img/next_euro.png)

*Every bar is the same quantity: what the median-forecast strategy would have gained over the same 730 days, as a share of the perfect-foresight ceiling. Closing the forecast gap is worth €16,424 over the two years; lifting the two-cycle warranty cap, €223. Measuring the cap against the ceiling instead answers a different question, what a trader with perfect knowledge would gain from it, and that figure is €1,151.*

![Reliability diagram of four forecasters](docs/img/calibration.png)

The production model is close to calibrated with slightly narrow tails: its 90% range covered 85.6% of prices, against 73.3% for raw LightGBM quantile. It is weakest on extremes: pinball loss rises to 8.20 on days above €200/MWh, and its range did not reach the -€500/MWh floor on 1 May 2026.

![The Forecast tab of the dashboard](docs/img/forecast_tab.png)

*The Forecast tab: calibration against the diagonal, error and money at stake by delivery hour, and what the model leans on. Over the 29 traded days to 14 September 2026 the 50% band held 31.7% of prices, the 80% band 56.8% and the 90% band 67.4%, so the ranges ran narrow at the end of the hold-out. Yesterday's price for the same quarter-hour carries 17.5% of the model's gain.*

### Trading performance

1 MW / 2 MWh, 90% round trip, €8 wear per MWh discharged, at most two cycles a day. Schedules are committed before the 12:00 gate and settled at the realised price.

| Strategy | Annual P&L net of wear (€/MW) | Capture vs perfect foresight | Cycles/day | Max drawdown (€) | Hold-out capture |
|---|---|---|---|---|---|
| Perfect foresight (upper bound) | 82,961 | 100% | 1.74 | 0 | 100% |
| Median-forecast dispatch | 74,749 | 90.1% | 1.73 | 30 | 91.0% |
| Mean-forecast dispatch | 74,762 | 90.1% | 1.76 | 24 | 90.7% |
| Quantile-aware dispatch, q25 | 70,752 | 85.3% | 1.25 | 26 | 88.8% |
| Median dispatch on a naive forecast (yesterday's prices) | 64,418 | 77.6% | 1.73 | 174 | 84.0% |
| **Fixed shape, same calendar month (no forecast at all)** | **70,461** | **84.9%** | 1.85 | 37 | not scored |
| Fixed shape, every trailing day (no forecast at all) | 56,987 | 68.7% | 2.00 | 821 | not scored |

The two fixed-shape rows are not scored on the hold-out: it was scored once, before these arms existed, and reusing it would spend the one clean test this project has. The all-year fixed shape sits exactly on the two-cycle warranty cap on every one of the 730 days, so its 2.00 is a binding constraint rather than a comfortable margin.

![Forecast fan and battery schedule for one backtest day](docs/img/example_day.png)

*One validation day: the quantile forecast issued at 11:40 the day before, and the schedule the optimiser committed before the 12:00 gate.*

![The Trading tab of the dashboard](docs/img/trading_tab.png)

*The Trading tab: profit against the perfect-foresight ceiling, and where the forecast error cost money. Over the last 29 traded days the battery earned €8,755 of the €9,775 available, 89.6%, with no drawdown. Under-forecasting the 18:00 to 20:00 block cost €1,029, more than the whole €1,019 gap to perfect foresight, because the other blocks handed some of it back.*

I froze every choice on the validation window before trading the hold-out, 1 June to 14 September 2026, once. Against the same summer months of validation, 94.3%, the hold-out came in 3.4 points lower.

Where the money is lost: about 40% of the gap to perfect foresight falls between 15:00 and 21:00 local time. Within that window the loss sits later in summer, so the 18:00 to 21:00 evening block takes 35% of the summer gap but only 16% in winter. Anchoring the window to sunset instead of the clock explains no measurably more of the loss. Reading the two schedules inside that window, the cash they trade there cannot be told apart. The window's cost sits where the forecast came in below the outcome, 89% of it, while the battery reaches 15:00 slightly fuller than perfect foresight, so a charge reserve would not target it. The cheapest dispatch-side correction, valuing sales in that window at the forecast's q75, was tested against a bar fixed before the run and rejected: the daily profit difference against median dispatch was -€2.59 (95% interval -3.95 to -0.89), because cash after 21:00 fell by most of what the window gained and cash in the morning and at midday fell too.

### The decision-value experiment

![Forecast accuracy against captured profit for eight forecasters](docs/img/decision_value.png)

Better pinball loss mostly means more profit: capture rises from 77.6% for a naive forecast to 90.9% for QRA. Six of the eight forecasters on that chart sit above a rule that forecasts nothing, at 84.9%; the naive and seasonal-naive baselines, at 77.6% and 79.7%, sit below it. That compresses the useful part of the range far more than the chart suggests. Among the four most accurate models, capture differs by only 2.1 points and does not follow the accuracy order. Synthetic forecasts show why: with the same €16/MWh average error, a level shift lost €44 over two years and random noise €24,800, while pulling the evening one hour early lost €24,800 with an error of only €7/MWh.

So I tried the other direction: training the forecaster on the money. Keeping the production model as a fixed base, 100 extra trees are trained on a loss that rewards the battery's profit rather than the price error (SPO+, Elmachtoub and Grigas 2022), with the optimiser's linear relaxation solved 70,000 times per fit. Against a bar fixed before the run, on the 730 validation days it earned €2.93 a day more than median dispatch (95% interval +1.53 to +4.74), €2,138 over two years and 1.3 capture points, with the point's accuracy unchanged at 16.0 €/MWh: the corrections raise the morning and evening by about €1 and lower the night by about €1. The honest caveats: the ten best days carry 58% of the gain, two dark winter days a third, and 2025 alone only just clears zero. Checked afterwards: a cheap hour-of-day bias correction lost money, and other seeds gave the same result. The live pipeline still trades the median forecast.

Two more experiments then went after the evening loss from the forecasting side, each with its pass mark fixed before the run, and both failed it. Nine columns saying why an evening spikes (evening load and its ramp, afternoon solar, evening wind, residual load, recent spikes) changed profit by -€0.76 a day (95% interval -1.97 to +0.52): the model could already see a tight evening coming and still guessed low, because a model trained for average accuracy hedges. A separate model of the chance that the evening reaches €200 was good (AUC 0.91; of the days it put above 70%, 84% spiked) but betting on it in the dispatch, by raising the evening valuation or by holding the battery full for 17:00, earned nothing measurable (-€0.65 and +€0.27 a day, both within noise): when a spike is visible the optimiser already holds charge for it, and the money sits in which quarter-hour the peak lands. The three together say the evening loss is not a shortage of day-level signals about whether the evening will be tight, nor a confidence problem; what is left is consistent with timing within the evening on spike days, and moving it would take a forecast of the peak's shape, quarter-hour by quarter-hour. The plans, written before each run, are in `docs/plans/`.

### The benchmark that forecasts nothing

Every baseline above is a forecaster, including the naive one the table presents as its floor. So I built one
that is not. For each delivery day, take the mean price of every local time of day over the trailing 730 days
of the same calendar month, and solve the same optimiser once on that average profile. The schedule uses
nothing whatever about the day it trades: it is climatology, not prediction, and a test fails if one period of
the target day leaks into the average.

It captures 84.9%, against the production forecast's 90.1%. The floor was in the wrong place. Against the naive baseline this repository published, it moved seven points; against raw previous-day prices put through the same code and the same ceiling, which reach 80.5%, it moved 4.4. Either way the floor was too low.
Restricting the average to the same calendar month is what does the work: averaging every trailing day instead
captures 68.7%, with 83 losing days and a worst day of -€200.

What the forecast is worth is the 5.2 points between them, €11.75 a day. It is not spread evenly. Scoring each
day by how closely its ranking of quarter-hours matched the seasonal shape, then splitting the days into
fifths. The split by day type was fixed before the run; the two capture columns below were added after it,
and are reported as post-hoc on the results page:

| the day's ranking of periods | forecast capture | fixed rule | the forecast's edge |
|---|---|---|---|
| most ordinary fifth | 92.9% | 93.2% | -€0.94 a day |
| second | 93.1% | 90.8% | +€6.34 a day |
| middle | 91.6% | 84.8% | +€17.25 a day |
| fourth | 86.3% | 76.8% | +€16.19 a day |
| most broken fifth | 80.8% | 67.3% | +€19.91 a day |

On an ordinary day the forecast is worth nothing measurable and the fixed rule is fractionally ahead. What the
forecast buys is a slower failure: when a day departs from its pattern the forecast loses 12 points of capture
and the fixed rule loses 26. The 40% most ordinary days carry 9.2% of the forecast's total edge. Three of the
four predictions fixed before that run failed: a wider spread buys the forecast nothing once the day's own
ceiling is divided out, and spike days are worth less to it, not more.

Run again over the 2021-2023 gas crisis the fixed rule captured 84.5%, essentially unchanged, and season for
season identical, 83.2% in winter in both windows. A trivial yesterday's-prices forecast went the other way,
from 80.5% to 74.2%: the smooth estimator held while the noisy one degraded. Two of the three predictions I
registered before that run, that the rule would break and that a trivial forecast would overtake it, failed.

One limitation runs in the benchmark's favour and should be named. Day-ahead traded hourly products until
2025-10-01, so on a quarter-hourly delivery day roughly three quarters of the history behind its seasonal
shape carries no structure inside the hour: the shape's within-hour variation is €1.90/MWh against €8.41 in
the realised prices. That handicaps the fixed rule, so 84.9% is a floor on what it could do, not a ceiling.

This is not a new idea and the repository should say so: `docs/prior_work.md` records what was already known,
including a 2026 preprint that runs the same benchmark on French prices and reaches 78%, and a 2025 paper whose
comparable ordering disagrees with mine. My numbers sit inside the published range rather than outside it.

### Model health

I broke the pipeline on purpose and measured what it cost.

![Model health tab: regime shift and drift monitor](docs/img/model_health.png)

*The Model health tab: rolling 90% coverage through the 2021 to 2023 gas crisis for a model frozen on 2019 to 2020 prices against quarterly and monthly refits, and the drift monitor at the end of the hold-out, 67.5% coverage against its 74.0% alert line.*

- Fitted on 2019 to 2020 and never refitted, the model's 90% range covered 25.7% of prices through the 2021 to 2023 gas crisis and captured 28.9% of perfect foresight. Refitted every 28 days, as in production, it held 80.6% coverage and 80.7% capture.
- Refitting weekly instead of every 28 days cut mean pinball loss by 2.1%, in the evening peak as much as elsewhere, but brought no measurable gain in profit over two years of validation: €327 more, well inside the noise. The live pipeline keeps the 28-day cadence the backtests used.
- My drift monitor, with thresholds fixed on validation, first alerted 31 days into the hold-out on its pinball-loss signal; the coverage signal took 102 days.
- A missing weather feed at 11:40 cost 2.3 capture points, €3,742 over two years; falling back to a model trained without weather cut that to €1,973.
- With pipeline failures injected on 13% of days, my fallback chain still submitted a forecast before the 12:00 gate every day in my simulation, and kept €15,898 that a pipeline without fallbacks, and so without a position on those days, would have lost.

### The daily pipeline

The same code that ran the backtest runs as a daily job. It refreshes the feeds, waits
for tomorrow's load forecast and weather, forecasts at 11:40 and commits a schedule
before the 12:00 gate. The next day it settles what it committed and runs the drift
monitor on the settled record.

```mermaid
flowchart LR
  A[ingest prices] --> D[build inputs]
  B[ingest weather] --> D
  C[ingest fuels] --> D
  D --> E{wait for inputs<br/>deadline 11:30}
  E -->|feeds arrived| F[refit if due,<br/>forecast]
  E -->|timed out| G[forecast, degraded]
  F --> J[record days<br/>with no bid]
  G --> J
  J --> H[export dashboard]
  H --> L[check deadline]
  L --> I[settle yesterday]
  I --> K[check drift]
```

- **A late feed does not stop the bid.** The chain steps down from the production model
  to one without weather, then seasonal naive, then yesterday's prices, writing an
  incident that names the rung that ran. Every fit and every rung has a 120-second cap.
- **The model comes from the MLflow registry and is refit every 28 days.** A refit must
  forecast its day with finite quantiles before it is registered; one that fails leaves
  the serving version in place and writes an incident, so it never blocks the bid.
- **A registered model that will not load is a critical incident, not a silent
  substitution.** The chain still fits one on the spot and still bids, because a bid
  beats no bid, but the run says which version it could not load.
- **The interpreter is pinned, because the model is a pickle.** Unpickling under a
  different Python minor version segmentation faults rather than raising, and a process
  that dies there writes nothing. The loader now refuses a mismatch before unpickling.
- **The registry is rewritten on the way in and out.** MLflow stores absolute paths, so
  a store restored under another root would quietly fit a local model and call it
  production. A save refuses to publish a machine path.
- **Eight scheduled attempts run between 00:13 and 08:47 UTC,** far earlier than a 12:00
  Berlin gate needs, because GitHub fires these late: delays here ran 4h 14m to 5h 23m,
  then drifted to 7h 39m and cost three bids. No single margin survives a distribution
  that moves, so the attempts are dense instead, and sit on odd minutes because the top
  and half of the hour are the most contended slots.
- **An attempt that lands more than four hours out stands down without bidding,** and
  the next carries on, so the chain is one long poll made of short jobs. Inside the
  window it re-fetches the weather and rebuilds on each poke, until the feeds arrive or
  45 minutes before the gate.
- **Re-fetching matters as much as rebuilding.** The weather ingest masks anything
  stamped past the moment of the fetch plus its lead minus an archive lag, so the mask
  relaxes only when the fetch is repeated. Polling the rebuild alone cost three days of
  production forecasts once the schedule moved earlier.
- **State travels, market data does not.** About 7 MB of run records, plans,
  settlements, incidents and the registry live on a `live-state` branch, so each bid's
  time is in a commit. The data is rebuilt from source, 86 days on an ordinary run
  against eight years, sized from what that day actually reads.
- **A day the pipeline did not run is shown, not hidden.** Each gets its own incident
  and its own row, and every live figure is counted over days the desk actually bid. A
  day it ran and failed on is not an outage: that run wrote its own incident.
- **A reconstruction never passes for a bid.** The chain can forecast a past day from
  the data available before its gate, but it has no issue time, no gate verdict, is
  left out of drift, and is labelled "reconstructed". It does not close the gap: the
  battery still traded nothing.
- **The drift monitor runs on the live record** with thresholds fixed on validation:
  rolling 28-day coverage below 74% or pinball above 1.5 times its validation median.
  It needs 28 settled days before it can alert, and raises for review rather than
  refitting early.
- **The hold-out stays frozen:** the pipeline refuses any delivery day before
  `live_from`, the day after the hold-out was last scored.

**The record so far.** 19 delivery days from 16 September 2026, 17 of them bid: 12 on
time, 5 late, and 2 days reconstructed afterwards and counted separately. Eleven bids
used the production model, three a fallback after a feed was missing, three a fallback
caused by the polling bug above. Sixteen have settled, at €6,811 against €6,901
planned. Every failure behind those numbers is in the incident log or the commit
history. One is in neither and cannot be: a scheduled run that never starts writes
nothing at all, and closing that needs a heartbeat watched from outside the workflow.

## Three things I learned

- Timing beat accuracy in my backtest: an evening forecast one hour early, with a €7/MWh average error, cost as much as random noise at €16/MWh. Turned around, a forecaster trained on the battery's profit instead of its error earned 1.4% more with the same accuracy.
- Days with a negative price were 28% of my trading days but earned 37% of the battery's profit.
- On my hold-out, forecasts too low between 18:00 and 21:00 caused 49% of the shortfall against perfect foresight. On validation, telling the model why evenings spike, a model of the chance of a spike, selling that window at q75, bidding price limits from the fan, and dispatching against whole-evening price paths all left the evening loss where it was. The last of those settles something: with a profit linear in price, the expected value of a schedule over any set of paths is its value at their average, so the ordering across periods cannot reach the battery through the optimiser at all.

## Limitations

- **Day-ahead only:** no intraday re-trading and no balancing or reserve revenue. I looked into adding the intraday leg honestly and stopped: neither SMARD nor the ENTSO-E Transparency Platform publishes a free DE-LU intraday index at quarter-hour resolution. SMARD's wholesale category is day-ahead prices for each bidding zone, and ENTSO-E's A44 returns the two day-ahead auctions, SDAC and the EXAA 10:15 auction, whatever contract or auction parameters I asked for. Simulating intraday without those prices would have meant inventing them.
- **Price taker (T5):** one 1 MW battery with fixed-volume orders filled at the clearing price; no fleet, grid or market-impact effects. Bidding price limits from the forecast's own quantiles instead was tested and rejected: withholding one leg of a paired trade starves the other, and the imbalance charges for undelivered energy (-€44,271 and -€29,563 across the two arms) dwarfed what the limits saved. With those consequences left out the difference is within noise, so it is the pairing that breaks it, not the limits.
- **Wear is a flat €8 per MWh discharged** with a two-cycle cap, not a cell-ageing model, and each day starts and ends half full.
- **Outages sit outside the headline numbers (T4).** Settled at the German imbalance price, a random two-hour outage costs €44 on average, the worst window of a day €277.
- **The hold-out is 105 summer days (T6)** and public data has gaps. Failure rates in the deadline simulation are assumptions, not measured outages.
- **Measured feeds are read at their latest revised values.** SMARD revises actual
  load and generation after publication, and every build downloads whatever is there
  now rather than what stood on the day. The features that use them are lagged, so a
  day's own values are never read, but a backtest and a reconstruction both see
  figures slightly tidier than the desk had. Forecast feeds do not have this problem:
  the weather comes from an archive of forecasts at a fixed lead, and prices are final
  once the auction clears.
- **A scheduled run is not guaranteed to start.** GitHub delays and drops scheduled
  workflows, and a run that never begins writes no incident and opens no issue, so
  nothing inside the system can report it. Eight attempts a morning make a drop
  survivable; a morning on which every one is dropped is only caught by the sweep the
  next day, and closing that needs a heartbeat watched from outside the workflow.
- **The live model refits on a fixed schedule only.** The drift monitor runs daily on the live record but only flags a retraining review, and it needs 28 production-model days before it can alert, so a sudden jump in price level waits for the next 28-day refit.
- **The desk briefing is written by a template, not a model, by default (M5).** With a model switched on, every figure it writes must appear in the payload the page was built from, and prose that fails gets one rewrite before the template answers. What the check cannot tell is whether a figure is used in the right role: of 20 gpt-4o-mini briefings shown after it, 7 still misstated what a figure meant. A reflection layer with evals could close that gap; I left it out on purpose, because the template says less but everything it says is right.
- Not a trading recommendation.

A production system would add intraday re-optimisation and bid curves on top of what
is here.

## Architecture

```mermaid
flowchart LR
  A[SMARD, Open-Meteo,<br/>gas, carbon, ENTSO-E] --> B[Information set<br/>as of 11:40]
  B --> C[Features]
  C --> D[Quantile forecaster<br/>LightGBM + conformal]
  D --> E[Battery optimiser<br/>MILP, PuLP and CBC]
  E --> F[Settlement at<br/>realised prices]
  F --> G[Backtester and<br/>attribution]
  G --> J[Failure experiments,<br/>drift and incidents]
  J --> H
  G --> H[Exported artifacts]
  H --> I[FastAPI and<br/>React dashboard]
```

```
config/settings.yaml     market, data sources, hold-out, battery, strategies, live refits
src/ingest/              SMARD, Open-Meteo, fuel and ENTSO-E clients, data-quality checks
src/features/            features built only from what is known at 11:40, plus an opt-in group of evening spike drivers
src/forecasting/         information set, walk-forward harness, eight models, hold-out runner
src/trading/             battery, MILP optimiser (with an optional state-of-charge floor) and its HiGHS relaxation, settlement, strategies, backtest, attribution
src/health/              drift monitor, incident log, failure experiments (M1 to M5, T1 to T10, D1, D5)
src/narration/           the desk briefing: payload, deterministic writer, optional model, grounding check
src/pipeline/            the daily live run: readiness, fallback chain, plan, model registry, refits, drift check
dags/                    the Airflow DAG, thin: every task shells into src/
ops/                     launchd plists for running the DAG from login on a Mac
src/export/              dashboard artifacts: forecasts, P&L grid, attribution
api/                     read-only FastAPI service behind the dashboard
frontend/                React, TypeScript and recharts dashboard
notebooks/               builders for the evaluation notebooks and README figures
tests/                   network-free tests, including DST days, leakage and toy MILPs
docs/                    data guide, production notes, prior work, dashboard API contract, experiment plans, results, figures
```

## Reproduce it

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/); the dashboard also needs Node.js 20.19 or newer.

```bash
make setup      # uv sync --extra dev --extra eda
make data       # SMARD prices, load and generation; weather forecasts; gas and carbon
make entsoe     # optional: ENTSO-E price cross-check and imbalance prices
make forecast   # baselines and the eight-model walk-forward comparison
make backtest   # battery strategies, backtest suites, outage stress test
make holdout    # hold-out forecasts and backtest; runs once, refuses a second run
make report     # results report and evaluation notebooks
make figures    # README figures
make export     # dashboard artifacts, including the P&L grid for 104 battery settings
make experiments # Phase 6 failure experiments: regime shift, missing weather, deadline, drift
                # the follow-up experiments run one at a time: uv run python -m src.health.experiments.<name>
make health     # drift monitor and incident log from saved forecasts
make mlflow     # browse the recorded experiment runs at http://127.0.0.1:5001
make airflow-setup # create the Airflow environment, once
make airflow    # scheduler and UI at http://127.0.0.1:8080
make pipeline   # one live run without Airflow: make pipeline DAY=2026-09-17
make settle     # value a committed schedule once prices publish: make settle DAY=2026-09-17
make drift      # drift monitor on the live record through a day: make drift DAY=2026-09-17
make gaps       # record the delivery days the desk did not bid on: make gaps DAY=2026-09-21
make backfill   # reconstruct a missed day, marked as a reconstruction: make backfill DAY=2026-09-19
make history DAY=2026-09-23 # days of market history that run needs to download
make state-restore STORE=.live-state # bring the desk's state into a fresh checkout
make state-save STORE=.live-state    # carry it back out after a run
make launchd-install # macOS: switch the daily DAG on, keep Airflow running from login and the Mac awake for the run
make dashboard  # build the React app and serve it with the API at http://127.0.0.1:8000
make test       # ruff, strict mypy, pytest
```

SMARD and Open-Meteo need no key. For ENTSO-E, register on the [Transparency Platform](https://transparency.entsoe.eu), email transparency@entsoe.eu asking for Restful API access, generate a token in your account settings and set it as `ENTSOE_API_KEY` in `.env` at the repository root: copy `.env.example` to `.env` to start, which is gitignored. Downloads land in `data/raw/` and `data/processed/`, which are not committed.

The desk briefing needs no key: a deterministic writer produces it from the page's own numbers, and every test runs that way. To have a model write it instead, set `NARRATION_PROVIDER` to `openai` or `anthropic` and that provider's key (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`) in the same `.env`, or export them in your shell, which takes precedence over `.env`. A key alone does not switch the model on. Every figure the model writes is checked against the page before it is shown.

On an Apple M2 Pro with 10 cores, the eight-model comparison takes 88 minutes, the five validation backtest suites 17, and the hold-out forecasts 5.

| Source | Used for | Licence |
|---|---|---|
| [SMARD.de](https://www.smard.de), Bundesnetzagentur | DE-LU day-ahead prices; load, wind and solar | CC BY 4.0 |
| [Open-Meteo](https://open-meteo.com) Previous Runs API | Weather forecasts issued two days ahead, 12 German points | CC BY 4.0; free API for non-commercial use |
| Yahoo Finance, ticker TTF=F | Daily Dutch TTF gas settlement | Yahoo terms, personal use; not redistributed |
| [EEX](https://www.eex.com) EU ETS primary auctions | Daily carbon allowance price | EEX public reports; not redistributed |
| [ENTSO-E Transparency Platform](https://transparency.entsoe.eu) | Price cross-check; German imbalance prices | Platform terms of use; not redistributed |

Data: Bundesnetzagentur | SMARD.de, and weather data by Open-Meteo.com, both licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Background reading

The reasoning and evidence behind each choice, written while building this: [the data guide](docs/data_guide.md), [the production notes](docs/production_notes.md) with a measured result for each failure mode, [the model comparison](notebooks/model_comparison.ipynb), [the backtest notebook](notebooks/backtest_report.ipynb), [the full results](docs/results/) and [the experiment plans](docs/plans/). Each was written before its run, and four of them, the bid-curves plan and the three behind the benchmark that forecasts nothing, were committed on their own before any of their code, so those are the ones whose ordering the commit history proves.

## About me

**Hasan Rafiq.** PhD in Electrical Engineering, 9+ years of applied machine learning in energy, with production experience in load and generation forecasting, battery dispatch optimisation, fault detection & diagnosis, and asset health monitoring. [LinkedIn](https://www.linkedin.com/in/rafiqh)

Questions, corrections or things I have got wrong are welcome: [hassan.rafiq182@gmail.com](mailto:hassan.rafiq182@gmail.com)

