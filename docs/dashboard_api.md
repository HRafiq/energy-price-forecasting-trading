# Dashboard API

The read-only FastAPI service behind the dashboard. It serves the artifacts written by
`python -m src.export.artifacts` from `data/processed/dashboard/<run_id>/`. Battery
P&L comes from the exported grid, scaled by power; forecast scores (pinball loss,
calibration, error by hour) are aggregated from the exported forecasts on each
request. The one thing solved on request is a single day's schedule for the chosen
battery, with the same optimizer the backtest used.

Start it with `make api` (port 8000). Every endpoint returns JSON, and every one
is a `GET` apart from `POST /api/narrate`, which writes the desk briefing.

## Common query parameters

| Parameter | Values | Default |
|---|---|---|
| `run` | a run id from `/api/runs` | the latest run |
| `window` | `last30`, `last90`, `validation`, `holdout`, `all`; `last30` and `last90` end on the run's last day | `last30` |
| `duration` | battery duration in hours: 1, 2, 3 or 4 | 2 |
| `degradation` | wear price in € per MWh discharged: 0 to 25 | 8 |
| `strategy` | dispatch dial: `median`, `q25`, `q10` | `median` |
| `power` | battery power in MW: 0.5 to 5 in steps of 0.5 | 1 |

Every endpoint accepts `run`. Each error carries a readable `detail`:

| Status | When |
|---|---|
| 422 | An invalid value: a malformed date, a window, battery or power outside the grid, or a day that was not traded |
| 404 | An unknown run, or a well-formed date the run has no forecast for |
| 503 | A file not exported yet (the grid takes about an hour), a run being rewritten by an export, or a live solve that did not finish |

**Delivery days** have 96 quarter-hours, 92 on the spring clock change and 100 on the
autumn one, when the local times 02:00 to 02:45 appear twice. Each period carries a
unique `utc` stamp; use the position in `periods` as the x value and `time` only as a
label.

## Endpoints

### `GET /api/health`

```json
{"status": "ok", "run": "backtest-2026-09-14", "run_kind": "backtest", "traded_days": 835,
 "grid_available": true, "mode": "Battery results come from a pre-computed grid at 1 MW ..."}
```

### `GET /api/runs`

```json
[{"run_id": "backtest-2026-09-14", "run_kind": "backtest", "created_utc": "2026-09-14T18:00:00+00:00",
  "issued_utc": null, "source_commit": "8689442", "grid_available": true,
  "model": "lightgbm_conformal", "baseline_model": "naive_previous_day",
  "first_day": "2024-06-01", "last_day": "2026-09-14", "data_through": "2026-09-14",
  "holdout_start": "2026-06-01", "timezone": "Europe/Berlin",
  "traded_days": 835,
  "reference_battery": {"power_mw": 1.0, "capacity_mwh": 2.0, "round_trip_efficiency": 0.9,
                        "degradation_eur_per_mwh": 8.0, "initial_soc_fraction": 0.5,
                        "max_cycles_per_day": 2.0},
  "grid": {"durations_h": [1, 2, 3, 4], "degradation_eur_per_mwh": [0, 1, 2, "...", 25],
           "power_mw": {"min": 0.5, "max": 5.0, "step": 0.5},
           "strategies": {"median": "median_forecast", "q25": "quantile_q25", "q10": "quantile_q10"}},
  "mode": "..."}]
```

`run_kind` is `backtest` for a replay of history and `live` for a run the daily
pipeline produced. A live run also carries `issued_utc`, the UTC time it issued its
forecast (null for a backtest); both carry `data_through`, the last delivery day with
published prices, which on a live run can be behind the day it forecasts. A manifest
written before live runs carries none of the three and is served as a backtest, with
`issued_utc` null and `data_through` at the run's last day. `timezone` is the market's
timezone, which the dashboard uses to show `issued_utc` in market local time. The
header pill reads `run_kind`: "live · today's run", otherwise "backtest · historical
data"; a live run whose `data_through` is behind the delivery day on screen also gets a
muted note that those prices are not published yet.

### `GET /api/days`

Every day in the run, for the date picker.

```json
{"days": [{"date": "2024-06-01", "window": "validation", "traded": true, "skip_reason": null},
          {"date": "2026-09-13", "window": "holdout", "traded": false, "skip_reason": "missing price or forecast"}]}
```

### `GET /api/runs/{run_id}/summary?window&duration&degradation&strategy&power`

The KPI strip. P&L is net of wear and scaled by power; pinball loss does not depend on
the battery.

```json
{"window": {"key": "last30", "first_day": "2026-08-16", "last_day": "2026-09-14",
            "traded_days": 29, "skipped_days": ["2026-09-13"]},
 "battery": {"power_mw": 1.0, "capacity_mwh": 2.0, "duration_h": 2,
             "degradation_eur_per_mwh": 8, "strategy": "median"},
 "kpis": {"pnl_eur": 9412.5, "perfect_foresight_pnl_eur": 10321.0, "capture_ratio": 0.912,
          "cycles_per_day": 1.64, "pinball_eur_mwh": 6.81, "baseline_pinball_eur_mwh": 10.52}}
```

### `GET /api/forecast?date`

The quantile fan for one delivery day, in local time. `actual` is null where no price
was published.

```json
{"date": "2025-11-21", "window": "validation", "issued_local": "2025-11-20 11:40",
 "gate_local": "2025-11-20 12:00", "product_minutes": 15,
 "periods": [{"time": "00:00", "utc": "2025-11-20T23:00Z", "q05": 97.1, "q10": 99.0, "q25": 101.9, "q50": 108.7,
              "q75": 115.3, "q90": 125.4, "q95": 131.2, "actual": 101.6}]}
```

### `GET /api/dispatch?date&duration&degradation&strategy&power`

The schedule solved on request for one day. Positive `net_mw` sells, negative buys;
`soc_mwh` is the state of charge at the end of each period. `solve_ms` covers two
solves, the chosen strategy and perfect foresight; each is abandoned after 10 s.

```json
{"date": "2025-11-21", "strategy": "median", "solved_on_request": true, "solve_ms": 115,
 "battery": {"power_mw": 1.0, "capacity_mwh": 2.0, "degradation_eur_per_mwh": 8},
 "pnl_eur": 362.4, "perfect_foresight_pnl_eur": 397.0,
 "periods": [{"time": "04:00", "utc": "2025-11-21T03:00Z", "charge_mw": 0.84, "discharge_mw": 0.0, "net_mw": -0.84,
              "soc_mwh": 1.2, "price": 93.6}]}
```

### `GET /api/pnl?window&duration&degradation&strategy&power`

Cumulative P&L, running totals in € from the first traded day of the window, for perfect foresight, median dispatch and the selected strategy. Drawdown is computed on the daily values.

```json
{"window": {"...": "as in summary"}, "selected_strategy": "q25",
 "series": [{"date": "2026-08-16", "perfect_foresight": 312.0, "median": 280.4, "selected": 271.9}],
 "max_drawdown_eur": {"perfect_foresight": 0.0, "median": 0.0, "selected": 0.0}}
```

### `GET /api/calibration?window`

Share of realised prices at or below each forecast quantile, and the coverage of the
central intervals, production model.

```json
{"window": {"...": "..."},
 "quantiles": [{"level": 0.05, "empirical": 0.07}, {"level": 0.95, "empirical": 0.93}],
 "intervals": [{"nominal": 0.5, "coverage": 0.49}, {"nominal": 0.8, "coverage": 0.76},
               {"nominal": 0.9, "coverage": 0.86}]}
```

### `GET /api/error-analysis?window`

Mean absolute error of the median forecast per local hour, the cash perfect foresight
moves in that hour (value at stake, reference battery), and the Shapley split of median
dispatch's gap to perfect foresight by hour block and direction (reference battery). Neither depends on the battery controls. A block's share can be negative, so one block can exceed the whole gap.

```json
{"window": {"...": "..."},
 "by_hour": [{"hour": 18, "mae_eur_mwh": 31.2, "value_at_stake_eur_per_day": 38.5}],
 "asymmetry": {"reference": "1 MW / 2 MWh, €8 wear, median dispatch", "gap_eur": 3197.0, "days": 105,
               "blocks": [{"block": "18-20", "over_eur": 213.0, "under_eur": 1561.0}]}}
```

### `GET /api/feature-importance`

```json
{"model": "lightgbm_conformal", "trained_for_day": "2026-09-14", "importance": "gain",
 "features": [{"feature": "price_lag_1d", "label": "Price, same quarter-hour yesterday", "gain_share": 0.21}]}
```

## Model health

The Model health tab reads four endpoints under `/api/model-health/`; `/api/health`
stays the service status. They serve the run's `health/` folder, written by
`python -m src.export.artifacts --steps health` (part of the default steps): copies of
`m1_regime_experiment.json`, `m2_drift.json` and `d5_deadline.json` from
`data/processed/experiments/`, the incident log as `incidents.json` (one JSON list)
and `index.json`, which lists the files present and for each the source it was copied
from, `generated_utc` (the source's own `generated_utc` field when it has one, otherwise
its modification time), `source_modified_utc` and `exported_utc`. M2's entry also
records `last_target_day`, `run_last_day` and `matches_run`. A source that does not
exist is skipped with a note and any copy an earlier export left is removed, so an
endpoint answers 503 rather than serving data without a source. A re-export is picked
up on the next request without a restart: each health file's modification time in
nanoseconds and its size are part of the cache key, and `index.json` changes with
every export. The dashboard does not keep model health responses in its session cache.

Nothing is recomputed and nothing is rounded: values are served as the experiments
wrote them and the dashboard formats them once, rounding ties to the even digit like
the results docs. Every endpoint serves `generated_utc` for its source (`ops` per
section), which the tab shows as "as of <date>". `drift` and `ops.drift` also serve
`last_target_day`, `run_last_day` and `matches_run`; when M2 does not end on the run's
last day the tab shows a warning. On the September 2026 run the largest response is
`drift` with every day, about 170 KB (`regime` about 57 KB, `incidents`
about 32 KB at the default limit and 75 KB at 500, `ops` about 2 KB). M1's
`coverage_alert_threshold` is not served: the drift threshold comes from M2 alone.

| Status | When |
|---|---|
| 422 | An unknown `window` or incident `type`, a malformed `source`, `start` or `end`, `start` after `end`, `limit` outside 1 to 500, a negative `offset`; checked before the run's files are read, so a bad query on a run without a health export is still 422 |
| 404 | An unknown `run` |
| 503 | The file an endpoint needs is not in the run's health export, or it is being rewritten; `ops` answers 503 only when the run has no health export at all |

### `GET /api/model-health/regime`

The M1 regime shift backtest, 2021 to 2023: the same model family as production
(LightGBM with conformal ranges) without weather features, fitted once (`frozen`), refitted every 91 days (`quarterly`) or every 28 days
(`monthly`), with the naive previous-day baseline for context. `rolling_coverage_90`
holds one value per arm per date, the share of periods inside the 90% interval over
the 28 days ending that date. `periods` holds the per-year rows and the total.

```json
{"experiment": "m1_regime_shift", "generated_utc": "2026-09-15T15:40:53+00:00",
 "historical_backtest": true,
 "setup": {"model": "lightgbm_conformal", "training_days": 730, "calibration_days": 42,
           "first_target_day": "2021-01-01", "last_target_day": "2023-12-31",
           "feature_groups": ["calendar", "price_history", "load", "grid_operator_forecasts", "measured", "fuels"],
           "weather_features": false, "arms": [{"name": "frozen", "refit_every_days": null, "description": "..."}],
           "summary": "...", "spike_threshold_eur_mwh": 200.0, "battery": {"power_mw": 1.0, "...": "..."}},
 "coverage_target": 0.9,
 "rolling_coverage_90": {"window_days": 28, "dates": ["2021-01-28", "..."],
                         "arms": {"frozen": [0.888393, "..."], "quarterly": ["..."], "monthly": ["..."], "naive_previous_day": ["..."]}},
 "min_rolling_coverage_90": {"frozen": {"value": 0.001488, "window_end": "2022-12-15"}},
 "periods": [{"arm": "frozen", "period": "2022", "days": 365, "coverage_90": 0.0625, "coverage_50": 0.0162,
              "pinball": 80.023837, "mae_q50": 167.807163, "capture_ratio": 0.163978, "pnl_eur": 18874.706503,
              "perfect_foresight_pnl_eur": 115105.034176}]}
```

### `GET /api/model-health/drift?window`

The M2 drift monitor over the production model's saved forecasts. `window` is
`validation`, `holdout` or `all` (default) and filters `series`, `episodes` and
`windows`. Thresholds were fixed on validation days only and applied unchanged to the
hold-out. `pinball_ratio` is the rolling mean pinball loss over its validation median.

```json
{"experiment": "m2_drift", "generated_utc": "2026-09-15T15:16:52+00:00", "model": "lightgbm_conformal",
 "window": "all", "window_days": 28, "holdout_start": "2026-06-01", "rule": "Rolling 28 traded days, ...",
 "last_target_day": "2026-09-14", "run_last_day": "2026-09-14", "matches_run": true,
 "thresholds": {"coverage": 0.74, "pinball_ratio": 1.5, "pinball_median": 4.912904, "validation_days": 730,
                "coverage_alert_share": 0.047945, "pinball_alert_share": 0.042466},
 "windows": {"holdout": {"days": 105, "coverage_alert_share": 0.028571, "first_coverage_alert": "2026-09-11", "...": "..."}},
 "episodes": [{"signal": "coverage", "window": "holdout", "start": "2026-09-11", "end": "2026-09-14", "days": 3,
               "first_value": 0.727679, "extreme_value": 0.674851, "extreme_day": "2026-09-14", "open_at_end": true}],
 "series": [{"target_day": "2026-09-14", "window": "holdout", "coverage_90": 0.0, "pinball": 65.341323,
             "rolling_coverage_90": 0.674851, "rolling_pinball_ratio": 1.937204,
             "coverage_alert": true, "pinball_alert": true}]}
```

### `GET /api/model-health/incidents?start&end&type&source&limit&offset`

The incident log, newest first (delivery day, then detection time). `type` and
`source` may repeat (`type=drift&type=tail_miss`); `start` and `end` bound the
delivery day, inclusive. `limit` defaults to 50, `offset` to 0. `total` counts every
record matching all filters. `counts.type` applies every filter except `type`, and
`counts.source` every filter except `source`, so a filter menu shows what each choice
returns; every type is listed, with 0 when absent. `counts.provenance` counts the
records matching all filters by provenance, and `source_provenance` maps each source to
its category.

Each record carries `provenance`: `observed` for `source` `observed` (fixed rules over
saved backtest outputs) and `pipeline` (the live pipeline's own runs), `measured` for
`m2_drift` and `live_drift` (drift alerts measured on real saved forecasts, in the M2
experiment and in the live pipeline's daily check) and `simulated` for `d5_deadline`
(failure injection); a source outside these explicit sets counts as `simulated`.
`in_sample` is `true` when `metrics.in_sample` is 1 (the delivery day lies in the
validation window the threshold was fitted on), `false` when it is 0 and `null` when the
record does not say. The tab shows "observed", "observed: live pipeline", "measured:
drift monitor", "measured: live drift monitor" or "simulated: D5 deadline", with an
"in-sample" tag.

```json
{"generated_utc": "2026-09-15T15:43:04+00:00", "total": 116, "limit": 1, "offset": 0,
 "counts": {"type": {"data_gap": 1, "late_data": 76, "tail_miss": 11, "drift": 9, "pipeline": 19, "drawdown": 0},
            "source": {"d5_deadline": 95, "m2_drift": 9, "observed": 12},
            "provenance": {"observed": 12, "measured": 9, "simulated": 95}},
 "source_provenance": {"d5_deadline": "simulated", "m2_drift": "measured", "observed": "observed"},
 "incidents": [{"incident_id": "414e9c2b52e9d1ca", "delivery_day": "2026-09-14",
                "detected_utc": "2026-09-14T22:00:00Z", "type": "tail_miss", "severity": "warning",
                "detail": "Realised price left the 90% range in 96 of 96 periods (100.0%, rule cut-off 62.9%). ...",
                "action": "Logged for forecast review; no automatic action.", "status": "review",
                "source": "observed", "metrics": {"outside_share": 1.0, "in_sample": 0.0, "...": "..."},
                "provenance": "observed", "in_sample": false}]}
```

### `GET /api/model-health/ops`

The operations tiles. `deadline` is the D5 simulation, with failures injected on the
validation days: the share of days with a forecast before the gate with and without the
fallback chain, with the nominal `failure_rates`, the `seed` and, when D5 records them,
the `realised_failure_rates` (share of days each failure type was drawn; `null` in older
exports). `fallbacks` serves observed fallback activations (observed `data_gap` records
with fallback forecast periods, over the run's days) and D5's simulated fallback days
(over the D5 period) side by side; they cover different periods and kinds of evidence
and are never added. `drift` is the last day of the M2 series, the end of the hold-out. A section whose source is not exported yet
answers `{"available": false, "detail": "D5 not exported yet"}` (or `"M2 not exported
yet"`); `fallbacks` then still carries the observed count.

```json
{"run_id": "backtest-2026-09-14",
 "deadline": {"available": true, "source": "d5_deadline", "simulation": true,
              "generated_utc": "2026-09-15T15:43:04+00:00",
              "period": {"first_day": "2024-06-01", "last_day": "2026-05-31"},
              "issue_local": "11:40", "gate_local": "12:00",
              "failure_rates": {"weather_late": 0.1, "model_fails": 0.03, "prices_late": 0.01}, "seed": 6,
              "realised_failure_rates": {"weather_late": 0.090411, "model_fails": 0.026027, "prices_late": 0.016438},
              "days": 730, "on_time_share_with_chain": 1.0, "on_time_share_without_chain": 0.869863,
              "fallback_days": 95,
              "fallback_by_step": {"fallback_no_weather": 64, "naive_previous_day": 19, "seasonal_naive_previous_week": 12},
              "latest_submission_minutes_after_issue": 10.000142,
              "capture_full": 0.901014, "capture_chain": 0.895963, "capture_without_chain": 0.800145},
 "fallbacks": {"available": true, "simulated_days": 95, "simulated_source": "d5_deadline",
               "simulated_period": {"first_day": "2024-06-01", "last_day": "2026-05-31"},
               "simulated_generated_utc": "2026-09-15T16:12:55+00:00",
               "observed": 0, "observed_rule": "observed data_gap incidents with fallback forecast periods",
               "observed_period": {"first_day": "2024-06-01", "last_day": "2026-09-14"},
               "observed_generated_utc": "2026-09-15T16:12:55+00:00"},
 "drift": {"available": true, "source": "m2_drift", "generated_utc": "2026-09-15T15:16:52+00:00",
           "as_of": "2026-09-14", "window": "holdout", "window_days": 28, "holdout_start": "2026-06-01",
           "holdout_days": 105, "last_target_day": "2026-09-14", "run_last_day": "2026-09-14", "matches_run": true,
           "coverage": {"value": 0.674851, "threshold": 0.74, "alert": true},
           "pinball_ratio": {"value": 1.937204, "threshold": 1.5, "alert": true}}}
```

## Desk briefing

### `POST /api/narrate`

Writes three to five sentences about one tab, for the day and battery the page is
showing. The body carries the same values the other endpoints take as query
parameters, plus the tab and an optional follow-up question. Only the overview tab
is about a single day; the others show a window, so `date` may be left out and the
run's last day is used:

```json
{"tab": "overview", "date": "2026-09-14", "window": "last30", "run": null,
 "power": 1, "duration": 2, "degradation": 8, "strategy": "median",
 "question": "What changed vs yesterday?"}
```

```json
{"tab": "overview", "day": "2026-09-14", "window": "last30",
 "text": "On 2026-09-14 the price ran from 152 EUR/MWh at 16:00 to 740.01 at 19:45. ...",
 "provider": "template", "model": null, "grounded": true,
 "unsupported": [], "fell_back": false, "attempts": 1, "rejected": [],
 "fallback_reason": null,
 "follow_ups": {"why_this_dispatch": "Why this dispatch?",
                "what_changed": "What changed vs yesterday?",
                "explain_the_miss": "Explain the miss"}}
```

The briefing may state only numbers the deterministic core already produced. By
default the deterministic writer produces it. When a model is switched on (see
`provider` below), the model is given a payload built from these same endpoints and
nothing else, and every
figure it writes is looked up in that payload afterwards. A draft carrying a figure
that is not there is refused, and the model writes once more, shown its draft and
told which figures failed. If the rewrite passes it is returned, with `attempts`
at 2 and the first draft's figures still listed in `rejected`. If it fails too there
is no third draft: the deterministic writer answers, `fell_back` is `true`,
`rejected` names the figures from both drafts, and `fallback_reason` says the
briefing stated figures the page does not show. When the model cannot be reached,
nothing is retried, `fallback_reason` carries the error, and the endpoint still
returns a briefing. `attempts` counts the drafts asked for, so it is 1 when the
first one stood or the template is the writer.

Rounding is the only latitude the check gives. A payload holding 0.8957 supports
"89.57%", "89.6%" and "90%", but not "89%"; 964.46 supports "964" but not "965"; and
a count of 19 supports only "19". A percentage must come from a share, so 29 traded
days does not support "29% of perfect foresight". A clock time or a date the payload
holds may be quoted as written and is not read for numbers, but only as a whole
token: the "00:00" inside "200:000" is part of a figure and is checked as one. A
figure the check cannot value, spelled out in words or written in a numeral such as
"½", counts as unsupported rather than being passed over.

What it does not check is which key a sentence is talking about. A number that is in
the payload in another role is accepted: a 2 hour battery cycling 1.71 times a day
supports the sentence "it cycled 2 times a day", because 2 is in the payload. The
check catches figures that exist nowhere in the data, which is the failure that
matters here.

`question` is echoed back inside the answer, so its figures are stripped before it is
used, in digits and in words alike: a figure typed into the question does not appear
in the briefing as though the data held it.

`provider` says who wrote the returned text: `template` by default and after a
fallback, and `openai` or `anthropic` when `NARRATION_PROVIDER` names that provider
and its key is set, in the environment or in the gitignored `.env` (see
`.env.example`). A key alone leaves the template writing. `grounded` describes
the text actually returned. It is `true` in every response the endpoint
produces, because prose that fails the check is replaced rather than returned; read
`fell_back` and `fallback_reason` to see whether that happened.

| Status | When |
|---|---|
| 422 | A tab other than `overview`, `forecast`, `trading`, `model_health`, or a battery value outside the grid |
| 404 | An unknown run, or a well-formed date the run has no forecast for |
| 503 | The P&L grid or health export the tab needs has not been written yet |
