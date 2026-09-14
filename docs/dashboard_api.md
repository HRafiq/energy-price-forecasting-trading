# Dashboard API

The read-only FastAPI service behind the dashboard. It serves the artifacts written by
`python -m src.export.artifacts` from `data/processed/dashboard/<run_id>/`. Battery
P&L comes from the exported grid, scaled by power; forecast scores (pinball loss,
calibration, error by hour) are aggregated from the exported forecasts on each
request. The one thing solved on request is a single day's schedule for the chosen
battery, with the same optimizer the backtest used.

Start it with `make api` (port 8000). All endpoints are `GET` and return JSON.

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
{"status": "ok", "run": "backtest-2026-09-14", "traded_days": 835, "grid_available": true,
 "mode": "Battery results come from a pre-computed grid at 1 MW ..."}
```

### `GET /api/runs`

```json
[{"run_id": "backtest-2026-09-14", "created_utc": "2026-09-14T18:00:00+00:00",
  "source_commit": "8689442", "grid_available": true,
  "model": "lightgbm_conformal", "baseline_model": "naive_previous_day",
  "first_day": "2024-06-01", "last_day": "2026-09-14", "holdout_start": "2026-06-01",
  "traded_days": 835,
  "reference_battery": {"power_mw": 1.0, "capacity_mwh": 2.0, "round_trip_efficiency": 0.9,
                        "degradation_eur_per_mwh": 8.0, "initial_soc_fraction": 0.5,
                        "max_cycles_per_day": 2.0},
  "grid": {"durations_h": [1, 2, 3, 4], "degradation_eur_per_mwh": [0, 1, 2, "...", 25],
           "power_mw": {"min": 0.5, "max": 5.0, "step": 0.5},
           "strategies": {"median": "median_forecast", "q25": "quantile_q25", "q10": "quantile_q10"}},
  "mode": "..."}]
```

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
