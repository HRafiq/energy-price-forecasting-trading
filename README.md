# live-state

The daily desk's memory. Not code: this branch holds what the live pipeline
produced and cannot reproduce, so a scheduled run on a fresh machine can pick up
where the last one left off.

It has no history in common with the code branches, and nothing here is edited by
hand. The daily workflow checks it out, restores it, runs, and commits back
whatever changed.

```
data/processed/pipeline/runs          one record per delivery day: when the bid
                                      went out, against the gate, on which rung
data/processed/pipeline/plans         the schedule committed for each day
data/processed/pipeline/settlements   what each schedule earned once prices published
data/processed/forecasts/production    the forecasts those runs saved
data/processed/health                 the incident log and the drift summary
data/processed/experiments/m2_drift.json  the drift thresholds, fixed on validation
mlflow.db, mlruns/                    the model registry, so the version that has
                                      been serving keeps serving
```

Two things are deliberately absent. The market data is not here: it is rebuilt
from SMARD, Open-Meteo and the fuel feeds on every run, so keeping it would be
carrying something reproducible. And the fuel price file is not here at all,
because its source does not licence redistribution.

The registry records absolute paths, so it is stored with `/__repo_root__` where
the repository root belongs and the real root is written back on restore. That is
what lets a store written on one machine be read on another, and it keeps the
home directory of whoever logged the model off a public branch. A save reads the
whole store for machine paths and refuses rather than publishing one.

See `src/pipeline/state.py` and `src/pipeline/registry_paths.py` on the default
branch.
