"""T1 follow-up: does telling the model why evenings spike earn more?

    uv run python -m src.health.experiments.t1_spike_features
    uv run python -m src.health.experiments.t1_spike_features --quick

The plan, fixed before any of this was written, is
``docs/plans/spike_features_plan.md``. Two arms walk forward over the saved Phase 2
comparison run's days with the production refit cadence:

* ``production``: the production model as it is. It must reproduce the saved
  comparison forecasts, which proves both arms read the same inputs.
* ``spike_drivers``: the same model with the nine opt-in ``spike_drivers``
  columns added (``src.features.build.EXTRA_FEATURE_GROUPS``): evening load and
  its ramp, afternoon radiation, evening wind, the evening residual-load estimate
  and its standing against the last week, and the week's spike record.

Both are scored on the validation days and traded with median dispatch against
perfect foresight on the days both can trade. Criterion, fixed in advance: adopt
only if the paired daily profit difference has a 95% moving-block bootstrap
interval above zero. ``--quick`` runs the first 100 days and writes nothing to
``docs/``. The hold-out is never read.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import REPO_ROOT, Settings, load_settings
from src.features.build import EVENING_HOURS, EXTRA_FEATURE_GROUPS, FEATURE_GROUPS
from src.forecasting.evaluate import segment_scores
from src.forecasting.models.common import feature_names
from src.forecasting.models.gradient_boosting import LightGBMConformalModel
from src.forecasting.run_comparison import REFIT_EVERY_DAYS
from src.forecasting.walkforward import run_walk_forward
from src.health.experiments.d1_missing_weather import (
    _trade,
    cached_forecasts,
    check_days,
    common_traded_days,
    reproduction_gap,
)
from src.health.experiments.m1_refit_cadence import (
    daily_profit,
    paired_difference,
    verdict,
)
from src.health.experiments.m1_regime_shift import daily_metrics

__all__ = [
    "CANDIDATE",
    "CONTROL",
    "GROUP",
    "evening_bias",
    "main",
    "results_markdown",
]

PRODUCTION = "lightgbm_conformal"
CANDIDATE_MODEL = "lightgbm_conformal_spike_drivers"
CONTROL = "production"
CANDIDATE = "spike_drivers"
GROUP = "spike_drivers"
REPRODUCTION_TOLERANCE = 0.01
QUICK_DAYS = 100
RESULTS_PATH = REPO_ROOT / "docs" / "results" / "t1_spike_features.md"


def evening_bias(
    forecasts: pd.DataFrame, settings: Settings, *, spike_days_only: bool
) -> float:
    """Mean of (actual minus q50) over the evening hours, positive when too low."""
    threshold = settings.evaluation.spike_threshold_eur_mwh
    hour = pd.DatetimeIndex(forecasts.index).tz_convert(settings.market.timezone).hour
    evening = (hour >= EVENING_HOURS[0]) & (hour <= EVENING_HOURS[1])
    chosen = forecasts[evening & forecasts["actual"].notna()]
    if spike_days_only:
        spiky = forecasts.groupby("target_day")["actual"].max() >= threshold
        chosen = chosen[chosen["target_day"].map(spiky).astype(bool)]
    if chosen.empty:
        return float("nan")
    return float((chosen["actual"] - chosen["q50"]).mean())


def _scalar(value: Any) -> float:
    return float(np.asarray(value, dtype="float64"))


def summarise(
    forecasts: Mapping[str, pd.DataFrame],
    daily: Mapping[str, pd.DataFrame],
    pnl: Mapping[str, pd.DataFrame],
    settings: Settings,
) -> pd.DataFrame:
    rows = []
    for arm, frame in forecasts.items():
        scores = segment_scores(frame, settings)
        overall = scores.loc["All target days"]
        threshold = settings.evaluation.spike_threshold_eur_mwh
        spike_row = f"Days with a price above €{threshold:.0f}"
        spikes = scores.loc[spike_row] if spike_row in scores.index else None
        profit = daily_profit(pnl[arm])
        earned = float(profit["pnl_eur"].sum())
        ceiling = float(profit["perfect_foresight_pnl_eur"].sum())
        rows.append(
            {
                "arm": arm,
                "mean_pinball": _scalar(overall["mean pinball"]),
                "coverage_90": _scalar(overall["coverage 90%"]),
                "mae_median": _scalar(overall["MAE of median"]),
                "spike_pinball": _scalar(spikes["mean pinball"])
                if spikes is not None
                else float("nan"),
                "evening_bias_all": evening_bias(
                    frame, settings, spike_days_only=False
                ),
                "evening_bias_spikes": evening_bias(
                    frame, settings, spike_days_only=True
                ),
                "coverage_50": float(daily[arm]["coverage_50"].mean()),
                "pnl_eur": earned,
                "perfect_foresight_pnl_eur": ceiling,
                "capture": earned / ceiling if ceiling else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("arm")


def _signed(value: float, places: int = 2) -> str:
    return f"{value:+,.{places}f}"


def _interval(item: Mapping[str, float]) -> str:
    return (
        f"{_signed(item['mean'])} ({_signed(item['low'])} to {_signed(item['high'])})"
    )


def results_markdown(
    table: pd.DataFrame,
    comparison: Mapping[str, Mapping[str, float]],
    importance: Mapping[str, float],
    *,
    days: Sequence[date],
    scored: Sequence[date],
    traded: int,
    gap: float,
    seasons: Mapping[str, Mapping[str, float]],
) -> str:
    control, candidate = table.loc[CONTROL], table.loc[CANDIDATE]
    profit = comparison["pnl_eur"]
    adopted = profit["low"] > 0
    lines = [
        "# T1 follow-up: features for why an evening spikes",
        "",
        "Generated by `python -m src.health.experiments.t1_spike_features`. The "
        "production model walked forward over the "
        f"{len(days)} days of the saved Phase 2 comparison run ({days[0]} to "
        f"{days[-1]}), refit every 28 days, once as it is and once with the nine "
        "`spike_drivers` columns added: evening load and its ramp from the "
        "afternoon, afternoon radiation and evening wind from the weather forecast, "
        "the evening residual-load estimate and its standing against the last week, "
        "and the week's spike record. Scored on "
        f"{len(scored)} validation days ({scored[0]} to {scored[-1]}), traded on the "
        f"{traded} days both arms can trade: 1 MW / 2 MWh battery, median dispatch "
        "against perfect foresight. Prices and errors in €/MWh, profit in €.",
        "",
        f"The recomputed production arm reproduces the saved comparison forecasts "
        f"within {gap:.4f} €/MWh, so both arms read the same inputs and differ only "
        "in the nine columns.",
        "",
        "Criterion, fixed before the run: adopt `spike_drivers` only if the paired "
        "daily profit difference has a 95% moving-block bootstrap interval (7-day "
        "blocks, 5,000 draws) entirely above zero.",
        "",
        "| arm | pinball | pinball, spike days | 90% coverage | MAE of median | "
        "evening bias, all days | evening bias, spike days | capture | P&L |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for arm, r in ((CONTROL, control), (CANDIDATE, candidate)):
        lines.append(
            f"| {arm} | {r['mean_pinball']:.3f} | {r['spike_pinball']:.3f} | "
            f"{r['coverage_90']:.1%} | {r['mae_median']:.2f} | "
            f"{_signed(float(r['evening_bias_all']))} | "
            f"{_signed(float(r['evening_bias_spikes']))} | "
            f"{r['capture']:.2%} | {r['pnl_eur']:,.0f} |"
        )
    lines += [
        "",
        "Evening bias is the mean of the real price minus q50 over 17:00 to 20:59: "
        "positive means the forecast ran low.",
        "",
        "## spike_drivers minus production, paired by day",
        "",
        "| measure | mean daily difference | 95% interval | days | reading |",
        "|---|---|---|---|---|",
        f"| pinball, €/MWh (lower is better) | {comparison['pinball']['mean']:+.3f} | "
        f"{comparison['pinball']['low']:+.3f} to {comparison['pinball']['high']:+.3f} "
        f"| {int(comparison['pinball']['days'])} | "
        f"{verdict(comparison['pinball'], better='lower')} |",
        f"| median P&L, € per day (higher is better) | {profit['mean']:+.2f} | "
        f"{profit['low']:+.2f} to {profit['high']:+.2f} | {int(profit['days'])} | "
        f"{verdict(profit, better='higher')} |",
        f"| median P&L, June to September | {seasons['summer']['mean']:+.2f} | "
        f"{seasons['summer']['low']:+.2f} to {seasons['summer']['high']:+.2f} | "
        f"{int(seasons['summer']['days'])} | |",
        f"| median P&L, October to May | {seasons['winter']['mean']:+.2f} | "
        f"{seasons['winter']['low']:+.2f} to {seasons['winter']['high']:+.2f} | "
        f"{int(seasons['winter']['days'])} | |",
        "",
        "## What the model leaned on",
        "",
        "Share of the candidate's last refit's total gain taken by each new column:",
        "",
        "| column | gain share |",
        "|---|---|",
    ]
    for name, share in sorted(importance.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {name} | {share:.1%} |")
    total = candidate["pnl_eur"] - control["pnl_eur"]
    state = (
        "adopted: the whole interval is above zero"
        if adopted
        else "not adopted: the interval does not lie above zero"
    )
    lines += [
        "",
        "## Verdict",
        "",
        f"`spike_drivers` is {state}. "
        f"Over the {traded} traded days it made {total:+,.0f} € against the "
        f"production model, {_interval(profit)} € a day.",
        "",
    ]
    return "\n".join(lines)


def _season_split(
    control: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, dict[str, float]]:
    left = control.set_index("target_day")["pnl_eur"]
    right = candidate.set_index("target_day")["pnl_eur"]
    both = left.index.intersection(right.index)
    months = pd.to_datetime(pd.Index(both)).month
    out = {}
    for name, mask in (
        ("summer", np.isin(months, [6, 7, 8, 9])),
        ("winter", ~np.isin(months, [6, 7, 8, 9])),
    ):
        days = both[mask]
        if len(days) == 0:
            nan = float("nan")
            out[name] = {"mean": nan, "low": nan, "high": nan, "days": 0.0}
            continue
        out[name] = paired_difference(
            control[control["target_day"].isin(days)],
            candidate[candidate["target_day"].isin(days)],
            "pnl_eur",
        )
    return out


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial")
    partial.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(partial, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Add the spike-driver columns to the production model."
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1)
    )
    parser.add_argument(
        "--quick", action="store_true", help=f"first {QUICK_DAYS} days only"
    )
    args = parser.parse_args(argv)
    settings = load_settings(args.config)
    quantiles = settings.forecasting.quantiles
    comparison_dir = settings.data.processed_path / "forecasts" / "comparison"
    saved = pd.read_parquet(comparison_dir / f"{PRODUCTION}.parquet")
    days = sorted(saved["target_day"].unique())
    if args.quick:
        days = days[:QUICK_DAYS]
        saved = saved[saved["target_day"].isin(days)]
    check_days(days, settings)
    scored = [day for day in days if day >= settings.evaluation.validation_start]
    if not scored:
        parser.error("no validation days in the selected span")

    inputs = pd.read_parquet(settings.data.inputs_path)
    out = settings.data.processed_path / "experiments"
    prefix = "t1_spike_quick" if args.quick else "t1_spike"
    refit = REFIT_EVERY_DAYS[PRODUCTION]
    models: dict[str, LightGBMConformalModel] = {}

    def builder(arm: str) -> Callable[[], pd.DataFrame]:
        def build() -> pd.DataFrame:
            if arm == CONTROL:
                model = LightGBMConformalModel(settings)
            else:
                model = LightGBMConformalModel(
                    settings,
                    name=CANDIDATE_MODEL,
                    _names=feature_names([*FEATURE_GROUPS, GROUP]),
                )
            models[arm] = model
            print(f"{arm}: walking forward over {len(days)} days", flush=True)
            started = time.perf_counter()
            frame = run_walk_forward(
                inputs, model, days, settings, refit_every_days=refit
            )
            print(
                f"{arm}: done in {(time.perf_counter() - started) / 60:.1f} min",
                flush=True,
            )
            return frame

        return build

    forecasts: dict[str, pd.DataFrame] = {}
    for arm, model_name in ((CONTROL, PRODUCTION), (CANDIDATE, CANDIDATE_MODEL)):
        frame, reused = cached_forecasts(
            out / f"{prefix}_forecasts_{arm}.parquet", builder(arm), days, model_name
        )
        if reused:
            print(f"{arm}: reused saved forecasts", flush=True)
        forecasts[arm] = frame

    gap = reproduction_gap(forecasts[CONTROL], saved, quantiles)
    if gap > REPRODUCTION_TOLERANCE:
        raise RuntimeError(
            f"the recomputed production arm differs from the saved comparison by "
            f"{gap:.4f} €/MWh; the arms would not share their inputs"
        )
    print(f"production reproduces the saved run within {gap:.4f} €/MWh", flush=True)

    scored_frames = {
        arm: frame[frame["target_day"].isin(set(scored))]
        for arm, frame in forecasts.items()
    }
    traded_days = common_traded_days(scored_frames, scored, settings)
    daily = {
        arm: daily_metrics(frame, settings) for arm, frame in scored_frames.items()
    }
    pnl = {
        arm: _trade(frame, traded_days, settings, args.workers)
        for arm, frame in scored_frames.items()
    }
    table = summarise(scored_frames, daily, pnl, settings)
    comparison = {
        "pinball": paired_difference(daily[CONTROL], daily[CANDIDATE], "pinball"),
        "coverage_90": paired_difference(
            daily[CONTROL], daily[CANDIDATE], "coverage_90"
        ),
        "pnl_eur": paired_difference(
            daily_profit(pnl[CONTROL]), daily_profit(pnl[CANDIDATE]), "pnl_eur"
        ),
    }
    seasons = _season_split(daily_profit(pnl[CONTROL]), daily_profit(pnl[CANDIDATE]))
    # Importance is read off the last refit; a reused walk-forward has no model in
    # memory, so that refit is fit again for the reading.
    if CANDIDATE not in models:
        from src.forecasting.information import build_information_set
        from src.health.experiments.m1_refit_cadence import refit_days

        last_fit = refit_days(days, refit)[-1]
        model = LightGBMConformalModel(
            settings,
            name=CANDIDATE_MODEL,
            _names=feature_names([*FEATURE_GROUPS, GROUP]),
        )
        model.fit(
            build_information_set(inputs, last_fit, settings, model.fit_lookback_days)
        )
        models[CANDIDATE] = model
    gains = models[CANDIDATE].feature_importance()
    share = gains / gains.sum()
    importance = {
        name: float(share.get(name, 0.0)) for name in EXTRA_FEATURE_GROUPS[GROUP]
    }
    summary = {
        "table": table.reset_index().to_dict(orient="records"),
        "comparison": comparison,
        "seasons": seasons,
        "importance": importance,
        "reproduction_gap": gap,
        "traded_days": len(traded_days),
        "adopted": bool(comparison["pnl_eur"]["low"] > 0),
    }
    _write_json(summary, out / f"{prefix}_summary.json")
    for arm in pnl:
        pnl[arm].to_parquet(out / f"{prefix}_pnl_{arm}.parquet")
    page = results_markdown(
        table,
        comparison,
        importance,
        days=days,
        scored=scored,
        traded=len(traded_days),
        gap=gap,
        seasons=seasons,
    )
    print(page)
    if not args.quick:
        RESULTS_PATH.write_text(page, encoding="utf-8")
        print(f"wrote {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
