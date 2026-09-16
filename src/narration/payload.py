"""What the desk briefing is allowed to talk about.

The briefing narrates one tab of the dashboard for one day and window. It may use
nothing but the numbers the deterministic core already produced: this module
gathers them through the same functions the API serves, so a briefing can only
mention figures the deterministic core produced.

Every payload is a plain dictionary of scalars and short lists, small enough to
send whole to a model and to check a sentence against afterwards
(:mod:`src.narration.grounding`). What little arithmetic happens here is done on
the API's own numbers and stored, never left to the briefing: a shortfall, the
widest band, the change since the last traded day. A difference a sentence works
out for itself is a figure the payload does not contain, and the grounding check
would rightly throw it away.

    from src.narration.payload import build_payload
    payload = build_payload(run, health, tab="overview", day=day, window="last30")
"""

from __future__ import annotations

import math
from datetime import date, timedelta
from typing import Any

from api import health as health_api
from api import service

__all__ = [
    "TABS",
    "PayloadError",
    "build_payload",
    "payload_labels",
    "payload_numbers",
]

#: The tabs a briefing can be written for, in the order the dashboard shows them.
TABS = ("overview", "forecast", "trading", "model_health")
#: Periods of the day the briefing may point at, in local clock time.
BLOCKS = (
    (0, 6, "night"),
    (6, 12, "morning"),
    (12, 17, "afternoon"),
    (17, 24, "evening"),
)


class PayloadError(ValueError):
    """The briefing was asked for a tab or a day the run cannot answer."""


#: How far back to look for a day to compare with. A run skips days whose prices or
#: forecast are incomplete, so the day before is not always there to compare against.
LOOKBACK_DAYS = 7
#: Money is written to the cent; a share needs more room, or a capture ratio of
#: 0.8957 becomes 0.9 and the briefing says 90% where the page says 89.6%.
RATIO_PLACES = 4


def _round(value: Any, places: int = 2) -> Any:
    """Round a number for the payload, leaving anything else alone."""
    return round(value, places) if isinstance(value, int | float) else value


def _round_kpi(value: Any) -> Any:
    """Keep a share precise enough to survive being written as a percentage.

    Judged by the value rather than the name of the key: anything inside [-1, 1]
    keeps four places, so a capture ratio of 0.8957 does not reach the briefing as
    0.9 and get written as 90% under a page showing 89.6%. A share added later
    inherits this without having to be named here.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return value
    return _round(value, RATIO_PLACES if -1.0 <= float(value) <= 1.0 else 2)


def _widest_band(periods: list[dict[str, Any]]) -> float | None:
    """The widest q05 to q95 spread, or None when a quantile is missing."""
    spreads = [
        float(p["q95"]) - float(p["q05"])
        for p in periods
        if p.get("q95") is not None and p.get("q05") is not None
    ]
    return max(spreads) if spreads else None


def _day_shape(periods: list[dict[str, Any]]) -> dict[str, Any]:
    """The few facts about a day's fan a sentence can lean on."""
    priced = [p for p in periods if p.get("actual") is not None]
    if not priced:
        return {"periods": len(periods), "prices_published": False}
    peak = max(priced, key=lambda p: float(p["actual"]))
    trough = min(priced, key=lambda p: float(p["actual"]))
    return {
        "periods": len(periods),
        "prices_published": True,
        "highest_price_eur_mwh": _round(float(peak["actual"])),
        "highest_price_at": peak["time"],
        "lowest_price_eur_mwh": _round(float(trough["actual"])),
        "lowest_price_at": trough["time"],
        "widest_band_eur_mwh": _round(_widest_band(periods)),
    }


def _schedule_shape(periods: list[dict[str, Any]]) -> dict[str, Any]:
    """When the battery bought and sold, without repeating all 96 periods."""
    charging = [p for p in periods if float(p["charge_mw"]) > 1e-9]
    discharging = [p for p in periods if float(p["discharge_mw"]) > 1e-9]
    return {
        "charging_periods": len(charging),
        "discharging_periods": len(discharging),
        "first_charge_at": charging[0]["time"] if charging else None,
        "last_discharge_at": discharging[-1]["time"] if discharging else None,
        "cheapest_charge_price_eur_mwh": _round(
            min((float(p["price"]) for p in charging), default=None)
            if charging
            else None
        ),
        "dearest_discharge_price_eur_mwh": _round(
            max((float(p["price"]) for p in discharging), default=None)
            if discharging
            else None
        ),
    }


def _previous_day(
    run: service.Run, day: date, battery: dict[str, Any], earned: float
) -> dict[str, Any] | None:
    """The last traded day before this one, and the change in earnings since.

    A day the run did not trade is stepped over rather than reported as a gap: the
    comparison for 2026-09-14 is 2026-09-12, because 2026-09-13 has incomplete
    prices. The payload names the day it compared with, so the briefing says which
    day it means rather than calling it yesterday.

    The change is computed here rather than left to the briefing: a difference a
    sentence works out for itself is a number the payload does not contain, and the
    grounding check would rightly throw it away. The size is carried beside the
    signed change because a sentence says "42 EUR less", not "-42 EUR less".
    """
    for back in range(1, LOOKBACK_DAYS + 1):
        before = day - timedelta(days=back)
        try:
            schedule = service.dispatch(
                run,
                before,
                int(battery["duration_h"]),
                int(battery["degradation_eur_per_mwh"]),
                str(battery["strategy"]),
                float(battery["power_mw"]),
            )
        except (service.DayNotFoundError, service.RequestError):
            continue
        except service.SolverError:
            # One day the solver cannot handle is not a reason to drop the
            # comparison; the day before it will do.
            continue
        except service.ArtifactsMissingError:
            return None
        break
    else:
        return None
    change = earned - float(schedule["pnl_eur"])
    return {
        "day": str(before),
        "pnl_eur": _round(schedule["pnl_eur"]),
        "perfect_foresight_pnl_eur": _round(schedule["perfect_foresight_pnl_eur"]),
        "pnl_change_eur": _round(change),
        "pnl_change_size_eur": _round(abs(change)),
    }


def _overview(run: service.Run, day: date, battery: dict[str, Any]) -> dict[str, Any]:
    fan = service.forecast_day(run, day)
    schedule = service.dispatch(
        run,
        day,
        int(battery["duration_h"]),
        int(battery["degradation_eur_per_mwh"]),
        str(battery["strategy"]),
        float(battery["power_mw"]),
    )
    return {
        "day": fan["date"],
        "window_of_day": fan["window"],
        "issued_local": fan["issued_local"],
        "gate_local": fan["gate_local"],
        "forecast": _day_shape(fan["periods"]),
        "battery": schedule["battery"],
        "dispatch": {
            "pnl_eur": _round(schedule["pnl_eur"]),
            "perfect_foresight_pnl_eur": _round(schedule["perfect_foresight_pnl_eur"]),
            "shortfall_eur": _round(
                schedule["perfect_foresight_pnl_eur"] - schedule["pnl_eur"]
            ),
            **_schedule_shape(schedule["periods"]),
        },
        "previous_day": _previous_day(run, day, battery, float(schedule["pnl_eur"])),
    }


def _forecast(run: service.Run, window: str) -> dict[str, Any]:
    calibration = service.calibration(run, window)
    errors = service.error_analysis(run, window)
    by_hour = errors["by_hour"]
    scored = [
        hour
        for hour in by_hour
        if isinstance(hour.get("mae_eur_mwh"), int | float)
        and math.isfinite(float(hour["mae_eur_mwh"]))
    ]
    worst = max(scored, key=lambda h: float(h["mae_eur_mwh"])) if scored else None
    return {
        "window": calibration["window"],
        "intervals": calibration["intervals"],
        "worst_hour": worst,
        "asymmetry": errors["asymmetry"],
    }


def _trading(run: service.Run, window: str, battery: dict[str, Any]) -> dict[str, Any]:
    kpis = service.summary(
        run,
        window,
        int(battery["duration_h"]),
        int(battery["degradation_eur_per_mwh"]),
        str(battery["strategy"]),
        float(battery["power_mw"]),
    )
    series = service.pnl_series(
        run,
        window,
        int(battery["duration_h"]),
        int(battery["degradation_eur_per_mwh"]),
        str(battery["strategy"]),
        float(battery["power_mw"]),
    )
    return {
        "window": kpis["window"],
        "battery": kpis["battery"],
        "kpis": {key: _round_kpi(value) for key, value in kpis["kpis"].items()},
        "max_drawdown_eur": {
            key: _round(value) for key, value in series["max_drawdown_eur"].items()
        },
        "selected_strategy": series["selected_strategy"],
    }


def _model_health(health: health_api.Health) -> dict[str, Any]:
    ops = health_api.ops(health)
    recent = health_api.incidents(health, limit=5, offset=0)
    return {
        "ops": ops,
        "recent_incidents": [
            {
                key: record[key]
                for key in ("delivery_day", "type", "severity", "status", "provenance")
            }
            for record in recent["incidents"]
        ],
        "incident_counts": recent["counts"],
    }


def build_payload(
    run: service.Run,
    health: health_api.Health | None,
    *,
    tab: str,
    day: date,
    window: str,
    battery: dict[str, Any],
) -> dict[str, Any]:
    """Everything the briefing for one tab may mention, and nothing else."""
    if tab not in TABS:
        raise PayloadError(f"unknown tab {tab!r}; choose one of {', '.join(TABS)}")
    common = {
        "tab": tab,
        "run_id": run.run_id,
        "run_kind": run.manifest.get("run_kind", "backtest"),
        "model": run.manifest["model"],
    }
    if tab == "overview":
        return common | _overview(run, day, battery)
    if tab == "forecast":
        return common | _forecast(run, window)
    if tab == "trading":
        return common | _trading(run, window, battery)
    if health is None:
        raise PayloadError(
            "the model health tab needs a health export; run "
            "python -m src.export.artifacts --steps health"
        )
    return common | _model_health(health)


def payload_numbers(payload: Any) -> set[float]:
    """Every number anywhere in a payload, for checking prose against it.

    Numbers only. The digits inside a label are not numbers the briefing may
    quote: mining them would let "2026-09-14" license a claim of -14 EUR, and
    "20:30" license a count of 20. Labels are matched as whole strings instead,
    by :func:`payload_labels`.
    """
    found: set[float] = set()
    if isinstance(payload, bool):
        return found
    if isinstance(payload, int | float):
        found.add(float(payload))
    elif isinstance(payload, dict):
        for value in payload.values():
            found |= payload_numbers(value)
    elif isinstance(payload, list | tuple):
        for value in payload:
            found |= payload_numbers(value)
    return found


def payload_labels(payload: Any) -> set[str]:
    """The payload's own strings that carry digits, such as "19:45" or a date.

    A briefing may quote these as written; the grounding check blanks them out of
    the prose before reading it for numbers, so a label can never stand in for a
    figure.
    """
    found: set[str] = set()
    if isinstance(payload, str):
        if any(character.isdigit() for character in payload):
            found.add(payload)
    elif isinstance(payload, dict):
        for value in payload.values():
            found |= payload_labels(value)
    elif isinstance(payload, list | tuple):
        for value in payload:
            found |= payload_labels(value)
    return found
