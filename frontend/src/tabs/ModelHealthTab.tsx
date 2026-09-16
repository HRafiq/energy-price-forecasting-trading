import { useState } from "react";
import type { DriftWindow, OpsResponse, Period, RegimePeriod, RegimeResponse } from "../api";
import { api } from "../api";
import { DriftMonitorChart } from "../components/charts/DriftMonitorChart";
import { REGIME_ARMS, RegimeCoverageChart } from "../components/charts/RegimeCoverageChart";
import { DeskBriefing } from "../components/DeskBriefing";
import { IncidentLog } from "../components/IncidentLog";
import { Kpi } from "../components/Kpi";
import { Loadable } from "../components/Loadable";
import { Panel } from "../components/Panel";
import { dayMonthYear, longDay, monthFullYear, num, pct, utcDay } from "../format";
import { type AsyncState, dataOf, isReloading, useApi } from "../hooks";
import { BUSY_OPACITY, C } from "../theme";

const buttonStyle = (active: boolean) =>
  active
    ? { background: C.panel, color: C.text, border: `1px solid ${C.panelEdge}` }
    : { background: C.inset, color: C.muted, border: `1px solid ${C.panelEdge}` };

const FAILURE_LABELS: Record<string, string> = {
  weather_late: "weather late",
  model_fails: "model fails",
  prices_late: "prices late",
};

/** "1 Jun 2024 to 31 May 2026": the same day format on both ends. */
function span(period: Period): string {
  if (!period.first_day || !period.last_day) return "period not recorded";
  return `${dayMonthYear(period.first_day)} to ${dayMonthYear(period.last_day)}`;
}

/** "as of 15 Sep 2026", from the export index. */
function asOf(generatedUtc: string | null | undefined): string {
  return generatedUtc ? `as of ${utcDay(generatedUtc)}` : "export date not recorded";
}

function rates(values: Record<string, number> | null | undefined): string | null {
  if (!values) return null;
  const parts = Object.entries(values).map(([key, share]) => `${FAILURE_LABELS[key] ?? key.replace(/_/g, " ")} ${pct(share)}`);
  return parts.length > 0 ? parts.join(", ") : null;
}

function Warning({ children }: { children: string }) {
  return (
    <p className="text-xs mt-2 leading-relaxed" role="status" style={{ color: C.warn }}>
      ⚠ {children}
    </p>
  );
}

function driftMismatch(d: { matches_run: boolean; last_target_day: string | null; run_last_day: string | null }): string | null {
  if (d.matches_run) return null;
  const ends = d.last_target_day ? dayMonthYear(d.last_target_day) : "an unknown day";
  const run = d.run_last_day ? dayMonthYear(d.run_last_day) : "an unknown day";
  return `The drift monitor ends on ${ends} but this run ends on ${run}: it was computed for another run. Rerun M2, then the health export.`;
}

// ------------------------------------------------------------------------------
// Ops KPI row
// ------------------------------------------------------------------------------

function OpsRow({ state }: { state: AsyncState<OpsResponse> }) {
  const ops = dataOf(state);
  const failed = state.status === "error";
  const pending = failed ? "unavailable" : "…";
  const deadline = ops?.deadline;
  const fallbacks = ops?.fallbacks;
  const drift = ops?.drift;

  let deadlineTile;
  if (deadline?.available) {
    const gate = deadline.gate_local ? `the ${deadline.gate_local} gate` : "the gate";
    const nominal = rates(deadline.failure_rates);
    const realised = rates(deadline.realised_failure_rates);
    const assumptions = [
      nominal ? `nominal failure rates ${nominal}` : null,
      realised ? `realised ${realised}` : null,
      deadline.seed !== null && deadline.seed !== undefined ? `seed ${deadline.seed}` : null,
    ].filter((part): part is string => part !== null);
    deadlineTile = (
      <Kpi
        label={`Forecast ready before ${gate}, with the chain`}
        value={pct(deadline.on_time_share_with_chain)}
        note={`without the fallback chain ${pct(deadline.on_time_share_without_chain)}`}
        detail={
          `D5 simulation with injected failures · ${deadline.days} validation days, ${span(deadline.period)}` +
          (assumptions.length > 0 ? ` · ${assumptions.join(" · ")}` : "") +
          ` · ${asOf(deadline.generated_utc)}`
        }
      />
    );
  } else {
    deadlineTile = (
      <Kpi
        label="Forecast ready before the gate, with the chain"
        value={deadline ? "unavailable" : pending}
        note={deadline && !deadline.available ? deadline.detail : "D5 simulation with injected failures"}
        detail="Source: D5 deadline experiment"
        tone={C.muted}
      />
    );
  }

  let fallbackTile;
  if (fallbacks) {
    const observed = fallbacks.observed;
    const simulated = fallbacks.available
      ? `${fallbacks.simulated_days} simulated in D5, ${span(fallbacks.simulated_period)}, shown apart and not added`
      : `D5 simulation: ${fallbacks.detail}`;
    fallbackTile = (
      <Kpi
        label="Fallback activations, observed"
        value={observed === null ? "unavailable" : String(observed)}
        note={observed === null ? "incident log not exported" : `observed ${span(fallbacks.observed_period)}`}
        detail={`${simulated} · observed count ${asOf(fallbacks.observed_generated_utc)}`}
        tone={observed === null ? C.muted : undefined}
      />
    );
  } else {
    fallbackTile = (
      <Kpi
        label="Fallback activations, observed"
        value={pending}
        note="observed fallback incidents"
        detail="Sources: the incident log; D5 simulated days shown apart"
        tone={C.muted}
      />
    );
  }

  const mismatch = drift?.available ? driftMismatch(drift) : null;
  const driftDetail = drift?.available
    ? `M2 drift monitor · ${drift.window_days} traded days to ${longDay(drift.as_of)}, end of the ${drift.window === "holdout" ? "hold-out" : "validation window"} · ${asOf(drift.generated_utc)}`
    : "Source: M2 drift monitor";
  const alertNote = (alert: boolean) => (alert ? "in alert" : "no alert");
  const coverageLabel = drift?.available ? `Rolling ${drift.window_days}-day 90% coverage` : "Rolling 90% coverage";

  return (
    <div>
      <div
        className="grid grid-cols-1 sm:grid-cols-2 gap-3"
        style={{ opacity: state.status === "loading" && ops ? BUSY_OPACITY : 1 }}
      >
        {deadlineTile}
        {fallbackTile}
        {drift?.available ? (
          <Kpi
            label={coverageLabel}
            value={drift.coverage.value === null ? "n/a" : pct(drift.coverage.value)}
            note={`alert below ${pct(drift.coverage.threshold)} · ${alertNote(drift.coverage.alert)}`}
            detail={driftDetail}
            tone={drift.coverage.alert ? C.neg : undefined}
          />
        ) : (
          <Kpi
            label={coverageLabel}
            value={drift ? "unavailable" : pending}
            note={drift && !drift.available ? drift.detail : "alert below the M2 threshold"}
            detail={driftDetail}
            tone={C.muted}
          />
        )}
        {drift?.available ? (
          <Kpi
            label="Rolling pinball ratio"
            value={drift.pinball_ratio.value === null ? "n/a" : num(drift.pinball_ratio.value, 2)}
            note={`alert above ${num(drift.pinball_ratio.threshold, 2)} · ${alertNote(drift.pinball_ratio.alert)}`}
            detail={driftDetail}
            tone={drift.pinball_ratio.alert ? C.neg : undefined}
          />
        ) : (
          <Kpi
            label="Rolling pinball ratio"
            value={drift ? "unavailable" : pending}
            note={drift && !drift.available ? drift.detail : "alert above the M2 threshold"}
            detail={driftDetail}
            tone={C.muted}
          />
        )}
      </div>
      {mismatch ? <Warning>{mismatch}</Warning> : null}
      {failed ? (
        <p className="text-xs mt-2" role="status" style={{ color: C.muted }}>
          Could not load the operations tiles: {state.error}
        </p>
      ) : null}
    </div>
  );
}

// ------------------------------------------------------------------------------
// Regime shift
// ------------------------------------------------------------------------------

function refitPhrase(arms: RegimeResponse["setup"]["arms"]): string | null {
  const refit = (arms ?? []).filter((a) => a.name !== "naive_previous_day" && a.refit_every_days !== null);
  if (refit.length === 0) return null;
  const days = refit.map((a) => a.refit_every_days);
  const list = days.length === 1 ? `${days[0]}` : `${days.slice(0, -1).join(", ")} and ${days[days.length - 1]}`;
  return `the refit arms retrain every ${list} days`;
}

function regimeSub(r: RegimeResponse | null): string {
  if (!r) return "Historical backtest through the 2021 to 2023 price shock";
  const s = r.setup;
  const weather = s.weather_features ? "with weather features" : "without weather features";
  const first = s.first_target_day ? monthFullYear(s.first_target_day) : "?";
  const last = s.last_target_day ? monthFullYear(s.last_target_day) : "?";
  const refit = refitPhrase(s.arms);
  const target = Math.round(r.coverage_target * 100);
  return (
    `Historical backtest, ${first} to ${last} · same model family as production (LightGBM with conformal ranges), ${weather} · ` +
    `${s.training_days ?? "?"} training days and ${s.calibration_days ?? "?"} calibration days per fit · ` +
    `frozen is fitted once before the first day${refit ? `; ${refit}` : ""} · ` +
    `rolling ${r.rolling_coverage_90.window_days}-day share of prices inside the 90% interval; the solid light line is the ${target}% target · ` +
    asOf(r.generated_utc)
  );
}

function RegimeTable({ periods }: { periods: RegimePeriod[] }) {
  const columns = ["2021", "2022", "2023", "total"];
  const cell = (arm: string, period: string) => periods.find((p) => p.arm === arm && p.period === period);
  const arms = REGIME_ARMS.filter((a) => periods.some((p) => p.arm === a.key));
  const th = "px-2 py-1 text-right font-normal whitespace-nowrap";
  const value = (v: number | null | undefined) => (v === null || v === undefined ? "n/a" : pct(v));
  return (
    <div className="overflow-x-auto mt-3">
      <table className="w-full text-xs tabular-nums" style={{ color: C.muted, minWidth: 560 }}>
        <thead>
          <tr>
            <th className="px-2 py-1 text-left font-normal" rowSpan={2}>
              Arm
            </th>
            <th className="px-2 py-1 text-center font-normal" colSpan={4} style={{ borderBottom: `1px solid ${C.panelEdge}` }}>
              90% interval coverage
            </th>
            <th className="px-2 py-1 text-center font-normal" colSpan={4} style={{ borderBottom: `1px solid ${C.panelEdge}` }}>
              Capture vs perfect foresight
            </th>
          </tr>
          <tr>
            {[...columns, ...columns].map((c, i) => (
              <th key={`${c}-${i}`} className={th}>
                {c === "total" ? "Total" : c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {arms.map((arm) => (
            <tr key={arm.key} style={{ borderTop: `1px solid ${C.panelEdge}` }}>
              <td className="px-2 py-1.5 text-left whitespace-nowrap" style={{ color: C.text }}>
                <span style={{ color: arm.color }}>● </span>
                {arm.label}
              </td>
              {columns.map((c) => (
                <td key={`cov-${c}`} className="px-2 py-1.5 text-right" style={{ color: c === "total" ? C.text : C.muted }}>
                  {value(cell(arm.key, c)?.coverage_90)}
                </td>
              ))}
              {columns.map((c) => (
                <td key={`cap-${c}`} className="px-2 py-1.5 text-right" style={{ color: c === "total" ? C.text : C.muted }}>
                  {value(cell(arm.key, c)?.capture_ratio)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function RegimePanel({ run }: { run: string }) {
  const [showNaive, setShowNaive] = useState(false);
  const state = useApi(`regime|${run}`, (signal) => api.regime(run, signal));
  const r = dataOf(state);
  return (
    <Panel title="Regime shift: does refitting keep the ranges honest?" sub={regimeSub(r)} busy={isReloading(state)}>
      <Loadable state={state} height={280}>
        {(data) => (
          <>
            <div className="flex flex-wrap items-center gap-x-4 gap-y-2 mb-2 text-xs" style={{ color: C.muted }}>
              {REGIME_ARMS.filter((a) => a.key !== "naive_previous_day").map((a) => (
                <span key={a.key} className="whitespace-nowrap">
                  <span style={{ color: a.color }}>━</span> {a.label}
                </span>
              ))}
              <span className="whitespace-nowrap">
                <span style={{ color: C.target }}>━</span> {Math.round(data.coverage_target * 100)}% target
              </span>
              <button
                type="button"
                aria-pressed={showNaive}
                onClick={() => setShowNaive((v) => !v)}
                className="rounded-md px-2 py-1 whitespace-nowrap"
                style={buttonStyle(showNaive)}
              >
                <span style={{ color: C.armNaive }}>┈</span> Naive baseline, for context: {showNaive ? "shown" : "hidden"}
              </button>
            </div>
            <RegimeCoverageChart data={data} showNaive={showNaive} />
            <LowestCoverage data={data} />
            <RegimeTable periods={data.periods} />
          </>
        )}
      </Loadable>
    </Panel>
  );
}

function LowestCoverage({ data }: { data: RegimeResponse }) {
  const days = data.rolling_coverage_90.window_days;
  const parts = REGIME_ARMS.filter((a) => a.key !== "naive_previous_day").flatMap((a) => {
    const low = data.min_rolling_coverage_90[a.key];
    return low ? [`${a.label.toLowerCase()} ${pct(low.value)} (${days} days to ${longDay(low.window_end)})`] : [];
  });
  if (parts.length === 0) return null;
  return (
    <p className="text-xs mt-2 leading-relaxed" style={{ color: C.muted }}>
      Lowest rolling coverage: {parts.join(", ")}.
    </p>
  );
}

// ------------------------------------------------------------------------------
// Drift monitor
// ------------------------------------------------------------------------------

const DRIFT_WINDOWS: Array<{ key: DriftWindow; label: string }> = [
  { key: "all", label: "Validation and hold-out" },
  { key: "validation", label: "Validation" },
  { key: "holdout", label: "Hold-out" },
];

function DriftPanel({ run }: { run: string }) {
  const [windowKey, setWindowKey] = useState<DriftWindow>("all");
  const state = useApi(`drift|${run}|${windowKey}`, (signal) => api.drift(run, windowKey, signal));
  const d = dataOf(state);
  const sub = d
    ? `Production model's saved forecasts, ${d.series.length} traded days · thresholds fixed on the validation window only (${pct(d.thresholds.coverage)} coverage, pinball ratio ${num(d.thresholds.pinball_ratio, 2)}) and applied unchanged to the hold-out from ${longDay(d.holdout_start)} · red marks alert days · ${asOf(d.generated_utc)}`
    : "Rolling signals with thresholds fixed on the validation window only";
  const mismatch = d ? driftMismatch(d) : null;
  return (
    <Panel title="Drift monitor" sub={sub} busy={isReloading(state)}>
      {mismatch ? <Warning>{mismatch}</Warning> : null}
      <div role="group" aria-label="Drift window" className="flex flex-wrap gap-1 mb-3">
        {DRIFT_WINDOWS.map((w) => (
          <button
            key={w.key}
            type="button"
            aria-pressed={windowKey === w.key}
            onClick={() => setWindowKey(w.key)}
            className="text-xs rounded-md px-2.5 py-1"
            style={buttonStyle(windowKey === w.key)}
          >
            {w.label}
          </button>
        ))}
      </div>
      <Loadable state={state} height={380}>
        {(data) => <DriftMonitorChart data={data} />}
      </Loadable>
    </Panel>
  );
}

// ------------------------------------------------------------------------------
// Tab
// ------------------------------------------------------------------------------

export function ModelHealthTab({ run }: { run: string }) {
  const ops = useApi(`ops|${run}`, (signal) => api.ops(run, signal));
  return (
    <div className="space-y-4">
      <DeskBriefing tab="model_health" run={run} />
      <OpsRow state={ops} />
      <RegimePanel run={run} />
      <DriftPanel run={run} />
      <IncidentLog run={run} />
    </div>
  );
}
