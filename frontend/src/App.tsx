import { type ReactNode, useEffect, useMemo, useState } from "react";
import type { BatteryQuery, RunInfo, RunKind, SummaryResponse } from "./api";
import { api } from "./api";
import { type BatteryGrid, batteryGrid, defaultBattery } from "./battery";
import { type ControlValues, Controls } from "./components/Controls";
import { Kpi } from "./components/Kpi";
import { StatusMessage } from "./components/Loadable";
import { Pill } from "./components/Pill";
import { clock, eur, localClock, longDay, num, pct, shortDay, windowPhrase } from "./format";
import { type AsyncState, dataOf, useApi, useDebounced } from "./hooks";
import { ForecastTab } from "./tabs/ForecastTab";
import { LiveTab } from "./tabs/LiveTab";
import { ModelHealthTab } from "./tabs/ModelHealthTab";
import { OverviewTab } from "./tabs/OverviewTab";
import { TradingTab } from "./tabs/TradingTab";
import { BUSY_OPACITY, C, FONT_STACK } from "./theme";

const TABS = ["Overview", "Forecast", "Trading", "Model health", "Live"] as const;
type Tab = (typeof TABS)[number];

/** URL hash for a tab, so a tab can be linked to directly: "Model health" is #model-health. */
function tabSlug(tab: Tab): string {
  return tab.toLowerCase().replace(/ /g, "-");
}

function tabFromHash(): Tab {
  const slug = window.location.hash.replace(/^#/, "");
  return TABS.find((t) => tabSlug(t) === slug) ?? "Overview";
}

const SLIDER_DEBOUNCE_MS = 200;
const PANEL_ID = "tab-panel";

function tabId(tab: Tab): string {
  return `tab-${tab.toLowerCase().replace(/\s+/g, "-")}`;
}

function latestRun(runs: RunInfo[]): RunInfo | null {
  return [...runs].sort((a, b) => b.created_utc.localeCompare(a.created_utc))[0] ?? null;
}

function errorOf<T>(state: AsyncState<T>): string | null {
  return state.status === "error" ? state.error : null;
}

function KpiStrip({
  state,
  gridAvailable,
  runKind,
}: {
  state: AsyncState<SummaryResponse>;
  gridAvailable: boolean | undefined;
  runKind: RunKind | undefined;
}) {
  const s = dataOf(state);
  const k = s?.kpis;
  const failed = state.status === "error";
  const missing = failed ? "unavailable" : "…";
  return (
    <div className="mb-4">
      <div
        className="grid grid-cols-2 lg:grid-cols-4 gap-3"
        style={{ opacity: state.status === "loading" && s ? BUSY_OPACITY : 1 }}
      >
        <Kpi
          label={s ? `P&L, ${windowPhrase(s.window.key)}` : "P&L"}
          value={k ? eur(k.pnl_eur) : missing}
          note={s ? `net of degradation · ${s.window.traded_days} traded days` : "net of degradation"}
          tone={k ? (k.pnl_eur >= 0 ? C.good : C.neg) : C.muted}
        />
        <Kpi
          label="Capture ratio"
          value={k ? pct(k.capture_ratio) : missing}
          note={k ? `vs perfect foresight ${eur(k.perfect_foresight_pnl_eur)}` : "vs perfect foresight"}
          tone={k ? undefined : C.muted}
        />
        <Kpi
          label="Pinball loss"
          value={k ? num(k.pinball_eur_mwh, 2) : missing}
          note={k ? `€/MWh · baseline ${num(k.baseline_pinball_eur_mwh, 2)}` : "€/MWh"}
          tone={k ? undefined : C.muted}
        />
        <Kpi
          label="Cycles per day"
          value={k ? num(k.cycles_per_day, 2) : missing}
          note="avg per traded day"
          tone={k ? undefined : C.muted}
        />
      </div>
      {state.status === "error" ? (
        <p className="text-xs mt-2" role="status" style={{ color: C.muted }}>
          {gridAvailable === false
            ? runKind === "live"
              ? "A live run carries today's forecast and schedule, not the battery P&L grid. The KPIs are on the backtest run."
              : "P&L grid still exporting. The KPIs appear after the export finishes and the page is reloaded."
            : `Could not load the KPIs: ${state.error}`}
        </p>
      ) : null}
    </div>
  );
}

interface ShellProps {
  context: string;
  /** A muted aside beside the context line, such as prices not published yet. */
  note: string | null;
  runKind: RunKind;
  holdout: boolean;
  tab: Tab;
  onTab: (tab: Tab) => void;
  runId: string | null;
  children: ReactNode;
}

function Shell({ context, note, runKind, holdout, tab, onTab, runId, children }: ShellProps) {
  return (
    <div className="min-h-screen w-full" style={{ background: C.bg, color: C.text, fontFamily: FONT_STACK }}>
      <div className="max-w-6xl mx-auto px-3 sm:px-4 py-5">
        <header className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mb-4">
          <h1 className="text-lg font-semibold tracking-tight">DE-LU day-ahead battery trading</h1>
          <span className="text-xs tabular-nums" style={{ color: C.muted }}>
            {context}
          </span>
          {note ? (
            <span className="text-xs" style={{ color: C.muted }}>
              {note}
            </span>
          ) : null}
          <span className="flex gap-2 sm:ml-auto">
            <Pill
              label={runKind === "live" ? "live · today's run" : "backtest · historical data"}
              color={C.good}
            />
            {holdout ? <Pill label="hold-out" color={C.warn} /> : null}
          </span>
        </header>

        <div
          role="tablist"
          aria-label="Dashboard views"
          className="flex gap-1 mb-4 rounded-lg p-1 overflow-x-auto"
          style={{ background: C.inset, border: `1px solid ${C.panelEdge}` }}
        >
          {TABS.map((t, i) => (
            <button
              key={t}
              id={tabId(t)}
              type="button"
              role="tab"
              aria-selected={tab === t}
              aria-controls={PANEL_ID}
              tabIndex={tab === t ? 0 : -1}
              onClick={() => onTab(t)}
              onKeyDown={(e) => {
                if (e.key !== "ArrowRight" && e.key !== "ArrowLeft") return;
                e.preventDefault();
                const target = TABS[(i + (e.key === "ArrowRight" ? 1 : TABS.length - 1)) % TABS.length];
                if (target) {
                  onTab(target);
                  document.getElementById(tabId(target))?.focus();
                }
              }}
              className="flex-1 text-sm rounded-md px-3 py-1.5 whitespace-nowrap"
              style={
                tab === t
                  ? { background: C.panel, color: C.text, border: `1px solid ${C.panelEdge}` }
                  : { color: C.muted, border: "1px solid transparent" }
              }
            >
              {t}
            </button>
          ))}
        </div>

        {children}

        <footer className="mt-4 text-xs leading-relaxed" style={{ color: C.muted }}>
          Data: SMARD (Bundesnetzagentur) prices, load and generation; Open-Meteo weather forecasts → LightGBM with
          conformal ranges → MILP dispatch (PuLP/CBC) → walk-forward backtest
          {runId ? ` · run ${runId}` : ""}
        </footer>
      </div>
    </div>
  );
}

function Dashboard({ run, grid, tab, onTab }: { run: RunInfo; grid: BatteryGrid; tab: Tab; onTab: (tab: Tab) => void }) {
  const runId = run.run_id;
  const [controls, setControls] = useState<ControlValues>(() => ({
    ...defaultBattery(run, grid),
    strategy: "median",
    windowKey: "last30",
    date: null,
  }));
  const update = (patch: Partial<ControlValues>) => setControls((c) => ({ ...c, ...patch }));

  const days = useApi(`days|${runId}`, (signal) => api.days(runId, signal));
  const dayList = dataOf(days)?.days ?? null;
  const daysFailed = days.status === "error";

  // Open on the last traded day; if the calendar fails or is empty, use the run's last day.
  useEffect(() => {
    if (controls.date !== null || (!dayList && !daysFailed)) return;
    const traded = (dayList ?? []).filter((d) => d.traded).map((d) => d.date).sort();
    const last = traded[traded.length - 1] ?? run.last_day;
    setControls((c) => ({ ...c, date: last }));
  }, [dayList, daysFailed, controls.date, run.last_day]);

  // Sliders are debounced; selects apply at once.
  const power = useDebounced(controls.power, SLIDER_DEBOUNCE_MS);
  const duration = useDebounced(controls.duration, SLIDER_DEBOUNCE_MS);
  const degradation = useDebounced(controls.degradation, SLIDER_DEBOUNCE_MS);
  const strategy = controls.strategy;
  const battery: BatteryQuery = useMemo(
    () => ({ power, duration, degradation, strategy }),
    [power, duration, degradation, strategy],
  );
  const { windowKey, date } = controls;

  const summary = useApi(["summary", runId, windowKey, power, duration, degradation, strategy].join("|"), (signal) =>
    api.summary(runId, windowKey, battery, signal),
  );
  const forecast = useApi(date ? `forecast|${runId}|${date}` : null, (signal) =>
    date ? api.forecast(runId, date, signal) : Promise.reject(new Error("no day")),
  );

  const fc = dataOf(forecast);
  const selectedDay = date ? dayList?.find((d) => d.date === date) : undefined;
  const isHoldout = (selectedDay?.window ?? fc?.window) === "holdout";

  let context: string;
  if (!date) context = "Loading calendar…";
  else if (fc && fc.date === date)
    context = `Forecast issued ${clock(fc.issued_local)} · gate ${clock(fc.gate_local)} · delivery ${longDay(date)}`;
  else if (forecast.status === "error") context = `Delivery ${longDay(date)} · forecast unavailable: ${forecast.error}`;
  else context = `Delivery ${longDay(date)} · loading forecast…`;

  // A live run says when today's forecast was issued, in market local time, and
  // flags a delivery day whose prices the market has not published yet.
  const runKind: RunKind = run.run_kind ?? "backtest";
  const isLive = runKind === "live";
  if (isLive && run.issued_utc) context += ` · run issued ${localClock(run.issued_utc, run.timezone ?? "UTC")} local`;
  const priceNote =
    isLive && date && run.data_through && run.data_through < date
      ? `prices for ${shortDay(date)} not published yet`
      : null;

  // The battery and strategy controls and the P&L KPIs do not apply to Model health
  // or Live, so those tabs hide them; the control values are kept for the others.
  const tradingControls = tab !== "Model health" && tab !== "Live";

  return (
    <Shell context={context} note={priceNote} runKind={runKind} holdout={isHoldout} tab={tab} onTab={onTab} runId={runId}>
      {tradingControls ? (
        <KpiStrip
          state={summary}
          gridAvailable={run.grid_available}
          runKind={run.run_kind}
        />
      ) : null}

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {tradingControls ? (
          <Controls
            values={controls}
            onChange={update}
            run={run}
            grid={grid}
            days={dayList}
            daysError={errorOf(days)}
          />
        ) : null}
        <main
          id={PANEL_ID}
          role="tabpanel"
          aria-labelledby={tabId(tab)}
          className={`${tradingControls ? "lg:col-span-3" : "lg:col-span-4"} min-w-0`}
        >
          {tab === "Overview" && <OverviewTab run={runId} date={date} battery={battery} forecast={forecast} />}
          {tab === "Forecast" && <ForecastTab run={runId} windowKey={windowKey} />}
          {tab === "Trading" && (
            <TradingTab
              run={runId}
              windowKey={windowKey}
              battery={battery}
              gridAvailable={run.grid_available}
              runKind={run.run_kind}
            />
          )}
          {tab === "Model health" && <ModelHealthTab run={runId} />}
          {tab === "Live" && <LiveTab />}
        </main>
      </div>
    </Shell>
  );
}

export default function App() {
  const [tab, setTab] = useState<Tab>(tabFromHash);

  // Keep the hash and the tab in step without adding history entries.
  useEffect(() => {
    const hash = `#${tabSlug(tab)}`;
    if (window.location.hash !== hash) window.history.replaceState(null, "", hash);
  }, [tab]);
  useEffect(() => {
    const onHash = () => setTab(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const runs = useApi("runs", (signal) => api.runs(signal));
  const runList = dataOf(runs);
  const run = useMemo(() => (runList ? latestRun(runList) : null), [runList]);
  const grid = useMemo(() => (run ? batteryGrid(run) : null), [run]);
  const runsError = errorOf(runs) ?? (runList && runList.length === 0 ? "no runs exported yet" : null);

  // Battery controls start from the run's reference battery, so nothing is queried until the run is known.
  if (!run || !grid) {
    const message = runsError ? `Could not load the run: ${runsError}` : "Loading run…";
    return (
      <Shell context={message} note={null} runKind="backtest" holdout={false} tab={tab} onTab={setTab} runId={null}>
        <main
          id={PANEL_ID}
          role="tabpanel"
          aria-labelledby={tabId(tab)}
          className="rounded-lg"
          style={{ background: C.panel, border: `1px solid ${C.panelEdge}` }}
        >
          <StatusMessage height={160}>{message}</StatusMessage>
        </main>
      </Shell>
    );
  }
  return <Dashboard key={run.run_id} run={run} grid={grid} tab={tab} onTab={setTab} />;
}
