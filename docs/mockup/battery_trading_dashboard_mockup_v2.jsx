import { useState, useMemo } from "react";
import {
  ComposedChart, Area, Line, Bar, XAxis, YAxis, Tooltip,
  CartesianGrid, ReferenceLine, ResponsiveContainer, Legend,
} from "recharts";

// ----------------------------------------------------------------
// Design tokens — "grid control room at night". Custom colors are
// inline styles (no Tailwind arbitrary values in this environment).
// ----------------------------------------------------------------
const C = {
  bg: "#0d1522",
  panel: "#141f30",
  panelEdge: "#1f2d42",
  text: "#e6ecf5",
  muted: "#7f8ea6",
  faint: "#54637b",
  price: "#f2b23e",
  actual: "#ffffff",
  band90: "rgba(79, 143, 197, 0.18)",
  band50: "rgba(79, 143, 197, 0.38)",
  bandLine: "#4f8fc5",
  charge: "#4f8fc5",
  discharge: "#f2b23e",
  soc: "#79c398",
  pf: "#8f9fb8",
  median: "#c97b5a",
  quantile: "#79c398",
  neg: "#e0685c",
  good: "#79c398",
  warn: "#e8a13c",
  narr: "#1a2740",
};

// ----------------------------------------------------------------
// Mock data
// ----------------------------------------------------------------
const MEDIAN = [62, 55, 48, 45, 44, 48, 68, 95, 88, 72, 45, 18, -6, -14, -4, 22, 58, 92, 138, 164, 141, 105, 84, 70];
const SPREAD = [8, 8, 7, 7, 7, 8, 12, 16, 13, 11, 12, 15, 18, 20, 16, 12, 13, 18, 26, 32, 24, 15, 11, 9];
const ACTUAL = [64, 53, 47, 46, 42, 50, 74, 99, 84, 69, 41, 12, -9, -18, -2, 27, 63, 99, 149, 181, 149, 108, 81, 68];

const forecastData = MEDIAN.map((m, h) => ({
  h: `${String(h).padStart(2, "0")}:00`,
  band90: [m - 1.65 * SPREAD[h], m + 1.65 * SPREAD[h]],
  band50: [m - 0.67 * SPREAD[h], m + 0.67 * SPREAD[h]],
  median: m,
  actual: ACTUAL[h],
}));

const calibrationData = [10, 20, 30, 40, 50, 60, 70, 80, 90].map((nom, i) => ({
  nominal: nom,
  empirical: [9, 19, 31, 42, 52, 60, 71, 79, 87][i],
}));

const errorByHour = MEDIAN.map((m, h) => ({
  h: `${String(h).padStart(2, "0")}`,
  mae: +(Math.abs(ACTUAL[h] - m) * 0.6 + SPREAD[h] * 0.35).toFixed(1),
  valueAtStake: h >= 18 && h <= 20 ? 3 : h >= 11 && h <= 14 ? 2.4 : 0.7,
}));

const featureImportance = [
  { f: "Wind forecast D+1", w: 100 },
  { f: "Solar forecast D+1", w: 84 },
  { f: "Load forecast D+1", w: 71 },
  { f: "Price lag 24h", w: 58 },
  { f: "Gas price (TTF)", w: 44 },
  { f: "Price lag 168h", w: 37 },
  { f: "Hour of day", w: 33 },
  { f: "Carbon (EUA)", w: 21 },
];

const pnlData = Array.from({ length: 30 }, (_, i) => {
  const d = i + 1;
  const wobble = Math.sin(d * 1.7) * 18;
  const dip = d === 17 ? -55 : 0;
  return {
    day: d,
    perfect: Math.round(80 * d + wobble),
    quantile: Math.round(66 * d + wobble * 0.9 + dip),
    median: Math.round(58 * d + wobble * 0.8 + dip * 1.4),
  };
});

// Over- vs under-forecast cost, aggregated into hour blocks
const asymmetryData = [
  { block: "00–05", over: 4, under: 6 },
  { block: "06–10", over: 12, under: 18 },
  { block: "11–14", over: 22, under: 9 },
  { block: "15–17", over: 10, under: 14 },
  { block: "18–20", over: 31, under: 118 },
  { block: "21–23", over: 8, under: 12 },
];

// Regime shift: 90% coverage over time, frozen vs rolling-retrained model
const regimeData = [
  ["2020 Q1", 89, 89], ["2020 Q2", 90, 90], ["2020 Q3", 88, 89], ["2020 Q4", 89, 88],
  ["2021 Q1", 86, 88], ["2021 Q2", 83, 87], ["2021 Q3", 76, 85], ["2021 Q4", 68, 82],
  ["2022 Q1", 63, 80], ["2022 Q2", 61, 84], ["2022 Q3", 58, 86], ["2022 Q4", 64, 88],
  ["2023 Q1", 70, 89], ["2023 Q2", 74, 90],
].map(([q, frozen, rolling]) => ({ q, frozen, rolling }));

const incidents = [
  { date: "12 Sep", type: "Tail miss", detail: "19:00 price €181 exceeded q0.95 (€164+). Spike driven by low wind + FR export.", action: "Logged for spike-classifier training set", status: "review", tone: C.warn },
  { date: "08 Sep", type: "Data gap", detail: "ENTSO-E wind D+1 forecast missing at 11:30 pull.", action: "Fallback feature set used; forecast issued 11:47", status: "resolved", tone: C.good },
  { date: "02 Sep", type: "Late data", detail: "Load actuals revised +3.1% two days after publication.", action: "Training snapshot rebuilt; drift check passed", status: "resolved", tone: C.good },
  { date: "28 Aug", type: "Pipeline", detail: "DST-style short-day handling test failed on 23-hour synthetic day.", action: "Fixed index alignment in feature builder; test added", status: "resolved", tone: C.good },
  { date: "17 Aug", type: "Drawdown", detail: "−€55 day: charged at 14:00, prices fell further; forecast valley too early.", action: "Attribution written; valley-timing error tracked", status: "resolved", tone: C.good },
];

// ----------------------------------------------------------------
// Building blocks
// ----------------------------------------------------------------
function Panel({ title, sub, children, className = "" }) {
  return (
    <section className={`rounded-lg p-4 ${className}`} style={{ background: C.panel, border: `1px solid ${C.panelEdge}` }}>
      <header className="mb-3">
        <h2 className="text-sm font-semibold" style={{ color: C.text }}>{title}</h2>
        {sub && <p className="text-xs mt-0.5" style={{ color: C.muted }}>{sub}</p>}
      </header>
      {children}
    </section>
  );
}

function Narration({ children, tab }) {
  return (
    <section className="rounded-lg p-4" style={{ background: C.narr, border: `1px solid #2a3d5f` }}>
      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-sm font-semibold" style={{ color: C.text }}>Desk briefing</span>
        <span className="text-xs" style={{ color: C.faint }}>generated from today's numbers · deterministic core, LLM narration</span>
      </div>
      <p className="text-sm leading-relaxed" style={{ color: "#c3cfe0" }}>{children}</p>
      <div className="flex gap-2 mt-3">
        {["Why this dispatch?", "What changed vs yesterday?", "Explain the miss"].map((q) => (
          <button key={q} className="text-xs rounded-full px-3 py-1"
            style={{ background: "rgba(79,143,197,0.15)", color: "#9dc3e2", border: "1px solid rgba(79,143,197,0.35)" }}>
            {q}
          </button>
        ))}
      </div>
    </section>
  );
}

function Kpi({ label, value, note, tone }) {
  return (
    <div className="rounded-lg px-4 py-3 flex-1 min-w-0" style={{ background: C.panel, border: `1px solid ${C.panelEdge}` }}>
      <div className="text-xs" style={{ color: C.muted }}>{label}</div>
      <div className="text-xl font-semibold mt-1 tabular-nums" style={{ color: tone || C.text }}>{value}</div>
      <div className="text-xs mt-0.5" style={{ color: C.faint }}>{note}</div>
    </div>
  );
}

function Slider({ label, value, setValue, min, max, step, unit }) {
  return (
    <label className="block">
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: C.muted }}>{label}</span>
        <span className="tabular-nums" style={{ color: C.text }}>{value} {unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => setValue(+e.target.value)} className="w-full" style={{ accentColor: C.price }} />
    </label>
  );
}

const tooltipStyle = { background: "#0b1220", border: `1px solid ${C.panelEdge}`, borderRadius: 8, fontSize: 12, color: "#e6ecf5" };
const axisTick = { fill: C.faint, fontSize: 10 };

// ----------------------------------------------------------------
// Charts
// ----------------------------------------------------------------
function FanChart() {
  return (
    <ResponsiveContainer width="100%" height={250}>
      <ComposedChart data={forecastData} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="h" tick={axisTick} interval={3} axisLine={false} tickLine={false} />
        <YAxis tick={axisTick} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle}
          formatter={(v, n) => [Array.isArray(v) ? `${v[0].toFixed(0)} to ${v[1].toFixed(0)} €` : `${v} €/MWh`, n]} />
        <ReferenceLine y={0} stroke={C.neg} strokeDasharray="4 3" />
        <Area dataKey="band90" name="90% interval" fill={C.band90} stroke="none" isAnimationActive={false} />
        <Area dataKey="band50" name="50% interval" fill={C.band50} stroke="none" isAnimationActive={false} />
        <Line dataKey="median" name="Median forecast" stroke={C.bandLine} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line dataKey="actual" name="Realised" stroke={C.actual} strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

function DispatchChart({ powerMW, durationH }) {
  const baseDispatch = MEDIAN.map((m, h) => (h >= 11 && h <= 13 ? -1 : h === 18 || h === 19 ? 1 : 0));
  let soc = 0.2;
  const socPath = baseDispatch.map((mw) => {
    soc = Math.min(2, Math.max(0, soc - mw * (mw < 0 ? 0.95 : 1 / 0.95)));
    return soc;
  });
  const data = MEDIAN.map((m, h) => ({
    h: `${String(h).padStart(2, "0")}:00`,
    price: ACTUAL[h],
    charge: baseDispatch[h] < 0 ? baseDispatch[h] * powerMW : 0,
    discharge: baseDispatch[h] > 0 ? baseDispatch[h] * powerMW : 0,
    soc: +(socPath[h] * (durationH / 2) * powerMW).toFixed(2),
  }));
  return (
    <ResponsiveContainer width="100%" height={230}>
      <ComposedChart data={data} margin={{ top: 4, right: 0, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="h" tick={axisTick} interval={3} axisLine={false} tickLine={false} />
        <YAxis yAxisId="mw" tick={axisTick} axisLine={false} tickLine={false} />
        <YAxis yAxisId="p" orientation="right" tick={axisTick} axisLine={false} tickLine={false} />
        <Tooltip contentStyle={tooltipStyle} />
        <ReferenceLine yAxisId="mw" y={0} stroke={C.faint} />
        <Bar yAxisId="mw" dataKey="charge" name="Charge (MW)" fill={C.charge} isAnimationActive={false} />
        <Bar yAxisId="mw" dataKey="discharge" name="Discharge (MW)" fill={C.discharge} isAnimationActive={false} />
        <Line yAxisId="mw" dataKey="soc" name="SoC (MWh)" stroke={C.soc} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line yAxisId="p" dataKey="price" name="Price (€/MWh)" stroke={C.actual} strokeWidth={1} strokeDasharray="4 3" dot={false} isAnimationActive={false} />
        <Legend wrapperStyle={{ fontSize: 11, color: C.muted }} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ----------------------------------------------------------------
// Tabs
// ----------------------------------------------------------------
function OverviewTab({ powerMW, durationH }) {
  return (
    <div className="space-y-4">
      <Narration>
        The optimizer bought 3 MWh through the 12:00–14:00 negative-price valley (avg −€10/MWh) and sold into the
        evening ramp at 18:00–19:00 (avg €165/MWh). It held one cycle in reserve because the q0.75 forecast showed
        spike risk after 20:00 that did not materialise. Today closed €41 behind perfect foresight — almost all of it
        from the 19:00 spike settling €17 above the q0.95 band.
      </Narration>
      <Panel title="Price forecast vs outcome" sub="Quantile fan (q0.05–q0.95) issued before the 12:00 gate · white line is the realised price">
        <FanChart />
      </Panel>
      <Panel title="Dispatch schedule & state of charge" sub="Charge in the valley, discharge into the spike">
        <DispatchChart powerMW={powerMW} durationH={durationH} />
      </Panel>
    </div>
  );
}

function ForecastTab() {
  return (
    <div className="space-y-4">
      <Narration>
        Calibration is healthy overall: 90% intervals covered 87% of outcomes this window, with mild over-coverage in
        the 30–40% bands. Errors concentrate in hours 18–20, where the value at stake is also highest — average error
        there is 2.6× the flat-hour error. Wind forecast remains the dominant feature; its importance rose again this
        month as wind variability increased.
      </Narration>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel title="Calibration" sub="Nominal vs empirical coverage, rolling 90 days">
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={calibrationData} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" />
              <XAxis dataKey="nominal" tick={axisTick} axisLine={false} tickLine={false} />
              <YAxis tick={axisTick} axisLine={false} tickLine={false} domain={[0, 100]} />
              <Tooltip contentStyle={tooltipStyle} />
              <Line dataKey="nominal" name="Perfect" stroke={C.faint} strokeDasharray="4 4" dot={false} isAnimationActive={false} />
              <Line dataKey="empirical" name="Model" stroke={C.quantile} strokeWidth={2} dot={{ r: 3, fill: C.quantile }} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </Panel>
        <Panel title="Error by delivery hour" sub="MAE (€/MWh) — the expensive hours are the spread-defining ones">
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart data={errorByHour} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
              <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
              <XAxis dataKey="h" tick={axisTick} interval={3} axisLine={false} tickLine={false} />
              <YAxis tick={axisTick} axisLine={false} tickLine={false} />
              <Tooltip contentStyle={tooltipStyle} />
              <Bar dataKey="mae" name="MAE" fill={C.bandLine} isAnimationActive={false} />
              <Line dataKey="valueAtStake" name="Value at stake (rel.)" stroke={C.price} strokeWidth={2} dot={false} isAnimationActive={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </Panel>
      </div>
      <Panel title="Feature importance" sub="LightGBM gain, median model, current training window">
        <div className="space-y-2">
          {featureImportance.map((f) => (
            <div key={f.f} className="flex items-center gap-3">
              <div className="text-xs w-36 shrink-0" style={{ color: C.muted }}>{f.f}</div>
              <div className="flex-1 rounded-full h-2" style={{ background: "#0b1220" }}>
                <div className="h-2 rounded-full" style={{ width: `${f.w}%`, background: C.bandLine }} />
              </div>
              <div className="text-xs tabular-nums w-8 text-right" style={{ color: C.faint }}>{f.w}</div>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function TradingTab({ scale }) {
  const scaled = pnlData.map((d) => ({
    day: d.day,
    perfect: Math.round(d.perfect * scale),
    quantile: Math.round(d.quantile * scale),
    median: Math.round(d.median * scale),
  }));
  return (
    <div className="space-y-4">
      <Narration>
        Quantile-aware dispatch leads median dispatch by €{Math.round(240 * scale).toLocaleString()} over the window,
        with the gap opening on spike days — it holds capacity when the upper quantiles flag risk. Day 17 was the
        largest drawdown: both strategies charged into a valley that kept falling. Capture ratio is stable at 82–84%;
        the remaining gap to perfect foresight is concentrated in five spike days.
      </Narration>
      <Panel title="Cumulative P&L by strategy" sub="Dashed grey line is the perfect-foresight ceiling">
        <ResponsiveContainer width="100%" height={250}>
          <ComposedChart data={scaled} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
            <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="day" tick={axisTick} axisLine={false} tickLine={false} />
            <YAxis tick={axisTick} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v, n) => [`€ ${v.toLocaleString()}`, n]} />
            <Line dataKey="perfect" name="Perfect foresight" stroke={C.pf} strokeWidth={1.5} strokeDasharray="6 4" dot={false} isAnimationActive={false} />
            <Line dataKey="quantile" name="Quantile-aware" stroke={C.quantile} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="median" name="Median-forecast" stroke={C.median} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.muted }} />
          </ComposedChart>
        </ResponsiveContainer>
      </Panel>
      <Panel title="Cost of forecast error by direction" sub="Where over- and under-forecasting actually lose money (€, backtest window)">
        <ResponsiveContainer width="100%" height={220}>
          <ComposedChart data={asymmetryData} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
            <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="block" tick={axisTick} axisLine={false} tickLine={false} />
            <YAxis tick={axisTick} axisLine={false} tickLine={false} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v, n) => [`€ ${v}`, n]} />
            <Bar dataKey="over" name="Over-forecast cost" fill={C.median} isAnimationActive={false} />
            <Bar dataKey="under" name="Under-forecast cost" fill={C.bandLine} isAnimationActive={false} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.muted }} />
          </ComposedChart>
        </ResponsiveContainer>
        <p className="text-xs mt-2 leading-relaxed" style={{ color: C.faint }}>
          Under-forecasting the 18–20 block dominates: missing a spike after the battery is empty is the single most
          expensive error. Over-forecasting midday costs little — the valley is forgiving.
        </p>
      </Panel>
    </div>
  );
}

function ModelHealthTab() {
  return (
    <div className="space-y-4">
      <Narration>
        The frozen pre-2021 model's 90% coverage collapsed to 58% during the 2022 gas crisis; quarterly retraining held
        it above 80% and recovered to 86% within two quarters. One data incident this month (missing wind forecast)
        triggered the fallback path — forecast quality degraded but the 12:00 gate was met. The 12 Sep tail miss is
        queued for the spike-classifier training set.
      </Narration>
      <Panel title="Regime shift experiment" sub="90% interval coverage over time: frozen pre-2021 model vs quarterly retraining, through the gas crisis">
        <ResponsiveContainer width="100%" height={240}>
          <ComposedChart data={regimeData} margin={{ top: 4, right: 8, left: -8, bottom: 0 }}>
            <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="q" tick={axisTick} interval={1} axisLine={false} tickLine={false} />
            <YAxis tick={axisTick} axisLine={false} tickLine={false} domain={[50, 100]} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v, n) => [`${v}%`, n]} />
            <ReferenceLine y={90} stroke={C.faint} strokeDasharray="4 4" label={{ value: "target 90%", fill: C.faint, fontSize: 10, position: "insideTopRight" }} />
            <Line dataKey="frozen" name="Frozen model" stroke={C.neg} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="rolling" name="Quarterly retrain" stroke={C.quantile} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Legend wrapperStyle={{ fontSize: 11, color: C.muted }} />
          </ComposedChart>
        </ResponsiveContainer>
      </Panel>
      <Panel title="Incident log" sub="Failures injected or observed, and what was done about them">
        <div className="space-y-3">
          {incidents.map((it, i) => (
            <div key={i} className="flex gap-3 items-start rounded-md p-3" style={{ background: "#0f1929", border: `1px solid ${C.panelEdge}` }}>
              <div className="text-xs tabular-nums shrink-0 w-12 pt-0.5" style={{ color: C.faint }}>{it.date}</div>
              <div className="min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-semibold" style={{ color: C.text }}>{it.type}</span>
                  <span className="text-xs rounded-full px-2 py-px" style={{ color: it.tone, border: `1px solid ${it.tone}55`, background: `${it.tone}18` }}>{it.status}</span>
                </div>
                <p className="text-xs mt-1 leading-relaxed" style={{ color: C.muted }}>{it.detail}</p>
                <p className="text-xs mt-1 leading-relaxed" style={{ color: C.soc }}>→ {it.action}</p>
              </div>
            </div>
          ))}
        </div>
      </Panel>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-3">
        <Kpi label="Pipeline SLA" value="30 / 30 days" note="forecast ready before 12:00 gate" tone={C.good} />
        <Kpi label="Fallback activations" value="1" note="seasonal-naive path, 08 Sep" tone={C.warn} />
        <Kpi label="Rolling coverage (90%)" value="87 %" note="drift alert threshold: 82%" />
      </div>
    </div>
  );
}

// ----------------------------------------------------------------
// App shell
// ----------------------------------------------------------------
const TABS = ["Overview", "Forecast", "Trading", "Model health"];

export default function BatteryTradingDashboard() {
  const [tab, setTab] = useState("Overview");
  const [powerMW, setPowerMW] = useState(1);
  const [durationH, setDurationH] = useState(2);
  const [riskQ, setRiskQ] = useState("0.25");
  const [degCost, setDegCost] = useState(8);

  const scale = powerMW * (durationH / 2);
  const kpis = useMemo(() => ({
    pnl: Math.round(1987 * scale - degCost * 12 * scale),
    capture: Math.max(60, Math.round(83 - (degCost - 8) * 0.6)),
    pinball: 4.21,
    cycles: (1.1 * (2 / durationH)).toFixed(1),
  }), [scale, degCost, durationH]);

  return (
    <div className="min-h-screen w-full" style={{ background: C.bg, color: C.text, fontFamily: "ui-sans-serif, system-ui, sans-serif" }}>
      <div className="max-w-6xl mx-auto px-4 py-5">

        <header className="flex flex-wrap items-baseline gap-x-4 gap-y-1 mb-4">
          <h1 className="text-lg font-semibold tracking-tight">Day-ahead battery trading — DE-LU</h1>
          <span className="text-xs" style={{ color: C.muted }}>Forecast issued 11:40 · gate 12:00 · delivery Fri 12 Sep</span>
          <span className="text-xs rounded-full px-2.5 py-0.5 ml-auto"
            style={{ background: "rgba(121,195,152,0.12)", color: C.good, border: "1px solid rgba(121,195,152,0.3)" }}>
            backtest · mock data
          </span>
        </header>

        {/* Tabs */}
        <nav className="flex gap-1 mb-4 rounded-lg p-1" style={{ background: "#0b1220", border: `1px solid ${C.panelEdge}` }}>
          {TABS.map((t) => (
            <button key={t} onClick={() => setTab(t)}
              className="flex-1 text-sm rounded-md px-3 py-1.5 transition-colors"
              style={tab === t
                ? { background: C.panel, color: C.text, border: `1px solid ${C.panelEdge}` }
                : { color: C.muted, border: "1px solid transparent" }}>
              {t}
            </button>
          ))}
        </nav>

        {/* KPI strip stays visible on every tab */}
        <div className="flex flex-wrap gap-3 mb-4">
          <Kpi label="P&L, 30-day backtest" value={`€ ${kpis.pnl.toLocaleString()}`} note="net of degradation" tone={C.good} />
          <Kpi label="Capture ratio" value={`${kpis.capture} %`} note="vs perfect foresight" />
          <Kpi label="Pinball loss" value={kpis.pinball} note="€/MWh · baseline 6.90" />
          <Kpi label="Cycles per day" value={kpis.cycles} note="avg over backtest" />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
          {/* Persistent controls */}
          <Panel title="Battery & strategy" sub="Re-runs the backtest" className="lg:col-span-1 h-fit">
            <div className="space-y-4">
              <Slider label="Power" value={powerMW} setValue={setPowerMW} min={0.5} max={5} step={0.5} unit="MW" />
              <Slider label="Duration" value={durationH} setValue={setDurationH} min={1} max={4} step={1} unit="h" />
              <Slider label="Degradation cost" value={degCost} setValue={setDegCost} min={0} max={25} step={1} unit="€/MWh" />
              <label className="block">
                <div className="text-xs mb-1" style={{ color: C.muted }}>Dispatch against quantile</div>
                <select value={riskQ} onChange={(e) => setRiskQ(e.target.value)}
                  className="w-full rounded-md px-2 py-1.5 text-sm"
                  style={{ background: "#0b1220", color: C.text, border: `1px solid ${C.panelEdge}` }}>
                  <option value="0.5">q0.50 — median (neutral)</option>
                  <option value="0.25">q0.25 — conservative buy</option>
                  <option value="0.10">q0.10 — very conservative</option>
                </select>
              </label>
              <p className="text-xs leading-relaxed" style={{ color: C.faint }}>
                Battery: {powerMW} MW / {(powerMW * durationH).toFixed(1)} MWh, 90% round-trip efficiency.
                Re-optimised daily on the {riskQ === "0.5" ? "median" : `q${riskQ}`} forecast.
              </p>
            </div>
          </Panel>

          {/* Tab content */}
          <div className="lg:col-span-3">
            {tab === "Overview" && <OverviewTab powerMW={powerMW} durationH={durationH} />}
            {tab === "Forecast" && <ForecastTab />}
            {tab === "Trading" && <TradingTab scale={scale} />}
            {tab === "Model health" && <ModelHealthTab />}
          </div>
        </div>

        <footer className="mt-4 text-xs" style={{ color: C.faint }}>
          Mock data for layout review. Live build: OPSD / ENTSO-E → LightGBM quantile models (MLflow-tracked) → MILP dispatch (PuLP/CBC) → walk-forward backtest · Airflow-orchestrated in the live phase.
        </footer>
      </div>
    </div>
  );
}
