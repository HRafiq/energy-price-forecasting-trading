// Typed client for the read-only dashboard API (docs/dashboard_api.md).
// Every number on the page comes from one of these calls.

export type WindowKey = "last30" | "last90" | "validation" | "holdout" | "all";
export type StrategyKey = "median" | "q25" | "q10";
export type DayWindow = "validation" | "holdout";

export interface BatteryQuery {
  duration: number;
  degradation: number;
  strategy: StrategyKey;
  power: number;
}

export interface ReferenceBattery {
  power_mw: number;
  capacity_mwh: number;
  round_trip_efficiency: number;
  degradation_eur_per_mwh: number;
  initial_soc_fraction: number;
  max_cycles_per_day: number;
}

export interface RunGrid {
  durations_h: number[];
  degradation_eur_per_mwh: Array<number | string>;
  power_mw: { min: number; max: number; step: number };
  strategies: Record<string, string>;
}

export interface Health {
  status: string;
  run: string;
  traded_days: number;
  mode: string;
  grid_available?: boolean;
}

export interface RunInfo {
  run_id: string;
  created_utc: string;
  model: string;
  baseline_model: string;
  first_day: string;
  last_day: string;
  holdout_start: string;
  traded_days: number;
  reference_battery: ReferenceBattery;
  grid: RunGrid;
  mode: string;
  /** Present in the live API, not in docs/dashboard_api.md. */
  source_commit?: string;
  /** False while the P&L grid export is still running (summary and pnl answer 503). */
  grid_available?: boolean;
}

export interface DayEntry {
  date: string;
  window: DayWindow;
  traded: boolean;
  skip_reason: string | null;
}

export interface DaysResponse {
  days: DayEntry[];
}

export interface WindowInfo {
  key: WindowKey;
  first_day: string;
  last_day: string;
  traded_days: number;
  skipped_days: string[];
}

export interface SummaryResponse {
  window: WindowInfo;
  battery: {
    power_mw: number;
    capacity_mwh: number;
    duration_h: number;
    degradation_eur_per_mwh: number;
    strategy: StrategyKey;
  };
  kpis: {
    pnl_eur: number;
    perfect_foresight_pnl_eur: number;
    capture_ratio: number;
    cycles_per_day: number;
    pinball_eur_mwh: number;
    baseline_pinball_eur_mwh: number;
  };
}

export interface ForecastPeriod {
  /** Local clock label; repeats 02:00 to 02:45 on the autumn clock change, so never a key. */
  time: string;
  /** ISO timestamp, unique per period; added by the backend. */
  utc?: string;
  q05: number;
  q10: number;
  q25: number;
  q50: number;
  q75: number;
  q90: number;
  q95: number;
  actual: number | null;
}

export interface ForecastResponse {
  date: string;
  window: DayWindow;
  issued_local: string;
  gate_local: string;
  product_minutes: number;
  periods: ForecastPeriod[];
}

export interface DispatchPeriod {
  /** Local clock label; repeats on the autumn clock change, so never a key. */
  time: string;
  /** ISO timestamp, unique per period; added by the backend. */
  utc?: string;
  charge_mw: number;
  discharge_mw: number;
  net_mw: number;
  soc_mwh: number;
  price: number | null;
}

export interface DispatchResponse {
  date: string;
  strategy: StrategyKey;
  solved_on_request: boolean;
  solve_ms: number;
  battery: { power_mw: number; capacity_mwh: number; degradation_eur_per_mwh: number };
  pnl_eur: number;
  perfect_foresight_pnl_eur: number;
  periods: DispatchPeriod[];
}

export interface PnlPoint {
  date: string;
  perfect_foresight: number;
  median: number;
  selected: number;
}

export interface PnlResponse {
  window: WindowInfo;
  selected_strategy: StrategyKey;
  series: PnlPoint[];
  max_drawdown_eur: { perfect_foresight: number; median: number; selected: number };
}

export interface CalibrationResponse {
  window: WindowInfo;
  quantiles: Array<{ level: number; empirical: number }>;
  intervals: Array<{ nominal: number; coverage: number }>;
}

export interface HourError {
  hour: number;
  mae_eur_mwh: number;
  value_at_stake_eur_per_day: number;
}

export interface AsymmetryBlock {
  block: string;
  over_eur: number;
  under_eur: number;
}

export interface ErrorAnalysisResponse {
  window: WindowInfo;
  by_hour: HourError[];
  asymmetry: { reference: string; gap_eur: number; days?: number; blocks: AsymmetryBlock[] };
}

export interface FeatureImportanceResponse {
  model: string;
  trained_for_day: string;
  importance: string;
  features: Array<{ feature: string; label: string; gain_share: number }>;
}

// ------------------------------------------------------------------------------
// Transport
// ------------------------------------------------------------------------------

export class ApiError extends Error {
  readonly status: number;
  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

type Params = Record<string, string | number | undefined>;

// The API serves read-only artifacts, so an identical GET returns the same body.
// Successful responses are kept in memory, keyed by URL, so switching tabs does
// not refetch. Errors are never cached; the oldest entry is evicted first.
const CACHE_LIMIT = 200;
const responseCache = new Map<string, unknown>();

function remember(url: string, body: unknown): void {
  responseCache.delete(url);
  responseCache.set(url, body);
  if (responseCache.size > CACHE_LIMIT) {
    const oldest = responseCache.keys().next();
    if (!oldest.done) responseCache.delete(oldest.value);
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/** FastAPI sends {"detail": "..."} or, for validation errors, {"detail": [{"msg": ...}]}. */
function detailText(body: unknown): string | null {
  if (!isRecord(body)) return null;
  const detail = body.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    const parts = detail.map((item) => {
      if (!isRecord(item)) return String(item);
      const loc = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "query").join(".") : "";
      const msg = typeof item.msg === "string" ? item.msg : JSON.stringify(item);
      return loc ? `${loc}: ${msg}` : msg;
    });
    return parts.join("; ");
  }
  return null;
}

async function getJson<T>(path: string, params: Params, signal: AbortSignal): Promise<T> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined) query.set(key, String(value));
  }
  const qs = query.toString();
  const url = qs ? `${path}?${qs}` : path;
  if (responseCache.has(url)) {
    const hit = responseCache.get(url);
    remember(url, hit);
    return hit as T;
  }
  const response = await fetch(url, {
    signal,
    headers: { Accept: "application/json" },
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`.trim();
    try {
      const text = detailText(await response.json());
      if (text) message = text;
    } catch {
      // Body was not JSON; keep the status line.
    }
    throw new ApiError(response.status, message);
  }
  const body = (await response.json()) as T;
  remember(url, body);
  return body;
}

function batteryParams(b: BatteryQuery): Params {
  return { duration: b.duration, degradation: b.degradation, strategy: b.strategy, power: b.power };
}

// ------------------------------------------------------------------------------
// Endpoints
// ------------------------------------------------------------------------------

export const api = {
  health: (signal: AbortSignal) => getJson<Health>("/api/health", {}, signal),

  runs: (signal: AbortSignal) => getJson<RunInfo[]>("/api/runs", {}, signal),

  days: (run: string, signal: AbortSignal) => getJson<DaysResponse>("/api/days", { run }, signal),

  summary: (run: string, windowKey: WindowKey, battery: BatteryQuery, signal: AbortSignal) =>
    getJson<SummaryResponse>(
      `/api/runs/${encodeURIComponent(run)}/summary`,
      { window: windowKey, ...batteryParams(battery) },
      signal,
    ),

  forecast: (run: string, date: string, signal: AbortSignal) =>
    getJson<ForecastResponse>("/api/forecast", { run, date }, signal),

  dispatch: (run: string, date: string, battery: BatteryQuery, signal: AbortSignal) =>
    getJson<DispatchResponse>("/api/dispatch", { run, date, ...batteryParams(battery) }, signal),

  pnl: (run: string, windowKey: WindowKey, battery: BatteryQuery, signal: AbortSignal) =>
    getJson<PnlResponse>("/api/pnl", { run, window: windowKey, ...batteryParams(battery) }, signal),

  calibration: (run: string, windowKey: WindowKey, signal: AbortSignal) =>
    getJson<CalibrationResponse>("/api/calibration", { run, window: windowKey }, signal),

  errorAnalysis: (run: string, windowKey: WindowKey, signal: AbortSignal) =>
    getJson<ErrorAnalysisResponse>("/api/error-analysis", { run, window: windowKey }, signal),

  featureImportance: (run: string, signal: AbortSignal) =>
    getJson<FeatureImportanceResponse>("/api/feature-importance", { run }, signal),
};
