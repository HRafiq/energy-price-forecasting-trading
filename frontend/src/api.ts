// Typed client for the read-only dashboard API (docs/dashboard_api.md).
// Every number on the page comes from one of these calls.

export type WindowKey = "last30" | "last90" | "validation" | "holdout" | "all";
export type StrategyKey = "median" | "q25" | "q10";
export type DayWindow = "validation" | "holdout";
/** A replay of history, or a run the daily pipeline produced today. */
export type RunKind = "backtest" | "live";

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
  /** Absent in an API older than the live pipeline; treat as "backtest". */
  run_kind?: RunKind;
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
  /** Absent in an API older than the live pipeline; treat as "backtest". */
  run_kind?: RunKind;
  /** When a live run issued its forecast, UTC; null for a backtest. */
  issued_utc?: string | null;
  /** Last delivery day with published prices; a live run can forecast past it. */
  data_through?: string | null;
  /** The market's timezone, for showing issue times in market local time. */
  timezone?: string;
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
// Model health (/api/model-health/*)
// ------------------------------------------------------------------------------

export type RegimeArm = "frozen" | "quarterly" | "monthly" | "naive_previous_day";
export type DriftWindow = "validation" | "holdout" | "all";
export type IncidentType = "data_gap" | "late_data" | "tail_miss" | "drift" | "pipeline" | "drawdown";

export interface RegimePeriod {
  arm: string;
  period: string;
  days: number | null;
  coverage_90: number | null;
  coverage_50: number | null;
  pinball: number | null;
  mae_q50: number | null;
  capture_ratio: number | null;
  pnl_eur: number | null;
  perfect_foresight_pnl_eur: number | null;
}

export interface RegimeResponse {
  experiment: string;
  generated_utc: string | null;
  historical_backtest: boolean;
  setup: {
    summary: string | null;
    model: string | null;
    training_days: number | null;
    calibration_days: number | null;
    first_target_day: string | null;
    last_target_day: string | null;
    feature_groups: string[];
    weather_features: boolean;
    arms: Array<{ name: string; refit_every_days: number | null; description?: string }> | null;
  };
  coverage_target: number;
  rolling_coverage_90: { window_days: number; dates: string[]; arms: Record<string, Array<number | null>> };
  min_rolling_coverage_90: Record<string, { value: number; window_end: string }>;
  periods: RegimePeriod[];
}

export interface DriftPoint {
  target_day: string;
  window: "validation" | "holdout";
  coverage_90: number | null;
  pinball: number | null;
  rolling_coverage_90: number | null;
  rolling_pinball_ratio: number | null;
  coverage_alert: boolean;
  pinball_alert: boolean;
}

export interface DriftEpisode {
  signal: "coverage" | "pinball";
  window: "validation" | "holdout";
  start: string;
  end: string;
  days: number;
  first_value: number;
  extreme_value: number;
  extreme_day: string;
  open_at_end: boolean;
}

export interface DriftResponse {
  experiment: string;
  generated_utc: string | null;
  model: string | null;
  window: DriftWindow;
  window_days: number;
  holdout_start: string;
  rule: string | null;
  /** M2's last target day, the run's last day, and whether they agree. */
  last_target_day: string | null;
  run_last_day: string | null;
  matches_run: boolean;
  thresholds: { coverage: number; pinball_ratio: number; pinball_median?: number };
  windows: Record<string, { days: number; coverage_alert_share?: number; pinball_alert_share?: number }>;
  episodes: DriftEpisode[];
  series: DriftPoint[];
}

/** observed: fixed rules over saved outputs, live pipeline runs; measured: drift monitor, M2 and live; simulated: failure injection. */
export type Provenance = "observed" | "measured" | "simulated";

export interface Incident {
  incident_id: string;
  delivery_day: string;
  detected_utc: string;
  type: IncidentType;
  severity: "info" | "warning" | "critical";
  detail: string;
  action: string;
  status: "resolved" | "review";
  source: string;
  metrics: Record<string, number>;
  provenance: Provenance;
  /** True when the day lies in the window its threshold was fitted on; null when not recorded. */
  in_sample: boolean | null;
}

export interface IncidentsResponse {
  generated_utc: string | null;
  total: number;
  limit: number;
  offset: number;
  counts: { type: Record<string, number>; source: Record<string, number>; provenance: Record<Provenance, number> };
  source_provenance: Record<string, Provenance>;
  incidents: Incident[];
}

export interface IncidentQuery {
  type?: IncidentType;
  source?: string;
  limit: number;
  offset?: number;
}

interface Unavailable {
  available: false;
  detail: string;
}

export interface Period {
  first_day: string | null;
  last_day: string | null;
}

export type OpsDeadline =
  | Unavailable
  | {
      available: true;
      source: string;
      simulation: boolean;
      generated_utc: string | null;
      period: Period;
      issue_local: string | null;
      gate_local: string | null;
      failure_rates: Record<string, number> | null;
      seed: number | null;
      /** Share of days each failure type was actually drawn; absent in older exports. */
      realised_failure_rates?: Record<string, number> | null;
      days: number;
      on_time_share_with_chain: number;
      on_time_share_without_chain: number;
      fallback_days: number;
      fallback_by_step: Record<string, number>;
      latest_submission_minutes_after_issue: number;
    };

interface ObservedFallbacks {
  observed: number | null;
  observed_rule: string;
  observed_period: Period;
  observed_generated_utc: string | null;
}

export type OpsFallbacks =
  | (Unavailable & ObservedFallbacks)
  | ({
      available: true;
      simulated_days: number;
      simulated_source: string;
      simulated_period: Period;
      simulated_generated_utc: string | null;
    } & ObservedFallbacks);

export interface DriftSignalNow {
  value: number | null;
  threshold: number;
  alert: boolean;
}

export type OpsDrift =
  | Unavailable
  | {
      available: true;
      source: string;
      generated_utc: string | null;
      as_of: string;
      window: "validation" | "holdout";
      window_days: number;
      holdout_start: string;
      holdout_days: number;
      last_target_day: string | null;
      run_last_day: string | null;
      matches_run: boolean;
      coverage: DriftSignalNow;
      pinball_ratio: DriftSignalNow;
    };

export interface OpsResponse {
  run_id: string;
  deadline: OpsDeadline;
  fallbacks: OpsFallbacks;
  drift: OpsDrift;
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
// Model health responses are not cached: a health export can be rerun while the
// page is open, and the next visit to the tab should show it without a reload.
const CACHE_LIMIT = 200;
const UNCACHED_PREFIXES = ["/api/model-health/"];
const responseCache = new Map<string, unknown>();

function cacheable(path: string): boolean {
  return !UNCACHED_PREFIXES.some((prefix) => path.startsWith(prefix));
}

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
  const cached = cacheable(path);
  if (cached && responseCache.has(url)) {
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
  if (cached) remember(url, body);
  return body;
}

/** The one write in the API: the briefing is composed per request, so nothing is cached. */
async function postJson<T>(path: string, body: unknown, signal: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    signal,
    headers: { Accept: "application/json", "Content-Type": "application/json" },
    body: JSON.stringify(body),
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
  return (await response.json()) as T;
}

/** The tabs a briefing can be written about, as the API names them. */
export type NarrateTab = "overview" | "forecast" | "trading" | "model_health";

export interface NarrateRequest {
  tab: NarrateTab;
  /** The delivery day; omitted on tabs that show a window rather than a day. */
  date?: string;
  window?: WindowKey;
  run?: string;
  duration?: number;
  degradation?: number;
  strategy?: StrategyKey;
  power?: number;
  question?: string;
}

export interface NarrateResponse {
  tab: NarrateTab;
  day: string;
  window: WindowKey;
  text: string;
  /** "template" by default and after a fallback; "openai" or "anthropic" when switched on. */
  provider: string;
  model: string | null;
  /** Every figure in `text` was found in the payload the briefing was given. */
  grounded: boolean;
  unsupported: string[];
  /** The deterministic writer answered in place of the model. */
  fell_back: boolean;
  /** Drafts asked of the provider: 2 when the first was refused and rewritten. */
  attempts: number;
  /** Figures a model wrote that are not in the payload, over every refused draft. */
  rejected: string[];
  /** Why the deterministic writer answered, or null when a model draft stood. */
  fallback_reason: string | null;
  follow_ups: Record<string, string>;
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

  regime: (run: string, signal: AbortSignal) => getJson<RegimeResponse>("/api/model-health/regime", { run }, signal),

  drift: (run: string, windowKey: DriftWindow, signal: AbortSignal) =>
    getJson<DriftResponse>("/api/model-health/drift", { run, window: windowKey }, signal),

  incidents: (run: string, query: IncidentQuery, signal: AbortSignal) =>
    getJson<IncidentsResponse>(
      "/api/model-health/incidents",
      { run, type: query.type, source: query.source, limit: query.limit, offset: query.offset },
      signal,
    ),

  ops: (run: string, signal: AbortSignal) => getJson<OpsResponse>("/api/model-health/ops", { run }, signal),

  narrate: (request: NarrateRequest, signal: AbortSignal) =>
    postJson<NarrateResponse>("/api/narrate", request, signal),
};
