import { useEffect, useId, useState } from "react";
import type { Incident, IncidentQuery, IncidentType, IncidentsResponse, Provenance } from "../api";
import { api } from "../api";
import { longDay, utcDay } from "../format";
import { dataOf, isReloading, useApi } from "../hooks";
import { C } from "../theme";
import { StatusMessage } from "./Loadable";
import { Panel } from "./Panel";
import { Pill } from "./Pill";

const PAGE = 20;

const TYPE_LABELS: Record<IncidentType, string> = {
  data_gap: "Data gap",
  late_data: "Late data",
  tail_miss: "Tail miss",
  drift: "Drift",
  pipeline: "Pipeline",
  drawdown: "Drawdown",
};

const SEVERITY_COLOR: Record<Incident["severity"], string> = {
  critical: C.neg,
  warning: C.warn,
  info: C.muted,
};

/** Readable names of the sources that write incidents. */
const SOURCE_NAMES: Record<string, string> = {
  m2_drift: "drift monitor",
  live_drift: "live drift monitor",
  pipeline: "live pipeline",
  desk_offline: "desk offline",
  d5_deadline: "D5 deadline",
};

const PROVENANCE_ORDER: Provenance[] = ["observed", "measured", "simulated"];

const selectStyle = { background: C.inset, color: C.text, border: `1px solid ${C.panelEdge}` };

function typeLabel(type: string): string {
  return TYPE_LABELS[type as IncidentType] ?? type;
}

/** "observed", "observed: live pipeline", "measured: drift monitor", "simulated: D5 deadline". */
export function sourceLabel(source: string, provenance: Provenance | undefined): string {
  const category = provenance ?? (source === "observed" ? "observed" : "simulated");
  if (source === "observed") return "observed";
  return `${category}: ${SOURCE_NAMES[source] ?? source.replace(/_/g, " ")}`;
}

function IncidentRow({ incident }: { incident: Incident }) {
  const resolved = incident.status === "resolved";
  return (
    <li className="py-3" style={{ borderTop: `1px solid ${C.panelEdge}` }}>
      <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
        <span className="text-sm font-medium tabular-nums" style={{ color: C.text }}>
          {longDay(incident.delivery_day)}
        </span>
        <Pill label={typeLabel(incident.type)} color={C.bandLine} />
        {/* Severity and provenance wrap as one unit, so the separator never starts a line. */}
        <span className="text-xs whitespace-nowrap">
          <span style={{ color: SEVERITY_COLOR[incident.severity] }}>{incident.severity}</span>
          <span style={{ color: C.muted }}> · {sourceLabel(incident.source, incident.provenance)}</span>
        </span>
        {incident.in_sample ? (
          <span title="Its day lies in the validation window its threshold was fitted on">
            <Pill label="in-sample" color={C.muted} />
          </span>
        ) : null}
        <span className="ml-auto">
          {resolved ? (
            <Pill label="resolved" color={C.good} />
          ) : (
            <Pill label="review" color={C.warn} />
          )}
        </span>
      </div>
      <p className="text-sm leading-relaxed mt-1.5 break-words" style={{ color: C.narrText }}>
        {incident.detail}
      </p>
      <p className="text-xs leading-relaxed mt-0.5 break-words" style={{ color: C.muted }}>
        Action: {incident.action}
      </p>
    </li>
  );
}

function Counts({ data }: { data: IncidentsResponse }) {
  const entries = Object.entries(data.counts.type).filter(([, n]) => n > 0);
  if (entries.length === 0) return null;
  return (
    <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs tabular-nums" style={{ color: C.muted }}>
      {entries.map(([type, n]) => (
        <span key={type} className="whitespace-nowrap">
          {typeLabel(type)} <span style={{ color: C.text }}>{n}</span>
        </span>
      ))}
    </div>
  );
}

function provenanceSummary(data: IncidentsResponse): string {
  const counts = data.counts.provenance;
  if (!counts) return "";
  return PROVENANCE_ORDER.map((p) => `${counts[p] ?? 0} ${p}`).join(", ");
}

export function IncidentLog({ run }: { run: string }) {
  const typeId = useId();
  const sourceId = useId();
  const [type, setType] = useState<IncidentType | "">("");
  const [source, setSource] = useState("");
  const [offset, setOffset] = useState(0);
  const [rows, setRows] = useState<Incident[]>([]);
  const query: IncidentQuery = { type: type || undefined, source: source || undefined, limit: PAGE, offset };
  const state = useApi(["incidents", run, type, source, offset].join("|"), (signal) =>
    api.incidents(run, query, signal),
  );
  // Filter menus come from an unfiltered request, so every choice stays listed.
  const facets = useApi(`incidents|${run}|facets`, (signal) => api.incidents(run, { limit: 1 }, signal));
  const data = dataOf(state);
  const all = dataOf(facets);

  // Pages are fetched with offset and appended; page one replaces the list.
  useEffect(() => {
    if (state.status !== "ok") return;
    const page = state.data;
    setRows((previous) => (page.offset === 0 ? page.incidents : [...previous.slice(0, page.offset), ...page.incidents]));
  }, [state]);

  const typeOptions = Object.keys(all?.counts.type ?? TYPE_LABELS) as IncidentType[];
  const sourceOptions = Object.keys(all?.counts.source ?? {});
  const sourceProvenance = all?.source_provenance ?? data?.source_provenance ?? {};
  const busy = isReloading(state);
  const loadingMore = busy && offset > 0;

  const legend =
    "observed: fixed rules over saved backtest outputs, and the live pipeline's own runs; measured: the drift monitor over real saved forecasts, in M2 and live; simulated: failures injected in the D5 experiment";
  let sub = `Newest first · ${legend}`;
  if (data) {
    const filtered = type || source ? " matching the filters" : "";
    const split = provenanceSummary(data);
    const stamp = data.generated_utc ? `as of ${utcDay(data.generated_utc)}` : "export date not recorded";
    sub = `${data.total} incidents${filtered}${split ? ` (${split})` : ""}, newest first · ${legend} · ${stamp}`;
  }

  const changeFilter = (apply: () => void) => {
    apply();
    setOffset(0);
  };

  return (
    <Panel title="Incident log" sub={sub} busy={busy && !loadingMore}>
      <div className="flex flex-wrap items-end gap-3 mb-2">
        <div>
          <label htmlFor={typeId} className="block text-xs mb-1" style={{ color: C.muted }}>
            Type
          </label>
          <select
            id={typeId}
            value={type}
            onChange={(e) => changeFilter(() => setType(e.target.value as IncidentType | ""))}
            className="rounded-md px-2 py-1.5 text-sm"
            style={selectStyle}
          >
            <option value="">All types</option>
            {typeOptions.map((t) => (
              <option key={t} value={t}>
                {typeLabel(t)} ({data?.counts.type[t] ?? 0})
              </option>
            ))}
          </select>
        </div>
        <div>
          <label htmlFor={sourceId} className="block text-xs mb-1" style={{ color: C.muted }}>
            Source
          </label>
          <select
            id={sourceId}
            value={source}
            onChange={(e) => changeFilter(() => setSource(e.target.value))}
            className="rounded-md px-2 py-1.5 text-sm"
            style={selectStyle}
          >
            <option value="">All sources</option>
            {sourceOptions.map((s) => (
              <option key={s} value={s}>
                {sourceLabel(s, sourceProvenance[s])} ({data?.counts.source[s] ?? 0})
              </option>
            ))}
          </select>
        </div>
      </div>

      {state.status === "error" ? (
        <StatusMessage height={120}>Could not load the incident log: {state.error}</StatusMessage>
      ) : !data ? (
        <StatusMessage height={120}>Loading…</StatusMessage>
      ) : (
        <div
          style={{ opacity: busy && !loadingMore ? 0.55 : 1, transition: "opacity 120ms" }}
          aria-busy={busy || undefined}
        >
          <Counts data={data} />
          {rows.length === 0 ? (
            <StatusMessage height={80}>No incidents match these filters.</StatusMessage>
          ) : (
            <ul className="mt-2">
              {rows.map((incident) => (
                <IncidentRow key={incident.incident_id} incident={incident} />
              ))}
            </ul>
          )}
          <div className="flex flex-wrap items-center gap-3 pt-3" style={{ borderTop: `1px solid ${C.panelEdge}` }}>
            <span className="text-xs tabular-nums" style={{ color: C.muted }}>
              Showing {rows.length} of {data.total}
            </span>
            {rows.length < data.total ? (
              <button
                type="button"
                onClick={() => setOffset(rows.length)}
                disabled={busy}
                className="text-xs rounded-md px-3 py-1.5"
                style={selectStyle}
              >
                {loadingMore ? "Loading…" : `Show ${Math.min(PAGE, data.total - rows.length)} more`}
              </button>
            ) : null}
          </div>
        </div>
      )}
    </Panel>
  );
}
