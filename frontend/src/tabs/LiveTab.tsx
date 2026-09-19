import type { LiveDay, LiveResponse } from "../api";
import { api } from "../api";
import { Kpi } from "../components/Kpi";
import { Loadable } from "../components/Loadable";
import { Panel } from "../components/Panel";
import { Pill } from "../components/Pill";
import { eur, longDay, num, pct, utcDay } from "../format";
import { useApi } from "../hooks";
import { C } from "../theme";

const STEP_LABELS: Record<string, string> = {
  production: "production model",
  no_weather: "model without weather",
  seasonal_naive: "last week's prices",
  naive: "yesterday's prices",
};

function stepLabel(step: string | null): string {
  if (!step) return "no forecast";
  return STEP_LABELS[step] ?? step.replace(/_/g, " ");
}

/** "13:21", from "2026-09-16 13:21". */
function clockOf(local: string | null): string {
  return local ? local.slice(11) : "n/a";
}

function money(value: number | null): string {
  return value === null ? "n/a" : eur(value);
}

function Totals({ live }: { live: LiveResponse }) {
  const t = live.totals;
  const settledNote = t.settled_days > 0 ? `${t.settled_days} settled ${t.settled_days === 1 ? "day" : "days"}` : "no day settled yet";
  const settledDetail =
    t.settled_days > 0 ? `planned value of those days ${eur(t.planned_value_settled_days_eur)}` : undefined;
  const onTimeShare = t.days > 0 ? pct(t.on_time_days / t.days) : "n/a";
  const productionShare = t.days > 0 ? pct(t.production_days / t.days) : "n/a";
  const drift = live.drift;
  const driftValue = drift ? (drift.warming_up ? `${drift.scored_days} of ${drift.window_days}` : drift.status ?? "n/a") : "not run";
  const driftNote = drift
    ? drift.warming_up
      ? "scored days, warming up"
      : "window full"
    : "not run yet";
  const driftDetail = drift
    ? [
        drift.warming_up ? `alerts need ${drift.window_days} scored days` : null,
        drift.latest
          ? `coverage ${pct(drift.latest.rolling_coverage_90)} · pinball ${num(drift.latest.rolling_pinball_ratio, 2)}x its validation median`
          : null,
        drift.through ? `checked through ${longDay(drift.through)}` : null,
      ]
        .filter((part): part is string => part !== null)
        .join(" · ")
    : "the daily drift check has not run";
  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
      <Kpi label="Live delivery days" value={String(t.days)} note={`from ${longDay(live.live_from)}`} />
      <Kpi
        label={`Bids before the ${live.gate_local} gate`}
        value={onTimeShare}
        note={`${t.on_time_days} of ${t.days} days`}
        tone={t.days > 0 && t.on_time_days < t.days ? C.warn : undefined}
      />
      <Kpi
        label="Production model used"
        value={productionShare}
        note={`${t.production_days} of ${t.days} days`}
        detail={t.days > t.production_days ? "the other days used a fallback rung" : undefined}
        tone={t.days > 0 && t.production_days < t.days ? C.warn : undefined}
      />
      <Kpi label="Settled profit" value={money(t.settled_days > 0 ? t.settled_pnl_eur : null)} note={settledNote} detail={settledDetail} />
      <Kpi
        label="Drift monitor"
        value={driftValue}
        note={driftNote}
        detail={driftDetail}
        tone={drift?.status === "in alert" ? C.warn : undefined}
      />
    </div>
  );
}

function DayRow({ day }: { day: LiveDay }) {
  const late = !day.on_time;
  const fallback = day.step !== "production";
  const open = day.incidents.filter((i) => i.status === "review").length;
  return (
    <tr style={{ borderTop: `1px solid ${C.panelEdge}` }}>
      <td className="px-2 py-1.5 text-left whitespace-nowrap" style={{ color: C.text }}>
        {longDay(day.target_day)}
      </td>
      <td className="px-2 py-1.5 text-left whitespace-nowrap">
        {clockOf(day.issued_local)}{" "}
        <Pill label={late ? "after the gate" : "on time"} color={late ? C.neg : C.good} />
      </td>
      <td className="px-2 py-1.5 text-left whitespace-nowrap">
        {stepLabel(day.step)}
        {day.model_version ? ` v${day.model_version}` : ""}
        {day.refit === "refitted" ? <span style={{ color: C.good }}> · refit</span> : null}
        {fallback && day.feeds_missing.length > 0 ? (
          <span style={{ color: C.muted }}> · missing {day.feeds_missing.join(", ")}</span>
        ) : null}
      </td>
      <td className="px-2 py-1.5 text-right" style={{ color: C.muted }}>
        {money(day.planned_value_eur)}
      </td>
      <td className="px-2 py-1.5 text-right" style={{ color: day.settled ? C.text : C.muted }}>
        {day.settled ? money(day.pnl_eur) : "not settled"}
      </td>
      <td className="px-2 py-1.5 text-right" style={{ color: C.muted }}>
        {day.cycles === null ? "n/a" : num(day.cycles, 2)}
      </td>
      <td className="px-2 py-1.5 text-left whitespace-nowrap">
        {day.incidents.length === 0 ? (
          <span style={{ color: C.muted }}>none</span>
        ) : (
          <span title={day.incidents.map((i) => i.detail).join("\n")}>
            {day.incidents.length} {day.incidents.length === 1 ? "incident" : "incidents"}
            {open > 0 ? <span style={{ color: C.warn }}> · {open} open</span> : null}
          </span>
        )}
      </td>
    </tr>
  );
}

const th = "px-2 py-1 font-normal";

function Record({ data }: { data: LiveResponse }) {
  return (
    <div className="space-y-4">
      <Totals live={data} />
      <Panel
        title="The live record"
        sub={`One row per delivery day the pipeline ran for, newest first, times in ${data.timezone} · exported ${utcDay(data.generated_utc)}. Planned value is what the schedule was worth on its own forecast; settled profit is what it earned at the published prices, net of wear.`}
      >
        {(() =>
            data.days.length === 0 ? (
              <p className="text-xs" style={{ color: C.muted }}>
                No live delivery day recorded yet.
              </p>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs tabular-nums" style={{ color: C.muted, minWidth: 720 }}>
                  <thead>
                    <tr>
                      <th scope="col" className={`${th} text-left`}>Delivery day</th>
                      <th scope="col" className={`${th} text-left`}>Bid issued</th>
                      <th scope="col" className={`${th} text-left`}>Forecast by</th>
                      <th scope="col" className={`${th} text-right`}>Planned</th>
                      <th scope="col" className={`${th} text-right`}>Settled</th>
                      <th scope="col" className={`${th} text-right`}>Cycles</th>
                      <th scope="col" className={`${th} text-left`}>Incidents</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...data.days].reverse().map((day) => (
                      <DayRow key={day.target_day} day={day} />
                    ))}
                  </tbody>
                </table>
              </div>
            ))()}
      </Panel>
      {data.drift && data.drift.skipped.length > 0 ? (
        <p className="text-xs" style={{ color: C.muted }}>
          Days the drift monitor skipped: {data.drift.skipped.map((s) => `${longDay(s.day)} (${s.reason})`).join("; ")}.
        </p>
      ) : null}
    </div>
  );
}

export function LiveTab() {
  const live = useApi("live", (signal) => api.live(signal));
  return (
    <Loadable state={live} height={200}>
      {(data) => <Record data={data} />}
    </Loadable>
  );
}
