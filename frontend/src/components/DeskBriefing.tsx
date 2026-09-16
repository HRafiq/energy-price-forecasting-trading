import { useState } from "react";
import { api, type BatteryQuery, type NarrateRequest, type NarrateResponse, type NarrateTab, type WindowKey } from "../api";
import { useApi } from "../hooks";
import { C } from "../theme";

/** The questions offered before the first briefing comes back and names its own. */
const DEFAULT_FOLLOW_UPS: Record<string, string> = {
  why_this_dispatch: "Why this dispatch?",
  what_changed: "What changed vs yesterday?",
  explain_the_miss: "Explain the miss",
};

interface DeskBriefingProps {
  tab: NarrateTab;
  run: string | null;
  /** Only the overview tab is about one day; elsewhere the API takes the last one. */
  date?: string | null;
  windowKey?: WindowKey;
  battery?: BatteryQuery;
}

/**
 * A few sentences about the page, written from the same numbers the page shows.
 *
 * The prose is composed on request and checked before it arrives: a figure that is
 * not in the payload is dropped by the API, which answers with its deterministic
 * writer instead and says so. This component therefore never shows an unchecked
 * number, and it labels who wrote what the reader is looking at.
 */
export function DeskBriefing({ tab, run, date = null, windowKey, battery }: DeskBriefingProps) {
  const [question, setQuestion] = useState<string | null>(null);
  const batteryKey = battery ? [battery.power, battery.duration, battery.degradation, battery.strategy].join(",") : "";
  const key = run ? ["narrate", tab, run, date ?? "", windowKey ?? "", batteryKey, question ?? ""].join("|") : null;

  const state = useApi<NarrateResponse>(key, (signal) => {
    if (!run) return Promise.reject(new Error("no run"));
    const request: NarrateRequest = { tab, run };
    if (date) request.date = date;
    if (windowKey) request.window = windowKey;
    if (battery) {
      request.duration = battery.duration;
      request.degradation = battery.degradation;
      request.strategy = battery.strategy;
      request.power = battery.power;
    }
    if (question) request.question = question;
    return api.narrate(request, signal);
  });

  const briefing = state.status === "ok" ? state.data : state.status === "loading" ? state.previous : null;
  const followUps = briefing?.follow_ups ?? DEFAULT_FOLLOW_UPS;
  const waiting = state.status === "loading";
  const writer = briefing
    ? briefing.provider === "template"
      ? "written from this page's numbers"
      : `written by ${briefing.model ?? briefing.provider}, every figure checked against this page`
    : "reading this page";

  return (
    <section className="rounded-lg p-4" style={{ background: C.narr, border: `1px solid ${C.narrEdge}` }}>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 mb-2">
        <span className="text-sm font-semibold" style={{ color: C.text }}>
          Desk briefing
        </span>
        <span className="text-xs" style={{ color: C.muted }}>
          {waiting ? "writing..." : writer}
        </span>
      </div>

      {state.status === "error" ? (
        <p className="text-sm leading-relaxed" style={{ color: C.muted }}>
          No briefing: {state.error}
        </p>
      ) : (
        <p className="text-sm leading-relaxed" style={{ color: C.narrText, opacity: waiting && briefing ? 0.55 : 1 }}>
          {briefing ? briefing.text : "Reading the numbers on this page."}
        </p>
      )}

      {briefing?.fell_back && (
        <p className="text-xs mt-2" style={{ color: C.muted }}>
          {briefing.rejected.length > 0
            ? `Written here instead: the model's draft named ${briefing.rejected.join(", ")}, which is not a number on this page.`
            : `Written here instead: ${briefing.fallback_reason ?? "the model did not answer"}.`}
        </p>
      )}

      <div className="flex flex-wrap gap-2 mt-3">
        {Object.entries(followUps).map(([id, label]) => {
          const active = question === label;
          return (
            <button
              key={id}
              type="button"
              disabled={!run}
              onClick={() => setQuestion(active ? null : label)}
              title={active ? "Back to the briefing" : label}
              className="text-xs rounded-full px-3 py-1"
              style={{
                background: active ? "rgba(79,143,197,0.22)" : "rgba(79,143,197,0.08)",
                color: "#9dc3e2",
                border: `1px solid rgba(79,143,197,${active ? 0.6 : 0.25})`,
                opacity: run ? 1 : 0.5,
                cursor: run ? "pointer" : "default",
              }}
            >
              {label}
            </button>
          );
        })}
      </div>
    </section>
  );
}
