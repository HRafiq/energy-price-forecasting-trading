import { C } from "../theme";

const FOLLOW_UPS = ["Why this dispatch?", "What changed vs yesterday?", "Explain the miss"];

/** Placeholder until the narration phase; it never shows invented prose. */
export function DeskBriefing() {
  return (
    <section className="rounded-lg p-4" style={{ background: C.narr, border: `1px solid ${C.narrEdge}` }}>
      <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0.5 mb-2">
        <span className="text-sm font-semibold" style={{ color: C.text }}>
          Desk briefing
        </span>
        <span className="text-xs" style={{ color: C.muted }}>
          placeholder · narration not yet enabled
        </span>
      </div>
      <p className="text-sm leading-relaxed" style={{ color: C.narrText }}>
        Desk briefing arrives in a later phase. It will narrate only the numbers shown on this page.
      </p>
      <div className="flex flex-wrap gap-2 mt-3">
        {FOLLOW_UPS.map((q) => (
          <button
            key={q}
            type="button"
            disabled
            title="Available when the desk briefing ships"
            className="text-xs rounded-full px-3 py-1"
            style={{
              background: "rgba(79,143,197,0.08)",
              color: "#9dc3e2",
              border: "1px solid rgba(79,143,197,0.25)",
              opacity: 0.5,
            }}
          >
            {q}
          </button>
        ))}
      </div>
    </section>
  );
}
