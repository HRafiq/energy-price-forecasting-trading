import { C } from "../theme";

interface KpiProps {
  label: string;
  value: string;
  note: string;
  tone?: string;
}

export function Kpi({ label, value, note, tone }: KpiProps) {
  return (
    <div
      className="rounded-lg px-4 py-3 min-w-0"
      style={{ background: C.panel, border: `1px solid ${C.panelEdge}` }}
    >
      <div className="text-xs truncate" style={{ color: C.muted }}>
        {label}
      </div>
      <div className="text-xl font-semibold mt-1 tabular-nums" style={{ color: tone ?? C.text }}>
        {value}
      </div>
      <div className="text-xs mt-0.5 truncate" style={{ color: C.muted }}>
        {note}
      </div>
    </div>
  );
}
