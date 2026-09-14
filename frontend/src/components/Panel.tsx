import type { ReactNode } from "react";
import { BUSY_OPACITY, C } from "../theme";

interface PanelProps {
  title: string;
  sub?: ReactNode;
  /** Dims the subtitle while a new request replaces the result it describes. */
  busy?: boolean;
  className?: string;
  children: ReactNode;
}

export function Panel({ title, sub, busy = false, className = "", children }: PanelProps) {
  return (
    <section
      className={`rounded-lg p-4 min-w-0 ${className}`}
      style={{ background: C.panel, border: `1px solid ${C.panelEdge}` }}
      aria-busy={busy || undefined}
    >
      <header className="mb-3">
        <h2 className="text-sm font-semibold" style={{ color: C.text }}>
          {title}
        </h2>
        {sub ? (
          <p
            className="text-xs mt-0.5 leading-relaxed"
            style={{ color: C.muted, opacity: busy ? BUSY_OPACITY : 1, transition: "opacity 120ms" }}
          >
            {sub}
          </p>
        ) : null}
      </header>
      {children}
    </section>
  );
}
