// Design tokens from the approved dashboard mockup: "grid control room at night".
// Custom colours are applied as inline styles; Tailwind handles spacing and layout.

export const C = {
  bg: "#0d1522",
  panel: "#141f30",
  panelEdge: "#1f2d42",
  inset: "#0b1220",
  narr: "#1a2740",
  narrEdge: "#2a3d5f",
  narrText: "#c3cfe0",
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
  // Regime shift arms, fixed by identity (validated for dark-surface and colour-blind separation).
  armFrozen: "#d0735a",
  armQuarterly: "#4f8fc5",
  armMonthly: "#35a877",
  // Naive baseline: lighter than the target line and drawn dotted; 6.2:1 on the panel.
  armNaive: "#8f9fb8",
  // The 90% coverage target line: near-white, solid, labelled directly.
  target: "#c3cfe0",
} as const;

/** Opacity of a panel's subtitle and body while a new request replaces the shown result. */
export const BUSY_OPACITY = 0.55;

export const FONT_STACK = "ui-sans-serif, system-ui, sans-serif";

export const tooltipStyle = {
  background: C.inset,
  border: `1px solid ${C.panelEdge}`,
  borderRadius: 8,
  fontSize: 12,
  color: C.text,
};

export const tooltipLabelStyle = { color: C.muted };

export const axisTick = { fill: C.faint, fontSize: 10 };

/** Axis ticks of the Model health charts: muted (4.98:1 on the panel) rather than faint. */
export const healthAxisTick = { fill: C.muted, fontSize: 10 };

export const legendStyle = { fontSize: 11, color: C.muted };
