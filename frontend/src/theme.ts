// Design tokens for the dashboard. Custom colours are applied as inline styles;
// Tailwind handles spacing and layout.
//
// The series hues are the ones the README figures already draw with, so a chart
// in the repository and the same chart on the page are the same colours. Blue,
// orange and aqua were validated as a categorical set on this surface: the worst
// adjacent pair separates by 27.6 unsimulated and 9.2 for deuteranopia. Yellow is
// deliberately not a peer hue here, because it sits 13.7 from orange, close
// enough that full-colour vision struggles to tell them apart.
//
// Status colours never carry meaning alone: each one ships beside a word. The
// fixed status steps are tuned for marks rather than text (warning reads 1.79:1
// on this surface), so the tokens below are darkened to stay legible as words
// while keeping the same meaning.

export const C = {
  bg: "#f2f2ef",
  panel: "#fcfcfb",
  panelEdge: "#e1e0d9",
  inset: "#f7f7f4",
  narr: "#eef3fa",
  narrEdge: "#cfdcec",
  narrText: "#23303f",
  text: "#0b0b0b",
  muted: "#52514e",
  faint: "#898781",
  price: "#eb6834",
  actual: "#0b0b0b",
  band90: "rgba(42, 120, 214, 0.14)",
  band50: "rgba(42, 120, 214, 0.30)",
  bandLine: "#2a78d6",
  charge: "#2a78d6",
  discharge: "#eb6834",
  soc: "#1baf7a",
  pf: "#898781",
  median: "#eb6834",
  quantile: "#1baf7a",
  neg: "#c02a2a",
  good: "#0f7a2e",
  warn: "#a35a00",
  // Regime shift arms, fixed by identity and validated as a set on this surface.
  armFrozen: "#eb6834",
  armQuarterly: "#2a78d6",
  armMonthly: "#1baf7a",
  // Naive baseline: recessive, drawn dotted and labelled directly, so it reads as
  // a reference rather than a fourth series competing with the three hues.
  armNaive: "#898781",
  // The 90% coverage target line: dark, solid, labelled directly.
  target: "#52514e",
} as const;

/** A hex token at partial opacity, so a fill and its edge come from one colour. */
export function withAlpha(hex: string, alpha: number): string {
  const value = hex.replace("#", "");
  const r = parseInt(value.slice(0, 2), 16);
  const g = parseInt(value.slice(2, 4), 16);
  const b = parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/** Opacity of a panel's subtitle and body while a new request replaces the shown result. */
export const BUSY_OPACITY = 0.55;

export const FONT_STACK = "ui-sans-serif, system-ui, sans-serif";

export const tooltipStyle = {
  background: C.panel,
  border: `1px solid ${C.panelEdge}`,
  borderRadius: 8,
  fontSize: 12,
  color: C.text,
};

export const tooltipLabelStyle = { color: C.muted };

export const axisTick = { fill: C.faint, fontSize: 10 };

/** Axis ticks of the Model health charts: muted rather than faint. */
export const healthAxisTick = { fill: C.muted, fontSize: 10 };

export const legendStyle = { fontSize: 11, color: C.muted };
