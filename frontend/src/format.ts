import type { StrategyKey, WindowKey } from "./api";

const MINUS = "−";

const eurWhole = new Intl.NumberFormat("en-GB", { maximumFractionDigits: 0 });

/** € total with thousands separators and no decimals: "€ 9,413", "−€ 55". */
export function eur(value: number): string {
  const rounded = Math.round(value);
  const sign = rounded < 0 ? MINUS : "";
  return `${sign}€ ${eurWhole.format(Math.abs(rounded))}`;
}

/** Share (0..1) as a percentage with one decimal: "91.2%". */
export function pct(share: number): string {
  return `${(share * 100).toFixed(1)}%`;
}

/** Plain number with fixed decimals and a real minus sign. */
export function num(value: number, decimals: number): string {
  const text = Math.abs(value).toFixed(decimals);
  return value < 0 && Number(text) !== 0 ? `${MINUS}${text}` : text;
}

/** Price in €/MWh with one decimal. */
export function price(value: number): string {
  return `${num(value, 1)} €/MWh`;
}

/** MW figures on the controls: "0.5", "1", "2.5". */
export function mw(value: number): string {
  return Number.isInteger(value) ? String(value) : value.toFixed(1);
}

// Fixed three-letter names: Intl's en-GB output varies by ICU version ("Sep" vs "Sept").
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

function isoDate(date: string): Date {
  return new Date(`${date}T00:00:00Z`);
}

/** "Fri 21 Nov 2025" */
export function longDay(date: string): string {
  const d = isoDate(date);
  if (Number.isNaN(d.getTime())) return date;
  return `${WEEKDAYS[d.getUTCDay()]} ${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** "21 Nov" */
export function shortDay(date: string): string {
  const d = isoDate(date);
  if (Number.isNaN(d.getTime())) return date;
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`;
}

/** "Nov 25" */
export function monthYear(date: string): string {
  const d = isoDate(date);
  if (Number.isNaN(d.getTime())) return date;
  return `${MONTHS[d.getUTCMonth()]} ${String(d.getUTCFullYear()).slice(2)}`;
}

/** "2025-11-20 11:40" -> "11:40" */
export function clock(localTimestamp: string): string {
  const match = /(\d{2}:\d{2})/.exec(localTimestamp);
  return match?.[1] ?? localTimestamp;
}

/** "18-20" -> "18 to 20" */
export function blockLabel(block: string): string {
  return block.replace("-", " to ");
}

export function hourLabel(hour: number): string {
  return String(hour).padStart(2, "0");
}

export const WINDOW_OPTIONS: Array<{ key: WindowKey; label: string; phrase: string }> = [
  { key: "last30", label: "Last 30 days", phrase: "last 30 days" },
  { key: "last90", label: "Last 90 days", phrase: "last 90 days" },
  { key: "validation", label: "Validation window", phrase: "validation window" },
  { key: "holdout", label: "Hold-out", phrase: "hold-out" },
  { key: "all", label: "All days", phrase: "all days" },
];

export function windowPhrase(key: WindowKey): string {
  return WINDOW_OPTIONS.find((w) => w.key === key)?.phrase ?? key;
}

export const STRATEGY_OPTIONS: Array<{ key: StrategyKey; label: string; short: string }> = [
  { key: "median", label: "q0.50: median (neutral)", short: "median" },
  { key: "q25", label: "q0.25: conservative buy", short: "q0.25" },
  { key: "q10", label: "q0.10: very conservative", short: "q0.10" },
];

export function strategyShort(key: StrategyKey): string {
  return STRATEGY_OPTIONS.find((s) => s.key === key)?.short ?? key;
}
