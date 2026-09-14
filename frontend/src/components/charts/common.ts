/** Tooltip value text for numbers and [low, high] ranges. */
export function tooltipText(value: unknown, format: (n: number) => string): string {
  if (typeof value === "number") return format(value);
  if (Array.isArray(value) && value.length === 2) {
    const [lo, hi] = value as unknown[];
    if (typeof lo === "number" && typeof hi === "number") return `${format(lo)} to ${format(hi)}`;
  }
  return value === null || value === undefined ? "n/a" : String(value);
}

/**
 * Intraday charts use the period index as the x value because local time labels
 * repeat on the autumn DST day (02:00 to 02:45 appear twice). These helpers map
 * indices back to time labels.
 */
export interface IntradayAxis {
  ticks: number[];
  label: (index: unknown) => string;
}

export function intradayAxis(times: string[], everyHours: number): IntradayAxis {
  const seen = new Map<string, number>();
  const occurrence: number[] = [];
  const ticks: number[] = [];
  times.forEach((time, i) => {
    const n = (seen.get(time) ?? 0) + 1;
    seen.set(time, n);
    occurrence.push(n);
    const match = /^(\d{2}):00$/.exec(time);
    if (match?.[1] !== undefined && n === 1 && Number(match[1]) % everyHours === 0) ticks.push(i);
  });
  const label = (index: unknown): string => {
    const i = typeof index === "number" ? index : Number(index);
    const time = times[i];
    if (time === undefined) return "";
    return (occurrence[i] ?? 1) > 1 ? `${time} (repeated hour)` : time;
  };
  return { ticks, label };
}

/** A price axis from the data: a padded domain and round-number ticks inside it. */
export interface PriceAxis {
  domain: [number, number];
  ticks: number[];
}

/**
 * Fits the axis to the data instead of starting at zero, so a fan around €100 is
 * not squashed. Ticks are multiples of 1, 2 or 5 times a power of ten. Nulls are ignored.
 */
export function priceAxis(values: Array<number | null>): PriceAxis {
  let lo = Infinity;
  let hi = -Infinity;
  for (const v of values) {
    if (v === null || !Number.isFinite(v)) continue;
    lo = Math.min(lo, v);
    hi = Math.max(hi, v);
  }
  if (lo > hi) {
    lo = 0;
    hi = 100;
  }
  const pad = Math.max((hi - lo) * 0.05, 5);
  const domain: [number, number] = [Math.floor((lo - pad) / 10) * 10, Math.ceil((hi + pad) / 10) * 10];
  const rough = (domain[1] - domain[0]) / 5;
  const magnitude = 10 ** Math.floor(Math.log10(rough));
  const step = ([1, 2, 5, 10].find((m) => m * magnitude >= rough) ?? 10) * magnitude;
  const ticks: number[] = [];
  for (let t = Math.ceil(domain[0] / step) * step; t <= domain[1]; t += step) ticks.push(Number(t.toFixed(6)));
  return { domain, ticks };
}
