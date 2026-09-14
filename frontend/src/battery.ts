import type { RunInfo } from "./api";

/** The battery settings the run's pre-computed grid covers, read from /api/runs. */
export interface BatteryGrid {
  durations: number[];
  wear: number[];
  power: { min: number; max: number; step: number };
}

export interface BatterySettings {
  power: number;
  duration: number;
  degradation: number;
}

function sortedNumbers(values: Array<number | string>): number[] {
  const numbers = values.filter((v): v is number => typeof v === "number" && Number.isFinite(v));
  return [...new Set(numbers)].sort((a, b) => a - b);
}

function nearest(options: number[], target: number): number {
  let best = options[0] ?? target;
  for (const option of options) {
    if (Math.abs(option - target) < Math.abs(best - target)) best = option;
  }
  return best;
}

export function batteryGrid(run: RunInfo): BatteryGrid {
  const ref = run.reference_battery;
  const durations = sortedNumbers(run.grid.durations_h);
  const wear = sortedNumbers(run.grid.degradation_eur_per_mwh);
  const { min, max, step } = run.grid.power_mw;
  return {
    durations: durations.length > 0 ? durations : [ref.capacity_mwh / ref.power_mw],
    wear: wear.length > 0 ? wear : [ref.degradation_eur_per_mwh],
    power: step > 0 && max >= min ? { min, max, step } : { min: ref.power_mw, max: ref.power_mw, step: 1 },
  };
}

/** The reference battery at 1 MW, snapped onto the grid. */
export function defaultBattery(run: RunInfo, grid: BatteryGrid): BatterySettings {
  const ref = run.reference_battery;
  const { min, max, step } = grid.power;
  const clamped = Math.min(max, Math.max(min, 1));
  const power = Number((min + Math.round((clamped - min) / step) * step).toFixed(6));
  return {
    power,
    duration: nearest(grid.durations, ref.capacity_mwh / ref.power_mw),
    degradation: nearest(grid.wear, ref.degradation_eur_per_mwh),
  };
}
