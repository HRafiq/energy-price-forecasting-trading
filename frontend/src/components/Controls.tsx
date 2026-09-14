import { useId } from "react";
import type { DayEntry, RunInfo, StrategyKey, WindowKey } from "../api";
import type { BatteryGrid } from "../battery";
import { longDay, mw, num, STRATEGY_OPTIONS, strategyShort, WINDOW_OPTIONS } from "../format";
import { C } from "../theme";
import { Panel } from "./Panel";
import { Slider } from "./Slider";

export interface ControlValues {
  power: number;
  duration: number;
  degradation: number;
  strategy: StrategyKey;
  windowKey: WindowKey;
  date: string | null;
}

interface ControlsProps {
  values: ControlValues;
  onChange: (patch: Partial<ControlValues>) => void;
  run: RunInfo;
  grid: BatteryGrid;
  days: DayEntry[] | null;
  daysError: string | null;
}

const selectStyle = { background: C.inset, color: C.text, border: `1px solid ${C.panelEdge}` };

function DayPicker({
  date,
  run,
  days,
  onPick,
}: {
  date: string | null;
  run: RunInfo;
  days: DayEntry[] | null;
  onPick: (date: string) => void;
}) {
  const inputId = useId();
  const traded = (days ?? []).filter((d) => d.traded).map((d) => d.date).sort();
  const prev = date ? [...traded].reverse().find((d) => d < date) : undefined;
  const next = date ? traded.find((d) => d > date) : undefined;
  const entry = date ? days?.find((d) => d.date === date) : undefined;
  const first = run.first_day;
  const last = run.last_day;

  const navButton = (label: string, target: string | undefined, aria: string) => (
    <button
      type="button"
      aria-label={aria}
      disabled={!target}
      onClick={() => target && onPick(target)}
      className="rounded-md px-2 py-1.5 text-sm"
      style={{ ...selectStyle, opacity: target ? 1 : 0.4 }}
    >
      {label}
    </button>
  );

  return (
    <div>
      <label htmlFor={inputId} className="block text-xs mb-1" style={{ color: C.muted }}>
        Delivery day
      </label>
      <div className="flex gap-1.5">
        {navButton("‹", prev, "Previous traded day")}
        <input
          id={inputId}
          type="date"
          value={date ?? ""}
          min={first}
          max={last}
          disabled={!date}
          onChange={(e) => {
            const value = e.target.value;
            if (value && value >= first && value <= last) onPick(value);
          }}
          className="flex-1 min-w-0 rounded-md px-2 py-1.5 text-sm tabular-nums"
          style={selectStyle}
        />
        {navButton("›", next, "Next traded day")}
      </div>
      {date ? (
        <p className="text-xs mt-1" style={{ color: C.muted }}>
          {longDay(date)}
          {entry ? ` · ${entry.window === "holdout" ? "hold-out" : "validation"}` : ""}
          {entry && !entry.traded ? ` · not traded: ${entry.skip_reason ?? "no reason recorded"}` : ""}
        </p>
      ) : null}
    </div>
  );
}

export function Controls({ values, onChange, run, grid, days, daysError }: ControlsProps) {
  const capacity = values.power * values.duration;
  const efficiency = run.reference_battery.round_trip_efficiency;
  return (
    <Panel title="Battery & strategy" sub="Pick a battery, a dispatch dial and a window" className="lg:col-span-1 h-fit">
      <div className="space-y-4">
        <Slider
          label="Power"
          value={values.power}
          onChange={(power) => onChange({ power })}
          min={grid.power.min}
          max={grid.power.max}
          step={grid.power.step}
          unit="MW"
          display={mw}
        />
        <Slider
          label="Duration"
          value={values.duration}
          onChange={(duration) => onChange({ duration })}
          options={grid.durations}
          unit="h"
        />
        <Slider
          label="Degradation cost"
          value={values.degradation}
          onChange={(degradation) => onChange({ degradation })}
          options={grid.wear}
          unit="€/MWh"
        />
        <label className="block">
          <div className="text-xs mb-1" style={{ color: C.muted }}>
            Dispatch against quantile
          </div>
          <select
            value={values.strategy}
            onChange={(e) => {
              const option = STRATEGY_OPTIONS.find((s) => s.key === e.target.value);
              if (option) onChange({ strategy: option.key });
            }}
            className="w-full rounded-md px-2 py-1.5 text-sm"
            style={selectStyle}
          >
            {STRATEGY_OPTIONS.map((s) => (
              <option key={s.key} value={s.key}>
                {s.label}
              </option>
            ))}
          </select>
        </label>
        <label className="block">
          <div className="text-xs mb-1" style={{ color: C.muted }}>
            Window
          </div>
          <select
            value={values.windowKey}
            onChange={(e) => {
              const option = WINDOW_OPTIONS.find((w) => w.key === e.target.value);
              if (option) onChange({ windowKey: option.key });
            }}
            className="w-full rounded-md px-2 py-1.5 text-sm"
            style={selectStyle}
          >
            {WINDOW_OPTIONS.map((w) => (
              <option key={w.key} value={w.key}>
                {w.label}
              </option>
            ))}
          </select>
        </label>
        <DayPicker date={values.date} run={run} days={days} onPick={(date) => onChange({ date })} />
        {daysError ? (
          <p className="text-xs" style={{ color: C.muted }}>
            Could not load days: {daysError}
          </p>
        ) : null}

        <div className="space-y-2 pt-1">
          <p className="text-xs leading-relaxed" style={{ color: C.muted }}>
            Battery: {mw(values.power)} MW / {num(capacity, 1)} MWh, {Math.round(efficiency * 100)}% round-trip
            efficiency. Re-optimised daily on the {strategyShort(values.strategy)} forecast.
          </p>
          <p className="text-xs leading-relaxed" style={{ color: C.muted }}>
            {run.mode}
          </p>
        </div>
      </div>
    </Panel>
  );
}
