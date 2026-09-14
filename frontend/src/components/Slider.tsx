import { C } from "../theme";

interface SliderBase {
  label: string;
  value: number;
  onChange: (value: number) => void;
  unit: string;
  display?: (value: number) => string;
}

/** Either an evenly stepped range or a fixed list of allowed values. */
type SliderProps = SliderBase & ({ min: number; max: number; step: number } | { options: number[] });

export function Slider(props: SliderProps) {
  const { label, value, onChange, unit, display } = props;
  const text = `${display ? display(value) : value} ${unit}`;
  let range: { min: number; max: number; step: number; position: number; toValue: (raw: number) => number };
  if ("options" in props) {
    const { options } = props;
    range = {
      min: 0,
      max: Math.max(0, options.length - 1),
      step: 1,
      position: Math.max(0, options.indexOf(value)),
      toValue: (raw) => options[raw] ?? value,
    };
  } else {
    range = { min: props.min, max: props.max, step: props.step, position: value, toValue: (raw) => raw };
  }
  return (
    <label className="block">
      <div className="flex justify-between text-xs mb-1">
        <span style={{ color: C.muted }}>{label}</span>
        <span className="tabular-nums" style={{ color: C.text }}>
          {text}
        </span>
      </div>
      <input
        type="range"
        min={range.min}
        max={range.max}
        step={range.step}
        value={range.position}
        aria-valuetext={text}
        onChange={(e) => onChange(range.toValue(Number(e.target.value)))}
        className="w-full"
        style={{ accentColor: C.price }}
      />
    </label>
  );
}
