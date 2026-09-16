import { withAlpha } from "../theme";

interface PillProps {
  label: string;
  color: string;
}

/** Status pill: the fill and edge are the label's own colour, faded. */
export function Pill({ label, color }: PillProps) {
  return (
    <span
      className="text-xs rounded-full px-2.5 py-0.5 whitespace-nowrap"
      style={{ background: withAlpha(color, 0.12), color, border: `1px solid ${withAlpha(color, 0.35)}` }}
    >
      {label}
    </span>
  );
}
