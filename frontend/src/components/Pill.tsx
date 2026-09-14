interface PillProps {
  label: string;
  color: string;
  rgb: string;
}

/** Status pill; `rgb` is the colour as "r,g,b" for the translucent fill and edge. */
export function Pill({ label, color, rgb }: PillProps) {
  return (
    <span
      className="text-xs rounded-full px-2.5 py-0.5 whitespace-nowrap"
      style={{ background: `rgba(${rgb},0.12)`, color, border: `1px solid rgba(${rgb},0.3)` }}
    >
      {label}
    </span>
  );
}
