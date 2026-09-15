import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  usePlotArea,
  useYAxisScale,
  XAxis,
  YAxis,
} from "recharts";
import type { RegimeResponse } from "../../api";
import { longDay, monthYear, pctPoints } from "../../format";
import { useNarrow } from "../../hooks";
import { C, healthAxisTick, tooltipLabelStyle, tooltipStyle } from "../../theme";
import { tooltipText } from "./common";

/** Arms in drawing order, each with a colour fixed by identity. */
export const REGIME_ARMS = [
  { key: "frozen", label: "Frozen", color: C.armFrozen },
  { key: "quarterly", label: "Refit quarterly", color: C.armQuarterly },
  { key: "monthly", label: "Refit monthly", color: C.armMonthly },
  { key: "naive_previous_day", label: "Naive baseline", color: C.armNaive },
] as const;

export type RegimeArmKey = (typeof REGIME_ARMS)[number]["key"];

/** Dash pattern of the naive baseline: dotted, unlike the solid arms and the target. */
export const NAIVE_DASH = "1 3";
const LABEL_GAP_PX = 13;

export function armLabel(key: string): string {
  return REGIME_ARMS.find((a) => a.key === key)?.label ?? key;
}

/** First date of each January and July in the series. */
function halfYearTicks(dates: string[]): string[] {
  const ticks: string[] = [];
  let lastMonth = "";
  for (const d of dates) {
    const month = d.slice(0, 7);
    if (month !== lastMonth && (month.endsWith("-01") || month.endsWith("-07"))) ticks.push(d);
    lastMonth = month;
  }
  return ticks;
}

interface EndLabel {
  text: string;
  color: string;
  value: number;
  /** Draw a coloured dot before the text; the target label has none. */
  dot?: boolean;
}

/**
 * Direct labels at the right edge, one per line's last value, pushed apart so no
 * two overlap: sorted top to bottom, each at least LABEL_GAP_PX below the one above,
 * then shifted up as a group if the last one falls below the plot.
 */
function EndLabels({ labels }: { labels: EndLabel[] }) {
  const area = usePlotArea();
  const scale = useYAxisScale();
  if (!area || !scale) return null;
  const placed = labels
    .map((label) => ({ ...label, y: Number(scale(label.value)) }))
    .filter((label) => Number.isFinite(label.y))
    .sort((a, b) => a.y - b.y);
  for (let i = 1; i < placed.length; i++) {
    const above = placed[i - 1];
    const current = placed[i];
    if (above && current && current.y - above.y < LABEL_GAP_PX) current.y = above.y + LABEL_GAP_PX;
  }
  const bottom = area.y + area.height;
  const last = placed[placed.length - 1];
  const overflow = last ? Math.max(0, last.y - bottom) : 0;
  const x = area.x + area.width + 6;
  return (
    <g aria-hidden="true">
      {placed.map((label) => (
        <text key={label.text} x={x} y={label.y - overflow} dy={4} fontSize={10} fill={C.muted}>
          {label.dot === false ? (
            <tspan fill={label.color}>{label.text}</tspan>
          ) : (
            <>
              <tspan fill={label.color}>● </tspan>
              {label.text}
            </>
          )}
        </text>
      ))}
    </g>
  );
}

export function RegimeCoverageChart({ data, showNaive }: { data: RegimeResponse; showNaive: boolean }) {
  const narrow = useNarrow();
  const { dates, arms, window_days: windowDays } = data.rolling_coverage_90;
  const rows = dates.map((date, i) => {
    const row: Record<string, string | number | null> = { date };
    for (const arm of REGIME_ARMS) {
      const v = arms[arm.key]?.[i];
      row[arm.key] = v === null || v === undefined ? null : v * 100;
    }
    return row;
  });
  const target = data.coverage_target * 100;
  const shown = REGIME_ARMS.filter((a) => a.key !== "naive_previous_day" || showNaive).filter((a) => arms[a.key]);
  const endLabels: EndLabel[] = shown.flatMap((arm) => {
    const values = arms[arm.key] ?? [];
    const last = values[values.length - 1];
    return last === null || last === undefined ? [] : [{ text: arm.label, color: arm.color, value: last * 100 }];
  });
  // The target is labelled at the right edge with the arms, so the spacing keeps it clear of them.
  const targetText = `${Math.round(target)}% target`;
  endLabels.push({ text: targetText, color: C.target, value: target, dot: false });
  return (
    <ResponsiveContainer width="100%" height={narrow ? 240 : 280}>
      <ComposedChart data={rows} margin={{ top: 8, right: narrow ? 8 : 104, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="date"
          ticks={halfYearTicks(dates)}
          interval={0}
          tickFormatter={(d: string) => monthYear(d)}
          tick={healthAxisTick}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tickFormatter={(v: number) => `${v}%`}
          tick={healthAxisTick}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={tooltipLabelStyle}
          labelFormatter={(label) => `${windowDays} days to ${longDay(String(label))}`}
          formatter={(value) => tooltipText(value, pctPoints)}
          isAnimationActive={false}
        />
        <ReferenceLine
          y={target}
          stroke={C.target}
          strokeWidth={1}
          ifOverflow="extendDomain"
          label={
            narrow ? { value: targetText, position: "insideTopRight", fill: C.target, fontSize: 10 } : undefined
          }
        />
        {shown.map((arm) => {
          const naive = arm.key === "naive_previous_day";
          return (
            <Line
              key={arm.key}
              dataKey={arm.key}
              name={arm.label}
              stroke={arm.color}
              strokeWidth={1.5}
              strokeDasharray={naive ? NAIVE_DASH : undefined}
              strokeLinecap={naive ? "round" : undefined}
              dot={false}
              activeDot={{ r: 3 }}
              connectNulls={false}
              isAnimationActive={false}
            />
          );
        })}
        {narrow ? null : <EndLabels labels={endLabels} />}
      </ComposedChart>
    </ResponsiveContainer>
  );
}
