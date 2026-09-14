import { CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { CalibrationResponse } from "../../api";
import { axisTick, C, legendStyle, tooltipLabelStyle, tooltipStyle } from "../../theme";
import { tooltipText } from "./common";

const IDEAL: readonly [{ x: number; y: number }, { x: number; y: number }] = [
  { x: 0, y: 0 },
  { x: 100, y: 100 },
];

export function CalibrationChart({ quantiles }: { quantiles: CalibrationResponse["quantiles"] }) {
  const data = [...quantiles]
    .sort((a, b) => a.level - b.level)
    .map((q) => ({
      nominal: Math.round(q.level * 1000) / 10,
      empirical: Math.round(q.empirical * 1000) / 10,
    }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" />
        <XAxis
          dataKey="nominal"
          type="number"
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tick={axisTick}
          axisLine={false}
          tickLine={false}
          tickFormatter={(v: number) => `${v}%`}
        />
        <YAxis
          tick={axisTick}
          axisLine={false}
          tickLine={false}
          domain={[0, 100]}
          ticks={[0, 25, 50, 75, 100]}
          tickFormatter={(v: number) => `${v}%`}
        />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={tooltipLabelStyle}
          labelFormatter={(label) => `Quantile level ${String(label)}%`}
          formatter={(value) => tooltipText(value, (n) => `${n.toFixed(1)}%`)}
          isAnimationActive={false}
        />
        <ReferenceLine segment={IDEAL} stroke={C.faint} strokeDasharray="4 4" />
        <Line
          dataKey="empirical"
          name="Share of prices at or below"
          stroke={C.quantile}
          strokeWidth={2}
          dot={{ r: 3, fill: C.quantile }}
          isAnimationActive={false}
        />
        <Legend wrapperStyle={legendStyle} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
