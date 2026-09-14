import { Bar, CartesianGrid, ComposedChart, Legend, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { AsymmetryBlock } from "../../api";
import { blockLabel, eur } from "../../format";
import { axisTick, C, legendStyle, tooltipLabelStyle, tooltipStyle } from "../../theme";
import { tooltipText } from "./common";

export function AsymmetryChart({ blocks }: { blocks: AsymmetryBlock[] }) {
  const data = blocks.map((b) => ({ block: b.block, over: b.over_eur, under: b.under_eur }));
  return (
    <ResponsiveContainer width="100%" height={220}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
        <XAxis dataKey="block" tick={axisTick} axisLine={false} tickLine={false} />
        <YAxis tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v: number) => eur(v)} width={70} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={tooltipLabelStyle}
          labelFormatter={(label) => `Hours ${blockLabel(String(label))}`}
          formatter={(value) => tooltipText(value, eur)}
          isAnimationActive={false}
        />
        <ReferenceLine y={0} stroke={C.faint} />
        <Bar dataKey="over" name="Over-forecast cost" fill={C.median} isAnimationActive={false} />
        <Bar dataKey="under" name="Under-forecast cost" fill={C.bandLine} isAnimationActive={false} />
        <Legend wrapperStyle={legendStyle} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
