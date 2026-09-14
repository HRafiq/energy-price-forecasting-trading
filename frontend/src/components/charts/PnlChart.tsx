import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { PnlPoint, StrategyKey } from "../../api";
import { eur, longDay, monthYear, shortDay, strategyShort } from "../../format";
import { axisTick, C, legendStyle, tooltipLabelStyle, tooltipStyle } from "../../theme";
import { tooltipText } from "./common";

export function PnlChart({ series, selected }: { series: PnlPoint[]; selected: StrategyKey }) {
  const long = series.length > 120;
  const showSelected = selected !== "median";
  return (
    <ResponsiveContainer width="100%" height={260}>
      <ComposedChart data={series} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="date"
          tick={axisTick}
          axisLine={false}
          tickLine={false}
          minTickGap={28}
          tickFormatter={(d: string) => (long ? monthYear(d) : shortDay(d))}
        />
        <YAxis tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v: number) => eur(v)} width={70} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={tooltipLabelStyle}
          labelFormatter={(label) => longDay(String(label))}
          formatter={(value) => tooltipText(value, eur)}
          isAnimationActive={false}
        />
        <Line
          dataKey="perfect_foresight"
          name="Perfect foresight"
          stroke={C.pf}
          strokeWidth={1.5}
          strokeDasharray="6 4"
          dot={false}
          isAnimationActive={false}
        />
        {showSelected ? (
          <Line
            dataKey="selected"
            name={`Quantile-aware (${strategyShort(selected)})`}
            stroke={C.quantile}
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        ) : null}
        <Line
          dataKey="median"
          name={showSelected ? "Median forecast" : "Median forecast (selected)"}
          stroke={C.median}
          strokeWidth={2}
          dot={false}
          isAnimationActive={false}
        />
        <Legend wrapperStyle={legendStyle} />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
