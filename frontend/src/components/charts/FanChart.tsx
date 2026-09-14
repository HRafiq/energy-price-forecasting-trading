import { Area, CartesianGrid, ComposedChart, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ForecastPeriod } from "../../api";
import { price } from "../../format";
import { useNarrow } from "../../hooks";
import { axisTick, C, tooltipLabelStyle, tooltipStyle } from "../../theme";
import { intradayAxis, priceAxis, tooltipText } from "./common";

export function FanChart({ periods }: { periods: ForecastPeriod[] }) {
  const narrow = useNarrow();
  const data = periods.map((p, i) => ({
    i,
    band90: [p.q05, p.q95],
    band50: [p.q25, p.q75],
    median: p.q50,
    actual: p.actual,
  }));
  const axis = intradayAxis(
    periods.map((p) => p.time),
    narrow ? 6 : 3,
  );
  const prices = priceAxis(periods.flatMap((p) => [p.q05, p.q95, p.actual]));
  const zeroInRange = prices.domain[0] <= 0 && prices.domain[1] >= 0;
  return (
    <ResponsiveContainer width="100%" height={250}>
      <ComposedChart data={data} margin={{ top: 4, right: 8, left: -12, bottom: 0 }}>
        <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
        <XAxis
          dataKey="i"
          type="category"
          ticks={axis.ticks}
          interval={0}
          tickFormatter={axis.label}
          tick={axisTick}
          axisLine={false}
          tickLine={false}
        />
        <YAxis domain={prices.domain} ticks={prices.ticks} tick={axisTick} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={tooltipStyle}
          labelStyle={tooltipLabelStyle}
          labelFormatter={axis.label}
          formatter={(value) => tooltipText(value, price)}
          isAnimationActive={false}
        />
        {zeroInRange ? <ReferenceLine y={0} stroke={C.neg} strokeDasharray="4 3" /> : null}
        <Area dataKey="band90" name="90% interval" fill={C.band90} stroke="none" isAnimationActive={false} />
        <Area dataKey="band50" name="50% interval" fill={C.band50} stroke="none" isAnimationActive={false} />
        <Line dataKey="median" name="Median forecast" stroke={C.bandLine} strokeWidth={2} dot={false} isAnimationActive={false} />
        <Line
          dataKey="actual"
          name="Realised"
          stroke={C.actual}
          strokeWidth={1.5}
          strokeDasharray="5 3"
          dot={false}
          connectNulls={false}
          isAnimationActive={false}
        />
      </ComposedChart>
    </ResponsiveContainer>
  );
}
