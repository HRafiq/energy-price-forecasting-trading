import { Bar, CartesianGrid, ComposedChart, Legend, Line, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { DispatchPeriod } from "../../api";
import { num, price } from "../../format";
import { useNarrow } from "../../hooks";
import { axisTick, C, legendStyle, tooltipLabelStyle, tooltipStyle } from "../../theme";
import { intradayAxis, priceAxis, tooltipText } from "./common";

// Three stacked charts, one y-scale each, sharing the period index on x.
// A fixed axis width and equal margins keep the x positions aligned.
const AXIS_WIDTH = 44;
const MARGIN = { top: 4, right: 8, left: 0, bottom: 0 };
const SYNC_ID = "dispatch";

interface DispatchChartProps {
  periods: DispatchPeriod[];
  capacityMwh: number;
}

function Caption({ children }: { children: string }) {
  return (
    <div className="text-xs mt-2" style={{ color: C.muted, paddingLeft: AXIS_WIDTH }}>
      {children}
    </div>
  );
}

export function DispatchChart({ periods, capacityMwh }: DispatchChartProps) {
  const narrow = useNarrow();
  const data = periods.map((p, i) => ({
    i,
    charge: p.charge_mw > 0 ? -p.charge_mw : 0,
    discharge: p.discharge_mw,
    price: p.price,
    soc: capacityMwh > 0 ? (p.soc_mwh / capacityMwh) * 100 : null,
  }));
  const axis = intradayAxis(
    periods.map((p) => p.time),
    narrow ? 6 : 3,
  );
  const prices = priceAxis(periods.map((p) => p.price));
  const tooltip = (format: (n: number) => string) => (
    <Tooltip
      contentStyle={tooltipStyle}
      labelStyle={tooltipLabelStyle}
      labelFormatter={axis.label}
      formatter={(value) => tooltipText(value, format)}
      isAnimationActive={false}
    />
  );
  return (
    <div>
      <ResponsiveContainer width="100%" height={160}>
        <ComposedChart data={data} syncId={SYNC_ID} stackOffset="sign" barCategoryGap="8%" margin={{ ...MARGIN, top: 0 }}>
          <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="i" type="category" hide />
          <YAxis width={AXIS_WIDTH} tick={axisTick} axisLine={false} tickLine={false} />
          {tooltip((n) => `${num(n, 2)} MW`)}
          <ReferenceLine y={0} stroke={C.faint} />
          <Bar stackId="mw" dataKey="charge" name="Charge (MW)" fill={C.charge} isAnimationActive={false} />
          <Bar stackId="mw" dataKey="discharge" name="Discharge (MW)" fill={C.discharge} isAnimationActive={false} />
          <Legend verticalAlign="top" wrapperStyle={{ ...legendStyle, paddingBottom: 4 }} />
        </ComposedChart>
      </ResponsiveContainer>
      <Caption>Realised price, €/MWh</Caption>
      <ResponsiveContainer width="100%" height={100}>
        <ComposedChart data={data} syncId={SYNC_ID} margin={MARGIN}>
          <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="i" type="category" hide />
          <YAxis width={AXIS_WIDTH} domain={prices.domain} ticks={prices.ticks} tick={axisTick} axisLine={false} tickLine={false} />
          {tooltip(price)}
          <Line
            dataKey="price"
            name="Realised price"
            stroke={C.actual}
            strokeOpacity={0.7}
            strokeWidth={1.5}
            dot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <Caption>{`State of charge, % of ${num(capacityMwh, 1)} MWh`}</Caption>
      <ResponsiveContainer width="100%" height={90}>
        <ComposedChart data={data} syncId={SYNC_ID} margin={MARGIN}>
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
          <YAxis
            width={AXIS_WIDTH}
            domain={[0, 100]}
            ticks={[0, 50, 100]}
            tickFormatter={(v: number) => `${v}%`}
            tick={axisTick}
            axisLine={false}
            tickLine={false}
          />
          {tooltip((n) => `${num(n, 0)}%`)}
          <Line dataKey="soc" name="State of charge" stroke={C.soc} strokeWidth={2} dot={false} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
