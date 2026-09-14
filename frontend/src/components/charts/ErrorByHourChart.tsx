import { Bar, CartesianGrid, ComposedChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { HourError } from "../../api";
import { eur, hourLabel, price } from "../../format";
import { axisTick, C, tooltipLabelStyle, tooltipStyle } from "../../theme";
import { tooltipText } from "./common";

// Two stacked charts, one y-scale each, sharing the local hour on x.
const AXIS_WIDTH = 52;
const MARGIN = { top: 4, right: 8, left: 0, bottom: 0 };
const SYNC_ID = "error-by-hour";

function Caption({ children }: { children: string }) {
  return (
    <div className="text-xs mb-1" style={{ color: C.muted, paddingLeft: AXIS_WIDTH }}>
      {children}
    </div>
  );
}

export function ErrorByHourChart({ byHour }: { byHour: HourError[] }) {
  const data = [...byHour]
    .sort((a, b) => a.hour - b.hour)
    .map((h) => ({ hour: hourLabel(h.hour), mae: h.mae_eur_mwh, stake: h.value_at_stake_eur_per_day }));
  const tooltip = (format: (n: number) => string) => (
    <Tooltip
      contentStyle={tooltipStyle}
      labelStyle={tooltipLabelStyle}
      labelFormatter={(label) => `Hour ${String(label)}`}
      formatter={(value) => tooltipText(value, format)}
      isAnimationActive={false}
    />
  );
  return (
    <div>
      <Caption>Mean absolute error of the median forecast, €/MWh</Caption>
      <ResponsiveContainer width="100%" height={110}>
        <ComposedChart data={data} syncId={SYNC_ID} margin={MARGIN}>
          <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="hour" hide />
          <YAxis width={AXIS_WIDTH} tick={axisTick} axisLine={false} tickLine={false} />
          {tooltip(price)}
          <Bar dataKey="mae" name="MAE" fill={C.bandLine} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-2">
        <Caption>Value at stake, € per day, reference battery</Caption>
      </div>
      <ResponsiveContainer width="100%" height={125}>
        <ComposedChart data={data} syncId={SYNC_ID} margin={MARGIN}>
          <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="hour" tick={axisTick} interval={2} axisLine={false} tickLine={false} />
          <YAxis width={AXIS_WIDTH} tick={axisTick} axisLine={false} tickLine={false} tickFormatter={(v: number) => eur(v)} />
          {tooltip((n) => `${eur(n)} per day`)}
          <Bar dataKey="stake" name="Value at stake" fill={C.price} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
