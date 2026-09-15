import {
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceArea,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { DriftEpisode, DriftResponse } from "../../api";
import { longDay, monthYear, num, pct, pctPoints, shortDay } from "../../format";
import { useNarrow } from "../../hooks";
import { C, healthAxisTick as axisTick, tooltipLabelStyle, tooltipStyle } from "../../theme";

// Two stacked charts, one y-scale each, sharing the delivery day on x.
const AXIS_WIDTH = 44;
const MARGIN = { top: 8, right: 8, left: 0, bottom: 0 };
const SYNC_ID = "drift-monitor";
const ALERT_FILL = "rgba(224, 104, 92, 0.16)";

interface Row {
  date: string;
  coverage: number | null;
  coverageAlert: number | null;
  ratio: number | null;
  ratioAlert: number | null;
  inCoverageAlert: boolean;
  inRatioAlert: boolean;
  window: string;
}

function Caption({ children }: { children: string }) {
  return (
    <div className="text-xs mb-1" style={{ color: C.muted, paddingLeft: AXIS_WIDTH }}>
      {children}
    </div>
  );
}

/** First date of every third month (every month for a short series). */
function monthTicks(dates: string[]): string[] {
  const every = dates.length > 200 ? 3 : 1;
  const ticks: string[] = [];
  let lastMonth = "";
  for (const d of dates) {
    const month = d.slice(0, 7);
    if (month !== lastMonth && (Number(month.slice(5)) - 1) % every === 0) ticks.push(d);
    lastMonth = month;
  }
  return ticks;
}

function niceFloor(value: number, step: number): number {
  return Math.floor(value / step) * step;
}

function TooltipBody({
  active,
  payload,
  signal,
}: {
  active?: boolean;
  payload?: ReadonlyArray<{ payload?: Row }>;
  signal: "coverage" | "ratio";
}) {
  const row = payload?.[0]?.payload;
  if (!active || !row) return null;
  const value = signal === "coverage" ? row.coverage : row.ratio;
  const alert = signal === "coverage" ? row.inCoverageAlert : row.inRatioAlert;
  const text =
    value === null ? "n/a" : signal === "coverage" ? `${pctPoints(value)} coverage` : `pinball ratio ${num(value, 2)}`;
  return (
    <div style={{ ...tooltipStyle, padding: "6px 8px" }}>
      <div style={tooltipLabelStyle}>
        {longDay(row.date)} · {row.window === "holdout" ? "hold-out" : "validation"}
      </div>
      <div className="tabular-nums">{text}</div>
      {alert ? <div style={{ color: C.neg }}>alert day</div> : null}
    </div>
  );
}

function episodeAreas(episodes: DriftEpisode[], signal: DriftEpisode["signal"], dates: Set<string>) {
  return episodes
    .filter((e) => e.signal === signal && dates.has(e.start) && dates.has(e.end))
    .map((e) => (
      <ReferenceArea key={`${signal}-${e.start}`} x1={e.start} x2={e.end} fill={ALERT_FILL} stroke="none" ifOverflow="hidden" />
    ));
}

export function DriftMonitorChart({ data }: { data: DriftResponse }) {
  const narrow = useNarrow();
  const rows: Row[] = data.series.map((p) => {
    const coverage = p.rolling_coverage_90 === null ? null : p.rolling_coverage_90 * 100;
    const ratio = p.rolling_pinball_ratio;
    return {
      date: p.target_day,
      window: p.window,
      coverage,
      coverageAlert: p.coverage_alert ? coverage : null,
      ratio,
      ratioAlert: p.pinball_alert ? ratio : null,
      inCoverageAlert: p.coverage_alert,
      inRatioAlert: p.pinball_alert,
    };
  });
  const dates = rows.map((r) => r.date);
  const dateSet = new Set(dates);
  const ticks = monthTicks(dates);
  const long = rows.length > 200;
  const tickFormatter = (d: string) => (long ? monthYear(d) : shortDay(d));
  const coverageThreshold = data.thresholds.coverage * 100;
  const ratioThreshold = data.thresholds.pinball_ratio;
  const coverageValues = rows.flatMap((r) => (r.coverage === null ? [] : [r.coverage]));
  const ratioValues = rows.flatMap((r) => (r.ratio === null ? [] : [r.ratio]));
  const coverageLow = Math.max(0, niceFloor(Math.min(coverageThreshold, ...coverageValues) - 5, 10));
  const coverageTicks: number[] = [];
  for (let t = coverageLow; t <= 100; t += coverageLow < 50 ? 25 : 10) coverageTicks.push(t);
  const ratioHigh = Math.ceil((Math.max(ratioThreshold, ...ratioValues) + 0.1) * 2) / 2;
  const holdoutMarked = dateSet.has(data.holdout_start) && dates[0] !== data.holdout_start;
  const holdoutLine = holdoutMarked ? (
    <ReferenceLine
      x={data.holdout_start}
      stroke={C.muted}
      strokeDasharray="2 3"
      label={{ value: "hold-out starts", position: "insideTopRight", fill: C.muted, fontSize: 10 }}
    />
  ) : null;
  const height = narrow ? 150 : 170;

  return (
    <div>
      <Caption>{`Rolling ${data.window_days}-day 90% interval coverage`}</Caption>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={rows} syncId={SYNC_ID} margin={MARGIN}>
          <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
          <XAxis dataKey="date" ticks={ticks} interval={0} hide />
          <YAxis
            width={AXIS_WIDTH}
            domain={[coverageLow, 100]}
            ticks={coverageTicks}
            tickFormatter={(v: number) => `${v}%`}
            tick={axisTick}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<TooltipBody signal="coverage" />} isAnimationActive={false} />
          {episodeAreas(data.episodes, "coverage", dateSet)}
          <ReferenceLine
            y={coverageThreshold}
            stroke={C.neg}
            strokeDasharray="4 3"
            label={{ value: `alert below ${pct(data.thresholds.coverage)}`, position: "insideBottomLeft", fill: C.muted, fontSize: 10 }}
          />
          {holdoutLine}
          <Line dataKey="coverage" name="Rolling coverage" stroke={C.bandLine} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line
            dataKey="coverageAlert"
            name="Alert days"
            stroke={C.neg}
            strokeWidth={2}
            dot={false}
            activeDot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="mt-3">
        <Caption>{`Rolling ${data.window_days}-day pinball loss over its validation median`}</Caption>
      </div>
      <ResponsiveContainer width="100%" height={height + 20}>
        <ComposedChart data={rows} syncId={SYNC_ID} margin={MARGIN}>
          <CartesianGrid stroke={C.panelEdge} strokeDasharray="2 4" vertical={false} />
          <XAxis
            dataKey="date"
            ticks={ticks}
            interval={narrow && long ? 1 : 0}
            tickFormatter={tickFormatter}
            tick={axisTick}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            width={AXIS_WIDTH}
            domain={[0, ratioHigh]}
            ticks={Array.from({ length: Math.round(ratioHigh * 2) + 1 }, (_, i) => i / 2)}
            tickFormatter={(v: number) => num(v, 1)}
            tick={axisTick}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip content={<TooltipBody signal="ratio" />} isAnimationActive={false} />
          {episodeAreas(data.episodes, "pinball", dateSet)}
          <ReferenceLine
            y={ratioThreshold}
            stroke={C.neg}
            strokeDasharray="4 3"
            label={{ value: `alert above ${num(ratioThreshold, 2)}`, position: "insideTopLeft", fill: C.muted, fontSize: 10 }}
          />
          {holdoutLine}
          <Line dataKey="ratio" name="Pinball ratio" stroke={C.bandLine} strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line
            dataKey="ratioAlert"
            name="Alert days"
            stroke={C.neg}
            strokeWidth={2}
            dot={false}
            activeDot={false}
            connectNulls={false}
            isAnimationActive={false}
          />
        </ComposedChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-xs" style={{ color: C.muted, paddingLeft: AXIS_WIDTH }}>
        <span>
          <span style={{ color: C.bandLine }}>━</span> rolling signal
        </span>
        <span>
          <span style={{ color: C.neg }}>━</span> alert days, shaded episodes
        </span>
        <span>
          <span style={{ color: C.neg }}>┅</span> threshold
        </span>
      </div>
    </div>
  );
}
