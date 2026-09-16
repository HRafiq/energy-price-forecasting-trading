import type { BatteryQuery, DispatchResponse, ForecastResponse } from "../api";
import { api } from "../api";
import { DispatchChart } from "../components/charts/DispatchChart";
import { FanChart } from "../components/charts/FanChart";
import { DeskBriefing } from "../components/DeskBriefing";
import { Loadable } from "../components/Loadable";
import { Panel } from "../components/Panel";
import { clock, eur, mw, num, strategyShort } from "../format";
import { type AsyncState, dataOf, isReloading, useApi } from "../hooks";

interface OverviewTabProps {
  run: string | null;
  date: string | null;
  battery: BatteryQuery;
  forecast: AsyncState<ForecastResponse>;
}

function dispatchSub(d: DispatchResponse | null): string {
  if (!d) return "Charge bars below zero, discharge above; realised price and state of charge in the strips below";
  return (
    `Day P&L ${eur(d.pnl_eur)} vs perfect foresight ${eur(d.perfect_foresight_pnl_eur)} · ` +
    `${mw(d.battery.power_mw)} MW / ${num(d.battery.capacity_mwh, 1)} MWh, €${d.battery.degradation_eur_per_mwh} wear, ` +
    `${strategyShort(d.strategy)} dispatch · solved on request in ${d.solve_ms} ms`
  );
}

export function OverviewTab({ run, date, battery, forecast }: OverviewTabProps) {
  const key =
    run && date
      ? ["dispatch", run, date, battery.power, battery.duration, battery.degradation, battery.strategy].join("|")
      : null;
  const dispatch = useApi(key, (signal) => {
    if (!run || !date) return Promise.reject(new Error("no day selected"));
    return api.dispatch(run, date, battery, signal);
  });
  const fc = dataOf(forecast);

  return (
    <div className="space-y-4">
      <DeskBriefing tab="overview" run={run} date={date} battery={battery} />
      <Panel
        title="Price forecast vs outcome"
        busy={isReloading(forecast)}
        sub={
          fc
            ? `Quantile fan issued ${clock(fc.issued_local)} before the ${clock(fc.gate_local)} gate · 90% band q0.05 to q0.95, 50% band q0.25 to q0.75, median line · dashed white line is the realised price (€/MWh)`
            : "Quantile fan with the realised price"
        }
      >
        <Loadable state={forecast} height={250}>
          {(f) => <FanChart periods={f.periods} />}
        </Loadable>
      </Panel>
      <Panel title="Dispatch schedule & state of charge" busy={isReloading(dispatch)} sub={dispatchSub(dataOf(dispatch))}>
        <Loadable state={dispatch} height={400}>
          {(d) => <DispatchChart periods={d.periods} capacityMwh={d.battery.capacity_mwh} />}
        </Loadable>
      </Panel>
    </div>
  );
}
