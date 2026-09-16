import type { BatteryQuery, PnlResponse, RunKind, WindowKey } from "../api";
import { api } from "../api";
import { AsymmetryChart } from "../components/charts/AsymmetryChart";
import { asymmetryNote } from "../components/charts/asymmetry";
import { PnlChart } from "../components/charts/PnlChart";
import { DeskBriefing } from "../components/DeskBriefing";
import { Loadable, StatusMessage } from "../components/Loadable";
import { Panel } from "../components/Panel";
import { eur, strategyShort, windowPhrase } from "../format";
import { dataOf, isReloading, useApi } from "../hooks";
import { C } from "../theme";

interface TradingTabProps {
  run: string | null;
  windowKey: WindowKey;
  battery: BatteryQuery;
  /** False while the P&L grid export is still running. */
  gridAvailable: boolean | undefined;
  runKind: RunKind | undefined;
}

function pnlSub(p: PnlResponse | null): string {
  const base = "Dashed grey line is the perfect-foresight ceiling";
  if (!p) return base;
  const dd = p.max_drawdown_eur;
  const parts = [`median ${eur(dd.median)}`];
  if (p.selected_strategy !== "median") parts.unshift(`${strategyShort(p.selected_strategy)} ${eur(dd.selected)}`);
  return `${base} · max drawdown: ${parts.join(", ")} · ${p.window.traded_days} traded days`;
}

export function TradingTab({
  run,
  windowKey,
  battery,
  gridAvailable,
  runKind,
}: TradingTabProps) {
  const pnlKey = run
    ? ["pnl", run, windowKey, battery.power, battery.duration, battery.degradation, battery.strategy].join("|")
    : null;
  const pnl = useApi(pnlKey, (signal) =>
    run ? api.pnl(run, windowKey, battery, signal) : Promise.reject(new Error("no run")),
  );
  const errors = useApi(run ? `err|${run}|${windowKey}` : null, (signal) =>
    run ? api.errorAnalysis(run, windowKey, signal) : Promise.reject(new Error("no run")),
  );
  const shown = dataOf(errors);

  return (
    <div className="space-y-4">
      <DeskBriefing tab="trading" run={run} windowKey={windowKey} battery={battery} />
      <Panel title="Cumulative P&L by strategy" busy={isReloading(pnl)} sub={pnlSub(dataOf(pnl))}>
        {pnl.status === "error" && gridAvailable === false ? (
          <StatusMessage height={260}>
            {runKind === "live"
              ? "A live run carries today's forecast and schedule, not the battery P&L grid. The cumulative P&L is on the backtest run."
              : "P&L grid still exporting. The chart appears after the export finishes and the page is reloaded."}
          </StatusMessage>
        ) : (
          <Loadable state={pnl} height={260}>
            {(p) => <PnlChart series={p.series} selected={p.selected_strategy} />}
          </Loadable>
        )}
      </Panel>
      <Panel
        title="Cost of forecast error by direction"
        busy={isReloading(errors)}
        sub={
          shown
            ? `Shapley split of median dispatch's gap to perfect foresight by hour block · ${shown.asymmetry.reference} · gap ${eur(shown.asymmetry.gap_eur)} over the ${windowPhrase(shown.window.key)}`
            : "Where over- and under-forecasting lose money (€)"
        }
      >
        <Loadable state={errors} height={220}>
          {(e) => (
            <>
              <AsymmetryChart blocks={e.asymmetry.blocks} />
              <p className="text-xs mt-2 leading-relaxed" style={{ color: C.muted }}>
                {asymmetryNote(e.asymmetry.blocks, e.asymmetry.gap_eur)} Reference battery, independent of the controls.
              </p>
            </>
          )}
        </Loadable>
      </Panel>
    </div>
  );
}
