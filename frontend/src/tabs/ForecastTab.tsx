import type { WindowInfo, WindowKey } from "../api";
import { api } from "../api";
import { CalibrationChart } from "../components/charts/CalibrationChart";
import { ErrorByHourChart } from "../components/charts/ErrorByHourChart";
import { FeatureImportanceBars } from "../components/charts/FeatureImportanceBars";
import { DeskBriefing } from "../components/DeskBriefing";
import { Loadable } from "../components/Loadable";
import { Panel } from "../components/Panel";
import { longDay, pct, shortDay, windowPhrase } from "../format";
import { dataOf, isReloading, useApi } from "../hooks";
import { C } from "../theme";

function windowSpan(w: WindowInfo | undefined, fallback: WindowKey): string {
  if (!w) return windowPhrase(fallback);
  return `${windowPhrase(w.key)}, ${shortDay(w.first_day)} to ${longDay(w.last_day)}`;
}

export function ForecastTab({ run, windowKey }: { run: string | null; windowKey: WindowKey }) {
  const calibration = useApi(run ? `cal|${run}|${windowKey}` : null, (signal) =>
    run ? api.calibration(run, windowKey, signal) : Promise.reject(new Error("no run")),
  );
  const errors = useApi(run ? `err|${run}|${windowKey}` : null, (signal) =>
    run ? api.errorAnalysis(run, windowKey, signal) : Promise.reject(new Error("no run")),
  );
  const importance = useApi(run ? `fi|${run}` : null, (signal) =>
    run ? api.featureImportance(run, signal) : Promise.reject(new Error("no run")),
  );
  const fi = dataOf(importance);

  return (
    <div className="space-y-4">
      <DeskBriefing tab="forecast" run={run} windowKey={windowKey} />
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <Panel
          title="Calibration"
          busy={isReloading(calibration)}
          sub={`Share of realised prices at or below each quantile vs its level; dashed diagonal is perfect calibration · ${windowSpan(dataOf(calibration)?.window, windowKey)}`}
        >
          <Loadable state={calibration} height={220}>
            {(c) => (
              <>
                <CalibrationChart quantiles={c.quantiles} />
                <div className="grid grid-cols-3 gap-2 mt-3">
                  {[...c.intervals]
                    .sort((a, b) => a.nominal - b.nominal)
                    .map((i) => (
                      <div
                        key={i.nominal}
                        className="rounded-md px-2 py-1.5"
                        style={{ background: C.inset, border: `1px solid ${C.panelEdge}` }}
                      >
                        <div className="text-xs" style={{ color: C.muted }}>
                          {Math.round(i.nominal * 100)}% interval
                        </div>
                        <div className="text-sm font-semibold tabular-nums" style={{ color: C.text }}>
                          {pct(i.coverage)}
                        </div>
                      </div>
                    ))}
                </div>
              </>
            )}
          </Loadable>
        </Panel>
        <Panel
          title="Error by delivery hour"
          busy={isReloading(errors)}
          sub={`By local hour: MAE of the median forecast (top) and value at stake, the cash perfect foresight moves per day with the reference battery (bottom) · ${windowSpan(dataOf(errors)?.window, windowKey)}`}
        >
          <Loadable state={errors} height={260}>
            {(e) => <ErrorByHourChart byHour={e.by_hour} />}
          </Loadable>
        </Panel>
      </div>
      <Panel
        title="Feature importance"
        busy={isReloading(importance)}
        sub={
          fi
            ? `LightGBM ${fi.importance} share · ${fi.model} · trained for ${longDay(fi.trained_for_day)}`
            : "LightGBM gain share, production model"
        }
      >
        <Loadable state={importance} height={120}>
          {(f) => <FeatureImportanceBars features={f.features} />}
        </Loadable>
      </Panel>
    </div>
  );
}
