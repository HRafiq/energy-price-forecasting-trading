import { DeskBriefing } from "../components/DeskBriefing";
import { Panel } from "../components/Panel";
import { C } from "../theme";

export function ModelHealthTab() {
  return (
    <div className="space-y-4">
      <DeskBriefing />
      <Panel title="Model health" sub="Arrives in a later phase">
        <p className="text-sm leading-relaxed" style={{ color: C.muted }}>
          This tab arrives in a later phase with the regime-shift experiment, drift monitoring and the incident log.
          Nothing is shown here until those are backed by real records.
        </p>
      </Panel>
    </div>
  );
}
