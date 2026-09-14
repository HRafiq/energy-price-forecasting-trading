import type { FeatureImportanceResponse } from "../../api";
import { pct } from "../../format";
import { C } from "../../theme";

export function FeatureImportanceBars({ features }: { features: FeatureImportanceResponse["features"] }) {
  const sorted = [...features].sort((a, b) => b.gain_share - a.gain_share);
  const top = sorted[0]?.gain_share ?? 0;
  if (sorted.length === 0) {
    return (
      <p className="text-xs" style={{ color: C.muted }}>
        No features reported for this model.
      </p>
    );
  }
  return (
    <div className="space-y-2">
      {sorted.map((f) => (
        <div key={f.feature} className="flex items-center gap-3" title={f.feature}>
          <div className="text-xs w-40 sm:w-56 shrink-0 truncate" style={{ color: C.muted }}>
            {f.label}
          </div>
          <div className="flex-1 rounded-full h-2" style={{ background: C.inset }}>
            <div
              className="h-2 rounded-full"
              style={{ width: `${top > 0 ? (f.gain_share / top) * 100 : 0}%`, background: C.bandLine }}
            />
          </div>
          <div className="text-xs tabular-nums w-14 text-right" style={{ color: C.muted }}>
            {pct(f.gain_share)}
          </div>
        </div>
      ))}
    </div>
  );
}
