import type { AsymmetryBlock } from "../../api";
import { blockLabel, eur, pct } from "../../format";

/** One sentence naming the costliest block and direction, computed from the data. */
export function asymmetryNote(blocks: AsymmetryBlock[], gapEur: number): string {
  let best: { block: string; direction: "Over" | "Under"; cost: number } | null = null;
  for (const b of blocks) {
    if (!best || b.over_eur > best.cost) best = { block: b.block, direction: "Over", cost: b.over_eur };
    if (b.under_eur > best.cost) best = { block: b.block, direction: "Under", cost: b.under_eur };
  }
  if (!best || best.cost <= 0) {
    return "No block and direction carries a positive share of the gap to perfect foresight in this window.";
  }
  const lead = `${best.direction}-forecasting the ${blockLabel(best.block)} block is the costliest error: ${eur(best.cost)}`;
  if (gapEur <= 0) return `${lead}.`;
  const share = best.cost / gapEur;
  if (share <= 1) return `${lead}, ${pct(share)} of the ${eur(gapEur)} gap to perfect foresight.`;
  // Shapley shares can be negative, so one block can exceed the whole gap.
  return `${lead}, more than the whole ${eur(gapEur)} gap to perfect foresight; negative bars are blocks where the error happened to help and offset the rest.`;
}
