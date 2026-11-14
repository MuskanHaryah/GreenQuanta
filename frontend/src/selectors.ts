// Choosing entries from API numbers (no science here: no energy, validity or gap is computed).
import type { Kind } from "./api";

interface Ranked { kind: Kind; valid_real: boolean; diff_r1_pct: number | null }

/** The best baseline: valid_real, lowest diff_r1_pct (linear joules, the optimised objective; never curve kWh). */
export function bestBaseline<T extends Ranked>(rows: Array<[string, T]>): [string, T] | undefined {
  return rows
    .filter(([, r]) => r.kind === "baseline" && r.valid_real && r.diff_r1_pct != null)
    .sort(([a, x], [b, y]) => x.diff_r1_pct! - y.diff_r1_pct! || a.localeCompare(b))[0];
}

/** Signed relative difference of two API numbers in % of the reference (negative = lower than the reference). */
export const relDiffPct = (value: number, reference: number) => (100 * (value - reference)) / reference;

export const KIND_GROUPS: Array<{ id: string; label: string; kinds: Kind[] }> = [
  { id: "quantum", label: "Quantum", kinds: ["quantum"] },
  { id: "exact", label: "Exact", kinds: ["exact"] },
  { id: "baselines", label: "Baselines", kinds: ["baseline"] },
];
