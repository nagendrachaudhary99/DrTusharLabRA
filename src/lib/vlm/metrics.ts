import { mulberry32 } from "@/lib/trial/physics";
import { LOBES } from "@/lib/trial/spec";
import { wilson, type Rate } from "@/lib/trial/simulate";
import { LOBE_OR_NONE, armKey, type CaseResult, type GroundTruth } from "./schema";

/**
 * Deterministic scoring for the VLM benchmark (pure; runs on the server in the `score` node and in
 * the browser for the live running tally). Proportions get 95% Wilson CIs, the size MAE gets a
 * seeded percentile-bootstrap CI, detection AUC uses the Mann-Whitney U statistic on confidence.
 */

export type TaskId = "detection" | "localization" | "size";
export const TASKS: { id: TaskId; label: string; short: string }[] = [
  { id: "detection", label: "Detection (nodule yes/no)", short: "Det" },
  { id: "localization", label: "Localization (lobe)", short: "Loc" },
  { id: "size", label: "Size (±max(2 mm, 30%))", short: "Size" },
];

export type StratumRow = { key: string; label: string; n: number } & Record<TaskId, Rate | null>;

export type VlmMetrics = {
  n: number;
  nPositive: number;
  nNegative: number;
  overall: Record<TaskId, Rate | null>;
  sensitivity: Rate | null;
  specificity: Rate | null;
  auc: number | null;
  size: { mae: number; lo: number; hi: number; bias: number; n: number } | null;
  bySize: StratumRow[];
  byType: StratumRow[];
  byArm: StratumRow[];
  byLobe: StratumRow[];
  confusion: { tp: number; fp: number; fn: number; tn: number };
  /** truth lobe (rows, positives only) × predicted lobe/none (cols). */
  lobeMatrix: { truth: string; counts: number[] }[];
  sizeScatter: { caseId: string; truth: number; pred: number }[];
  weakSpots: { factor: string; stratum: string; task: TaskId; rate: number; lo: number; hi: number; n: number; gapPp: number }[];
  sources: { live: number; simulated: number };
};

type Row = { gt: GroundTruth; r: CaseResult };

const rate = (key: string, label: string, flags: (boolean | null)[]): Rate | null => {
  const xs = flags.filter((f): f is boolean => f !== null);
  if (!xs.length) return null;
  const k = xs.filter(Boolean).length;
  return { key, label, n: xs.length, k, ...wilson(k, xs.length) };
};

function stratify(rows: Row[], keyOf: (g: GroundTruth) => string, labelOf: (k: string) => string, order?: string[]): StratumRow[] {
  const groups = new Map<string, Row[]>();
  for (const x of rows) groups.set(keyOf(x.gt), [...(groups.get(keyOf(x.gt)) ?? []), x]);
  const keys = order ? order.filter((k) => groups.has(k)) : [...groups.keys()].sort();
  return keys.map((key) => {
    const g = groups.get(key)!;
    const label = labelOf(key);
    return {
      key,
      label,
      n: g.length,
      detection: rate(key, label, g.map((x) => x.r.correct.detection)),
      localization: rate(key, label, g.map((x) => x.r.correct.localization)),
      size: rate(key, label, g.map((x) => x.r.correct.size)),
    };
  });
}

/** Probability a random positive gets a higher "nodule present" score than a random negative. */
function aucOf(rows: Row[]) {
  const score = (x: Row) => (x.r.answer.nodulePresent ? 0.5 + x.r.answer.confidence / 2 : 0.5 - x.r.answer.confidence / 2);
  const pos = rows.filter((x) => x.gt.hasNodule).map(score);
  const neg = rows.filter((x) => !x.gt.hasNodule).map(score);
  if (!pos.length || !neg.length) return null;
  let u = 0;
  for (const p of pos) for (const q of neg) u += p > q ? 1 : p === q ? 0.5 : 0;
  return u / (pos.length * neg.length);
}

function bootstrapMae(errs: number[], B = 1000) {
  const rand = mulberry32(1234);
  const mean = (a: number[]) => a.reduce((s, x) => s + x, 0) / a.length;
  const abs = errs.map(Math.abs);
  const stats: number[] = [];
  for (let b = 0; b < B; b++) stats.push(mean(abs.map(() => abs[Math.floor(rand() * abs.length)])));
  stats.sort((a, b) => a - b);
  return { mae: mean(abs), lo: stats[Math.floor(0.025 * B)], hi: stats[Math.floor(0.975 * B)], bias: mean(errs), n: errs.length };
}

export function computeMetrics(cases: { id: string; groundTruth: GroundTruth }[], results: CaseResult[]): VlmMetrics {
  const byId = new Map(results.map((r) => [r.caseId, r]));
  const rows: Row[] = cases.flatMap((c) => (byId.has(c.id) ? [{ gt: c.groundTruth, r: byId.get(c.id)! }] : []));
  const pos = rows.filter((x) => x.gt.hasNodule);
  const neg = rows.filter((x) => !x.gt.hasNodule);
  const overall = {
    detection: rate("all", "All cases", rows.map((x) => x.r.correct.detection)),
    localization: rate("all", "All positives", rows.map((x) => x.r.correct.localization)),
    size: rate("all", "All positives", rows.map((x) => x.r.correct.size)),
  };
  const confusion = {
    tp: pos.filter((x) => x.r.answer.nodulePresent).length,
    fn: pos.filter((x) => !x.r.answer.nodulePresent).length,
    fp: neg.filter((x) => x.r.answer.nodulePresent).length,
    tn: neg.filter((x) => !x.r.answer.nodulePresent).length,
  };
  const errs = pos.map((x) => x.r.sizeErrorMm).filter((e): e is number => e !== null);

  const bySize = stratify(rows, (g) => String(g.hasNodule ? g.diameterMm : 0), (k) => (k === "0" ? "0 mm (no nodule)" : `${k} mm`), undefined);
  bySize.sort((a, b) => +a.key - +b.key);
  const byType = stratify(rows, (g) => g.type, (k) => (k === "none" ? "none (negative control)" : k), ["solid", "part-solid", "ground-glass", "none"]);
  const byArm = stratify(rows, armKey, (k) => k);
  const byLobe = stratify(rows, (g) => g.lobe, (k) => (k === "none" ? "none (negative control)" : k), [...LOBE_OR_NONE]);

  // Weak spots: strata ≥15 pp below the task's overall accuracy with at least 3 cases.
  const weakSpots: VlmMetrics["weakSpots"] = [];
  for (const [factor, strata] of [
    ["size", bySize],
    ["type", byType],
    ["scanner", byArm],
    ["lobe", byLobe],
  ] as const)
    for (const s of strata)
      for (const t of TASKS) {
        const r = s[t.id];
        const o = overall[t.id];
        if (!r || !o || r.n < 3) continue;
        const gap = (o.rate - r.rate) * 100;
        if (gap >= 15) weakSpots.push({ factor, stratum: s.label, task: t.id, rate: r.rate, lo: r.lo, hi: r.hi, n: r.n, gapPp: +gap.toFixed(1) });
      }
  weakSpots.sort((a, b) => b.gapPp - a.gapPp);

  return {
    n: rows.length,
    nPositive: pos.length,
    nNegative: neg.length,
    overall,
    sensitivity: rate("sens", "Sensitivity", pos.map((x) => x.r.answer.nodulePresent)),
    specificity: rate("spec", "Specificity", neg.map((x) => !x.r.answer.nodulePresent)),
    auc: aucOf(rows),
    size: errs.length ? bootstrapMae(errs) : null,
    bySize,
    byType,
    byArm,
    byLobe,
    confusion,
    lobeMatrix: LOBES.map((truth) => ({
      truth,
      counts: LOBE_OR_NONE.map((p) => pos.filter((x) => x.gt.lobe === truth && (x.r.answer.nodulePresent ? x.r.answer.lobe : "none") === p).length),
    })).filter((r) => r.counts.some(Boolean)),
    sizeScatter: pos.map((x) => ({ caseId: x.r.caseId, truth: x.gt.diameterMm, pred: x.r.answer.nodulePresent ? x.r.answer.diameterMm : 0 })),
    weakSpots: weakSpots.slice(0, 8),
    sources: { live: rows.filter((x) => x.r.source === "live").length, simulated: rows.filter((x) => x.r.source === "simulated").length },
  };
}
