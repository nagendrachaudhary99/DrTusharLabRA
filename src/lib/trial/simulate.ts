import { detectability, gaussian, mulberry32, noiseSigmaHU, LUNG_HU, NODULE_HU } from "./physics";
import type { ScannerArm, TrialSpec } from "./spec";

export type VirtualLesion = {
  lesionId: string;
  patientId: string;
  arm: number;
  diameterMm: number;
  sizeStratum: number;
  type: TrialSpec["noduleTypes"][number];
  lobe: TrialSpec["lobes"][number];
  juxtavascular: boolean;
  /** position inside the lobe (normalised 0..1) — used by the phantom renderer */
  pos: [number, number, number];
  contrastHU: number;
  sigmaHU?: number;
  cnr?: number;
  dprime?: number;
  score?: number;
  detected?: boolean;
};

export type VirtualPatient = {
  patientId: string;
  arm: number;
  age: number;
  sex: "F" | "M";
  bmi: number;
  emphysema: number;
  falsePositiveScores?: number[];
};

export type Cohort = { patients: VirtualPatient[]; lesions: VirtualLesion[] };

export function synthesizeCohort(spec: TrialSpec): Cohort {
  const rand = mulberry32(spec.seed);
  const pick = <T>(xs: readonly T[]) => xs[Math.floor(rand() * xs.length)];
  const patients: VirtualPatient[] = [];
  const lesions: VirtualLesion[] = [];
  spec.scanners.forEach((_, arm) => {
    for (let i = 0; i < spec.patientsPerArm; i++) {
      const patientId = `VP-${arm + 1}-${String(i + 1).padStart(4, "0")}`;
      // Screening-like demographics (age 50-80), log-normal BMI.
      patients.push({
        patientId,
        arm,
        age: Math.round(50 + rand() * 30),
        sex: rand() < 0.45 ? "F" : "M",
        bmi: +Math.min(45, Math.max(17, Math.exp(Math.log(26.5) + 0.17 * gaussian(rand)))).toFixed(1),
        emphysema: +Math.max(0, Math.min(1, 0.15 + 0.15 * gaussian(rand))).toFixed(2),
      });
      const n = 1 + Math.floor(rand() * spec.lesionsPerPatient);
      for (let j = 0; j < n; j++) {
        const sizeStratum = pick(spec.sizesMm);
        const type = pick(spec.noduleTypes);
        lesions.push({
          lesionId: `${patientId}-L${j + 1}`,
          patientId,
          arm,
          sizeStratum,
          diameterMm: +Math.max(2, sizeStratum * (1 + 0.08 * gaussian(rand))).toFixed(1),
          type,
          lobe: pick(spec.lobes),
          juxtavascular: rand() < spec.juxtavascularFraction,
          pos: [rand(), rand(), rand()],
          contrastHU: NODULE_HU[type] + 25 * gaussian(rand) - LUNG_HU,
        });
      }
    }
  });
  return { patients, lesions };
}

export function simulateImaging(spec: TrialSpec, cohort: Cohort): Cohort {
  const byId = new Map(cohort.patients.map((p) => [p.patientId, p]));
  const lesions = cohort.lesions.map((l) => {
    const arm = spec.scanners[l.arm];
    const sigmaHU = noiseSigmaHU(arm, byId.get(l.patientId)!.bmi);
    const { cnr, dprime } = detectability({ diameterMm: l.diameterMm, contrastHU: l.contrastHU, sigmaHU, kernel: arm.kernel });
    return { ...l, sigmaHU: +sigmaHU.toFixed(1), cnr: +cnr.toFixed(2), dprime: +dprime.toFixed(2) };
  });
  return { ...cohort, lesions };
}

/** Model-specific logit penalties: how each simulated AI responds to the virtual acquisition. */
function domainPenalty(profile: TrialSpec["modelUnderTest"], arm: ScannerArm) {
  switch (profile) {
    case "vendor-overfit":
      return (arm.vendor === "Siemens" ? 0 : 1.7) + (arm.kernel === "standard" ? 0 : 0.6);
    case "lowdose-fragile":
      return arm.dose === "standard" ? 0 : arm.dose === "low" ? 1.5 : 2.8;
    default:
      return arm.dose === "ultra-low" ? 0.4 : 0;
  }
}

export const OPERATING_THRESHOLD = 0.5;

/** Expected (noise-free) logit of the simulated detector for one lesion under one acquisition. */
export function detectionLogit(
  profile: TrialSpec["modelUnderTest"],
  arm: ScannerArm,
  l: Pick<VirtualLesion, "dprime" | "type" | "juxtavascular" | "diameterMm">,
) {
  // Psychometric curve fitted so a robust model reaches ~95% on 8 mm solid nodules at routine dose.
  let logit = 1.9 * Math.log(l.dprime ?? 1) - 5.7;
  logit -= domainPenalty(profile, arm);
  if (l.type === "ground-glass") logit -= 0.4;
  if (l.type === "part-solid") logit -= 0.3;
  if (l.juxtavascular) logit -= 0.8;
  if (l.diameterMm < 5) logit -= 0.6;
  return logit;
}

export function evaluateModel(spec: TrialSpec, cohort: Cohort): Cohort {
  const rand = mulberry32(spec.seed ^ 0x9e3779b9);
  const lesions = cohort.lesions.map((l) => {
    const arm = spec.scanners[l.arm];
    const logit = detectionLogit(spec.modelUnderTest, arm, l);
    const score = 1 / (1 + Math.exp(-(logit + 0.7 * gaussian(rand))));
    return { ...l, score: +score.toFixed(4), detected: score >= OPERATING_THRESHOLD };
  });
  const patients = cohort.patients.map((p) => {
    const arm = spec.scanners[p.arm];
    // False-positive candidates scale with noise level, emphysema and domain shift.
    const sigma = noiseSigmaHU(arm, p.bmi);
    const lambda = 1.2 * (sigma / 12) ** 0.8 * (1 + 2 * p.emphysema) * (1 + 0.4 * domainPenalty(spec.modelUnderTest, arm));
    const nFp = poisson(lambda * 3, rand); // candidate pool; most fall below threshold
    const falsePositiveScores = Array.from({ length: nFp }, () => +(1 / (1 + Math.exp(-(-1.7 + 1.1 * gaussian(rand))))).toFixed(4));
    return { ...p, falsePositiveScores };
  });
  return { patients, lesions };
}

function poisson(lambda: number, rand: () => number) {
  const L = Math.exp(-lambda);
  let k = 0;
  let p = 1;
  do {
    k++;
    p *= rand();
  } while (p > L && k < 100);
  return k - 1;
}

/* ───────────────────────────── statistics ───────────────────────────── */

/** One scanner arm end to end (imaging chain → model inference). Seeded per arm so parallel `Send` workers stay reproducible. */
export function simulateArm(spec: TrialSpec, cohort: Cohort, arm: number): Cohort {
  const sub = { patients: cohort.patients.filter((p) => p.arm === arm), lesions: cohort.lesions.filter((l) => l.arm === arm) };
  return evaluateModel({ ...spec, seed: spec.seed + 7919 * (arm + 1) }, simulateImaging(spec, sub));
}

export type Rate = { key: string; label: string; n: number; k: number; rate: number; lo: number; hi: number };

export function wilson(k: number, n: number, z = 1.96) {
  if (!n) return { rate: 0, lo: 0, hi: 0 };
  const p = k / n;
  const d = 1 + (z * z) / n;
  const c = (p + (z * z) / (2 * n)) / d;
  const h = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
  return { rate: p, lo: Math.max(0, c - h), hi: Math.min(1, c + h) };
}

function stratify(lesions: VirtualLesion[], keyOf: (l: VirtualLesion) => string, labelOf: (k: string) => string = (k) => k, order?: string[]) {
  const groups = new Map<string, VirtualLesion[]>();
  for (const l of lesions) {
    const k = keyOf(l);
    groups.set(k, [...(groups.get(k) ?? []), l]);
  }
  const keys = order?.filter((k) => groups.has(k)) ?? [...groups.keys()];
  return keys.map((key): Rate => {
    const g = groups.get(key)!;
    const k = g.filter((l) => l.detected).length;
    return { key, label: labelOf(key), n: g.length, k, ...wilson(k, g.length) };
  });
}

export const armLabel = (a: ScannerArm) => `${a.vendor} · ${a.kernel} · ${a.dose}`;

const FROC_FP_POINTS = [0.125, 0.25, 0.5, 1, 2, 4, 8];

/** FROC curve + LUNA16-style competition performance metric (mean sensitivity at 7 FP/scan levels). */
export function froc(lesions: VirtualLesion[], patients: VirtualPatient[]) {
  const nScans = patients.length;
  const fps = patients.flatMap((p) => p.falsePositiveScores ?? []).sort((a, b) => b - a);
  const tps = lesions.map((l) => l.score ?? 0).sort((a, b) => b - a);
  const thresholds = [...new Set([...fps, ...tps])].sort((a, b) => b - a);
  const curve: { fpPerScan: number; sensitivity: number; threshold: number }[] = [];
  let ti = 0;
  let fi = 0;
  for (const t of thresholds) {
    while (ti < tps.length && tps[ti] >= t) ti++;
    while (fi < fps.length && fps[fi] >= t) fi++;
    curve.push({ fpPerScan: fi / nScans, sensitivity: ti / tps.length, threshold: t });
  }
  const sensAt = (fp: number) => {
    let best = 0;
    for (const c of curve) if (c.fpPerScan <= fp) best = Math.max(best, c.sensitivity);
    return best;
  };
  const points = FROC_FP_POINTS.map((fp) => ({ fpPerScan: fp, sensitivity: sensAt(fp) }));
  const cpm = points.reduce((s, p) => s + p.sensitivity, 0) / points.length;
  // Downsample curve for transport.
  const step = Math.max(1, Math.floor(curve.length / 120));
  return { points, cpm, curve: curve.filter((_, i) => i % step === 0 && curve[i].fpPerScan <= 8) };
}

export type Finding = { severity: "critical" | "warning" | "info"; title: string; detail: string };

export function analyze(spec: TrialSpec, cohort: Cohort) {
  const { lesions, patients } = cohort;
  const overallK = lesions.filter((l) => l.detected).length;
  const overall = { n: lesions.length, k: overallK, ...wilson(overallK, lesions.length) };
  const fpAtOp = patients.reduce((s, p) => s + (p.falsePositiveScores ?? []).filter((x) => x >= OPERATING_THRESHOLD).length, 0) / Math.max(1, patients.length);

  const bySize = stratify(
    lesions,
    (l) => String(l.sizeStratum),
    (k) => `${k} mm`,
    spec.sizesMm.map(String).sort((a, b) => +a - +b),
  );
  const byType = stratify(lesions, (l) => l.type, undefined, [...spec.noduleTypes]);
  const byLobe = stratify(lesions, (l) => l.lobe, undefined, [...spec.lobes]);
  const byArm = stratify(
    lesions,
    (l) => String(l.arm),
    (k) => armLabel(spec.scanners[+k]),
    spec.scanners.map((_, i) => String(i)),
  );
  const byJuxta = stratify(lesions, (l) => (l.juxtavascular ? "juxtavascular" : "isolated"));

  // Size × arm heat map.
  const heat = spec.scanners.map((arm, a) => ({
    arm: armLabel(arm),
    cells: [...spec.sizesMm]
      .sort((x, y) => x - y)
      .map((s) => {
        const g = lesions.filter((l) => l.arm === a && l.sizeStratum === s);
        const k = g.filter((l) => l.detected).length;
        return { size: s, n: g.length, ...wilson(k, g.length) };
      }),
  }));

  const frocAll = froc(lesions, patients);
  const frocByArm = spec.scanners.map((arm, a) => ({
    arm: armLabel(arm),
    ...froc(
      lesions.filter((l) => l.arm === a),
      patients.filter((p) => p.arm === a),
    ),
  }));

  // Automated failure-mode discovery: strata whose CI lies entirely below the overall CI.
  const findings: Finding[] = [];
  const groups: [string, Rate[]][] = [
    ["scanner protocol", byArm],
    ["nodule size", bySize],
    ["nodule type", byType],
    ["lobe", byLobe],
    ["vessel attachment", byJuxta],
  ];
  for (const [dim, rates] of groups) {
    for (const r of rates) {
      if (r.n < 15) continue;
      const gap = overall.rate - r.rate;
      if (r.hi < overall.lo && gap > 0.12) {
        findings.push({
          severity: gap > 0.25 ? "critical" : "warning",
          title: `Under-performance on ${dim}: ${r.label}`,
          detail: `Sensitivity ${(r.rate * 100).toFixed(1)}% (95% CI ${(r.lo * 100).toFixed(1)}–${(r.hi * 100).toFixed(1)}%, n=${r.n}) vs overall ${(overall.rate * 100).toFixed(1)}% — a ${(gap * 100).toFixed(1)} pp gap with non-overlapping CIs.`,
        });
      }
    }
  }
  const armRates = byArm.map((r) => r.rate);
  const spread = Math.max(...armRates) - Math.min(...armRates);
  if (spec.scanners.length > 1 && spread > 0.15) {
    findings.push({
      severity: "critical",
      title: "Scanner/protocol domain shift detected",
      detail: `Sensitivity varies by ${(spread * 100).toFixed(1)} pp across acquisition arms. The model's performance is not invariant to vendor / kernel / dose — harmonization or multi-site fine-tuning is indicated before deployment.`,
    });
  }
  const sortedSize = [...bySize].sort((a, b) => +a.key - +b.key);
  const limit = sortedSize.find((r) => r.rate >= 0.8);
  findings.push({
    severity: "info",
    title: "Detection size limit",
    detail: limit
      ? `Sensitivity first reaches ≥80% at the ${limit.label} stratum. Lung-RADS v2022 treats solid nodules ≥6 mm as actionable (category 3+), so sub-threshold sensitivity below that size matters less clinically than misses at ≥6 mm.`
      : "Sensitivity never reaches 80% in any size stratum tested — the model is not fit for purpose under these acquisition conditions.",
  });
  // Power check: CI half-width per smallest stratum.
  const minN = Math.min(...bySize.map((r) => r.n));
  if (minN < 60) {
    findings.push({
      severity: "info",
      title: "Precision note",
      detail: `Smallest size stratum has n=${minN} lesions; 95% CI half-width can exceed ±12 pp. Increase patients per arm for tighter subgroup estimates (virtual patients are cheap).`,
    });
  }

  return {
    overall,
    fpPerScanAtOperatingPoint: +fpAtOp.toFixed(2),
    nPatients: patients.length,
    nLesions: lesions.length,
    bySize,
    byType,
    byLobe,
    byArm,
    byJuxta,
    heat,
    froc: frocAll,
    frocByArm,
    findings: findings.sort((a, b) => ["critical", "warning", "info"].indexOf(a.severity) - ["critical", "warning", "info"].indexOf(b.severity)),
  };
}

export type TrialAnalysis = ReturnType<typeof analyze>;
