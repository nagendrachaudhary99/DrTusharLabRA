import { z } from "zod";
import { mulberry32 } from "@/lib/trial/physics";
import { LOBES, NODULE_TYPES, VENDORS } from "@/lib/trial/spec";
import {
  DATASETS,
  FACTORS,
  LOCATIONS,
  LUNG_RADS,
  factorLevels,
  factorValue,
  filterHosts,
  filterNodules,
  lungRadsCategory,
  noduleHistograms,
  type Facets,
  type Factor,
  type Nodule,
} from "./index";

/**
 * Pure planner tools over the synthetic nodule index. The same functions back the
 * LangChain agent tools (server), the demo-mode rule-based planner and the
 * "build from facets" button in the browser, so results are identical everywhere.
 */

/**
 * LLM-facing facet schema. Ranges are flat min/max fields (not tuples) so the JSON schema
 * stays within what every OpenAI-compatible tool-calling endpoint accepts.
 */
export const FacetsSchema = z
  .object({
    minDiameterMm: z.number().min(0).max(40).optional().describe("Inclusive lower bound of nodule diameter (mm)"),
    maxDiameterMm: z.number().min(0).max(40).optional().describe("Inclusive upper bound of nodule diameter (mm)"),
    types: z.array(z.enum(NODULE_TYPES)).optional().describe("Nodule attenuation types; omit for all"),
    lobes: z.array(z.enum(LOBES)).optional().describe("RUL, RML, RLL, LUL, LLL. 'upper lobes' = RUL+LUL; omit for all"),
    locations: z.array(z.enum(LOCATIONS)).optional(),
    lungRads: z.array(z.enum(LUNG_RADS)).optional().describe("Lung-RADS v2022 categories to include"),
    datasets: z.array(z.enum(DATASETS)).optional().describe("Synthetic source-dataset stand-ins"),
    vendors: z.array(z.enum(VENDORS)).optional(),
    minAge: z.number().int().min(18).max(100).optional(),
    maxAge: z.number().int().min(18).max(100).optional(),
    sex: z.enum(["F", "M"]).optional(),
    minPackYears: z.number().min(0).optional(),
    minMalignancy: z.number().int().min(1).max(5).optional(),
    excludeCalcified: z.boolean().optional().describe("Drop benign-calcified nodules (Lung-RADS 1)"),
  })
  .describe("Faceted filter over the nodule index");
export type FacetsInput = z.infer<typeof FacetsSchema>;

export function toFacets(i: FacetsInput): Facets {
  const { minDiameterMm, maxDiameterMm, minAge, maxAge, ...rest } = i;
  const f: Facets = {};
  for (const [k, v] of Object.entries(rest)) if (v !== undefined && !(Array.isArray(v) && v.length === 0)) (f as Record<string, unknown>)[k] = v;
  if (minDiameterMm !== undefined || maxDiameterMm !== undefined) f.diameterMm = [minDiameterMm ?? 0, maxDiameterMm ?? 40];
  if (minAge !== undefined || maxAge !== undefined) f.ageRange = [minAge ?? 18, maxAge ?? 100];
  return f;
}

export const FactorEnum = z.enum(FACTORS);

export const BuildCohortSchema = z.object({
  facets: FacetsSchema,
  targetSize: z.number().int().min(10).max(1500).describe("Total cases (nodule-positive + negative controls)"),
  prevalence: z.number().min(0).max(1).describe("Fraction of cases that carry a nodule, e.g. 0.3"),
  stratifyBy: z.array(FactorEnum).max(3).describe("Factors defining strata for quotas; [] for none"),
  allocation: z.enum(["balanced", "natural"]).describe("balanced = equal quota per stratum; natural = index distribution"),
  seed: z.number().int().describe("Sampling seed (default 42)"),
});

/** Internal build arguments (facets in canonical range form). */
export type BuildCohortArgs = Omit<z.infer<typeof BuildCohortSchema>, "facets"> & { facets: Facets };

const median = (xs: number[]) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[Math.floor(s.length / 2)];
};

/* query_index */
export function queryIndex(facets: Facets) {
  const ns = filterNodules(facets);
  const hosts = filterHosts(facets);
  return {
    matchingNodules: ns.length,
    matchingPatients: new Set(ns.map((n) => n.patientId)).size,
    matchingNegativeHostScans: hosts.length,
    diameterMm: { min: ns.length ? Math.min(...ns.map((n) => n.diameterMm)) : null, median: median(ns.map((n) => n.diameterMm)), max: ns.length ? Math.max(...ns.map((n) => n.diameterMm)) : null },
    ageYears: { min: ns.length ? Math.min(...ns.map((n) => n.age)) : null, median: median(ns.map((n) => n.age)), max: ns.length ? Math.max(...ns.map((n) => n.age)) : null },
    histograms: noduleHistograms(ns),
  };
}

/* stratify */
function strataKeys(facets: Facets, factors: Factor[]) {
  let keys: string[][] = [[]];
  for (const f of factors) keys = keys.flatMap((k) => factorLevels(f, facets).map((l) => [...k, l]));
  return keys.map((k) => k.join(" · "));
}
const keyOf = (n: Nodule, factors: Factor[]) => factors.map((f) => factorValue(n, f)).join(" · ");

export function stratify(facets: Facets, factors: Factor[], targetPositives?: number) {
  const ns = filterNodules(facets);
  const keys = factors.length ? strataKeys(facets, factors) : ["all"];
  const counts: Record<string, number> = Object.fromEntries(keys.map((k) => [k, 0]));
  for (const n of ns) {
    const k = factors.length ? keyOf(n, factors) : "all";
    if (k in counts) counts[k]++;
  }
  const quota = targetPositives ? Math.ceil(targetPositives / keys.length) : null;
  const strata = keys.map((k) => ({ stratum: k, available: counts[k], ...(quota !== null ? { balancedQuota: quota, underFilled: counts[k] < quota } : {}) }));
  const under = strata.filter((s) => "underFilled" in s && s.underFilled);
  return {
    factors,
    totalAvailable: ns.length,
    nStrata: keys.length,
    balancedQuotaPerStratum: quota,
    strata,
    underFilledStrata: under.map((s) => s.stratum),
    note: under.length
      ? `${under.length} stratum/strata cannot meet a balanced quota of ${quota}; build_cohort will take all available donors there and redistribute the shortfall.`
      : quota !== null
        ? "All strata can meet a balanced quota."
        : "Pass targetPositives to check quota feasibility.",
  };
}

/* build_cohort */
export type ManifestRow = {
  caseId: string;
  label: "positive" | "negative";
  hostScanId: string;
  donorNoduleId: string | null;
  stratum: string | null;
  age: number;
  sex: "F" | "M";
  vendor: string;
  sliceThicknessMm: number;
  kernel: string;
  hostDataset: string;
  diameterMm: number | null;
  type: string | null;
  lobe: string | null;
  location: string | null;
  lungRads: string | null;
  malignancy: number | null;
  donorDataset: string | null;
};

export type CohortSummary = {
  targetSize: number;
  nCases: number;
  nPositive: number;
  nNegative: number;
  targetPrevalence: number;
  achievedPrevalence: number;
  allocation: "balanced" | "natural";
  stratifyBy: Factor[];
  seed: number;
  poolNodules: number;
  poolHosts: number;
  strata: { stratum: string; available: number; quota: number; selected: number; underFilled: boolean }[];
  positivesHistograms: ReturnType<typeof noduleHistograms>;
  warnings: string[];
  meetsSpec: boolean;
};

function shuffle<T>(xs: T[], rand: () => number) {
  const a = [...xs];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

export function buildCohort(args: BuildCohortArgs): { summary: CohortSummary; manifest: ManifestRow[] } {
  const { facets, targetSize, prevalence, allocation, seed } = args;
  const stratifyBy = [...new Set(args.stratifyBy)];
  const rand = mulberry32(seed);
  const warnings: string[] = [];
  const pool = shuffle(filterNodules(facets), rand);
  const hostPool = shuffle(filterHosts(facets), rand);

  let nCases = targetSize;
  if (hostPool.length < nCases) {
    warnings.push(`Only ${hostPool.length} host scans match the demographic/acquisition facets; cohort capped at ${hostPool.length} cases.`);
    nCases = hostPool.length;
  }
  let nPos = Math.round(nCases * prevalence);
  if (pool.length < nPos) {
    warnings.push(`Only ${pool.length} donor nodules match the facets (needed ${nPos}); positives capped.`);
    nPos = pool.length;
  }

  // Group donors by stratum.
  const keys = stratifyBy.length ? strataKeys(facets, stratifyBy) : ["all"];
  const groups = new Map<string, Nodule[]>(keys.map((k) => [k, []]));
  for (const n of pool) groups.get(stratifyBy.length ? keyOf(n, stratifyBy) : "all")?.push(n);

  const alloc = new Map<string, number>(keys.map((k) => [k, 0]));
  const ideal = keys.length ? nPos / keys.length : 0;
  if (allocation === "balanced" && stratifyBy.length) {
    // Water-filling: equal quotas, shortfall from small strata redistributed to the rest.
    let rem = nPos;
    let open = keys.filter((k) => groups.get(k)!.length > 0);
    while (rem > 0 && open.length) {
      const share = Math.floor(rem / open.length);
      let extra = rem - share * open.length;
      let used = 0;
      const next: string[] = [];
      for (const k of open) {
        const want = share + (extra > 0 ? 1 : 0);
        if (extra > 0) extra--;
        const give = Math.min(want, groups.get(k)!.length - alloc.get(k)!);
        alloc.set(k, alloc.get(k)! + give);
        used += give;
        if (alloc.get(k)! < groups.get(k)!.length) next.push(k);
      }
      rem -= used;
      if (!used) break;
      open = next;
    }
  } else {
    // Natural: simple random sample of the filtered pool (stratum shares follow the index).
    for (const n of pool.slice(0, nPos)) {
      const k = stratifyBy.length ? keyOf(n, stratifyBy) : "all";
      alloc.set(k, (alloc.get(k) ?? 0) + 1);
    }
  }

  const positives: { n: Nodule; stratum: string }[] = [];
  for (const k of keys) for (const n of groups.get(k)!.slice(0, alloc.get(k)!)) positives.push({ n, stratum: k });

  const strata = keys.map((k) => {
    const quota = allocation === "balanced" ? Math.ceil(ideal) : Math.round((nPos * groups.get(k)!.length) / Math.max(1, pool.length));
    return { stratum: k, available: groups.get(k)!.length, quota, selected: alloc.get(k)!, underFilled: groups.get(k)!.length < Math.floor(ideal) && allocation === "balanced" };
  });
  const under = strata.filter((s) => s.underFilled);
  if (under.length) warnings.push(`Under-filled strata (fewer donors than the balanced quota ${Math.floor(ideal)}): ${under.map((s) => `${s.stratum} (${s.available})`).join(", ")}.`);

  const nNeg = nCases - positives.length;
  const hosts = hostPool.slice(0, nCases);
  const cases = shuffle(
    hosts.map((h, i) => ({ h, pos: i < positives.length ? positives[i] : null })),
    rand,
  );
  const manifest: ManifestRow[] = cases.map(({ h, pos }, i) => ({
    caseId: `C${String(i + 1).padStart(4, "0")}`,
    label: pos ? "positive" : "negative",
    hostScanId: h.id,
    donorNoduleId: pos?.n.id ?? null,
    stratum: pos?.stratum ?? null,
    age: h.age,
    sex: h.sex,
    vendor: h.vendor,
    sliceThicknessMm: h.sliceThicknessMm,
    kernel: h.kernel,
    hostDataset: h.sourceDataset,
    diameterMm: pos?.n.diameterMm ?? null,
    type: pos?.n.type ?? null,
    lobe: pos?.n.lobe ?? null,
    location: pos?.n.location ?? null,
    lungRads: pos?.n.lungRads ?? null,
    malignancy: pos?.n.malignancy ?? null,
    donorDataset: pos?.n.sourceDataset ?? null,
  }));

  const achieved = nCases ? positives.length / nCases : 0;
  const summary: CohortSummary = {
    targetSize,
    nCases,
    nPositive: positives.length,
    nNegative: nNeg,
    targetPrevalence: prevalence,
    achievedPrevalence: +achieved.toFixed(4),
    allocation,
    stratifyBy,
    seed,
    poolNodules: pool.length,
    poolHosts: hostPool.length,
    strata,
    positivesHistograms: noduleHistograms(positives.map((p) => p.n)),
    warnings,
    meetsSpec: nCases === targetSize && Math.abs(achieved - prevalence) <= 1 / Math.max(1, targetSize) + 1e-9 && under.length === 0,
  };
  return { summary, manifest };
}

/** Compact view of a build result for the LLM (no id list — that stays server-side). */
export function summarizeForAgent(s: CohortSummary) {
  return {
    nCases: s.nCases,
    nPositive: s.nPositive,
    nNegative: s.nNegative,
    achievedPrevalence: s.achievedPrevalence,
    meetsSpec: s.meetsSpec,
    strata: s.strata.map(({ stratum, available, selected }) => ({ stratum, available, selected })),
    sizeBinsOfPositives: s.positivesHistograms.sizeBin,
    typesOfPositives: s.positivesHistograms.type,
    warnings: s.warnings,
  };
}

/* lung_rads */
const MANAGEMENT: Record<string, string> = {
  "1": "continue annual LDCT",
  "2": "continue annual LDCT",
  "3": "6-month LDCT",
  "4A": "3-month LDCT; PET/CT may be used if solid component ≥8 mm",
  "4B": "diagnostic chest CT, PET/CT and/or tissue sampling",
  "4X": "managed as 4B",
};

export const LungRadsSchema = z.object({
  diameterMm: z.number().min(0).max(60).describe("Mean diameter in mm"),
  type: z.enum(NODULE_TYPES),
  solidComponentMm: z.number().min(0).max(60).optional().describe("Part-solid only: solid component diameter"),
  spiculated: z.boolean().optional(),
});

export function lungRads(a: z.infer<typeof LungRadsSchema>) {
  const r = lungRadsCategory(a);
  return { ...r, management: MANAGEMENT[r.category], version: "Lung-RADS v2022 (baseline)" };
}

/* NL description of a cohort spec — used for the "Send to Trial Designer" link. */
const LOBE_PHRASE: Record<string, string> = { RUL: "right upper lobe", RML: "right middle lobe", RLL: "right lower lobe", LUL: "left upper lobe", LLL: "left lower lobe" };

export const FACTOR_LABEL: Record<Factor, string> = {
  sizeBin: "size bins (<6, 6-8, 8-15, ≥15 mm)",
  type: "nodule type",
  lobe: "lobe",
  location: "location",
  lungRads: "Lung-RADS category",
  dataset: "source dataset",
  vendor: "scanner vendor",
  sex: "sex",
  ageBin: "age band",
};

export function describeSpec(a: BuildCohortArgs, s?: CohortSummary) {
  const f = a.facets;
  const parts: string[] = [];
  const nPos = s?.nPositive ?? Math.round(a.targetSize * a.prevalence);
  parts.push(`Virtual trial on a ${s?.nCases ?? a.targetSize}-case cohort from the nodule index (${nPos} nodule-positive, ${Math.round((s?.achievedPrevalence ?? a.prevalence) * 100)}% prevalence)`);
  const nodule: string[] = [];
  nodule.push(f.types?.length ? f.types.join(" and ") : "solid, part-solid and ground-glass");
  nodule.push("nodules");
  if (f.diameterMm) nodule.push(`${f.diameterMm[0]}-${f.diameterMm[1]} mm`);
  if (f.lobes?.length) nodule.push(`in the ${f.lobes.map((l) => LOBE_PHRASE[l]).join(", ")}`);
  if (f.locations?.length) nodule.push(`(${f.locations.join(", ")})`);
  parts.push(nodule.join(" "));
  if (f.ageRange) parts.push(`patients aged ${f.ageRange[0]}-${f.ageRange[1]}`);
  if (f.lungRads?.length) parts.push(`Lung-RADS ${f.lungRads.join("/")}`);
  if (f.vendors?.length) parts.push(`scanners ${f.vendors.join(", ")}`);
  if (a.stratifyBy.length) parts.push(`${a.allocation} across ${a.stratifyBy.map((x) => FACTOR_LABEL[x]).join(" × ")}`);
  parts.push(`seed ${a.seed}`);
  return parts.join(", ") + ".";
}
