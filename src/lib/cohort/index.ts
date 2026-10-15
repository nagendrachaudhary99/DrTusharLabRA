import { gaussian, mulberry32 } from "@/lib/trial/physics";
import { KERNELS, LOBES, NODULE_TYPES, VENDORS } from "@/lib/trial/spec";

/**
 * SYNTHETIC nodule index for the Cohort Planner.
 *
 * This is a seeded, deterministic stand-in for the iTRIALSPACE nodule index. Attribute
 * distributions are loosely modelled on public lung CT datasets (LIDC-IDRI, LUNA16, NLST,
 * LNDb): log-normal diameters skewed small, solid-dominant attenuation, upper-lobe
 * predominance, malignancy correlated with size / spiculation / part-solid type.
 * NO real patient data is used. The real iTRIALSPACE index (Tushar et al., arXiv 2605.05761)
 * holds 13,140 real nodules with 54 attributes curated from 7 public CT sources.
 *
 * Pure module (no server deps) so the same index is rebuilt identically in the browser.
 */
export const INDEX_DISCLAIMER =
  "Synthetic index with attribute distributions modelled on public datasets; the real iTRIALSPACE index uses 13,140 real nodules (54 attributes, 7 public CT sources).";

export const DATASETS = ["LIDC-IDRI-like", "LUNA16-like", "NLST-like", "LNDb-like"] as const;
export const LOCATIONS = ["central", "peripheral", "juxtapleural", "juxtavascular"] as const;
export const MARGINS = ["smooth", "lobulated", "spiculated", "ill-defined"] as const;
export const CALCIFICATIONS = ["none", "central", "popcorn", "laminated", "diffuse"] as const;
export const LUNG_RADS = ["1", "2", "3", "4A", "4B", "4X"] as const;
export const SIZE_BINS = ["<6 mm", "6–8 mm", "8–15 mm", "≥15 mm"] as const;
export const SIZE_EDGES = [0, 6, 8, 15, Infinity];
export const AGE_BINS = ["<55", "55–64", "65–74", "≥75"] as const;

export type Dataset = (typeof DATASETS)[number];
export type Lobe = (typeof LOBES)[number];
export type NoduleType = (typeof NODULE_TYPES)[number];
export type Location = (typeof LOCATIONS)[number];
export type LungRads = (typeof LUNG_RADS)[number];

export type Nodule = {
  id: string;
  patientId: string;
  sourceDataset: Dataset;
  age: number;
  sex: "F" | "M";
  smokingPackYears: number;
  diameterMm: number;
  solidComponentMm: number;
  volumeMm3: number;
  type: NoduleType;
  lobe: Lobe;
  location: Location;
  spiculation: number;
  margin: (typeof MARGINS)[number];
  calcification: (typeof CALCIFICATIONS)[number];
  malignancy: number;
  lungRads: LungRads;
  vendor: (typeof VENDORS)[number];
  sliceThicknessMm: number;
  kernel: (typeof KERNELS)[number];
};

/** Nodule-free host scans: negative controls and insertion backgrounds. */
export type HostScan = {
  id: string;
  sourceDataset: Dataset;
  age: number;
  sex: "F" | "M";
  smokingPackYears: number;
  vendor: (typeof VENDORS)[number];
  sliceThicknessMm: number;
  kernel: (typeof KERNELS)[number];
};

export const N_NODULES = 3000;
export const N_HOSTS = 2000;
const INDEX_SEED = 13140;

const pick = <T,>(rand: () => number, items: readonly T[], weights: number[]) => {
  const total = weights.reduce((a, b) => a + b, 0);
  let r = rand() * total;
  for (let i = 0; i < items.length; i++) if ((r -= weights[i]) <= 0) return items[i];
  return items[items.length - 1];
};
const clamp = (x: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, x));
const round1 = (x: number) => Math.round(x * 10) / 10;

/* ── Lung-RADS v2022 (baseline screening), consistent with src/data/knowledge-base.json ── */
export type LungRadsInput = {
  diameterMm: number;
  type: NoduleType;
  solidComponentMm?: number;
  benignCalcification?: boolean;
  juxtapleuralSmooth?: boolean;
  spiculated?: boolean;
};

export function lungRadsCategory(n: LungRadsInput): { category: LungRads; rule: string } {
  if (n.benignCalcification) return { category: "1", rule: "benign calcification pattern → category 1" };
  let base: { category: LungRads; rule: string };
  if (n.type === "solid") {
    const d = n.diameterMm;
    if (n.juxtapleuralSmooth && d < 10) base = { category: "2", rule: "juxtapleural smooth solid nodule <10 mm → 2" };
    else if (d < 6) base = { category: "2", rule: "solid <6 mm → 2" };
    else if (d < 8) base = { category: "3", rule: "solid 6 to <8 mm → 3" };
    else if (d < 15) base = { category: "4A", rule: "solid 8 to <15 mm → 4A" };
    else base = { category: "4B", rule: "solid ≥15 mm → 4B" };
  } else if (n.type === "part-solid") {
    const s = n.solidComponentMm ?? n.diameterMm * 0.4;
    if (n.diameterMm < 6) base = { category: "2", rule: "part-solid <6 mm total → 2" };
    else if (s < 6) base = { category: "3", rule: "part-solid ≥6 mm total, solid component <6 mm → 3" };
    else if (s < 8) base = { category: "4A", rule: "part-solid, solid component 6 to <8 mm → 4A" };
    else base = { category: "4B", rule: "part-solid, solid component ≥8 mm → 4B" };
  } else {
    base = n.diameterMm < 30 ? { category: "2", rule: "ground-glass <30 mm → 2" } : { category: "3", rule: "ground-glass ≥30 mm → 3" };
  }
  if (n.spiculated && ["3", "4A", "4B"].includes(base.category)) return { category: "4X", rule: `${base.rule}; spiculation upgrades category 3/4 → 4X` };
  return base;
}

export const sizeBinOf = (d: number) => SIZE_BINS[SIZE_EDGES.findIndex((e, i) => d >= e && d < SIZE_EDGES[i + 1])];
export const ageBinOf = (a: number) => (a < 55 ? AGE_BINS[0] : a < 65 ? AGE_BINS[1] : a < 75 ? AGE_BINS[2] : AGE_BINS[3]);

/* ── Generation ── */
const DATASET_PROFILE: Record<Dataset, { weight: number; ageMu: number; ageSd: number; ageMin: number; ageMax: number; smoker: number; slices: number[]; vendorW: number[] }> = {
  "LIDC-IDRI-like": { weight: 0.35, ageMu: 62, ageSd: 12, ageMin: 30, ageMax: 90, smoker: 0.7, slices: [0.625, 1.25, 1.5, 2, 2.5, 3], vendorW: [0.45, 0.25, 0.15, 0.15] },
  "LUNA16-like": { weight: 0.2, ageMu: 61, ageSd: 11, ageMin: 35, ageMax: 88, smoker: 0.7, slices: [0.625, 1, 1.25, 2, 2.5], vendorW: [0.45, 0.25, 0.15, 0.15] },
  "NLST-like": { weight: 0.3, ageMu: 63, ageSd: 5, ageMin: 55, ageMax: 74, smoker: 1, slices: [1, 1.25, 2, 2.5], vendorW: [0.35, 0.35, 0.15, 0.15] },
  "LNDb-like": { weight: 0.15, ageMu: 66, ageSd: 11, ageMin: 35, ageMax: 90, smoker: 0.55, slices: [0.625, 1, 1.25], vendorW: [0.2, 0.55, 0.15, 0.1] },
};

function demographics(rand: () => number, ds: Dataset) {
  const p = DATASET_PROFILE[ds];
  const age = Math.round(clamp(p.ageMu + p.ageSd * gaussian(rand), p.ageMin, p.ageMax));
  const sex: "F" | "M" = rand() < (ds === "NLST-like" ? 0.41 : 0.46) ? "F" : "M";
  const smoker = rand() < p.smoker;
  const packYears = smoker ? Math.round(clamp(ds === "NLST-like" ? 30 + Math.abs(18 * gaussian(rand)) : 10 + Math.abs(30 * gaussian(rand)), 1, 150)) : 0;
  const vendor = pick(rand, VENDORS, p.vendorW);
  const sliceThicknessMm = p.slices[Math.floor(rand() * p.slices.length)];
  const kernel = pick(rand, KERNELS, [0.3, 0.4, 0.3]);
  return { age, sex, smokingPackYears: packYears, vendor, sliceThicknessMm, kernel };
}

let cache: { nodules: Nodule[]; hosts: HostScan[]; byId: Map<string, Nodule>; hostById: Map<string, HostScan> } | null = null;

export function getIndex() {
  if (cache) return cache;
  const rand = mulberry32(INDEX_SEED);
  const nodules: Nodule[] = [];
  let patient = 0;
  let left = 0;
  let demo: ReturnType<typeof demographics> = demographics(rand, "LIDC-IDRI-like");
  let ds: Dataset = "LIDC-IDRI-like";
  for (let i = 0; i < N_NODULES; i++) {
    // 1–3 nodules per patient (multi-nodule patients are common in LIDC/NLST).
    if (left <= 0) {
      ds = pick(rand, DATASETS, DATASETS.map((d) => DATASET_PROFILE[d].weight));
      demo = demographics(rand, ds);
      patient++;
      left = pick(rand, [1, 2, 3], [0.6, 0.28, 0.12]);
    }
    left--;
    const type = pick(rand, NODULE_TYPES, ds === "NLST-like" ? [0.74, 0.1, 0.16] : [0.8, 0.08, 0.12]);
    // Log-normal diameter skewed small (LIDC-like median ~6 mm); subsolid lesions skew larger.
    const mu = Math.log(type === "solid" ? 6 : type === "part-solid" ? 11 : 9);
    const diameterMm = round1(clamp(Math.exp(mu + 0.5 * gaussian(rand)), 3, 30));
    const solidComponentMm = type === "solid" ? diameterMm : type === "part-solid" ? round1(diameterMm * clamp(0.2 + 0.6 * rand(), 0.15, 0.85)) : 0;
    const volumeMm3 = Math.round((Math.PI / 6) * diameterMm ** 3 * clamp(0.85 + 0.3 * rand(), 0.7, 1.2));
    const lobe = pick(rand, LOBES, [0.3, 0.1, 0.18, 0.25, 0.17]);
    const location = pick(rand, LOCATIONS, [0.12, 0.4, 0.28, 0.2]);
    const calcified = type === "solid" && rand() < (diameterMm < 8 ? 0.11 : 0.04);
    const calcification = calcified ? pick(rand, CALCIFICATIONS.slice(1), [0.35, 0.2, 0.2, 0.25]) : "none";
    // Spiculation rises with size; GGNs are rarely spiculated.
    const spicLatent = (diameterMm - 6) / 6 + (type === "ground-glass" ? -1.2 : 0) + (calcified ? -1.5 : 0) + 0.9 * gaussian(rand);
    const spiculation = clamp(Math.round(1 + spicLatent), 1, 5);
    const margin: Nodule["margin"] =
      spiculation >= 4 ? "spiculated" : type === "ground-glass" ? (rand() < 0.6 ? "ill-defined" : "smooth") : spiculation === 3 ? "lobulated" : "smooth";
    // Malignancy (1–5, LIDC-style rating) ~ size, spiculation, part-solid, age, upper lobe.
    const malLatent = calcified
      ? -3
      : 0.16 * (diameterMm - 6) + 0.45 * (spiculation - 1) + (type === "part-solid" ? 0.9 : type === "ground-glass" ? 0.2 : 0) +
        (lobe === "RUL" || lobe === "LUL" ? 0.25 : 0) + 0.02 * (demo.age - 60) + 0.7 * gaussian(rand);
    const malignancy = clamp(Math.round(1.6 + malLatent), 1, 5);
    const { category } = lungRadsCategory({
      diameterMm,
      type,
      solidComponentMm,
      benignCalcification: calcified,
      juxtapleuralSmooth: location === "juxtapleural" && margin === "smooth",
      spiculated: spiculation >= 4,
    });
    nodules.push({
      id: `N${String(i + 1).padStart(5, "0")}`,
      patientId: `P${String(patient).padStart(5, "0")}`,
      sourceDataset: ds,
      ...demo,
      diameterMm,
      solidComponentMm,
      volumeMm3,
      type,
      lobe,
      location,
      spiculation,
      margin,
      calcification,
      malignancy,
      lungRads: category,
    });
  }
  const hosts: HostScan[] = [];
  for (let i = 0; i < N_HOSTS; i++) {
    const hds = pick(rand, DATASETS, DATASETS.map((d) => DATASET_PROFILE[d].weight));
    hosts.push({ id: `B${String(i + 1).padStart(5, "0")}`, sourceDataset: hds, ...demographics(rand, hds) });
  }
  cache = { nodules, hosts, byId: new Map(nodules.map((n) => [n.id, n])), hostById: new Map(hosts.map((h) => [h.id, h])) };
  return cache;
}

/* ── NoduleMap embedding: 2-D PCA of standardized numeric + one-hot attributes ── */
let embedCache: Float32Array | null = null;

export function getEmbedding(): Float32Array {
  if (embedCache) return embedCache;
  const { nodules } = getIndex();
  const rows = nodules.map((n) => [
    Math.log(n.diameterMm),
    n.spiculation,
    n.malignancy,
    n.age,
    n.smokingPackYears,
    n.sliceThicknessMm,
    ...NODULE_TYPES.map((t) => (n.type === t ? 2 : 0)),
    ...LOCATIONS.map((l) => (n.location === l ? 1 : 0)),
    ...LOBES.map((l) => (n.lobe === l ? 0.6 : 0)),
    n.calcification !== "none" ? 1.5 : 0,
  ]);
  const d = rows[0].length;
  const N = rows.length;
  const mean = new Array(d).fill(0);
  const sd = new Array(d).fill(0);
  for (const r of rows) r.forEach((v, j) => (mean[j] += v / N));
  for (const r of rows) r.forEach((v, j) => (sd[j] += (v - mean[j]) ** 2 / N));
  // Numeric columns are z-scored; one-hot columns are only centred so their weights act as importance.
  const X = rows.map((r) => r.map((v, j) => (j < 6 ? (v - mean[j]) / (Math.sqrt(sd[j]) || 1) : v - mean[j])));
  const C = Array.from({ length: d }, () => new Array(d).fill(0));
  for (const r of X) for (let a = 0; a < d; a++) for (let b = a; b < d; b++) C[a][b] += (r[a] * r[b]) / N;
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a];
  // Top-2 eigenvectors by power iteration with deflation (deterministic start vector).
  const comps: number[][] = [];
  for (let k = 0; k < 2; k++) {
    let v = Array.from({ length: d }, (_, j) => 1 + 0.1 * j);
    for (let it = 0; it < 200; it++) {
      const w = C.map((row) => row.reduce((s, c, j) => s + c * v[j], 0));
      for (const p of comps) {
        const dot = w.reduce((s, x, j) => s + x * p[j], 0);
        for (let j = 0; j < d; j++) w[j] -= dot * p[j];
      }
      const norm = Math.hypot(...w) || 1;
      v = w.map((x) => x / norm);
    }
    comps.push(v);
  }
  const out = new Float32Array(N * 2);
  const jit = mulberry32(7);
  X.forEach((r, i) => {
    out[2 * i] = r.reduce((s, x, j) => s + x * comps[0][j], 0) + 0.06 * gaussian(jit);
    out[2 * i + 1] = r.reduce((s, x, j) => s + x * comps[1][j], 0) + 0.06 * gaussian(jit);
  });
  embedCache = out;
  return out;
}

/* ── Faceted filtering ── */
export type Facets = {
  diameterMm?: [number, number];
  types?: NoduleType[];
  lobes?: Lobe[];
  locations?: Location[];
  lungRads?: LungRads[];
  datasets?: Dataset[];
  vendors?: (typeof VENDORS)[number][];
  ageRange?: [number, number];
  sex?: "F" | "M";
  minPackYears?: number;
  minMalignancy?: number;
  excludeCalcified?: boolean;
};

const has = <T,>(list: readonly T[] | undefined, v: T) => !list || list.length === 0 || list.includes(v);

export function matchesNodule(n: Nodule, f: Facets) {
  return (
    (!f.diameterMm || (n.diameterMm >= f.diameterMm[0] && n.diameterMm <= f.diameterMm[1])) &&
    has(f.types, n.type) &&
    has(f.lobes, n.lobe) &&
    has(f.locations, n.location) &&
    has(f.lungRads, n.lungRads) &&
    has(f.datasets, n.sourceDataset) &&
    has(f.vendors, n.vendor) &&
    matchesDemo(n, f) &&
    (f.minMalignancy === undefined || n.malignancy >= f.minMalignancy) &&
    (!f.excludeCalcified || n.calcification === "none")
  );
}

/** Demographic / acquisition facets that also apply to nodule-free host scans. */
export function matchesDemo(h: Pick<HostScan, "age" | "sex" | "smokingPackYears">, f: Facets) {
  return (
    (!f.ageRange || (h.age >= f.ageRange[0] && h.age <= f.ageRange[1])) &&
    (!f.sex || h.sex === f.sex) &&
    (f.minPackYears === undefined || h.smokingPackYears >= f.minPackYears)
  );
}

export function matchesHost(h: HostScan, f: Facets) {
  return matchesDemo(h, f) && has(f.datasets, h.sourceDataset) && has(f.vendors, h.vendor);
}

export const filterNodules = (f: Facets) => getIndex().nodules.filter((n) => matchesNodule(n, f));
export const filterHosts = (f: Facets) => getIndex().hosts.filter((h) => matchesHost(h, f));

/* ── Stratification factors ── */
export const FACTORS = ["sizeBin", "type", "lobe", "location", "lungRads", "dataset", "vendor", "sex", "ageBin"] as const;
export type Factor = (typeof FACTORS)[number];

export const factorValue = (n: Nodule, f: Factor): string =>
  f === "sizeBin"
    ? sizeBinOf(n.diameterMm)
    : f === "type"
      ? n.type
      : f === "lobe"
        ? n.lobe
        : f === "location"
          ? n.location
          : f === "lungRads"
            ? n.lungRads
            : f === "dataset"
              ? n.sourceDataset
              : f === "vendor"
                ? n.vendor
                : f === "sex"
                  ? n.sex
                  : ageBinOf(n.age);

/** Levels of a factor that are admissible under the facets (zero-count levels are kept so they can be flagged). */
export function factorLevels(f: Factor, facets: Facets): string[] {
  const sel = <T extends string>(all: readonly T[], chosen?: T[]) => (chosen?.length ? all.filter((x) => chosen.includes(x)) : [...all]);
  switch (f) {
    case "sizeBin": {
      const [lo, hi] = facets.diameterMm ?? [0, Infinity];
      return SIZE_BINS.filter((_, i) => SIZE_EDGES[i] <= hi && SIZE_EDGES[i + 1] > lo);
    }
    case "type":
      return sel(NODULE_TYPES, facets.types);
    case "lobe":
      return sel(LOBES, facets.lobes);
    case "location":
      return sel(LOCATIONS, facets.locations);
    case "lungRads":
      return sel(LUNG_RADS, facets.lungRads);
    case "dataset":
      return sel(DATASETS, facets.datasets);
    case "vendor":
      return sel(VENDORS, facets.vendors);
    case "sex":
      return facets.sex ? [facets.sex] : ["F", "M"];
    case "ageBin": {
      const [lo, hi] = facets.ageRange ?? [0, 200];
      const edges = [0, 55, 65, 75, 200];
      return AGE_BINS.filter((_, i) => edges[i] <= hi && edges[i + 1] > lo);
    }
  }
}

export function histogram<T>(items: T[], key: (x: T) => string, levels?: readonly string[]): Record<string, number> {
  const h: Record<string, number> = {};
  for (const l of levels ?? []) h[l] = 0;
  for (const x of items) {
    const k = key(x);
    h[k] = (h[k] ?? 0) + 1;
  }
  return h;
}

export function noduleHistograms(ns: Nodule[]) {
  return {
    sizeBin: histogram(ns, (n) => sizeBinOf(n.diameterMm), SIZE_BINS),
    type: histogram(ns, (n) => n.type, NODULE_TYPES),
    lobe: histogram(ns, (n) => n.lobe, LOBES),
    location: histogram(ns, (n) => n.location, LOCATIONS),
    lungRads: histogram(ns, (n) => n.lungRads, LUNG_RADS),
    dataset: histogram(ns, (n) => n.sourceDataset, DATASETS),
    malignancy: histogram(ns, (n) => String(n.malignancy), ["1", "2", "3", "4", "5"]),
  };
}
