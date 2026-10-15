import { mulberry32 } from "@/lib/trial/physics";
import { renderSlice, placeNodule, Z_EXTENT_MM, type RenderResult } from "@/lib/phantom/render";
import type { ScannerArm, TrialSpec } from "@/lib/trial/spec";
import { FOV_MM, IMAGE_PX, type GroundTruth } from "./schema";

/**
 * Factorial design → concrete benchmark cases. Pure and deterministic (seeded), so the same design
 * always yields the same images; rendering uses the in-browser procedural phantom.
 */

export type BenchDesign = {
  sizesMm: number[]; // 0 = negative control (no nodule)
  noduleTypes: TrialSpec["noduleTypes"];
  lobes: TrialSpec["lobes"];
  scanners: ScannerArm[];
  nCases: number;
  seed: number;
};

export type CaseSpec = {
  id: string;
  groundTruth: GroundTruth;
  bmi: number;
  pos: [number, number, number];
  /** Lobe used to choose the slice level for negatives (so negatives sample the same anatomy). */
  sliceLobe: TrialSpec["lobes"][number];
};

export function buildCases(d: BenchDesign): CaseSpec[] {
  const rand = mulberry32(d.seed * 977 + 13);
  // Full factorial grid; negatives (size 0) are crossed with lobe (slice level) × scanner only.
  const cells: { size: number; type: GroundTruth["type"]; lobe: TrialSpec["lobes"][number]; arm: ScannerArm }[] = [];
  for (const size of d.sizesMm)
    for (const type of size === 0 ? (["none"] as const) : d.noduleTypes)
      for (const lobe of d.lobes) for (const arm of d.scanners) cells.push({ size, type, lobe, arm });
  // Balanced sampling: shuffle the grid, then cycle through it so every cell appears before any repeats.
  const order = cells.map((c) => ({ c, r: rand() })).sort((a, b) => a.r - b.r).map((x) => x.c);
  // Interleave by size so small n still spans all size strata.
  const bySize = d.sizesMm.map((s) => order.filter((c) => c.size === s));
  const picked: typeof cells = [];
  for (let i = 0; picked.length < d.nCases && bySize.some((g) => g.length); i++) {
    const g = bySize[i % bySize.length];
    if (g.length) picked.push(g[Math.floor(i / bySize.length) % g.length]);
  }
  return picked.map((c, i) => ({
    id: `C${String(i + 1).padStart(2, "0")}`,
    bmi: +(21 + rand() * 11).toFixed(1),
    pos: [rand(), rand(), 0.2 + rand() * 0.6],
    sliceLobe: c.lobe,
    groundTruth: {
      hasNodule: c.size > 0,
      diameterMm: c.size,
      type: c.type,
      lobe: c.size > 0 ? c.lobe : "none",
      vendor: c.arm.vendor,
      kernel: c.arm.kernel,
      dose: c.arm.dose,
    },
  }));
}

/** Render the axial slice through the nodule centre (or the equivalent level for negatives). */
export function renderCase(c: CaseSpec, seed: number): RenderResult {
  const g = c.groundTruth;
  const scanner = { vendor: g.vendor, kernel: g.kernel, dose: g.dose };
  const zMm = placeNodule(c.sliceLobe, c.pos)[2] * Z_EXTENT_MM;
  const nodule = g.hasNodule && g.type !== "none" && g.lobe !== "none"
    ? { diameterMm: g.diameterMm, type: g.type, lobe: g.lobe, juxtavascular: false, spiculated: false, pos: c.pos }
    : null;
  const base = { size: IMAGE_PX, fovMm: FOV_MM, seed: seed + Number(c.id.slice(1)), bmi: c.bmi, nodule, scanner };
  const r = renderSlice({ ...base, zMm });
  const zc = r.noduleCenterZmm ?? zMm;
  return Math.abs(zc - zMm) > 0.5 ? renderSlice({ ...base, zMm: zc }) : r;
}
