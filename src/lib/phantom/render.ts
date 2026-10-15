import { KERNEL_BLUR_PX, LUNG_HU, NODULE_HU, gaussian, mulberry32, noiseSigmaHU } from "@/lib/trial/physics";
import type { ScannerArm, TrialSpec } from "@/lib/trial/spec";

/**
 * Procedural chest-CT phantom + lesion insertion + imaging-chain simulation.
 *
 * Pipeline mirrors a generative VIT:
 *   anatomy label map (the "conditioning mask" a ControlNet / MAISI model consumes)
 *   → lesion inserted into the mask with exact size / type / lobe
 *   → HU assignment + parenchymal texture
 *   → scanner simulation: kernel MTF blur, vendor-textured correlated noise at the dose-dependent σ
 *   → display windowing.
 * The diffusion model is replaced by a deterministic phantom so it runs in-browser in ~30 ms.
 */

export const LABELS = {
  air: 0,
  body: 1,
  fat: 2,
  bone: 3,
  heart: 4,
  vessel: 5,
  RUL: 10,
  RML: 11,
  RLL: 12,
  LUL: 13,
  LLL: 14,
  nodule: 20,
} as const;

export type PhantomParams = {
  size: number; // output pixels (square)
  fovMm: number;
  zMm: number; // slice position, 0 = apex, 260 = base
  seed: number;
  bmi: number;
  nodule: {
    diameterMm: number;
    type: TrialSpec["noduleTypes"][number];
    lobe: TrialSpec["lobes"][number];
    juxtavascular: boolean;
    spiculated: boolean;
    pos: [number, number, number];
  } | null;
  scanner: ScannerArm;
};

export const Z_EXTENT_MM = 260;

type Vec3 = [number, number, number];

// Lungs in normalised image coords: x ∈ [-1,1] (patient right = image left), y ∈ [-1,1] (anterior = top).
function lungShape(z: number) {
  // z normalised 0 (apex) → 1 (base). Lungs widen toward the base; the diaphragm dome invades from below.
  const grow = Math.sin(Math.min(1, 0.15 + z / 0.6) * (Math.PI / 2));
  return {
    R: { cx: -0.4, cy: 0.0, a: 0.1 + 0.27 * grow, b: 0.14 + 0.38 * grow },
    L: { cx: 0.42, cy: 0.02, a: 0.09 + 0.26 * grow, b: 0.13 + 0.37 * grow },
    dome: z > 0.72 ? (z - 0.72) / 0.28 : 0, // fraction of lung replaced by diaphragm/liver
  };
}

function lobeAt(x: number, y: number, z: number, right: boolean) {
  // Oblique fissure: posterior-inferior region is the lower lobe.
  const t = z + 0.45 * y;
  if (t > 0.62) return right ? LABELS.RLL : LABELS.LLL;
  if (right && z > 0.48) return LABELS.RML;
  return right ? LABELS.RUL : LABELS.LUL;
}

function anatomyLabel(x: number, y: number, z: number): number {
  const bodyA = 0.9;
  const bodyB = 0.66;
  const rb = (x / bodyA) ** 2 + ((y - 0.02) / bodyB) ** 2;
  if (rb > 1) return LABELS.air;
  if (rb > 0.86) return LABELS.fat;
  // Spine & sternum
  if (x * x + (y - 0.5) ** 2 < 0.012) return LABELS.bone;
  if (Math.abs(x) < 0.05 && Math.abs(y + 0.58) < 0.03 && z > 0.2) return LABELS.bone;
  // Ribs: discrete arcs near the body wall, periodic in z
  if (rb > 0.72 && rb < 0.8 && Math.sin(z * 60 + Math.atan2(y, x) * 2) > 0.55) return LABELS.bone;
  // Heart / mediastinum
  const heartR = z < 0.35 ? 0.11 : 0.27 * Math.min(1, (z - 0.25) / 0.35);
  if ((x - 0.1) ** 2 + (y + 0.05) ** 2 < heartR * heartR) return LABELS.heart;
  if (Math.abs(x) < 0.12 && y < 0.4) return LABELS.body;
  const s = lungShape(z);
  for (const [k, e] of [
    ["R", s.R],
    ["L", s.L],
  ] as const) {
    const r = ((x - e.cx) / e.a) ** 2 + ((y - e.cy) / e.b) ** 2;
    // diaphragm dome rises from anterior-inferior
    if (r < 1 && !(s.dome > 0 && r < s.dome * 1.1 && y < 0.3)) return lobeAt(x, y, z, k === "R");
  }
  return LABELS.body;
}

/** Deterministic pulmonary vessel tree: branching 3D polylines from each hilum. */
function vesselTree(seed: number) {
  const rand = mulberry32(seed * 7 + 3);
  const pts: { p: Vec3; r: number }[] = [];
  const grow = (p: Vec3, d: Vec3, r: number, depth: number) => {
    const len = 0.1 + rand() * 0.12;
    const steps = 10;
    let q: Vec3 = [...p];
    for (let i = 0; i < steps; i++) {
      q = [q[0] + (d[0] * len) / steps, q[1] + (d[1] * len) / steps, q[2] + (d[2] * len) / steps];
      pts.push({ p: q, r: r * (1 - (0.3 * i) / steps) });
    }
    if (depth > 0 && r > 0.004) {
      for (let b = 0; b < 2; b++) {
        const nd: Vec3 = [d[0] + (rand() - 0.5) * 0.9, d[1] + (rand() - 0.5) * 0.9, d[2] + (rand() - 0.5) * 0.9];
        const n = Math.hypot(...nd);
        grow(q, [nd[0] / n, nd[1] / n, nd[2] / n], r * 0.72, depth - 1);
      }
    }
  };
  for (const side of [-1, 1]) {
    const hilum: Vec3 = [side * 0.2, 0.05, 0.45];
    for (let k = 0; k < 9; k++) {
      const ang = (k / 9) * Math.PI * 2 + rand();
      const d: Vec3 = [side * (0.7 + rand() * 0.3), Math.sin(ang) * 0.5, Math.cos(ang) * 0.9];
      const n = Math.hypot(...d);
      grow(hilum, [d[0] / n, d[1] / n, d[2] / n], 0.024, 7);
    }
  }
  return pts;
}

const vesselCache = new Map<number, ReturnType<typeof vesselTree>>();

/** Find a plausible nodule centre inside the requested lobe from normalised (u,v,w). */
export function placeNodule(lobe: keyof typeof LABELS, pos: [number, number, number]): Vec3 {
  const target = LABELS[lobe];
  const zRange: Record<string, [number, number]> = {
    RUL: [0.18, 0.45],
    LUL: [0.18, 0.55],
    RML: [0.52, 0.68],
    RLL: [0.6, 0.78],
    LLL: [0.6, 0.78],
  };
  const [z0, z1] = zRange[lobe] ?? [0.3, 0.6];
  const z = z0 + (z1 - z0) * pos[2];
  const candidates: Vec3[] = [];
  for (let gy = -0.6; gy <= 0.6; gy += 0.02)
    for (let gx = -0.85; gx <= 0.85; gx += 0.02) {
      // Require margin from lobe boundary so the nodule sits in parenchyma.
      if (
        anatomyLabel(gx, gy, z) === target &&
        anatomyLabel(gx + 0.05, gy, z) === target &&
        anatomyLabel(gx - 0.05, gy, z) === target &&
        anatomyLabel(gx, gy + 0.05, z) === target &&
        anatomyLabel(gx, gy - 0.05, z) === target
      )
        candidates.push([gx, gy, z]);
    }
  if (!candidates.length) return [lobe.startsWith("R") ? -0.4 : 0.4, 0, z];
  const i = Math.floor(((pos[0] * 0.7 + pos[1] * 0.3) % 1) * candidates.length);
  return candidates[Math.min(candidates.length - 1, i)];
}

function blur(src: Float32Array, n: number, sigma: number) {
  if (sigma < 0.15) return src;
  const rad = Math.ceil(sigma * 3);
  const k = Array.from({ length: 2 * rad + 1 }, (_, i) => Math.exp(-((i - rad) ** 2) / (2 * sigma * sigma)));
  const ks = k.reduce((a, b) => a + b, 0);
  const tmp = new Float32Array(src.length);
  const out = new Float32Array(src.length);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let i = -rad; i <= rad; i++) s += src[y * n + Math.min(n - 1, Math.max(0, x + i))] * k[i + rad];
      tmp[y * n + x] = s / ks;
    }
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      let s = 0;
      for (let i = -rad; i <= rad; i++) s += tmp[Math.min(n - 1, Math.max(0, y + i)) * n + x] * k[i + rad];
      out[y * n + x] = s / ks;
    }
  return out;
}

/** Vendor-specific noise texture (noise power spectrum shape) — grain correlation length in px. */
const VENDOR_GRAIN: Record<ScannerArm["vendor"], number> = { GE: 1.05, Siemens: 0.75, Philips: 0.9, Canon: 0.65 };

export type RenderResult = {
  hu: Float32Array; // simulated CT (HU)
  clean: Float32Array; // noise-free ground truth (HU)
  labels: Uint8Array;
  size: number;
  sigmaHU: number;
  noduleCenterPx: [number, number] | null;
  noduleRadiusPxInSlice: number;
  noduleCenterZmm: number | null;
};

export function renderSlice(p: PhantomParams): RenderResult {
  const n = p.size;
  const mmPerPx = p.fovMm / n;
  const z = p.zMm / Z_EXTENT_MM;
  const labels = new Uint8Array(n * n);
  const clean = new Float32Array(n * n);
  const rand = mulberry32(p.seed * 131 + Math.round(p.zMm * 10));
  const toNorm = (i: number) => (i / n) * 2 - 1; // px → normalised (FOV covers [-1,1])
  const normPerMm = 2 / p.fovMm;

  // 1. Anatomy label map
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) labels[j * n + i] = anatomyLabel(toNorm(i) * 1.05, toNorm(j) * 1.05, z);

  // 2. Vessels (only drawn inside lungs)
  if (!vesselCache.has(p.seed)) vesselCache.set(p.seed, vesselTree(p.seed));
  const vessels = vesselCache.get(p.seed)!;
  const vesselMask = new Float32Array(n * n);
  const zNorm = z;
  const zScale = 0.9; // vessel tree z units → normalised z
  for (const { p: vp, r } of vessels) {
    const dz = (vp[2] - zNorm) / zScale;
    const rr = r * r - dz * dz * 0.25;
    if (rr <= 0) continue;
    const rad = Math.sqrt(rr);
    const cx = ((vp[0] / 1.05 + 1) / 2) * n;
    const cy = ((vp[1] / 1.05 + 1) / 2) * n;
    const rp = Math.max(0.6, (rad / 2) * n);
    for (let j = Math.floor(cy - rp - 1); j <= cy + rp + 1; j++)
      for (let i = Math.floor(cx - rp - 1); i <= cx + rp + 1; i++) {
        if (i < 0 || j < 0 || i >= n || j >= n) continue;
        const d = Math.hypot(i + 0.5 - cx, j + 0.5 - cy);
        if (d < rp + 0.5) {
          const l = labels[j * n + i];
          if (l >= 10 && l <= 14) vesselMask[j * n + i] = Math.max(vesselMask[j * n + i], Math.min(1, rp + 0.5 - d));
        }
      }
  }

  // 3. Lesion insertion into the label map
  let noduleCenterPx: [number, number] | null = null;
  let noduleRadiusPxInSlice = 0;
  let noduleCenterZmm: number | null = null;
  const noduleFrac = new Float32Array(n * n);
  if (p.nodule) {
    let c = placeNodule(p.nodule.lobe, p.nodule.pos);
    if (p.nodule.juxtavascular) {
      // Snap next to the closest vessel at that height.
      let best = vessels[0];
      let bd = Infinity;
      for (const v of vessels) {
        const d = Math.hypot(v.p[0] - c[0], v.p[1] - c[1], (v.p[2] - c[2]) * 0.5);
        if (v.r > 0.006 && d < bd) {
          bd = d;
          best = v;
        }
      }
      const off = (p.nodule.diameterMm / 2) * normPerMm * 1.05 + best.r;
      c = [best.p[0] + off * 0.7, best.p[1] + off * 0.7, best.p[2]];
    }
    noduleCenterZmm = c[2] * Z_EXTENT_MM;
    const rMm = p.nodule.diameterMm / 2;
    const dzMm = p.zMm - noduleCenterZmm;
    const rSliceMm = rMm * rMm - dzMm * dzMm > 0 ? Math.sqrt(rMm * rMm - dzMm * dzMm) : 0;
    const cx = ((c[0] / 1.05 + 1) / 2) * n;
    const cy = ((c[1] / 1.05 + 1) / 2) * n;
    noduleCenterPx = [cx, cy];
    noduleRadiusPxInSlice = rSliceMm / mmPerPx;
    if (rSliceMm > 0) {
      const rp = rSliceMm / mmPerPx;
      const spikeRand = mulberry32(p.seed + 99);
      const spikes = Array.from({ length: 9 }, () => ({ a: spikeRand() * Math.PI * 2, w: 0.08 + spikeRand() * 0.1, h: 0.4 + spikeRand() * 0.8 }));
      for (let j = Math.floor(cy - rp * 2.5); j <= cy + rp * 2.5; j++)
        for (let i = Math.floor(cx - rp * 2.5); i <= cx + rp * 2.5; i++) {
          if (i < 0 || j < 0 || i >= n || j >= n) continue;
          const dx = i + 0.5 - cx;
          const dy = j + 0.5 - cy;
          let rEff = rp;
          if (p.nodule.spiculated) {
            const ang = Math.atan2(dy, dx);
            for (const s of spikes) {
              const da = Math.atan2(Math.sin(ang - s.a), Math.cos(ang - s.a));
              rEff += rp * s.h * Math.exp(-(da * da) / (2 * s.w * s.w * 0.15));
            }
          }
          const d = Math.hypot(dx, dy);
          const f = Math.max(0, Math.min(1, rEff + 0.5 - d)); // anti-aliased partial volume
          if (f > 0) {
            noduleFrac[j * n + i] = Math.max(noduleFrac[j * n + i], f);
            if (f > 0.5) labels[j * n + i] = LABELS.nodule;
          }
        }
    }
  }

  // 4. HU assignment with tissue texture
  const tex = new Float32Array(n * n);
  for (let k = 0; k < n * n; k++) tex[k] = gaussian(rand);
  const texture = blur(tex, n, 2.2);
  for (let k = 0; k < n * n; k++) {
    const l = labels[k];
    let hu = -1000;
    if (l === LABELS.body) hu = 45;
    else if (l === LABELS.fat) hu = -95;
    else if (l === LABELS.bone) hu = 650;
    else if (l === LABELS.heart) hu = 40;
    else if ((l >= 10 && l <= 14) || l === LABELS.nodule) hu = LUNG_HU + 60 * texture[k];
    if (l === LABELS.air) hu = -1000;
    if (vesselMask[k] > 0) hu = hu * (1 - vesselMask[k]) + 40 * vesselMask[k];
    clean[k] = hu;
  }
  // Nodule HU (partial-volume blended); part-solid = GGO halo + solid core.
  if (p.nodule && noduleCenterPx) {
    const [cx, cy] = noduleCenterPx;
    for (let k = 0; k < n * n; k++) {
      const f = noduleFrac[k];
      if (!f) continue;
      let target = NODULE_HU[p.nodule.type];
      if (p.nodule.type === "part-solid") {
        const i = k % n;
        const j = Math.floor(k / n);
        const d = Math.hypot(i + 0.5 - cx, j + 0.5 - cy);
        target = d < noduleRadiusPxInSlice * 0.5 ? NODULE_HU.solid : NODULE_HU["ground-glass"] + 100;
      }
      if (p.nodule.type === "ground-glass") target += 40 * texture[k];
      clean[k] = clean[k] * (1 - f) + target * f;
    }
  }

  // 5. Imaging chain: kernel MTF (blur / edge enhancement) + correlated quantum noise.
  const kernelSigma = KERNEL_BLUR_PX[p.scanner.kernel] * (0.7 / mmPerPx);
  let img = blur(clean, n, kernelSigma);
  if (p.scanner.kernel === "sharp") {
    const soft = blur(img, n, 1.2);
    img = img.map((v, k) => v + 0.6 * (v - soft[k]));
  }
  const sigmaHU = noiseSigmaHU(p.scanner, p.bmi);
  const raw = new Float32Array(n * n);
  for (let k = 0; k < n * n; k++) raw[k] = gaussian(rand);
  const corr = blur(raw, n, VENDOR_GRAIN[p.scanner.vendor] * (p.scanner.kernel === "soft" ? 1.4 : p.scanner.kernel === "sharp" ? 0.6 : 1));
  let s2 = 0;
  for (let k = 0; k < n * n; k++) s2 += corr[k] * corr[k];
  const scale = sigmaHU / Math.sqrt(s2 / (n * n));
  const hu = new Float32Array(n * n);
  for (let k = 0; k < n * n; k++) hu[k] = labels[k] === LABELS.air && clean[k] <= -999 ? -1000 + corr[k] * scale * 0.3 : img[k] + corr[k] * scale;

  return { hu, clean, labels, size: n, sigmaHU, noduleCenterPx, noduleRadiusPxInSlice, noduleCenterZmm };
}

export const WINDOWS = {
  lung: { level: -600, width: 1500, label: "Lung (W1500 / L-600)" },
  mediastinum: { level: 40, width: 400, label: "Mediastinum (W400 / L40)" },
  bone: { level: 400, width: 1800, label: "Bone (W1800 / L400)" },
} as const;

/** Dark-mode categorical slots (validated palette) for the five lobes. */
export const LABEL_COLORS: Record<number, [number, number, number]> = {
  [LABELS.air]: [8, 10, 14],
  [LABELS.body]: [70, 72, 78],
  [LABELS.fat]: [110, 100, 70],
  [LABELS.bone]: [225, 225, 215],
  [LABELS.heart]: [130, 60, 70],
  [LABELS.vessel]: [200, 80, 80],
  [LABELS.RUL]: [0x39, 0x87, 0xe5],
  [LABELS.RML]: [0xd9, 0x59, 0x26],
  [LABELS.RLL]: [0x19, 0x9e, 0x70],
  [LABELS.LUL]: [0xc9, 0x85, 0x00],
  [LABELS.LLL]: [0xd5, 0x51, 0x81],
  [LABELS.nodule]: [255, 255, 255],
};

export function toImageData(r: RenderResult, mode: "ct" | "mask" | "overlay" | "clean", win: { level: number; width: number }) {
  const n = r.size;
  const out = new ImageData(n, n);
  const lo = win.level - win.width / 2;
  for (let k = 0; k < n * n; k++) {
    const src = mode === "clean" ? r.clean[k] : r.hu[k];
    const g = Math.max(0, Math.min(255, ((src - lo) / win.width) * 255));
    let [R, G, B] = [g, g, g];
    if (mode === "mask") [R, G, B] = LABEL_COLORS[r.labels[k]] ?? [0, 0, 0];
    if (mode === "overlay" && r.labels[k] === LABELS.nodule) [R, G, B] = [g * 0.4 + 230 * 0.6, g * 0.4 + 60 * 0.6, g * 0.4 + 60 * 0.6];
    out.data[k * 4] = R;
    out.data[k * 4 + 1] = G;
    out.data[k * 4 + 2] = B;
    out.data[k * 4 + 3] = 255;
  }
  return out;
}
