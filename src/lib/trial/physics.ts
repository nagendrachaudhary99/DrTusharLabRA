import type { ScannerArm } from "./spec";

/**
 * Simplified imaging-chain model shared by the trial simulator and the
 * Virtual Patient Studio renderer. Values are illustrative, chosen to reproduce
 * qualitative behaviour (noise ∝ 1/√dose, sharper kernels → higher noise,
 * larger patients → higher noise), not calibrated to any specific scanner.
 */
export const VENDOR_BASE_NOISE_HU: Record<ScannerArm["vendor"], number> = {
  GE: 12.5,
  Siemens: 11,
  Philips: 12,
  Canon: 11.5,
};
export const KERNEL_NOISE: Record<ScannerArm["kernel"], number> = { soft: 0.75, standard: 1, sharp: 1.9 };
/** Relative dose; noise scales with 1/sqrt(dose). */
export const DOSE_FRACTION: Record<ScannerArm["dose"], number> = { standard: 1, low: 0.25, "ultra-low": 0.1 };
/** Gaussian blur sigma in pixels approximating each kernel's MTF. */
export const KERNEL_BLUR_PX: Record<ScannerArm["kernel"], number> = { soft: 1.1, standard: 0.7, sharp: 0.25 };

export const NODULE_HU: Record<"solid" | "part-solid" | "ground-glass", number> = {
  solid: 30,
  "part-solid": -350,
  "ground-glass": -600,
};
export const LUNG_HU = -860;
/** Anatomical "clutter" (vessels, fissures, parenchymal texture) adds in quadrature with quantum noise. */
export const ANATOMICAL_CLUTTER_HU = 60;

export function noiseSigmaHU(arm: ScannerArm, bmi = 25) {
  return VENDOR_BASE_NOISE_HU[arm.vendor] * KERNEL_NOISE[arm.kernel] * (1 / Math.sqrt(DOSE_FRACTION[arm.dose])) * Math.pow(bmi / 25, 1.2);
}

/**
 * Rose-model style detectability index d' = CNR × sqrt(N_pixels), with a
 * kernel-dependent blur loss for small objects (partial-volume effect) and
 * anatomical clutter in the noise term.
 */
export function detectability(opts: { diameterMm: number; contrastHU: number; sigmaHU: number; kernel: ScannerArm["kernel"]; pixelMm?: number }) {
  const px = opts.pixelMm ?? 0.7;
  const rPx = opts.diameterMm / 2 / px;
  const blur = KERNEL_BLUR_PX[opts.kernel];
  const pve = (rPx * rPx) / (rPx * rPx + 2 * blur * blur + 0.5);
  const cnr = (opts.contrastHU * pve) / Math.hypot(opts.sigmaHU, ANATOMICAL_CLUTTER_HU);
  return { cnr, dprime: cnr * Math.sqrt(Math.PI * rPx * rPx) };
}

/** Deterministic PRNG so every virtual trial is exactly reproducible from its seed. */
export function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function gaussian(rand: () => number) {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}
