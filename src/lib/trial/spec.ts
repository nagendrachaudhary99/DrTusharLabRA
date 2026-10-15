import { z } from "zod";

export const VENDORS = ["GE", "Siemens", "Philips", "Canon"] as const;
export const KERNELS = ["soft", "standard", "sharp"] as const;
export const DOSES = ["standard", "low", "ultra-low"] as const;
export const LOBES = ["RUL", "RML", "RLL", "LUL", "LLL"] as const;
export const NODULE_TYPES = ["solid", "part-solid", "ground-glass"] as const;
export const MODEL_PROFILES = ["robust-baseline", "vendor-overfit", "lowdose-fragile"] as const;

export const LOBE_NAMES: Record<(typeof LOBES)[number], string> = {
  RUL: "Right upper lobe",
  RML: "Right middle lobe",
  RLL: "Right lower lobe",
  LUL: "Left upper lobe",
  LLL: "Left lower lobe",
};

export const MODEL_PROFILE_INFO: Record<(typeof MODEL_PROFILES)[number], { name: string; blurb: string }> = {
  "robust-baseline": {
    name: "NoduleNet-R (robust baseline)",
    blurb: "Trained on multi-vendor, multi-dose data. Expected to degrade mainly on very small / ground-glass nodules.",
  },
  "vendor-overfit": {
    name: "NoduleNet-S (single-site model)",
    blurb: "Trained only on Siemens standard-kernel scans. A hidden domain-shift failure the virtual trial should expose.",
  },
  "lowdose-fragile": {
    name: "NoduleNet-HD (high-dose model)",
    blurb: "Trained on routine-dose diagnostic CT. Brittle under low-dose screening noise.",
  },
};

export const ScannerArm = z.object({
  vendor: z.enum(VENDORS),
  kernel: z.enum(KERNELS),
  dose: z.enum(DOSES).describe("standard≈routine dose, low≈LDCT screening (~25% dose), ultra-low≈~10% dose"),
});

export const TrialSpec = z.object({
  title: z.string().describe("Short trial title"),
  objective: z.string().describe("One-sentence primary objective"),
  modelUnderTest: z.enum(MODEL_PROFILES),
  sizesMm: z.array(z.number().min(2).max(40)).min(1).max(8).describe("Nodule diameter strata in mm"),
  noduleTypes: z.array(z.enum(NODULE_TYPES)).min(1),
  lobes: z.array(z.enum(LOBES)).min(1),
  juxtavascularFraction: z.number().min(0).max(1).describe("Fraction of nodules attached to vessels / pleura"),
  scanners: z.array(ScannerArm).min(1).max(8),
  patientsPerArm: z.number().int().min(10).max(600),
  lesionsPerPatient: z.number().int().min(1).max(3),
  seed: z.number().int(),
});

export type TrialSpec = z.infer<typeof TrialSpec>;
export type ScannerArm = z.infer<typeof ScannerArm>;

export const DEFAULT_SPEC: TrialSpec = {
  title: "Multi-vendor nodule detection stress test",
  objective: "Estimate per-nodule sensitivity of the detector across nodule size, type, lobe and CT scanner protocol.",
  modelUnderTest: "vendor-overfit",
  sizesMm: [4, 6, 8, 10, 15, 20],
  noduleTypes: ["solid", "part-solid", "ground-glass"],
  lobes: [...LOBES],
  juxtavascularFraction: 0.25,
  scanners: [
    { vendor: "Siemens", kernel: "standard", dose: "standard" },
    { vendor: "GE", kernel: "standard", dose: "standard" },
    { vendor: "Philips", kernel: "sharp", dose: "low" },
    { vendor: "Canon", kernel: "soft", dose: "low" },
  ],
  patientsPerArm: 120,
  lesionsPerPatient: 2,
  seed: 42,
};

export const EXAMPLE_PROMPTS = [
  "Stress-test our nodule detector on 4-12 mm nodules in the right upper lobe across GE vs Siemens scanners, standard and low dose. Include ground-glass nodules.",
  "Does the single-site model generalise? Compare Siemens, GE, Philips and Canon at standard dose with sharp vs soft kernels, all lobes, 150 patients per arm.",
  "Lung cancer screening scenario: low-dose and ultra-low-dose CT, solid and part-solid nodules 6-20 mm, both lower lobes, test the high-dose model.",
];
