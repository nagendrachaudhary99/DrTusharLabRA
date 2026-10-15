import { z } from "zod";
import { DOSES, KERNELS, LOBES, NODULE_TYPES, VENDORS } from "@/lib/trial/spec";

/**
 * Shared (client + server) contracts for the VLM benchmark.
 *
 * A "case" is one synthetic axial CT slice rendered in the browser from a factorial design cell,
 * plus its ground truth. The reader (a vision-language model) answers three tasks per image in a
 * single structured call: detection, lobe localization and diameter estimation.
 */

export const LOBE_OR_NONE = [...LOBES, "none"] as const;
export type LobeOrNone = (typeof LOBE_OR_NONE)[number];

export const MAX_CASES = 48;
/** ~256 px lung-window PNGs are 40–90 kB as base64; leave headroom but reject anything absurd. */
export const MAX_IMAGE_CHARS = 400_000;

/** Display geometry baked into every rendered slice; the prompt quotes it so size estimation is possible. */
export const IMAGE_PX = 256;
export const FOV_MM = 380;

export const GroundTruth = z.object({
  hasNodule: z.boolean(),
  diameterMm: z.number().min(0).max(40),
  type: z.enum([...NODULE_TYPES, "none"]),
  lobe: z.enum(LOBE_OR_NONE),
  vendor: z.enum(VENDORS),
  kernel: z.enum(KERNELS),
  dose: z.enum(DOSES),
});
export type GroundTruth = z.infer<typeof GroundTruth>;

export const VlmCase = z.object({
  id: z.string().min(1).max(40),
  imageDataUrl: z
    .string()
    .max(MAX_IMAGE_CHARS)
    .regex(/^data:image\/png;base64,[A-Za-z0-9+/=]+$/, "must be a base64 PNG data URL"),
  groundTruth: GroundTruth,
});
export type VlmCase = z.infer<typeof VlmCase>;

/** Structured output requested from the VLM — one object answers all three tasks. */
export const VlmAnswer = z.object({
  nodulePresent: z.boolean().describe("Task A (detection): is at least one lung nodule visible in this slice?"),
  confidence: z.number().min(0).max(1).describe("Probability (0-1) that a nodule is present"),
  lobe: z.enum(LOBE_OR_NONE).describe("Task B (localization): lobe containing the most conspicuous nodule, or 'none'"),
  diameterMm: z.number().min(0).max(60).describe("Task C (size): estimated longest axial diameter in mm; 0 if no nodule"),
  rationale: z.string().describe("One short sentence describing the finding and its location"),
});
export type VlmAnswer = z.infer<typeof VlmAnswer>;

export type ReaderSource = "live" | "simulated";

/** Per-case record streamed to the browser and kept in graph state. */
export type CaseResult = {
  caseId: string;
  answer: VlmAnswer;
  source: ReaderSource;
  latencyMs: number;
  /** null for negatives on location/size tasks (not applicable). */
  correct: { detection: boolean; localization: boolean | null; size: boolean | null };
  sizeErrorMm: number | null;
};

export const READER_PROMPT = [
  "You are a thoracic radiologist reading ONE axial chest CT slice for a research benchmark.",
  "Image conventions: radiological display — image LEFT is the PATIENT'S RIGHT; image top is anterior.",
  "Window: lung (W1500 / L-600).",
  `Geometry: the square image spans a ${FOV_MM} mm field of view (${(FOV_MM / IMAGE_PX).toFixed(2)} mm per pixel at ${IMAGE_PX} px), so use the body width (~${Math.round(FOV_MM * 0.86)} mm) as a ruler.`,
  "Lobes: right lung = RUL / RML / RLL, left lung = LUL / LLL. Lower lobes lie posterior (bottom of image) and toward the lung bases;",
  "the middle lobe (RML) is anterior in the right lung at lower slice levels.",
  "Tasks:",
  "(A) Detection — is there a lung nodule (focal round opacity: solid, part-solid or ground-glass) in this slice? Vessels seen in cross-section are NOT nodules.",
  "(B) Localization — which lobe contains it (RUL, RML, RLL, LUL, LLL), or 'none'.",
  "(C) Size — estimated diameter in millimetres (0 if none).",
  "Some images contain no nodule. Answer with the structured schema only.",
].join("\n");

/** Size task scoring rule: within ±2 mm or ±30 % of truth, whichever is larger. */
export const sizeTolerance = (truthMm: number) => Math.max(2, 0.3 * truthMm);

export function scoreAnswer(gt: GroundTruth, a: VlmAnswer): Pick<CaseResult, "correct" | "sizeErrorMm"> {
  const detection = a.nodulePresent === gt.hasNodule;
  if (!gt.hasNodule) return { correct: { detection, localization: null, size: null }, sizeErrorMm: null };
  const err = a.nodulePresent ? a.diameterMm - gt.diameterMm : -gt.diameterMm;
  return {
    correct: {
      detection,
      localization: a.nodulePresent && a.lobe === gt.lobe,
      size: a.nodulePresent && Math.abs(err) <= sizeTolerance(gt.diameterMm),
    },
    sizeErrorMm: +err.toFixed(1),
  };
}

export const armKey = (g: Pick<GroundTruth, "vendor" | "kernel" | "dose">) => `${g.vendor} ${g.kernel}/${g.dose}`;
