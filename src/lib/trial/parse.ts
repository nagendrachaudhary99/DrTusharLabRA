import { DEFAULT_SPEC, DOSES, KERNELS, LOBES, NODULE_TYPES, VENDORS, type TrialSpec } from "./spec";

/**
 * Deterministic natural-language → TrialSpec parser. Used in demo mode and as a
 * safety net if the LLM's structured output fails validation.
 */
export function heuristicParse(text: string, base: TrialSpec = DEFAULT_SPEC): TrialSpec {
  const t = text.toLowerCase();
  const spec: TrialSpec = structuredClone(base);

  // Sizes: ranges like "4-12 mm" / "4 to 12mm" or lists "6, 8 and 10 mm".
  const range = t.match(/(\d+(?:\.\d+)?)\s*(?:-|–|to)\s*(\d+(?:\.\d+)?)\s*mm/);
  if (range) {
    const [a, b] = [+range[1], +range[2]].sort((x, y) => x - y);
    const ladder = [3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30].filter((s) => s >= a && s <= b);
    spec.sizesMm = ladder.length ? ladder : [a, b];
  } else {
    const mm = [...t.matchAll(/(\d+(?:\.\d+)?)\s*mm/g)].map((m) => +m[1]).filter((x) => x >= 2 && x <= 40);
    if (mm.length) spec.sizesMm = [...new Set(mm)].sort((x, y) => x - y);
  }

  const lobeMap: [RegExp, (typeof LOBES)[number]][] = [
    [/right upper|\brul\b/, "RUL"],
    [/right middle|\brml\b/, "RML"],
    [/right lower|\brll\b/, "RLL"],
    [/left upper|\blul\b/, "LUL"],
    [/left lower|\blll\b/, "LLL"],
  ];
  let lobes = lobeMap.filter(([re]) => re.test(t)).map(([, l]) => l);
  if (/both lower|lower lobes/.test(t)) lobes = [...new Set([...lobes, "RLL", "LLL"] as const)];
  if (/both upper|upper lobes/.test(t)) lobes = [...new Set([...lobes, "RUL", "LUL"] as const)];
  if (lobes.length && !/all lobes/.test(t)) spec.lobes = lobes;

  const vendors = VENDORS.filter((v) => new RegExp(`\\b${v.toLowerCase()}\\b`).test(t));
  let kernels = KERNELS.filter((k) => new RegExp(`\\b${k}\\b`).test(t));
  let doses = DOSES.filter((d) =>
    d === "low" ? /\blow[- ]dose|\bldct\b/.test(t) : d === "ultra-low" ? /ultra[- ]?low/.test(t) : /standard dose|routine dose|standard and low/.test(t),
  );
  if (doses.includes("ultra-low") && !/(?<!ultra[- ])low[- ]dose/.test(t.replace(/ultra[- ]?low[- ]dose/g, ""))) doses = doses.filter((d) => d !== "low");
  if (vendors.length || kernels.length || doses.length) {
    const vs = vendors.length ? vendors : (["Siemens", "GE"] as const);
    if (!kernels.length) kernels = ["standard"];
    if (!doses.length) doses = ["standard"];
    const arms: TrialSpec["scanners"] = [];
    for (const vendor of vs) for (const kernel of kernels) for (const dose of doses) arms.push({ vendor, kernel, dose });
    spec.scanners = arms.slice(0, 8);
  }

  const types = NODULE_TYPES.filter((ty) =>
    ty === "ground-glass"
      ? /ground[- ]glass|\bggo|\bggn/.test(t)
      : ty === "part-solid"
        ? /part[- ]solid|subsolid/.test(t)
        : /\bsolid\b/.test(t.replace(/part[- ]solid/g, "")),
  );
  if (types.length) spec.noduleTypes = /include ground[- ]glass/.test(t) ? [...new Set([...spec.noduleTypes, ...types])] : types;

  const n = t.match(/(\d{2,3})\s*(?:virtual\s*)?patients/);
  if (n) spec.patientsPerArm = Math.min(600, Math.max(10, +n[1]));

  if (/high[- ]dose model|lowdose[- ]fragile|routine[- ]dose model/.test(t)) spec.modelUnderTest = "lowdose-fragile";
  else if (/single[- ]site|siemens[- ]only|overfit/.test(t)) spec.modelUnderTest = "vendor-overfit";
  else if (/robust|baseline/.test(t)) spec.modelUnderTest = "robust-baseline";

  spec.title = text.length > 70 ? text.slice(0, 67).trim() + "…" : text || spec.title;
  return spec;
}
