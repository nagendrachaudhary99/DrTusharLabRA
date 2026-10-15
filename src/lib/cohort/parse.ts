import { LOBES, NODULE_TYPES, VENDORS } from "@/lib/trial/spec";
import { DATASETS, LOCATIONS, LUNG_RADS, type Facets, type Factor } from "./index";
import type { BuildCohortArgs } from "./tools";

/** Deterministic rule-based request parser used in demo mode (no LLM). */
export function parseCohortRequest(text: string): BuildCohortArgs {
  const t = text.toLowerCase().replace(/[–—]/g, "-");
  const facets: Facets = {};

  const size = t.match(/(\d{2,4})\s*-?\s*(?:case|patient|subject|scan|participant)s?/) ?? t.match(/\bn\s*=\s*(\d{2,4})/);
  const targetSize = size ? Math.min(1500, Math.max(10, +size[1])) : 200;
  const prev = t.match(/(\d{1,3}(?:\.\d+)?)\s*%\s*(?:nodule\s*)?(?:prevalence|positive|positivity|with nodules)/) ?? t.match(/prevalence\s*(?:of|=|:)?\s*(\d{1,3}(?:\.\d+)?)\s*%/);
  const prevalence = prev ? Math.min(1, +prev[1] / 100) : 0.3;

  const types = new Set<(typeof NODULE_TYPES)[number]>();
  if (/(?<!part[- ]?)(?<!non[- ]?)\bsolid\b/.test(t)) types.add("solid");
  if (/part[- ]?solid|subsolid/.test(t)) types.add("part-solid");
  if (/ground[- ]?glass|\bggns?\b|non[- ]?solid|subsolid/.test(t)) types.add("ground-glass");
  if (types.size) facets.types = NODULE_TYPES.filter((x) => types.has(x));

  const dr = t.match(/(\d+(?:\.\d+)?)\s*(?:-|to)\s*(\d+(?:\.\d+)?)\s*mm/);
  const ge = t.match(/(?:≥|>=|>|over|above|at least|larger than|bigger than)\s*(\d+(?:\.\d+)?)\s*mm/);
  const le = t.match(/(?:≤|<=|<|under|below|smaller than|up to)\s*(\d+(?:\.\d+)?)\s*mm/);
  if (dr) facets.diameterMm = [+dr[1], +dr[2]];
  else if (ge || le) facets.diameterMm = [ge ? +ge[1] : 3, le ? +le[1] : 30];

  const lobes = new Set<(typeof LOBES)[number]>();
  if (/\bupper lobes?\b/.test(t) && !/(right|left) upper/.test(t)) ["RUL", "LUL"].forEach((l) => lobes.add(l as never));
  if (/\blower lobes?\b/.test(t) && !/(right|left) lower/.test(t)) ["RLL", "LLL"].forEach((l) => lobes.add(l as never));
  if (/right upper/.test(t)) lobes.add("RUL");
  if (/left upper/.test(t)) lobes.add("LUL");
  if (/right lower/.test(t)) lobes.add("RLL");
  if (/left lower/.test(t)) lobes.add("LLL");
  if (/middle lobe/.test(t)) lobes.add("RML");
  for (const l of LOBES) if (new RegExp(`\\b${l.toLowerCase()}\\b`).test(t)) lobes.add(l);
  if (lobes.size) facets.lobes = LOBES.filter((x) => lobes.has(x));

  const locs = LOCATIONS.filter((l) => t.includes(l));
  if (locs.length) facets.locations = locs;

  const age = t.match(/aged?\s*(\d{2})\s*(?:-|to)\s*(\d{2})/) ?? t.match(/(\d{2})\s*(?:-|to)\s*(\d{2})\s*(?:years|yrs|y\/o)/);
  const ageMin = t.match(/(?:aged?\s*)?(?:over|above|≥|>=|at least)\s*(\d{2})\s*(?:years|yrs)?/);
  if (age) facets.ageRange = [+age[1], +age[2]];
  else if (ageMin && /age|year|old/.test(t)) facets.ageRange = [+ageMin[1], 100];

  const lr = t.match(/lung-?rads\s*(4a|4b|4x|[1-3])(\s*(?:or|and)\s*(?:higher|above|more))?/);
  if (lr) {
    const c = lr[1].toUpperCase() as (typeof LUNG_RADS)[number];
    const i = LUNG_RADS.indexOf(c);
    facets.lungRads = lr[2] ? LUNG_RADS.slice(i) : [c];
  }
  const ds = DATASETS.filter((d) => t.includes(d.split("-")[0].toLowerCase()));
  if (ds.length) facets.datasets = ds;
  const vs = VENDORS.filter((v) => new RegExp(`\\b${v.toLowerCase()}\\b`).test(t));
  if (vs.length) facets.vendors = vs;
  if (/\b(women|female)\b/.test(t)) facets.sex = "F";
  else if (/\b(men|male)\b/.test(t)) facets.sex = "M";
  const py = t.match(/(\d+)\+?\s*pack[- ]?years?/);
  if (py) facets.minPackYears = +py[1];
  if (/non[- ]?calcified|exclude calcifi|no calcifi/.test(t)) facets.excludeCalcified = true;

  const stratifyBy: Factor[] = [];
  const strat = t.match(/(?:balanced|stratif\w*|equal\w*)\s+(?:across|by|over|between)\s+([^.;,]+)/);
  if (strat) {
    const s = strat[1];
    if (/size/.test(s)) stratifyBy.push("sizeBin");
    if (/type|attenuation/.test(s)) stratifyBy.push("type");
    if (/lobe/.test(s)) stratifyBy.push("lobe");
    if (/location/.test(s)) stratifyBy.push("location");
    if (/lung-?rads|categor/.test(s)) stratifyBy.push("lungRads");
    if (/dataset|source/.test(s)) stratifyBy.push("dataset");
    if (/vendor|scanner/.test(s)) stratifyBy.push("vendor");
    if (/\bsex\b|gender/.test(s)) stratifyBy.push("sex");
    if (/\bage\b/.test(s)) stratifyBy.push("ageBin");
  }
  const allocation = /natural|proportional|representative/.test(t) || !stratifyBy.length ? "natural" : "balanced";
  const seed = t.match(/seed\s*(\d+)/);
  return { facets, targetSize, prevalence, stratifyBy: stratifyBy.slice(0, 3), allocation, seed: seed ? +seed[1] : 42 };
}
