"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { ArrowRight, Database, Download, Hammer, Play, Square, X } from "lucide-react";
import { streamAgent } from "@/lib/client-stream";
import type { AgentEvent } from "@/lib/stream";
import { LOBES, NODULE_TYPES } from "@/lib/trial/spec";
import {
  DATASETS,
  INDEX_DISCLAIMER,
  LOCATIONS,
  LUNG_RADS,
  N_HOSTS,
  SIZE_BINS,
  filterHosts,
  getEmbedding,
  getIndex,
  matchesNodule,
  sizeBinOf,
  type Facets,
  type Factor,
  type Nodule,
} from "@/lib/cohort/index";
import { FACTOR_LABEL, buildCohort, describeSpec, type BuildCohortArgs, type CohortSummary, type ManifestRow } from "@/lib/cohort/tools";
import { NoduleMap } from "./nodule-map";

const EXAMPLES = [
  "Build a 300-case screening cohort, 30% prevalence, solid and part-solid nodules 6–20 mm, upper lobes, age 55–80, balanced across size bins",
  "Subsolid detection study: 250 cases, 40% prevalence, ground-glass and part-solid nodules ≥8 mm, balanced across nodule type and lobe",
  "Domain-shift cohort: 400 cases, 25% prevalence, solid nodules 4–12 mm from NLST-like and LIDC-IDRI-like sources only, balanced across scanner vendor",
  "Lung-RADS 4A or higher, juxtapleural and juxtavascular nodules, 150 cases, 50% prevalence, natural distribution",
];

type Cohort = {
  source: "agent" | "facets";
  mode?: "live" | "demo";
  args: BuildCohortArgs;
  summary: CohortSummary;
  manifest: ManifestRow[];
  answer?: string;
  trialRequest: string;
};
type TraceItem =
  | { kind: "tool"; tool: string; args: unknown; result?: unknown; detail?: string; t: number; done: boolean }
  | { kind: "note"; text: string; t: number }
  | { kind: "log"; text: string; t: number };

type MultiKey = "types" | "lobes" | "locations" | "lungRads" | "datasets";
const MULTI: { key: MultiKey; label: string; levels: readonly string[]; of: (n: Nodule) => string }[] = [
  { key: "types", label: "Nodule type", levels: NODULE_TYPES, of: (n) => n.type },
  { key: "lobes", label: "Lobe", levels: LOBES, of: (n) => n.lobe },
  { key: "locations", label: "Location", levels: LOCATIONS, of: (n) => n.location },
  { key: "lungRads", label: "Lung-RADS v2022", levels: LUNG_RADS, of: (n) => n.lungRads },
  { key: "datasets", label: "Source (synthetic stand-in)", levels: DATASETS, of: (n) => n.sourceDataset },
];

const COLOR_BY: { key: string; label: string; levels: readonly string[]; of: (n: Nodule) => string }[] = [
  { key: "type", label: "Type", levels: NODULE_TYPES, of: (n) => n.type },
  { key: "sizeBin", label: "Size bin", levels: SIZE_BINS, of: (n) => sizeBinOf(n.diameterMm) },
  { key: "lungRads", label: "Lung-RADS", levels: LUNG_RADS, of: (n) => n.lungRads },
  { key: "malignancy", label: "Malignancy", levels: ["1", "2", "3", "4", "5"], of: (n) => String(n.malignancy) },
  { key: "lobe", label: "Lobe", levels: LOBES, of: (n) => n.lobe },
  { key: "location", label: "Location", levels: LOCATIONS, of: (n) => n.location },
  { key: "dataset", label: "Source", levels: DATASETS, of: (n) => n.sourceDataset },
];

const HIST: { key: keyof CohortSummary["positivesHistograms"]; label: string; levels: readonly string[]; of: (n: Nodule) => string }[] = [
  { key: "sizeBin", label: "Size bin", levels: SIZE_BINS, of: (n) => sizeBinOf(n.diameterMm) },
  { key: "type", label: "Type", levels: NODULE_TYPES, of: (n) => n.type },
  { key: "lobe", label: "Lobe", levels: LOBES, of: (n) => n.lobe },
  { key: "lungRads", label: "Lung-RADS", levels: LUNG_RADS, of: (n) => n.lungRads },
];

const STRAT_FACTORS: Factor[] = ["sizeBin", "type", "lobe", "location", "lungRads", "dataset", "vendor"];

export function CohortClient() {
  const { nodules, byId } = useMemo(() => getIndex(), []);
  const embedding = useMemo(() => getEmbedding(), []);

  const [request, setRequest] = useState(EXAMPLES[0]);
  const [running, setRunning] = useState(false);
  const [trace, setTrace] = useState<TraceItem[]>([]);
  const [meta, setMeta] = useState<{ mode: string; model: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cohort, setCohort] = useState<Cohort | null>(null);
  const [facets, setFacets] = useState<Facets>({});
  const [colorBy, setColorBy] = useState("type");
  const [build, setBuild] = useState({ targetSize: 200, prevalence: 0.3, stratifyBy: ["sizeBin"] as Factor[], allocation: "balanced" as "balanced" | "natural", seed: 42 });
  const [openItem, setOpenItem] = useState<number | null>(null);
  const abort = useRef<AbortController | null>(null);
  const t0 = useRef(0);

  async function plan() {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setTrace([]);
    setError(null);
    setOpenItem(null);
    t0.current = performance.now();
    const t = () => performance.now() - t0.current;
    try {
      await streamAgent(
        "/api/cohort",
        { request },
        (e: AgentEvent) => {
          if (e.type === "meta") setMeta(e.data as { mode: string; model: string });
          if (e.type === "reset") setTrace([]);
          if (e.type === "log") setTrace((s) => [...s, { kind: "log", text: e.message, t: t() }]);
          if (e.type === "error") setError(e.message);
          if (e.type === "step") {
            const d = (e.data ?? {}) as { args?: unknown; result?: unknown; text?: string };
            if (e.node === "planner") setTrace((s) => [...s, { kind: "note", text: d.text ?? e.detail ?? "", t: t() }]);
            else if (e.status === "start") setTrace((s) => [...s, { kind: "tool", tool: e.node, args: d.args, t: t(), done: false }]);
            else
              setTrace((s) => {
                const i = s.findLastIndex((x) => x.kind === "tool" && x.tool === e.node && !x.done);
                if (i < 0) return s;
                const next = [...s];
                next[i] = { ...(next[i] as Extract<TraceItem, { kind: "tool" }>), done: true, detail: e.detail, result: d.result };
                return next;
              });
          }
          if (e.type === "result") {
            const r = e.data as Omit<Cohort, "source">;
            setCohort({ ...r, source: "agent" });
            setFacets(r.args.facets);
            setBuild({ targetSize: r.args.targetSize, prevalence: r.args.prevalence, stratifyBy: r.args.stratifyBy, allocation: r.args.allocation, seed: r.args.seed });
          }
        },
        ctrl.signal,
      );
    } catch (err) {
      if (!ctrl.signal.aborted) setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  function buildFromFacets() {
    const args: BuildCohortArgs = { facets, ...build };
    const r = buildCohort(args);
    setCohort({ source: "facets", args, ...r, trialRequest: describeSpec(args, r.summary) });
  }

  /* Live faceted search over the full index. */
  const matches = useMemo(() => {
    const m = new Uint8Array(nodules.length);
    nodules.forEach((n, i) => (m[i] = matchesNodule(n, facets) ? 1 : 0));
    return m;
  }, [nodules, facets]);
  const nMatch = useMemo(() => matches.reduce((a, b) => a + b, 0), [matches]);
  const nHosts = useMemo(() => filterHosts(facets).length, [facets]);
  const facetCounts = useMemo(() => {
    const out: Record<string, Record<string, number>> = {};
    for (const g of MULTI) {
      const without = { ...facets, [g.key]: undefined };
      const c: Record<string, number> = {};
      for (const n of nodules) if (matchesNodule(n, without)) c[g.of(n)] = (c[g.of(n)] ?? 0) + 1;
      out[g.key] = c;
    }
    return out;
  }, [nodules, facets]);

  const cohortIds = useMemo(() => new Set(cohort?.manifest.flatMap((r) => (r.donorNoduleId ? [r.donorNoduleId] : [])) ?? []), [cohort]);
  const color = COLOR_BY.find((c) => c.key === colorBy)!;

  function toggle(key: MultiKey, v: string) {
    setFacets((f) => {
      const cur = (f[key] as string[] | undefined) ?? [];
      const next = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
      return { ...f, [key]: next.length ? next : undefined } as Facets;
    });
  }
  const setRange = (key: "diameterMm" | "ageRange", i: 0 | 1, v: number, dflt: [number, number]) =>
    setFacets((f) => {
      const cur = [...(f[key] ?? dflt)] as [number, number];
      cur[i] = v;
      return { ...f, [key]: cur };
    });
  const extraFacets = Object.entries(facets).filter(([k, v]) => v !== undefined && !["types", "lobes", "locations", "lungRads", "datasets", "diameterMm", "ageRange"].includes(k));

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <header className="space-y-1">
        <p className="font-mono text-xs uppercase tracking-widest text-accent">Stage 1 · Multidataset nodule profiling · LangChain v1 tool-calling agent</p>
        <h1 className="text-2xl font-semibold tracking-tight">Nodule Index &amp; Cohort Planner</h1>
        <p className="max-w-3xl text-sm text-ink-2">
          Cohort by specification, the iTRIALSPACE way: describe the cohort you need and a <span className="font-mono text-ink">createAgent</span> planner calls
          tools over a nodule index (<span className="font-mono">query_index</span>, <span className="font-mono">stratify</span>,{" "}
          <span className="font-mono">build_cohort</span>, <span className="font-mono">lung_rads</span>) to assemble donor nodules and negative-control host scans
          that meet the target size, prevalence and per-stratum quotas. Explore the index with Retriever-style facets and a NoduleMap embedding.
        </p>
        <p className="flex items-start gap-2 rounded-md border border-[var(--status-warning)]/50 bg-surface-1 px-3 py-2 text-xs text-ink-2">
          <Database className="mt-0.5 h-3.5 w-3.5 shrink-0 text-[var(--status-warning)]" aria-hidden />
          <span>
            {INDEX_DISCLAIMER} This demo: {nodules.length.toLocaleString()} synthetic nodules ({Object.keys(nodules[0]).length} attributes) + {N_HOSTS.toLocaleString()} nodule-free host scans,
            generated deterministically (seed 13140).
          </span>
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
        {/* Left: NL request + agent trace */}
        <div className="card flex flex-col gap-4 p-4">
          <label className="block text-sm font-medium" htmlFor="cohort-req">
            Cohort request
          </label>
          <textarea
            id="cohort-req"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            rows={3}
            className="w-full resize-y rounded-lg border border-line bg-background p-3 text-sm outline-none focus:border-accent"
          />
          <div className="flex flex-wrap gap-2">
            {EXAMPLES.map((p, i) => (
              <button
                key={i}
                onClick={() => setRequest(p)}
                title={p}
                className="rounded-full border border-line px-3 py-1 text-xs text-ink-2 hover:border-accent hover:text-ink"
              >
                Example {i + 1}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-3">
            {running ? (
              <button onClick={() => abort.current?.abort()} className="flex items-center gap-2 rounded-lg bg-surface-2 px-4 py-2 text-sm font-medium">
                <Square className="h-4 w-4" /> Stop
              </button>
            ) : (
              <button onClick={plan} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:brightness-110">
                <Play className="h-4 w-4" /> Plan cohort
              </button>
            )}
            {meta && (
              <span className="rounded-full border border-line px-2 py-0.5 font-mono text-[11px] text-ink-3">
                {meta.mode === "live" && !(cohort?.source === "agent" && cohort.mode === "demo") ? `LLM agent: ${meta.model}` : "demo mode · rule-based planner"}
              </span>
            )}
          </div>
          {error && <p className="rounded-md border border-[var(--status-critical)] p-2 text-xs text-ink">⚠ {error}</p>}

          <div>
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-medium">Agent tool-call trace</h2>
              <span className="text-[11px] text-ink-3">click a call to inspect args / result</span>
            </div>
            <div className="max-h-[440px] min-h-48 space-y-1 overflow-auto rounded-lg border border-line bg-background p-2 font-mono text-[11px] leading-5">
              {trace.length ? (
                trace.map((it, i) => (
                  <div key={i}>
                    <div
                      className={`flex gap-2 ${it.kind === "tool" ? "cursor-pointer hover:bg-surface-2" : ""} rounded px-1`}
                      onClick={() => it.kind === "tool" && setOpenItem(openItem === i ? null : i)}
                    >
                      <span className="w-12 shrink-0 text-right text-ink-3">{(it.t / 1000).toFixed(2)}s</span>
                      {it.kind === "tool" ? (
                        <>
                          <span className={it.done ? "text-[var(--series-3)]" : "text-accent"}>{it.done ? "✓" : "▶"}</span>
                          <span className="shrink-0 text-ink">{it.tool}</span>
                          <span className="truncate text-ink-3">{it.done ? it.detail : "running…"}</span>
                        </>
                      ) : it.kind === "note" ? (
                        <>
                          <span className="text-[var(--series-7)]">◆</span>
                          <span className="whitespace-pre-wrap font-sans text-ink-2">{it.text}</span>
                        </>
                      ) : (
                        <>
                          <span className="text-[var(--status-warning)]">!</span>
                          <span className="text-ink-3">{it.text}</span>
                        </>
                      )}
                    </div>
                    {openItem === i && it.kind === "tool" && (
                      <pre className="ml-14 mt-1 max-h-64 overflow-auto whitespace-pre-wrap rounded border border-line bg-surface-1 p-2 text-ink-3">
                        {"args: " + JSON.stringify(it.args, null, 1) + "\n\nresult: " + JSON.stringify(it.result ?? null, null, 1).slice(0, 3000)}
                      </pre>
                    )}
                  </div>
                ))
              ) : (
                <span className="text-ink-3">Tool calls stream here: the agent decides which tools to call and with what arguments.</span>
              )}
            </div>
          </div>
        </div>

        {/* Right: facet panel */}
        <div className="card space-y-4 p-4">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-sm font-medium">Retriever · faceted filter (live, client-side)</h2>
            <span className="font-mono text-xs text-ink-2">
              <span className="text-ink">{nMatch.toLocaleString()}</span> / {nodules.length.toLocaleString()} nodules · {nHosts.toLocaleString()} host scans
            </span>
          </div>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <RangeField label="Diameter (mm)" value={facets.diameterMm} dflt={[3, 30]} min={3} max={30} step={0.5} onChange={(i, v) => setRange("diameterMm", i, v, [3, 30])} />
            <RangeField label="Patient age (y)" value={facets.ageRange} dflt={[30, 90]} min={30} max={90} step={1} onChange={(i, v) => setRange("ageRange", i, v, [30, 90])} />
          </div>
          {MULTI.map((g) => (
            <div key={g.key}>
              <div className="mb-1 text-[11px] text-ink-3">{g.label}</div>
              <div className="flex flex-wrap gap-1.5">
                {g.levels.map((l) => {
                  const on = ((facets[g.key] as string[] | undefined) ?? []).includes(l);
                  return (
                    <button
                      key={l}
                      onClick={() => toggle(g.key, l)}
                      className={`rounded-md border px-2 py-0.5 text-xs ${on ? "border-accent bg-surface-2 text-ink" : "border-line text-ink-2 hover:border-accent"}`}
                    >
                      {l} <span className="font-mono text-ink-3">{facetCounts[g.key][l] ?? 0}</span>
                    </button>
                  );
                })}
              </div>
            </div>
          ))}
          {extraFacets.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 text-xs">
              <span className="text-[11px] text-ink-3">Agent facets:</span>
              {extraFacets.map(([k, v]) => (
                <button
                  key={k}
                  onClick={() => setFacets((f) => ({ ...f, [k]: undefined }))}
                  className="flex items-center gap-1 rounded-md border border-line bg-surface-2 px-2 py-0.5 font-mono"
                >
                  {k}={JSON.stringify(v).replace(/"/g, "")} <X className="h-3 w-3" />
                </button>
              ))}
            </div>
          )}
          <div className="space-y-2 rounded-lg border border-line p-3 text-xs">
            <div className="flex items-center justify-between">
              <span className="font-medium text-ink">Build cohort from these facets</span>
              <button onClick={() => setFacets({})} className="text-[11px] text-accent">
                clear facets
              </button>
            </div>
            <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
              <NumField label="Cases" value={build.targetSize} min={10} max={1500} onChange={(v) => setBuild((b) => ({ ...b, targetSize: v }))} />
              <NumField label="Prevalence %" value={Math.round(build.prevalence * 100)} min={0} max={100} onChange={(v) => setBuild((b) => ({ ...b, prevalence: v / 100 }))} />
              <NumField label="Seed" value={build.seed} min={0} max={99999} onChange={(v) => setBuild((b) => ({ ...b, seed: v }))} />
              <label className="space-y-1">
                <span className="text-ink-3">Allocation</span>
                <select
                  value={build.allocation}
                  onChange={(e) => setBuild((b) => ({ ...b, allocation: e.target.value as "balanced" | "natural" }))}
                  className="w-full rounded border border-line bg-background px-1.5 py-1"
                >
                  <option value="balanced">balanced</option>
                  <option value="natural">natural</option>
                </select>
              </label>
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="text-ink-3">Stratify by (≤3):</span>
              {STRAT_FACTORS.map((f) => {
                const on = build.stratifyBy.includes(f);
                return (
                  <button
                    key={f}
                    onClick={() =>
                      setBuild((b) => ({ ...b, stratifyBy: on ? b.stratifyBy.filter((x) => x !== f) : b.stratifyBy.length < 3 ? [...b.stratifyBy, f] : b.stratifyBy }))
                    }
                    className={`rounded-md border px-2 py-0.5 font-mono ${on ? "border-accent bg-surface-2 text-ink" : "border-line text-ink-3"}`}
                  >
                    {f}
                  </button>
                );
              })}
            </div>
            <button onClick={buildFromFacets} className="flex items-center gap-2 rounded-lg border border-accent px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-2">
              <Hammer className="h-3.5 w-3.5" /> Build (same build_cohort tool, in-browser)
            </button>
          </div>
        </div>
      </section>

      {cohort && <CohortPanel cohort={cohort} />}

      <section className="grid gap-4 lg:grid-cols-[1.35fr_1fr]">
        <div className="card p-4">
          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
            <h2 className="text-sm font-medium">NoduleMap · 2-D PCA embedding of the index</h2>
            <label className="flex items-center gap-2 text-xs text-ink-3">
              colour by
              <select value={colorBy} onChange={(e) => setColorBy(e.target.value)} className="rounded border border-line bg-background px-1.5 py-1 text-ink">
                {COLOR_BY.map((c) => (
                  <option key={c.key} value={c.key}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <NoduleMap nodules={nodules} embedding={embedding} colorKey={color.of} levels={color.levels} matches={matches} cohort={cohortIds} />
          <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-ink-2">
            {color.levels.map((l, i) => (
              <span key={l} className="flex items-center gap-1">
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: `var(--series-${(i % 8) + 1})` }} />
                {l}
              </span>
            ))}
            <span className="text-ink-3">· faded = outside facets · ringed = cohort donor nodule</span>
          </div>
        </div>
        <div className="card space-y-4 p-4">
          <div className="flex items-baseline justify-between">
            <h2 className="text-sm font-medium">Distribution · {cohort ? "cohort positives vs index" : "facet matches vs index"}</h2>
            <span className="flex items-center gap-3 text-[11px] text-ink-3">
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-3 rounded-sm bg-[var(--text-muted)]" /> index
              </span>
              <span className="flex items-center gap-1">
                <span className="inline-block h-2 w-3 rounded-sm bg-accent" /> {cohort ? "cohort" : "matches"}
              </span>
            </span>
          </div>
          {HIST.map((h) => (
            <CompareHist
              key={h.key}
              label={h.label}
              levels={h.levels}
              index={count(nodules, h.of)}
              sel={cohort ? cohort.summary.positivesHistograms[h.key] : count(nodules.filter((_, i) => matches[i]), h.of)}
            />
          ))}
        </div>
      </section>

      {cohort && <Manifest cohort={cohort} byId={byId} />}
    </div>
  );
}

function count(ns: Nodule[], of: (n: Nodule) => string) {
  const c: Record<string, number> = {};
  for (const n of ns) c[of(n)] = (c[of(n)] ?? 0) + 1;
  return c;
}

function RangeField({
  label,
  value,
  dflt,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value?: [number, number];
  dflt: [number, number];
  min: number;
  max: number;
  step: number;
  onChange: (i: 0 | 1, v: number) => void;
}) {
  const [lo, hi] = value ?? dflt;
  return (
    <div className="space-y-1">
      <span className="flex justify-between text-ink-2">
        {label}
        <span className="font-mono text-ink">
          {lo}–{hi}
        </span>
      </span>
      <input type="range" min={min} max={max} step={step} value={lo} onChange={(e) => onChange(0, Math.min(+e.target.value, hi))} className="w-full" aria-label={`${label} min`} />
      <input type="range" min={min} max={max} step={step} value={hi} onChange={(e) => onChange(1, Math.max(+e.target.value, lo))} className="w-full" aria-label={`${label} max`} />
    </div>
  );
}

function NumField({ label, value, min, max, onChange }: { label: string; value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <label className="space-y-1">
      <span className="text-ink-3">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => onChange(Math.min(max, Math.max(min, Math.round(+e.target.value || 0))))}
        className="w-full rounded border border-line bg-background px-1.5 py-1 font-mono"
      />
    </label>
  );
}

function CompareHist({ label, levels, index, sel }: { label: string; levels: readonly string[]; index: Record<string, number>; sel: Record<string, number> }) {
  const ti = Object.values(index).reduce((a, b) => a + b, 0) || 1;
  const ts = Object.values(sel).reduce((a, b) => a + b, 0) || 1;
  const max = Math.max(...levels.map((l) => Math.max((index[l] ?? 0) / ti, (sel[l] ?? 0) / ts)), 0.01);
  return (
    <div>
      <div className="mb-1 text-[11px] text-ink-3">{label}</div>
      <div className="space-y-0.5">
        {levels.map((l) => {
          const a = (index[l] ?? 0) / ti;
          const b = (sel[l] ?? 0) / ts;
          return (
            <div key={l} className="grid grid-cols-[5rem_1fr_9rem] items-center gap-2 text-[11px]">
              <span className="truncate text-ink-2">{l}</span>
              <div className="space-y-px">
                <div className="h-1.5 rounded-sm bg-[var(--text-muted)]" style={{ width: `${(a / max) * 100}%` }} />
                <div className="h-1.5 rounded-sm bg-accent" style={{ width: `${(b / max) * 100}%` }} />
              </div>
              <span className="text-right font-mono text-ink-3">
                {(a * 100).toFixed(0)}% → <span className="text-ink">{(b * 100).toFixed(0)}%</span> <span className="text-ink-3">({sel[l] ?? 0})</span>
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function CohortPanel({ cohort }: { cohort: Cohort }) {
  const s = cohort.summary;
  const tiles = [
    { k: "Cases", v: s.nCases.toLocaleString(), sub: `target ${s.targetSize}` },
    { k: "Nodule-positive", v: s.nPositive.toLocaleString(), sub: `pool ${s.poolNodules} donors` },
    { k: "Negative controls", v: s.nNegative.toLocaleString(), sub: `pool ${s.poolHosts} host scans` },
    { k: "Prevalence", v: `${(s.achievedPrevalence * 100).toFixed(1)}%`, sub: `target ${(s.targetPrevalence * 100).toFixed(0)}%` },
    { k: "Spec", v: s.meetsSpec ? "met" : "partial", sub: s.meetsSpec ? "all quotas filled" : `${s.warnings.length} warning(s)` },
  ];
  return (
    <section className="space-y-4">
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
        {tiles.map((t) => (
          <div key={t.k} className="card p-3">
            <div className="text-[11px] text-ink-3">{t.k}</div>
            <div className={`font-mono text-xl ${t.k === "Spec" ? (s.meetsSpec ? "text-[var(--status-good)]" : "text-[var(--status-warning)]") : "text-ink"}`}>{t.v}</div>
            <div className="text-[10px] text-ink-3">{t.sub}</div>
          </div>
        ))}
      </div>
      <div className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
        <div className="card space-y-3 p-4 text-sm">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h2 className="text-sm font-medium">Cohort specification</h2>
            <span className="font-mono text-[11px] text-ink-3">
              {cohort.source === "agent" ? (cohort.mode === "live" ? "planned by LLM agent" : "planned in demo mode") : "built from facets"} · seed {s.seed}
            </span>
          </div>
          <p className="text-ink-2">{cohort.trialRequest}</p>
          {cohort.answer && <p className="whitespace-pre-wrap rounded-lg border border-line bg-background p-3 text-xs text-ink-2">{cohort.answer.replace(/\*\*/g, "")}</p>}
          {s.warnings.map((w, i) => (
            <p key={i} className="rounded-md border border-[var(--status-warning)]/60 p-2 text-xs text-ink-2">
              ⚠ {w}
            </p>
          ))}
          <Link
            href={`/trial?request=${encodeURIComponent(cohort.trialRequest)}`}
            className="inline-flex items-center gap-2 rounded-lg bg-accent px-3 py-1.5 text-xs font-medium text-white hover:brightness-110"
          >
            Send to Trial Designer <ArrowRight className="h-3.5 w-3.5" />
          </Link>
        </div>
        <div className="card p-4">
          <h2 className="mb-2 text-sm font-medium">
            Strata · {s.allocation}
            {s.stratifyBy.length ? ` across ${s.stratifyBy.map((f) => FACTOR_LABEL[f]).join(" × ")}` : " (no stratification)"}
          </h2>
          <div className="max-h-64 overflow-auto">
            <table className="w-full text-xs">
              <thead className="text-left text-ink-3">
                <tr>
                  <th className="py-1 font-normal">Stratum</th>
                  <th className="py-1 text-right font-normal">Available</th>
                  <th className="py-1 text-right font-normal">Quota</th>
                  <th className="py-1 text-right font-normal">Selected</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {s.strata.map((r) => (
                  <tr key={r.stratum} className="border-t border-line">
                    <td className="py-1 font-sans text-ink-2">{r.stratum}</td>
                    <td className="py-1 text-right text-ink-3">{r.available}</td>
                    <td className="py-1 text-right text-ink-3">{r.quota}</td>
                    <td className={`py-1 text-right ${r.underFilled ? "text-[var(--status-warning)]" : "text-ink"}`}>
                      {r.selected}
                      {r.underFilled ? " ⚠" : ""}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </section>
  );
}

const COLS: (keyof ManifestRow)[] = [
  "caseId",
  "label",
  "hostScanId",
  "donorNoduleId",
  "stratum",
  "age",
  "sex",
  "vendor",
  "kernel",
  "sliceThicknessMm",
  "hostDataset",
  "diameterMm",
  "type",
  "lobe",
  "location",
  "lungRads",
  "malignancy",
  "donorDataset",
];

function download(name: string, text: string, type: string) {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function Manifest({ cohort, byId }: { cohort: Cohort; byId: Map<string, Nodule> }) {
  const rows = cohort.manifest;
  const csv = () => {
    const esc = (v: unknown) => (v === null || v === undefined ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const lines = [COLS.join(","), ...rows.map((r) => COLS.map((c) => esc(r[c])).join(","))];
    download(`itrialspace-cohort-seed${cohort.summary.seed}.csv`, lines.join("\n"), "text/csv");
  };
  const json = () => {
    const donors = rows.flatMap((r) => (r.donorNoduleId ? [byId.get(r.donorNoduleId)!] : []));
    download(
      `itrialspace-cohort-seed${cohort.summary.seed}.json`,
      JSON.stringify({ disclaimer: INDEX_DISCLAIMER, spec: cohort.args, description: cohort.trialRequest, summary: cohort.summary, manifest: rows, donorNodules: donors }, null, 1),
      "application/json",
    );
  };
  return (
    <section className="card p-4">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-sm font-medium">
          Case manifest <span className="font-normal text-ink-3">· showing {Math.min(50, rows.length)} of {rows.length} cases</span>
        </h2>
        <div className="flex gap-2">
          <button onClick={csv} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1 text-xs hover:border-accent">
            <Download className="h-3.5 w-3.5" /> CSV
          </button>
          <button onClick={json} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1 text-xs hover:border-accent">
            <Download className="h-3.5 w-3.5" /> JSON
          </button>
        </div>
      </div>
      <div className="max-h-[420px] overflow-auto">
        <table className="w-full whitespace-nowrap text-xs">
          <thead className="sticky top-0 bg-surface-1 text-left text-ink-3">
            <tr>
              {["Case", "Label", "Host scan", "Age/Sex", "Scanner", "Donor nodule", "Size", "Type", "Lobe", "Location", "Lung-RADS", "Malig."].map((h) => (
                <th key={h} className="px-2 py-1 font-normal">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="font-mono">
            {rows.slice(0, 50).map((r) => (
              <tr key={r.caseId} className="border-t border-line">
                <td className="px-2 py-1 text-ink">{r.caseId}</td>
                <td className={`px-2 py-1 ${r.label === "positive" ? "text-accent" : "text-ink-3"}`}>{r.label}</td>
                <td className="px-2 py-1 text-ink-2">{r.hostScanId}</td>
                <td className="px-2 py-1 text-ink-2">
                  {r.age} {r.sex}
                </td>
                <td className="px-2 py-1 text-ink-2">
                  {r.vendor} {r.kernel} {r.sliceThicknessMm}mm
                </td>
                <td className="px-2 py-1 text-ink-2">{r.donorNoduleId ?? "—"}</td>
                <td className="px-2 py-1 text-ink-2">{r.diameterMm !== null ? `${r.diameterMm} mm` : ""}</td>
                <td className="px-2 py-1 text-ink-2">{r.type ?? ""}</td>
                <td className="px-2 py-1 text-ink-2">{r.lobe ?? ""}</td>
                <td className="px-2 py-1 text-ink-2">{r.location ?? ""}</td>
                <td className="px-2 py-1 text-ink-2">{r.lungRads ?? ""}</td>
                <td className="px-2 py-1 text-ink-2">{r.malignancy ?? ""}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
