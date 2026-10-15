"use client";

import { useMemo, useRef, useState } from "react";
import { Download, FlaskConical, Loader2, Play, Plus, Square, X } from "lucide-react";
import { CTCanvas } from "@/components/ct-canvas";
import { Markdown } from "@/components/markdown";
import { ConfusionSummary, SizeScatter, TaskForest } from "@/components/vlm-charts";
import { streamAgent } from "@/lib/client-stream";
import type { AgentEvent } from "@/lib/stream";
import { DOSES, KERNELS, LOBES, NODULE_TYPES, VENDORS, type ScannerArm, type TrialSpec } from "@/lib/trial/spec";
import { toImageData, WINDOWS, type RenderResult } from "@/lib/phantom/render";
import { buildCases, renderCase, type BenchDesign, type CaseSpec } from "@/lib/vlm/design";
import { computeMetrics, TASKS, type VlmMetrics } from "@/lib/vlm/metrics";
import { MAX_CASES, READER_PROMPT, armKey, type CaseResult } from "@/lib/vlm/schema";

const SIZE_OPTIONS = [0, 4, 6, 8, 10, 15, 20];
const ARM_PRESETS: ScannerArm[] = [
  { vendor: "Siemens", kernel: "standard", dose: "standard" },
  { vendor: "GE", kernel: "standard", dose: "low" },
  { vendor: "Philips", kernel: "sharp", dose: "low" },
  { vendor: "Canon", kernel: "soft", dose: "ultra-low" },
];
const PIPELINE = [
  { id: "plan", title: "plan", role: "design → Send() per case" },
  { id: "vlm_reader", title: "vlm_reader ×N", role: "1 structured multimodal call / image" },
  { id: "score", title: "score", role: "Wilson · bootstrap · AUC" },
  { id: "analyst", title: "analyst", role: "failure-mode summary (LLM)" },
] as const;

type Meta = { mode: "live" | "demo"; model: string; reader: string };
type BenchCase = CaseSpec & { imageDataUrl: string };

/** Lung-window PNG data URL of a rendered slice (what the VLM actually sees). */
function toPng(r: RenderResult) {
  const c = document.createElement("canvas");
  c.width = r.size;
  c.height = r.size;
  c.getContext("2d")!.putImageData(toImageData(r, "ct", WINDOWS.lung), 0, 0);
  return c.toDataURL("image/png");
}

const toggle = <T,>(xs: T[], x: T) => (xs.includes(x) ? xs.filter((y) => y !== x) : [...xs, x]);
const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
const gtLabel = (c: CaseSpec) =>
  c.groundTruth.hasNodule ? `${c.groundTruth.diameterMm} mm ${c.groundTruth.type} · ${c.groundTruth.lobe}` : "no nodule (control)";

export function VlmBenchClient() {
  const [sizes, setSizes] = useState<number[]>([0, 4, 6, 10, 20]);
  const [types, setTypes] = useState<TrialSpec["noduleTypes"]>(["solid", "ground-glass"]);
  const [lobes, setLobes] = useState<TrialSpec["lobes"]>([...LOBES]);
  const [arms, setArms] = useState<ScannerArm[]>(ARM_PRESETS.slice(0, 2));
  const [customArm, setCustomArm] = useState<ScannerArm>({ vendor: "GE", kernel: "sharp", dose: "ultra-low" });
  const [nCases, setNCases] = useState(24);
  const [seed, setSeed] = useState(7);

  const [phase, setPhase] = useState<"idle" | "rendering" | "reading" | "done">("idle");
  const [rendered, setRendered] = useState(0);
  const [cases, setCases] = useState<BenchCase[]>([]);
  const [renders, setRenders] = useState<Record<string, RenderResult>>({});
  const [reading, setReading] = useState<Set<string>>(new Set());
  const [results, setResults] = useState<Record<string, CaseResult>>({});
  const [status, setStatus] = useState<Record<string, "running" | "done">>({});
  const [prompt, setPrompt] = useState(READER_PROMPT);
  const [report, setReport] = useState("");
  const [serverMetrics, setServerMetrics] = useState<VlmMetrics | null>(null);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const gridRef = useRef<HTMLElement>(null);

  const design: BenchDesign = { sizesMm: [...sizes].sort((a, b) => a - b), noduleTypes: types, lobes, scanners: arms, nCases, seed };
  const valid = sizes.length > 0 && lobes.length > 0 && arms.length > 0 && (types.length > 0 || sizes.every((s) => s === 0));
  const gridCells = sizes.reduce((s, z) => s + (z === 0 ? 1 : types.length), 0) * lobes.length * arms.length;
  const running = phase === "rendering" || phase === "reading";

  async function run() {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setPhase("rendering");
    setRendered(0);
    setCases([]);
    setRenders({});
    setResults({});
    setReading(new Set());
    setStatus({});
    setReport("");
    setServerMetrics(null);
    setLogs([]);
    setError(null);
    setSelected(null);
    gridRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });

    // 1. Render every case in the browser (procedural phantom → lung-window PNG), yielding so the grid fills progressively.
    const specs = buildCases(design);
    const out: BenchCase[] = [];
    const rs: Record<string, RenderResult> = {};
    for (const spec of specs) {
      if (ctrl.signal.aborted) return setPhase("idle");
      const r = renderCase(spec, seed);
      rs[spec.id] = r;
      out.push({ ...spec, imageDataUrl: toPng(r) });
      setCases([...out]);
      setRendered(out.length);
      await new Promise((res) => setTimeout(res, 0));
    }
    setRenders(rs);

    // 2. Ship images + ground truth to the LangGraph route and stream answers back.
    setPhase("reading");
    try {
      await streamAgent(
        "/api/vlm",
        { cases: out.map((c) => ({ id: c.id, imageDataUrl: c.imageDataUrl, groundTruth: c.groundTruth })) },
        (e: AgentEvent) => {
          if (e.type === "meta") setMeta(e.data as Meta);
          if (e.type === "step") {
            if (e.node === "vlm_reader") {
              const d = e.data as { caseId: string; result?: CaseResult };
              setStatus((s) => ({ ...s, vlm_reader: "running" }));
              if (e.status === "start") setReading((s) => new Set(s).add(d.caseId));
              else if (d.result) {
                setReading((s) => {
                  const n = new Set(s);
                  n.delete(d.caseId);
                  return n;
                });
                setResults((r) => ({ ...r, [d.caseId]: d.result! }));
              }
            } else {
              setStatus((s) => ({ ...s, ...(e.node === "score" ? { vlm_reader: "done" } : {}), [e.node]: e.status === "start" ? "running" : "done" }));
              if (e.node === "plan" && e.status === "done") setPrompt((e.data as { prompt: string }).prompt);
            }
          }
          if (e.type === "log") setLogs((l) => [...l, `${e.node}: ${e.message}`]);
          if (e.type === "token") setReport((r) => r + e.text);
          if (e.type === "reset") setReport("");
          if (e.type === "result") setServerMetrics((e.data as { metrics: VlmMetrics }).metrics);
          if (e.type === "error") setError(e.message);
        },
        ctrl.signal,
      );
    } catch (err) {
      if (!ctrl.signal.aborted) setError((err as Error).message);
    } finally {
      setPhase("done");
    }
  }

  const resultList = useMemo(() => Object.values(results), [results]);
  // Running tally while answers stream in; the server's `score` node output replaces it when done (same code path).
  const liveMetrics = useMemo(() => (resultList.length ? computeMetrics(cases, resultList) : null), [cases, resultList]);
  const metrics = serverMetrics ?? liveMetrics;
  const simulatedCount = resultList.filter((r) => r.source === "simulated").length;
  const readerBadge = !meta
    ? null
    : meta.mode === "demo" || (resultList.length > 0 && simulatedCount === resultList.length)
      ? { text: "simulated reader (demo mode)", live: false }
      : { text: `live VLM: ${meta.model}${simulatedCount ? ` · ${simulatedCount} simulated fallback(s)` : ""}`, live: true };
  const sel = cases.find((c) => c.id === selected) ?? null;

  const download = (name: string, body: string, type: string) => {
    const url = URL.createObjectURL(new Blob([body], { type }));
    const el = document.createElement("a");
    el.href = url;
    el.download = name;
    el.click();
    URL.revokeObjectURL(url);
  };
  const exportCsv = () => {
    const cols = "caseId,hasNodule,diameterMm,type,lobe,vendor,kernel,dose,bmi,pred_nodule,pred_confidence,pred_lobe,pred_diameterMm,det_correct,loc_correct,size_correct,size_error_mm,source,latency_ms,rationale";
    const rows = cases.map((c) => {
      const g = c.groundTruth;
      const r = results[c.id];
      const a = r?.answer;
      return [
        c.id, g.hasNodule, g.diameterMm, g.type, g.lobe, g.vendor, g.kernel, g.dose, c.bmi,
        a?.nodulePresent ?? "", a?.confidence ?? "", a?.lobe ?? "", a?.diameterMm ?? "",
        r?.correct.detection ?? "", r?.correct.localization ?? "", r?.correct.size ?? "", r?.sizeErrorMm ?? "",
        r?.source ?? "", r?.latencyMs ?? "", `"${(a?.rationale ?? "").replace(/"/g, '""')}"`,
      ].join(",");
    });
    download("vlm-bench-cases.csv", [cols, ...rows].join("\n"), "text/csv");
  };
  const exportJson = () =>
    download(
      "vlm-bench-run.json",
      JSON.stringify(
        { design, reader: readerBadge?.text, prompt, cases: cases.map((c) => ({ id: c.id, groundTruth: c.groundTruth, bmi: c.bmi, pos: c.pos, result: results[c.id] ?? null })), metrics, report },
        null,
        2,
      ),
      "application/json",
    );

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <header className="space-y-1">
        <p className="font-mono text-xs uppercase tracking-widest text-accent">Stage 4 · Controlled model evaluation · LangGraph Send() map-reduce</p>
        <h1 className="text-2xl font-semibold tracking-tight">VLM Benchmark</h1>
        <p className="max-w-3xl text-sm text-ink-2">
          A real evaluation, not a simulation: choose a factorial design, render each virtual case in your browser, and send every slice to a vision-language
          model. One structured call per image answers three tasks — <b>detection</b>, <b>lobe localization</b> and <b>size</b> — which are scored against the
          known ground truth and stratified by nodule size, type and scanner arm.
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[1.25fr_1fr]">
        <div className="card space-y-4 p-4">
          <h2 className="text-sm font-medium">Factorial design</h2>
          <ChipGroup label="Nodule diameter (mm) · 0 = negative control">
            {SIZE_OPTIONS.map((s) => (
              <Chip key={s} on={sizes.includes(s)} onClick={() => setSizes(toggle(sizes, s))}>
                {s === 0 ? "0 (none)" : s}
              </Chip>
            ))}
          </ChipGroup>
          <ChipGroup label="Nodule type">
            {NODULE_TYPES.map((t) => (
              <Chip key={t} on={types.includes(t)} onClick={() => setTypes(toggle(types, t))}>
                {t}
              </Chip>
            ))}
          </ChipGroup>
          <ChipGroup label="Lobe">
            {LOBES.map((l) => (
              <Chip key={l} on={lobes.includes(l)} onClick={() => setLobes(toggle(lobes, l))}>
                {l}
              </Chip>
            ))}
          </ChipGroup>
          <div>
            <div className="mb-1.5 text-xs text-ink-3">Scanner arms (vendor · kernel · dose)</div>
            <div className="flex flex-wrap gap-2">
              {arms.map((a) => (
                <span key={armKey(a)} className="flex items-center gap-1.5 rounded-md border border-line bg-surface-2 px-2 py-1 font-mono text-[11px]">
                  {armKey(a)}
                  <button aria-label={`remove ${armKey(a)}`} onClick={() => setArms(arms.filter((x) => armKey(x) !== armKey(a)))} className="text-ink-3 hover:text-ink">
                    <X className="h-3 w-3" />
                  </button>
                </span>
              ))}
              {ARM_PRESETS.filter((p) => !arms.some((a) => armKey(a) === armKey(p))).map((p) => (
                <button
                  key={armKey(p)}
                  onClick={() => setArms([...arms, p])}
                  className="rounded-md border border-dashed border-line px-2 py-1 font-mono text-[11px] text-ink-3 hover:border-accent hover:text-ink"
                >
                  + {armKey(p)}
                </button>
              ))}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-[11px]">
              {(
                [
                  ["vendor", VENDORS],
                  ["kernel", KERNELS],
                  ["dose", DOSES],
                ] as const
              ).map(([k, opts]) => (
                <select
                  key={k}
                  aria-label={k}
                  value={customArm[k]}
                  onChange={(e) => setCustomArm({ ...customArm, [k]: e.target.value })}
                  className="rounded-md border border-line bg-background px-1.5 py-1"
                >
                  {opts.map((o) => (
                    <option key={o}>{o}</option>
                  ))}
                </select>
              ))}
              <button
                disabled={arms.length >= 6 || arms.some((a) => armKey(a) === armKey(customArm))}
                onClick={() => setArms([...arms, customArm])}
                className="flex items-center gap-1 rounded-md border border-line px-2 py-1 hover:border-accent disabled:opacity-40"
              >
                <Plus className="h-3 w-3" /> add arm
              </button>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <label className="space-y-1">
              <span className="flex justify-between text-ink-2">
                Cases (images sent to the VLM) <span className="font-mono text-ink">{nCases}</span>
              </span>
              <input type="range" min={4} max={MAX_CASES} step={1} value={nCases} onChange={(e) => setNCases(+e.target.value)} className="w-full" />
            </label>
            <label className="space-y-1">
              <span className="flex justify-between text-ink-2">
                Random seed <span className="font-mono text-ink">{seed}</span>
              </span>
              <input type="range" min={1} max={999} value={seed} onChange={(e) => setSeed(+e.target.value)} className="w-full" />
            </label>
          </div>
          <p className="text-[11px] text-ink-3">
            {gridCells} design cells · {nCases} cases sampled with size-balanced interleaving
            {nCases < gridCells ? " (fractional design — not every cell is covered)" : ""}.
          </p>
          <div className="flex flex-wrap items-center gap-3">
            {running ? (
              <button onClick={() => abort.current?.abort()} className="flex items-center gap-2 rounded-lg bg-surface-2 px-4 py-2 text-sm font-medium">
                <Square className="h-4 w-4" /> Stop
              </button>
            ) : (
              <button
                onClick={run}
                disabled={!valid}
                className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:brightness-110 disabled:opacity-40"
              >
                <Play className="h-4 w-4" /> Run benchmark
              </button>
            )}
            {readerBadge && (
              <span
                className={`rounded-full border px-2 py-0.5 font-mono text-[11px] ${readerBadge.live ? "border-[var(--series-3)] text-[var(--series-3)]" : "border-[var(--status-warning)] text-[var(--status-warning)]"}`}
              >
                {readerBadge.text}
              </span>
            )}
          </div>
          {error && <p className="rounded-md border border-[var(--status-critical)] p-2 text-xs text-ink">⚠ {error}</p>}
        </div>

        <div className="card flex flex-col gap-3 p-4">
          <h2 className="text-sm font-medium">LangGraph pipeline</h2>
          <ol className="space-y-1.5">
            {PIPELINE.map((n, i) => {
              const st = status[n.id];
              return (
                <li key={n.id} className="flex items-center gap-2 text-xs">
                  <span
                    className={`flex h-5 w-5 items-center justify-center rounded-full border font-mono text-[10px] ${st === "done" ? "border-[var(--series-3)] text-[var(--series-3)]" : st === "running" ? "border-accent text-accent" : "border-line text-ink-3"}`}
                  >
                    {st === "running" ? <Loader2 className="h-3 w-3 animate-spin" /> : st === "done" ? "✓" : i + 1}
                  </span>
                  <span className="font-mono text-ink">{n.title}</span>
                  <span className="text-ink-3">{n.role}</span>
                  {n.id === "vlm_reader" && phase !== "idle" && (
                    <span className="ml-auto font-mono text-ink-2">
                      {resultList.length}/{cases.length || nCases}
                      {reading.size ? ` · ${reading.size} in flight` : ""}
                    </span>
                  )}
                </li>
              );
            })}
          </ol>
          <div className="h-1.5 overflow-hidden rounded-full bg-surface-2">
            <div
              className="h-full bg-accent transition-all"
              style={{ width: `${phase === "rendering" ? (rendered / nCases) * 30 : phase === "idle" ? 0 : 30 + (resultList.length / Math.max(1, cases.length)) * 70}%` }}
            />
          </div>
          <p className="text-[11px] text-ink-3">
            {phase === "rendering" && `Rendering slices in the browser… ${rendered}/${nCases}`}
            {phase === "reading" && "Fan-out: one Send() per case; ≤6 concurrent model calls (semaphore)."}
            {phase === "done" && "Done. Click any case to inspect the exact prompt and raw structured answer."}
            {phase === "idle" && "Each image is a 256 px lung-window PNG of the axial slice through the nodule centre (or the matched level for controls)."}
          </p>
          {logs.length > 0 && (
            <div className="max-h-24 overflow-auto rounded-md border border-line bg-background p-2 font-mono text-[10px] text-[var(--status-warning)]">
              {logs.map((l, i) => (
                <div key={i}>! {l}</div>
              ))}
            </div>
          )}
          <MethodsNote />
        </div>
      </section>

      <section ref={gridRef} className="card scroll-mt-20 p-4">
        <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-sm font-medium">Cases · live answers</h2>
          {metrics && (
            <span className="font-mono text-[11px] text-ink-2">
              {TASKS.map((t) => `${t.short} ${metrics.overall[t.id] ? pct(metrics.overall[t.id]!.rate) : "–"}`).join(" · ")} · n={metrics.n}
            </span>
          )}
        </div>
        {cases.length === 0 ? (
          <p className="py-10 text-center text-xs text-ink-3">Configure a design and press Run — thumbnails appear here as they render, then fill in with the VLM&apos;s answers.</p>
        ) : (
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4 md:grid-cols-6 lg:grid-cols-8">
            {cases.map((c) => (
              <CaseTile key={c.id} c={c} r={results[c.id]} busy={reading.has(c.id)} active={selected === c.id} onClick={() => setSelected(c.id)} />
            ))}
          </div>
        )}
      </section>

      {sel && <Inspector c={sel} r={results[sel.id]} render={renders[sel.id]} prompt={prompt} onClose={() => setSelected(null)} />}

      {metrics && phase === "done" && (
        <>
          <Kpis m={metrics} />
          <section className="grid gap-4 lg:grid-cols-2">
            <TaskForest title="Accuracy by nodule size" rows={metrics.bySize} />
            <TaskForest title="Accuracy by nodule type" rows={metrics.byType} />
            <TaskForest title="Accuracy by scanner arm" rows={metrics.byArm} />
            <TaskForest title="Accuracy by lobe" rows={metrics.byLobe} />
          </section>
          <section className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
            <ConfusionSummary m={metrics} />
            <SizeScatter points={metrics.sizeScatter} onSelect={setSelected} />
          </section>
          <section className="grid gap-4 lg:grid-cols-[1.4fr_1fr]">
            <div className="card p-4">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-sm font-medium">Failure analysis · analyst node</h2>
                {status.analyst === "running" && <Loader2 className="h-4 w-4 animate-spin text-accent" />}
              </div>
              <Markdown>{report || "_Waiting for the analyst…_"}</Markdown>
            </div>
            <div className="card space-y-3 p-4">
              <h2 className="text-sm font-medium">Export</h2>
              <p className="text-xs text-ink-2">Per-case ground truth, raw answers and scoring, plus the full run (design, prompt, metrics, summary). Images are regenerable from design + seed.</p>
              <div className="flex flex-wrap gap-2">
                <button onClick={exportCsv} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:border-accent">
                  <Download className="h-3.5 w-3.5" /> Cases CSV ({cases.length})
                </button>
                <button onClick={exportJson} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:border-accent">
                  <Download className="h-3.5 w-3.5" /> Run JSON
                </button>
              </div>
            </div>
          </section>
        </>
      )}
    </div>
  );
}

const ChipGroup = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div>
    <div className="mb-1.5 text-xs text-ink-3">{label}</div>
    <div className="flex flex-wrap gap-1.5">{children}</div>
  </div>
);

const Chip = ({ on, onClick, children }: { on: boolean; onClick: () => void; children: React.ReactNode }) => (
  <button
    onClick={onClick}
    aria-pressed={on}
    className={`rounded-full border px-2.5 py-0.5 font-mono text-[11px] ${on ? "border-accent bg-surface-2 text-ink" : "border-line text-ink-3 hover:text-ink"}`}
  >
    {children}
  </button>
);

function Mark({ ok, label }: { ok: boolean | null | undefined; label: string }) {
  if (ok === null) return <span className="text-ink-3" title={`${label}: n/a (negative control)`}>{label}·</span>;
  if (ok === undefined) return <span className="text-ink-3">{label}?</span>;
  return (
    <span className={ok ? "text-[var(--series-3)]" : "text-[var(--status-critical)]"} title={`${label}: ${ok ? "correct" : "wrong"}`}>
      {label}
      {ok ? "✓" : "✗"}
    </span>
  );
}

function CaseTile({ c, r, busy, active, onClick }: { c: BenchCase; r?: CaseResult; busy: boolean; active: boolean; onClick: () => void }) {
  return (
    <button onClick={onClick} className="group space-y-1 text-left">
      <div className={`relative overflow-hidden rounded-md border ${active ? "border-accent" : "border-line group-hover:border-accent"}`}>
        {/* eslint-disable-next-line @next/next/no-img-element -- in-memory data URL */}
        <img src={c.imageDataUrl} alt={`case ${c.id}`} className="block h-auto w-full" />
        {busy && (
          <span className="absolute inset-0 flex items-center justify-center bg-black/40">
            <Loader2 className="h-4 w-4 animate-spin text-accent" />
          </span>
        )}
        {r?.source === "simulated" && <span className="absolute right-1 top-1 rounded bg-black/70 px-1 font-mono text-[9px] text-[var(--status-warning)]">SIM</span>}
        <span className="absolute left-1 top-1 rounded bg-black/60 px-1 font-mono text-[9px] text-ink-2">{c.id}</span>
      </div>
      <div className="flex justify-between font-mono text-[10px]">
        <Mark ok={r?.correct.detection} label="D" />
        <Mark ok={r ? r.correct.localization : undefined} label="L" />
        <Mark ok={r ? r.correct.size : undefined} label="S" />
      </div>
      <div className="truncate text-[10px] leading-tight text-ink-3">{gtLabel(c)}</div>
    </button>
  );
}

function Inspector({ c, r, render, prompt, onClose }: { c: BenchCase; r?: CaseResult; render?: RenderResult; prompt: string; onClose: () => void }) {
  const [showGt, setShowGt] = useState(false);
  return (
    <section className="card grid gap-4 p-4 lg:grid-cols-[280px_1fr_1fr]">
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-medium">Prompt / answer inspector · {c.id}</h2>
          <button onClick={onClose} aria-label="close inspector" className="text-ink-3 hover:text-ink">
            <X className="h-4 w-4" />
          </button>
        </div>
        {showGt && render ? (
          <CTCanvas result={render} mode="ct" window={WINDOWS.lung} showMarker label="ground-truth marker" />
        ) : (
          // eslint-disable-next-line @next/next/no-img-element -- exact bytes sent to the model
          <img src={c.imageDataUrl} alt={`case ${c.id} as sent to the VLM`} className="w-full rounded-lg border border-line" />
        )}
        <label className="flex items-center gap-2 text-[11px] text-ink-2">
          <input type="checkbox" checked={showGt} onChange={(e) => setShowGt(e.target.checked)} disabled={!render} /> show ground-truth marker (not sent to the model)
        </label>
        <div className="text-[11px] text-ink-2">
          <div>
            <span className="text-ink-3">Truth:</span> {gtLabel(c)}
          </div>
          <div>
            <span className="text-ink-3">Scanner:</span> {armKey(c.groundTruth)} · BMI {c.bmi}
          </div>
          {r && (
            <div>
              <span className="text-ink-3">Reader:</span> {r.source === "live" ? "live VLM" : "simulated reader"} · {(r.latencyMs / 1000).toFixed(1)} s
            </div>
          )}
        </div>
      </div>
      <div className="min-w-0">
        <div className="mb-1 text-xs text-ink-3">Exact prompt (text part; the PNG is attached as an image_url content part)</div>
        <pre className="max-h-80 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-background p-2 font-mono text-[10.5px] leading-4 text-ink-2">{prompt}</pre>
      </div>
      <div className="min-w-0 space-y-2">
        <div className="text-xs text-ink-3">Raw structured answer (zod schema VlmAnswer)</div>
        <pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded-lg border border-line bg-background p-2 font-mono text-[10.5px] leading-4 text-ink">
          {r ? JSON.stringify(r.answer, null, 2) : "(awaiting answer…)"}
        </pre>
        {r && (
          <div className="grid grid-cols-3 gap-2 text-center text-[11px]">
            {TASKS.map((t) => {
              const ok = r.correct[t.id];
              return (
                <div key={t.id} className="rounded-md border border-line p-1.5">
                  <div className="text-ink-3">{t.short}</div>
                  <div className={ok === null ? "text-ink-3" : ok ? "text-[var(--series-3)]" : "text-[var(--status-critical)]"}>
                    {ok === null ? "n/a" : ok ? "✓ correct" : "✗ wrong"}
                  </div>
                </div>
              );
            })}
          </div>
        )}
        {r?.sizeErrorMm != null && <div className="text-[11px] text-ink-2">Size error: {r.sizeErrorMm > 0 ? "+" : ""}{r.sizeErrorMm} mm</div>}
      </div>
    </section>
  );
}

function Kpis({ m }: { m: VlmMetrics }) {
  const ci = (r: { lo: number; hi: number } | null) => (r ? `95% CI ${pct(r.lo)}–${pct(r.hi)}` : "");
  const tiles = [
    { k: "Detection accuracy", v: m.overall.detection ? pct(m.overall.detection.rate) : "–", sub: ci(m.overall.detection) },
    { k: "Sensitivity / specificity", v: `${m.sensitivity ? pct(m.sensitivity.rate) : "–"} / ${m.specificity ? pct(m.specificity.rate) : "–"}`, sub: `${m.nPositive} pos · ${m.nNegative} neg` },
    { k: "Detection AUC", v: m.auc !== null ? m.auc.toFixed(2) : "–", sub: "from stated confidence" },
    { k: "Lobe accuracy", v: m.overall.localization ? pct(m.overall.localization.rate) : "–", sub: ci(m.overall.localization) },
    { k: "Size within tolerance", v: m.overall.size ? pct(m.overall.size.rate) : "–", sub: ci(m.overall.size) },
    { k: "Size MAE", v: m.size ? `${m.size.mae.toFixed(1)} mm` : "–", sub: m.size ? `bootstrap CI ${m.size.lo.toFixed(1)}–${m.size.hi.toFixed(1)} · bias ${m.size.bias > 0 ? "+" : ""}${m.size.bias.toFixed(1)}` : "" },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
      {tiles.map((t) => (
        <div key={t.k} className="card p-3">
          <div className="text-[11px] text-ink-3">{t.k}</div>
          <div className="font-mono text-xl text-ink">{t.v}</div>
          <div className="text-[10px] text-ink-3">{t.sub}</div>
        </div>
      ))}
    </div>
  );
}

function MethodsNote() {
  return (
    <details className="rounded-lg border border-line bg-background p-3 text-[11px] leading-relaxed text-ink-2">
      <summary className="flex cursor-pointer items-center gap-1.5 font-medium text-ink">
        <FlaskConical className="h-3.5 w-3.5 text-accent" /> Methods &amp; limitations
      </summary>
      <ul className="mt-2 list-disc space-y-1 pl-4">
        <li>
          Each case is a <b>single 2D axial slice</b> (256 px, lung window) through the nodule centre — no volumetric context, no adjacent slices to tell
          nodules from vessels.
        </li>
        <li>
          Images come from this app&apos;s <b>procedural phantom</b>, not the NodMAISI / MAISI generative models used by iTRIALSPACE; realism is limited.
        </li>
        <li>
          The reader is a <b>general-purpose VLM</b>, not a medical VLM. The iTRIALSPACE paper evaluates medical VLMs (e.g. BiomedCLIP, LLaVA-Med,
          MedGemma) at this stage and checks that synthetic-vs-real model rankings agree.
        </li>
        <li>Small n per stratum → wide Wilson CIs; subgroup gaps are hypotheses. Results illustrate the evaluation pipeline, not clinical performance.</li>
        <li>Demo mode / failed calls use a clearly badged deterministic simulated reader (accuracy depends on size, type and dose).</li>
      </ul>
    </details>
  );
}
