"use client";

import { useMemo, useRef, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { AlertOctagon, AlertTriangle, CheckCircle2, Download, Info, Loader2, Play, Plus, Square, Undo2, UserCheck, X } from "lucide-react";
import { AgentGraph, type GraphEdge, type GraphNode, type NodeStatus } from "@/components/agent-graph";
import { FrocChart, ForestPlot, Heatmap } from "@/components/charts";
import { Markdown } from "@/components/markdown";
import { CTCanvas } from "@/components/ct-canvas";
import { streamAgent } from "@/lib/client-stream";
import type { AgentEvent } from "@/lib/stream";
import {
  DOSES,
  EXAMPLE_PROMPTS,
  KERNELS,
  LOBES,
  MODEL_PROFILES,
  MODEL_PROFILE_INFO,
  NODULE_TYPES,
  VENDORS,
  type ScannerArm,
  type TrialSpec,
} from "@/lib/trial/spec";
import type { TrialAnalysis, VirtualLesion } from "@/lib/trial/simulate";
import { renderSlice, WINDOWS, placeNodule, Z_EXTENT_MM } from "@/lib/phantom/render";

const NODES: GraphNode[] = [
  { id: "protocol_architect", title: "Protocol Architect", role: "NL → typed protocol (LLM)", x: 85, y: 165 },
  { id: "protocol_reviewer", title: "Protocol Reviewer", role: "validity · power check", x: 255, y: 165 },
  { id: "human_approval", title: "Human Approval", role: "interrupt() · checkpoint", x: 425, y: 165 },
  { id: "evidence_librarian", title: "Evidence Librarian", role: "corrective-RAG subgraph", x: 620, y: 55 },
  { id: "cohort_synthesizer", title: "Cohort Synthesizer", role: "virtual patients + lesions", x: 620, y: 275 },
  { id: "simulate_arm", title: "Scanner-arm Workers", role: "Send: physics + AI per arm", x: 800, y: 275 },
  { id: "biostatistician", title: "Biostatistician", role: "reduce · Wilson CI · FROC", x: 990, y: 275 },
  { id: "report_writer", title: "Report Writer", role: "grounded report (LLM)", x: 1185, y: 165 },
];
const EDGES: GraphEdge[] = [
  { from: "protocol_architect", to: "protocol_reviewer" },
  { from: "protocol_reviewer", to: "protocol_architect", kind: "cycle" },
  { from: "protocol_reviewer", to: "human_approval", kind: "conditional" },
  { from: "human_approval", to: "protocol_architect", kind: "cycle", label: "researcher: send back", arc: 62 },
  { from: "human_approval", to: "evidence_librarian", kind: "conditional" },
  { from: "human_approval", to: "cohort_synthesizer", kind: "conditional" },
  { from: "cohort_synthesizer", to: "simulate_arm", kind: "conditional" },
  { from: "simulate_arm", to: "biostatistician" },
  { from: "biostatistician", to: "report_writer" },
  { from: "evidence_librarian", to: "report_writer" },
];
const NODE_IDS = new Set(NODES.map((n) => n.id));

type Evidence = { chunkId: string; title: string; source: string; url: string; snippet: string; rrf: number };
type Result = {
  approval: { by: "human" | "auto"; edited: boolean } | null;
  spec: TrialSpec;
  analysis: TrialAnalysis;
  report: string;
  evidence: Evidence[];
  lesions: VirtualLesion[];
  patients: { patientId: string; arm: number; age: number; sex: string; bmi: number; emphysema: number; fpCandidates: number; falsePositiveScores?: number[] }[];
};
type StepLog = { node: string; status: string; detail?: string; data?: unknown; t: number };
type Pending = { threadId: string; spec: TrialSpec; revisions: number };
type ArmCard = { arm: number; label: string; done: boolean; sensitivity?: number; meanNoiseHU?: number; lesions?: number };

export function TrialClient() {
  const sp = useSearchParams();
  const [request, setRequest] = useState(sp.get("request") ?? EXAMPLE_PROMPTS[1]);
  const [humanApproval, setHumanApproval] = useState(true);
  const [pending, setPending] = useState<Pending | null>(null);
  const [arms, setArms] = useState<Record<number, ArmCard>>({});
  const [profile, setProfile] = useState<(typeof MODEL_PROFILES)[number] | "auto">("auto");
  const [patients, setPatients] = useState(120);
  const [seed, setSeed] = useState(42);
  const [running, setRunning] = useState(false);
  const [status, setStatus] = useState<Record<string, NodeStatus>>({});
  const [steps, setSteps] = useState<StepLog[]>([]);
  const [report, setReport] = useState("");
  const [result, setResult] = useState<Result | null>(null);
  const [meta, setMeta] = useState<{ mode: string; model: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const abort = useRef<AbortController | null>(null);
  const graphRef = useRef<HTMLElement>(null);
  const t0 = useRef(0);

  function onEvent(e: AgentEvent) {
    const t = performance.now() - t0.current;
    if (e.type === "meta") setMeta(e.data as { mode: string; model: string });
    if (e.type === "step") {
      if (NODE_IDS.has(e.node) && e.node !== "simulate_arm") setStatus((s) => ({ ...s, [e.node]: e.status === "start" ? "running" : "done" }));
      if (e.node === "simulate_arm") {
        const d = e.data as Omit<ArmCard, "done">;
        setArms((a) => ({ ...a, [d.arm]: { ...a[d.arm], ...d, label: e.detail ?? "", done: e.status === "done" } }));
      }
      setSteps((s) => [...s, { node: e.node, status: e.status, detail: e.detail, data: e.data, t }]);
    }
    if (e.type === "log") setSteps((s) => [...s, { node: e.node, status: "log", detail: e.message, t }]);
    if (e.type === "token") setReport((r) => r + e.text);
    if (e.type === "reset") setReport("");
    if (e.type === "interrupt") {
      setPending(e.data as Pending);
      setStatus((s) => ({ ...s, human_approval: "waiting" }));
      setSteps((s) => [...s, { node: "human_approval", status: "log", detail: "paused · waiting for researcher approval", t }]);
    }
    if (e.type === "result") setResult(e.data as Result);
    if (e.type === "error") setError(e.message);
  }

  async function call(body: unknown) {
    abort.current?.abort();
    const ctrl = new AbortController();
    abort.current = ctrl;
    setRunning(true);
    setError(null);
    try {
      await streamAgent("/api/trial", body, onEvent, ctrl.signal);
    } catch (err) {
      if (!ctrl.signal.aborted) setError((err as Error).message);
    } finally {
      setRunning(false);
    }
  }

  async function run() {
    setStatus(humanApproval ? {} : { human_approval: "skipped" });
    setSteps([]);
    setArms({});
    setReport("");
    setResult(null);
    setPending(null);
    t0.current = performance.now();
    graphRef.current?.scrollIntoView({ behavior: "smooth", block: "center" });
    const overrides: Partial<TrialSpec> = { patientsPerArm: patients, seed };
    if (profile !== "auto") overrides.modelUnderTest = profile;
    await call({ request, overrides, humanApproval });
  }

  async function resume(decision: { action: "approve"; spec: TrialSpec } | { action: "revise"; feedback: string }) {
    if (!pending) return;
    const { threadId } = pending;
    setPending(null);
    setStatus((s) => ({ ...s, human_approval: "running" }));
    await call({ threadId, request, decision });
  }

  const selectedSteps = steps.filter((s) => s.node === selected && s.status === "done");
  const spec =
    result?.spec ??
    (steps.findLast((s) => ["protocol_architect", "human_approval"].includes(s.node) && s.status === "done" && (s.data as { spec?: TrialSpec })?.spec)
      ?.data as { spec: TrialSpec } | undefined)?.spec;
  const armList = Object.values(arms).sort((a, b) => a.arm - b.arm);
  const badges: Record<string, string> = armList.length ? { simulate_arm: `${armList.filter((a) => a.done).length}/${armList.length} arms` } : {};
  const graphStatus: Record<string, NodeStatus> = {
    ...status,
    ...(armList.length ? { simulate_arm: armList.every((a) => a.done) ? "done" : "running" } : {}),
  };

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <header className="space-y-1">
        <p className="font-mono text-xs uppercase tracking-widest text-accent">Stage 2 · Explicit trial specification · LangGraph multi-agent orchestration</p>
        <h1 className="text-2xl font-semibold tracking-tight">Virtual Imaging Trial Designer</h1>
        <p className="max-w-3xl text-sm text-ink-2">
          Describe a trial in plain English. Specialist agents in a LangGraph state machine turn it into a typed protocol, review it (and loop back if it is
          under-powered), <strong className="text-ink">pause for your approval</strong> (interrupt + checkpoint), synthesize a virtual cohort, fan out one
          worker per scanner arm in parallel (Send API), compute stratified statistics, and write a report grounded by a corrective-RAG subgraph.
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
        <div className="card space-y-4 p-4">
          <label className="block text-sm font-medium" htmlFor="req">
            Trial request
          </label>
          <textarea
            id="req"
            value={request}
            onChange={(e) => setRequest(e.target.value)}
            rows={4}
            className="w-full resize-y rounded-lg border border-line bg-background p-3 text-sm outline-none focus:border-accent"
          />
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_PROMPTS.map((p, i) => (
              <button
                key={i}
                onClick={() => setRequest(p)}
                className="rounded-full border border-line px-3 py-1 text-xs text-ink-2 hover:border-accent hover:text-ink"
              >
                Example {i + 1}
              </button>
            ))}
          </div>
          <fieldset>
            <legend className="mb-2 text-sm font-medium">AI model under test</legend>
            <div className="grid gap-2 sm:grid-cols-2">
              <button
                onClick={() => setProfile("auto")}
                className={`rounded-lg border p-2.5 text-left text-xs ${profile === "auto" ? "border-accent bg-surface-2" : "border-line"}`}
              >
                <div className="font-medium text-ink">Let the Architect decide</div>
                <div className="text-ink-3">Inferred from your request</div>
              </button>
              {MODEL_PROFILES.map((m) => (
                <button
                  key={m}
                  onClick={() => setProfile(m)}
                  className={`rounded-lg border p-2.5 text-left text-xs ${profile === m ? "border-accent bg-surface-2" : "border-line"}`}
                >
                  <div className="font-medium text-ink">{MODEL_PROFILE_INFO[m].name}</div>
                  <div className="text-ink-3">{MODEL_PROFILE_INFO[m].blurb}</div>
                </button>
              ))}
            </div>
          </fieldset>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <label className="space-y-1">
              <span className="flex justify-between text-ink-2">
                Virtual patients per arm <span className="font-mono text-ink">{patients}</span>
              </span>
              <input type="range" min={20} max={500} step={10} value={patients} onChange={(e) => setPatients(+e.target.value)} className="w-full" />
            </label>
            <label className="space-y-1">
              <span className="flex justify-between text-ink-2">
                Random seed <span className="font-mono text-ink">{seed}</span>
              </span>
              <input type="range" min={1} max={999} value={seed} onChange={(e) => setSeed(+e.target.value)} className="w-full" />
            </label>
          </div>
          <label className="flex items-start gap-2 rounded-lg border border-line p-2.5 text-xs">
            <input type="checkbox" checked={humanApproval} onChange={(e) => setHumanApproval(e.target.checked)} className="mt-0.5" />
            <span>
              <span className="flex items-center gap-1.5 font-medium text-ink">
                <UserCheck className="h-3.5 w-3.5 text-accent" /> Human-in-the-loop protocol approval
              </span>
              <span className="text-ink-3">The graph pauses after review (LangGraph interrupt). Approve, edit the protocol, or send it back with feedback.</span>
            </span>
          </label>
          <div className="flex items-center gap-3">
            {running ? (
              <button onClick={() => abort.current?.abort()} className="flex items-center gap-2 rounded-lg bg-surface-2 px-4 py-2 text-sm font-medium">
                <Square className="h-4 w-4" /> Stop
              </button>
            ) : (
              <button onClick={run} className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:brightness-110">
                <Play className="h-4 w-4" /> Run virtual trial
              </button>
            )}
            {meta && (
              <span className="rounded-full border border-line px-2 py-0.5 font-mono text-[11px] text-ink-3">
                {meta.mode === "live" ? `LLM: ${meta.model}` : "demo mode · deterministic agents"}
              </span>
            )}
          </div>
          {error && <p className="rounded-md border border-[var(--status-critical)] p-2 text-xs text-ink">⚠ {error}</p>}
        </div>

        <div className="card flex flex-col p-4">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-medium">{selected ? `State written by ${selected}` : "Execution trace"}</h2>
            <span className="text-[11px] text-ink-3">click a node in the graph below to inspect its output</span>
          </div>
          <div className="min-h-64 flex-1 overflow-auto rounded-lg border border-line bg-background p-2 font-mono text-[11px] leading-5 lg:max-h-[420px]">
            {selected ? (
              <>
                <div className="mb-1 flex justify-end text-ink-2">
                  <button className="text-accent" onClick={() => setSelected(null)}>
                    show trace
                  </button>
                </div>
                <pre className="whitespace-pre-wrap text-ink-3">
                  {selectedSteps.length ? JSON.stringify(selectedSteps.at(-1)!.data ?? {}, null, 1).slice(0, 4000) : "(not executed yet)"}
                </pre>
              </>
            ) : steps.length ? (
              steps.map((s, i) => (
                <div key={i} className="flex gap-2">
                  <span className="w-14 shrink-0 text-right text-ink-3">{(s.t / 1000).toFixed(2)}s</span>
                  <span className={s.status === "start" ? "text-accent" : s.status === "log" ? "text-[var(--status-warning)]" : "text-[var(--series-3)]"}>
                    {s.status === "start" ? "▶" : s.status === "log" ? "!" : "✓"}
                  </span>
                  <span className={NODE_IDS.has(s.node) ? "text-ink" : "pl-3 text-ink-2"}>
                    {NODE_IDS.has(s.node) ? s.node : `↳ rag:${s.node}`}
                  </span>
                  <span className="truncate text-ink-3">{s.detail}</span>
                </div>
              ))
            ) : (
              <span className="text-ink-3">Execution trace will stream here…</span>
            )}
          </div>
        </div>
      </section>

      <section ref={graphRef} className="card scroll-mt-20 p-4">
        <div className="mb-1 flex flex-wrap items-center justify-between gap-2">
          <h2 className="text-sm font-medium">Agent execution graph · LangGraph StateGraph</h2>
          <span className="text-[11px] text-ink-3">
            dashed = conditional edges · arcs = revise cycles · stacked node = parallel Send workers · librarian runs in parallel with the simulation branch
          </span>
        </div>
        <div className="overflow-x-auto">
          <div className="min-w-[900px]">
            <AgentGraph
              nodes={NODES}
              edges={EDGES}
              status={graphStatus}
              badges={badges}
              width={1280}
              height={320}
              onSelect={setSelected}
              selected={selected}
            />
          </div>
        </div>
        {armList.length > 0 && (
          <div className="mt-3 grid gap-2 sm:grid-cols-2 lg:grid-cols-4">
            {armList.map((a) => (
              <div key={a.arm} className={`rounded-lg border p-2.5 text-xs ${a.done ? "border-line" : "border-accent"}`}>
                <div className="flex items-center justify-between">
                  <span className="font-mono text-[10px] uppercase text-ink-3">worker {a.arm + 1} · Send</span>
                  {a.done ? <CheckCircle2 className="h-3.5 w-3.5 text-[var(--series-3)]" /> : <Loader2 className="h-3.5 w-3.5 animate-spin text-accent" />}
                </div>
                <div className="mt-0.5 font-medium text-ink">{a.label}</div>
                {a.done && (
                  <div className="mt-1 flex gap-3 font-mono text-ink-2">
                    <span>sens {(a.sensitivity! * 100).toFixed(1)}%</span>
                    <span>σ {a.meanNoiseHU} HU</span>
                    <span>n {a.lesions}</span>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </section>

      {pending && (
        <ApprovalPanel key={pending.threadId} pending={pending} onApprove={(s) => resume({ action: "approve", spec: s })} onRevise={(f) => resume({ action: "revise", feedback: f })} />
      )}

      {spec && !pending && <ProtocolCard spec={spec} approval={result?.approval ?? null} />}

      {(report || result) && (
        <section className="grid gap-4 lg:grid-cols-[1.2fr_1fr]">
          <div className="space-y-4">
            {result && <Kpis a={result.analysis} />}
            {result && <Findings a={result.analysis} />}
          </div>
          <div className="card p-4">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-sm font-medium">Trial report · Report Writer agent</h2>
              {running && status.report_writer === "running" && <Loader2 className="h-4 w-4 animate-spin text-accent" />}
            </div>
            <div className="max-h-[560px] overflow-auto pr-1">
              <Markdown>{report || "_Waiting for upstream agents…_"}</Markdown>
            </div>
          </div>
        </section>
      )}

      {result && <Results result={result} />}
    </div>
  );
}

function ProtocolCard({ spec, approval }: { spec: TrialSpec; approval: Result["approval"] }) {
  return (
    <section className="card grid gap-4 p-4 text-xs md:grid-cols-4">
      <div className="md:col-span-4 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-medium">Protocol · {spec.title}</h2>
        <span className="flex items-center gap-3 text-ink-3">
          {approval?.by === "human" && (
            <span className="flex items-center gap-1 text-[var(--series-3)]">
              <UserCheck className="h-3.5 w-3.5" /> approved by researcher{approval.edited ? " (edited)" : ""}
            </span>
          )}
          {MODEL_PROFILE_INFO[spec.modelUnderTest].name}
        </span>
      </div>
      <Field k="Nodule sizes" v={spec.sizesMm.map((s) => `${s} mm`).join(", ")} />
      <Field k="Nodule types" v={spec.noduleTypes.join(", ")} />
      <Field k="Lobes" v={spec.lobes.join(", ")} />
      <Field k="Cohort" v={`${spec.patientsPerArm} patients/arm · ≤${spec.lesionsPerPatient} lesions/patient · seed ${spec.seed}`} />
      <div className="md:col-span-4">
        <div className="mb-1 text-ink-3">Scanner arms</div>
        <div className="flex flex-wrap gap-2">
          {spec.scanners.map((a, i) => (
            <span key={i} className="rounded-md border border-line bg-surface-2 px-2 py-1 font-mono">
              {a.vendor} · {a.kernel} kernel · {a.dose} dose
            </span>
          ))}
        </div>
      </div>
    </section>
  );
}

/** Shown while the graph is paused at the human_approval interrupt: edit the typed protocol, approve, or send it back. */
function ApprovalPanel({ pending, onApprove, onRevise }: { pending: Pending; onApprove: (s: TrialSpec) => void; onRevise: (feedback: string) => void }) {
  const [s, setS] = useState<TrialSpec>(pending.spec);
  const [sizes, setSizes] = useState(pending.spec.sizesMm.join(", "));
  const [feedback, setFeedback] = useState("");
  const parsedSizes = sizes
    .split(/[,\s]+/)
    .map(Number)
    .filter((x) => Number.isFinite(x) && x >= 2 && x <= 40);
  const toggle = <T,>(list: T[], v: T) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);
  const setArm = (i: number, patch: Partial<ScannerArm>) => setS({ ...s, scanners: s.scanners.map((a, j) => (j === i ? { ...a, ...patch } : a)) });
  const valid = parsedSizes.length > 0 && s.noduleTypes.length > 0 && s.lobes.length > 0 && s.scanners.length > 0;
  const nLesions = Math.round(s.patientsPerArm * ((1 + s.lesionsPerPatient) / 2) * s.scanners.length);
  const chip = (on: boolean) => `rounded-full border px-2.5 py-0.5 ${on ? "border-accent bg-surface-2 text-ink" : "border-line text-ink-3"}`;
  const select = "rounded-md border border-line bg-background px-1.5 py-1";

  return (
    <section className="card space-y-4 border-[var(--status-warning)] p-4 text-xs">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-[var(--status-warning)]">Graph paused · interrupt() · thread {pending.threadId.slice(0, 8)}</p>
          <h2 className="text-sm font-medium text-ink">Review the protocol before any virtual patient is generated</h2>
          <p className="text-ink-3">
            The checkpointed state is held server-side. Approving resumes the same thread with <code>Command(&#123; resume &#125;)</code>; the edited
            protocol is re-validated against the Zod schema.
          </p>
        </div>
        <span className="rounded-md border border-line px-2 py-1 font-mono text-ink-2">≈ {nLesions.toLocaleString()} synthetic nodules</span>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <div className="space-y-3">
          <label className="block space-y-1">
            <span className="text-ink-3">Title</span>
            <input value={s.title} onChange={(e) => setS({ ...s, title: e.target.value })} className="w-full rounded-md border border-line bg-background px-2 py-1.5" />
          </label>
          <label className="block space-y-1">
            <span className="text-ink-3">Nodule diameters (mm, comma-separated)</span>
            <input value={sizes} onChange={(e) => setSizes(e.target.value)} className="w-full rounded-md border border-line bg-background px-2 py-1.5 font-mono" />
          </label>
          <div className="space-y-1">
            <span className="text-ink-3">Nodule types</span>
            <div className="flex flex-wrap gap-1.5">
              {NODULE_TYPES.map((t) => (
                <button key={t} onClick={() => setS({ ...s, noduleTypes: toggle(s.noduleTypes, t) })} className={chip(s.noduleTypes.includes(t))}>
                  {t}
                </button>
              ))}
            </div>
          </div>
          <div className="space-y-1">
            <span className="text-ink-3">Lobes</span>
            <div className="flex flex-wrap gap-1.5">
              {LOBES.map((l) => (
                <button key={l} onClick={() => setS({ ...s, lobes: toggle(s.lobes, l) })} className={chip(s.lobes.includes(l))}>
                  {l}
                </button>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <label className="space-y-1">
              <span className="flex justify-between text-ink-3">
                Patients per arm <span className="font-mono text-ink">{s.patientsPerArm}</span>
              </span>
              <input type="range" min={10} max={600} step={10} value={s.patientsPerArm} onChange={(e) => setS({ ...s, patientsPerArm: +e.target.value })} className="w-full" />
            </label>
            <label className="space-y-1">
              <span className="text-ink-3">AI model under test</span>
              <select value={s.modelUnderTest} onChange={(e) => setS({ ...s, modelUnderTest: e.target.value as TrialSpec["modelUnderTest"] })} className={`${select} w-full`}>
                {MODEL_PROFILES.map((m) => (
                  <option key={m} value={m}>
                    {MODEL_PROFILE_INFO[m].name}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <div className="space-y-3">
          <div className="space-y-1.5">
            <span className="text-ink-3">Scanner arms (each becomes one parallel worker)</span>
            {s.scanners.map((a, i) => (
              <div key={i} className="flex items-center gap-1.5">
                <select value={a.vendor} onChange={(e) => setArm(i, { vendor: e.target.value as ScannerArm["vendor"] })} className={select}>
                  {VENDORS.map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
                <select value={a.kernel} onChange={(e) => setArm(i, { kernel: e.target.value as ScannerArm["kernel"] })} className={select}>
                  {KERNELS.map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
                <select value={a.dose} onChange={(e) => setArm(i, { dose: e.target.value as ScannerArm["dose"] })} className={select}>
                  {DOSES.map((v) => (
                    <option key={v}>{v}</option>
                  ))}
                </select>
                <button aria-label="Remove arm" onClick={() => setS({ ...s, scanners: s.scanners.filter((_, j) => j !== i) })} className="text-ink-3 hover:text-ink">
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            {s.scanners.length < 8 && (
              <button
                onClick={() => setS({ ...s, scanners: [...s.scanners, { vendor: "Canon", kernel: "standard", dose: "standard" }] })}
                className="flex items-center gap-1 text-accent"
              >
                <Plus className="h-3.5 w-3.5" /> add arm
              </button>
            )}
          </div>
          <label className="block space-y-1">
            <span className="text-ink-3">…or send it back to the Protocol Architect with feedback</span>
            <textarea
              value={feedback}
              onChange={(e) => setFeedback(e.target.value)}
              rows={2}
              placeholder="e.g. Add ultra-low dose arms and focus on 4–8 mm ground-glass nodules"
              className="w-full resize-y rounded-md border border-line bg-background p-2"
            />
          </label>
        </div>
      </div>

      <div className="flex flex-wrap gap-2">
        <button
          disabled={!valid}
          onClick={() => onApprove({ ...s, sizesMm: [...new Set(parsedSizes)].sort((a, b) => a - b) })}
          className="flex items-center gap-2 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white hover:brightness-110 disabled:opacity-50"
        >
          <CheckCircle2 className="h-4 w-4" /> Approve &amp; run trial
        </button>
        <button
          disabled={feedback.trim().length < 2}
          onClick={() => onRevise(feedback.trim())}
          className="flex items-center gap-2 rounded-lg border border-line px-4 py-2 text-sm hover:border-accent disabled:opacity-50"
        >
          <Undo2 className="h-4 w-4" /> Send back with feedback
        </button>
      </div>
    </section>
  );
}

const Field = ({ k, v }: { k: string; v: string }) => (
  <div>
    <div className="text-ink-3">{k}</div>
    <div className="text-ink">{v}</div>
  </div>
);

function Kpis({ a }: { a: TrialAnalysis }) {
  const tiles = [
    { k: "Virtual patients", v: a.nPatients.toLocaleString() },
    { k: "Synthetic nodules", v: a.nLesions.toLocaleString() },
    { k: "Sensitivity", v: `${(a.overall.rate * 100).toFixed(1)}%`, sub: `95% CI ${(a.overall.lo * 100).toFixed(1)}–${(a.overall.hi * 100).toFixed(1)}%` },
    { k: "FP / scan", v: a.fpPerScanAtOperatingPoint.toFixed(2), sub: "at operating threshold 0.5" },
    { k: "CPM", v: a.froc.cpm.toFixed(3), sub: "mean sens. @ 1/8–8 FP/scan" },
  ];
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
      {tiles.map((t) => (
        <div key={t.k} className="card p-3">
          <div className="text-[11px] text-ink-3">{t.k}</div>
          <div className="font-mono text-xl text-ink">{t.v}</div>
          {t.sub && <div className="text-[10px] text-ink-3">{t.sub}</div>}
        </div>
      ))}
    </div>
  );
}

function Findings({ a }: { a: TrialAnalysis }) {
  const icon = {
    critical: <AlertOctagon className="h-4 w-4 shrink-0 text-[var(--status-critical)]" aria-label="critical" />,
    warning: <AlertTriangle className="h-4 w-4 shrink-0 text-[var(--status-warning)]" aria-label="warning" />,
    info: <Info className="h-4 w-4 shrink-0 text-accent" aria-label="info" />,
  };
  return (
    <div className="card p-4">
      <h2 className="mb-3 text-sm font-medium">Automatically discovered failure modes</h2>
      <ul className="space-y-2.5">
        {a.findings.map((f, i) => (
          <li key={i} className="flex gap-2.5">
            {icon[f.severity]}
            <div>
              <div className="text-sm text-ink">
                <span className="mr-1.5 font-mono text-[10px] uppercase text-ink-3">{f.severity}</span>
                {f.title}
              </div>
              <div className="text-xs text-ink-2">{f.detail}</div>
            </div>
          </li>
        ))}
      </ul>
    </div>
  );
}

function Results({ result }: { result: Result }) {
  const { analysis: a, spec } = result;
  const download = (name: string, body: string, type: string) => {
    const url = URL.createObjectURL(new Blob([body], { type }));
    const el = document.createElement("a");
    el.href = url;
    el.download = name;
    el.click();
    URL.revokeObjectURL(url);
  };
  const csv = () => {
    const cols = ["lesionId", "patientId", "arm", "diameterMm", "type", "lobe", "juxtavascular", "sigmaHU", "cnr", "dprime", "score", "detected"] as const;
    const rows = result.lesions.map((l) =>
      cols.map((c) => (c === "arm" ? `"${spec.scanners[l.arm].vendor} ${spec.scanners[l.arm].kernel} ${spec.scanners[l.arm].dose}"` : String(l[c]))).join(","),
    );
    download("itrialspace-lesions.csv", [cols.join(","), ...rows].join("\n"), "text/csv");
  };
  return (
    <>
      <section className="grid gap-4 lg:grid-cols-2">
        <ForestPlot title="Sensitivity by scanner arm" rows={a.byArm} overall={a.overall.rate} />
        <ForestPlot title="Sensitivity by nodule size" rows={a.bySize} overall={a.overall.rate} />
        <ForestPlot title="Sensitivity by nodule type & vessel attachment" rows={[...a.byType, ...a.byJuxta]} overall={a.overall.rate} />
        <ForestPlot title="Sensitivity by lobe" rows={a.byLobe} overall={a.overall.rate} />
      </section>
      <section className="grid gap-4 lg:grid-cols-2">
        <Heatmap rows={a.heat} />
        <FrocChart series={a.frocByArm} />
      </section>
      <SampleCases result={result} />
      <section className="grid gap-4 lg:grid-cols-[1fr_1fr]">
        <div className="card p-4">
          <h2 className="mb-2 text-sm font-medium">Evidence retrieved by the Librarian agent (corrective-RAG subgraph, graded)</h2>
          <ul className="space-y-2">
            {result.evidence.map((e, i) => (
              <li key={e.chunkId} className="text-xs">
                <a href={e.url} target="_blank" rel="noreferrer" className="font-medium text-ink hover:text-accent">
                  [{i + 1}] {e.title}
                </a>
                <span className="text-ink-3"> — {e.source}</span>
                <p className="line-clamp-2 text-ink-3">{e.snippet}</p>
              </li>
            ))}
          </ul>
          <Link
            href={`/copilot?q=${encodeURIComponent(`Why might a lung nodule detector under-perform on ${spec.scanners.map((s) => s.vendor).join("/")} scanners and ${spec.noduleTypes.join("/")} nodules, and how can a virtual imaging trial help?`)}`}
            className="mt-3 inline-block text-xs text-accent hover:underline"
          >
            Ask the Research Copilot about these results →
          </Link>
        </div>
        <div className="card p-4">
          <h2 className="mb-2 text-sm font-medium">Export</h2>
          <p className="mb-3 text-xs text-ink-2">
            Fully reproducible from the protocol + seed. Download the per-lesion ground truth and model outputs, then re-analyze with the Python package:{" "}
            <code className="font-mono">itrialspace-eval analyze itrialspace-lesions.csv --patients itrialspace-patients.csv</code>
          </p>
          <div className="flex flex-wrap gap-2">
            <button onClick={csv} className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:border-accent">
              <Download className="h-3.5 w-3.5" /> Lesions CSV ({result.lesions.length})
            </button>
            <button
              onClick={() =>
                download(
                  "itrialspace-patients.csv",
                  ["patientId,falsePositiveScores", ...result.patients.map((p) => `${p.patientId},${(p.falsePositiveScores ?? []).join(";")}`)].join("\n"),
                  "text/csv",
                )
              }
              className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:border-accent"
            >
              <Download className="h-3.5 w-3.5" /> Patients CSV (FP candidates)
            </button>
            <button
              onClick={() => download("itrialspace-protocol.json", JSON.stringify(spec, null, 2), "application/json")}
              className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:border-accent"
            >
              <Download className="h-3.5 w-3.5" /> Protocol JSON
            </button>
            <button
              onClick={() => download("itrialspace-report.md", result.report, "text/markdown")}
              className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:border-accent"
            >
              <Download className="h-3.5 w-3.5" /> Report (Markdown)
            </button>
            <button
              onClick={() => download("itrialspace-analysis.json", JSON.stringify(a, null, 2), "application/json")}
              className="flex items-center gap-1.5 rounded-lg border border-line px-3 py-1.5 text-xs hover:border-accent"
            >
              <Download className="h-3.5 w-3.5" /> Statistics JSON
            </button>
          </div>
        </div>
      </section>
    </>
  );
}

/** Renders a few virtual patients from the cohort at the slice through their nodule (missed vs detected). */
function SampleCases({ result }: { result: Result }) {
  const { spec } = result;
  const cases = useMemo(() => {
    // Round-robin across scanner arms so the gallery shows different acquisitions: 3 misses (≥6 mm) + 3 detections.
    const pickAcrossArms = (pool: VirtualLesion[], n: number) => {
      const byArm = spec.scanners.map((_, a) => pool.filter((l) => l.arm === a));
      const out: VirtualLesion[] = [];
      for (let i = 0; out.length < n && i < pool.length; i++) {
        const g = byArm[i % byArm.length];
        const l = g[Math.floor(i / byArm.length)];
        if (l) out.push(l);
      }
      return out;
    };
    const missed = pickAcrossArms(
      result.lesions.filter((l) => !l.detected && l.diameterMm >= 6),
      3,
    );
    const hit = pickAcrossArms(result.lesions.filter((l) => l.detected).reverse(), 6 - missed.length);
    return [...missed, ...hit].map((l) => {
      const p = result.patients.find((x) => x.patientId === l.patientId)!;
      const center = placeNodule(l.lobe, l.pos);
      const zMm = center[2] * Z_EXTENT_MM;
      const nodule = { diameterMm: l.diameterMm, type: l.type, lobe: l.lobe, juxtavascular: l.juxtavascular, spiculated: false, pos: l.pos };
      const r = renderSlice({ size: 192, fovMm: 380, zMm, seed: spec.seed, bmi: p.bmi, nodule, scanner: spec.scanners[l.arm] });
      const zc = r.noduleCenterZmm ?? zMm;
      const r2 =
        Math.abs(zc - zMm) > 0.5 ? renderSlice({ size: 192, fovMm: 380, zMm: zc, seed: spec.seed, bmi: p.bmi, nodule, scanner: spec.scanners[l.arm] }) : r;
      const arm = spec.scanners[l.arm];
      const q = new URLSearchParams({
        d: String(l.diameterMm),
        type: l.type,
        lobe: l.lobe,
        jv: l.juxtavascular ? "1" : "0",
        vendor: arm.vendor,
        kernel: arm.kernel,
        dose: arm.dose,
        bmi: String(p.bmi),
        pos: l.pos.join(","),
        seed: String(spec.seed),
      });
      return { l, r: r2, arm, href: `/studio?${q}` };
    });
  }, [result, spec]);
  return (
    <section className="card p-4">
      <div className="mb-3 flex flex-wrap items-baseline justify-between gap-2">
        <h2 className="text-sm font-medium">Sample virtual patients from this trial</h2>
        <span className="text-[11px] text-ink-3">Rendered in your browser from the cohort ground truth · click to open in the Virtual Patient Studio</span>
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        {cases.map(({ l, r, arm, href }) => (
          <Link key={l.lesionId} href={href} className="group space-y-1">
            <CTCanvas result={r} mode="ct" window={WINDOWS.lung} showMarker className="group-hover:border-accent" />
            <div className="flex items-center justify-between text-[11px]">
              <span className="font-mono text-ink-2">{l.lesionId.replace(/^VP-/, "")}</span>
              <span className={`flex items-center gap-1 ${l.detected ? "text-[var(--series-3)]" : "text-[var(--status-critical)]"}`}>
                {l.detected ? "✓ detected" : "✗ missed"}
              </span>
            </div>
            <div className="text-[10px] leading-tight text-ink-3">
              {l.diameterMm} mm {l.type} · {l.lobe}
              {l.juxtavascular ? " · juxtavascular" : ""}
              <br />
              {arm.vendor} {arm.kernel}/{arm.dose} · score {l.score?.toFixed(2)}
            </div>
          </Link>
        ))}
      </div>
    </section>
  );
}
