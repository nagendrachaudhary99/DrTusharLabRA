import Link from "next/link";
import {
  ArrowRight,
  Bot,
  BrainCircuit,
  Database,
  Eye,
  FlaskConical,
  GitBranch,
  Network,
  ScanLine,
  ShieldCheck,
  Terminal,
  UserCheck,
  Workflow,
} from "lucide-react";

/** The four stages of the iTRIALSPACE pipeline (Tushar et al., 2026), each mapped to a module of this prototype. */
const STAGES = [
  {
    n: "01",
    href: "/cohort",
    icon: Network,
    stage: "Multidataset nodule profiling",
    title: "Cohort Planner",
    tag: "LangChain createAgent · tool calling",
    body: "Faceted nodule index with a NoduleMap-style embedding view. A tool-calling agent turns “300-case screening cohort, 30% prevalence, 6–20 mm, upper lobes” into a stratified cohort manifest.",
  },
  {
    n: "02",
    href: "/trial",
    icon: FlaskConical,
    stage: "Explicit trial specification",
    title: "Trial Designer",
    tag: "LangGraph · interrupt · Send · subgraph",
    body: "Plain English → typed, power-checked protocol. The graph pauses for your approval, fans out one worker per scanner arm, and writes a report grounded by a corrective-RAG subgraph.",
  },
  {
    n: "03",
    href: "/studio",
    icon: ScanLine,
    stage: "Anatomy-aware mask insertion → synthesis",
    title: "Virtual Patient Studio",
    tag: "Conditioning mask · NiiVue 3D",
    body: "Design a nodule to the millimetre, insert it into a lobe's label map, and view the synthetic CT volume in 3D (NiiVue) on four simulated scanners. Export NIfTI for 3D Slicer.",
  },
  {
    n: "04",
    href: "/vlm-bench",
    icon: Eye,
    stage: "Controlled model evaluation",
    title: "VLM Benchmark",
    tag: "Live multimodal LLM · Send fan-out",
    body: "A real vision-language model reads every synthetic slice: detection, lobe localization and size. Stratified accuracy with CIs shows exactly where it fails, and how it compares to the simulated reader.",
  },
];

const PRINCIPLES = [
  { icon: UserCheck, k: "Human in the loop", v: "Agents propose; the researcher approves, edits or sends the protocol back before anything runs." },
  { icon: ShieldCheck, k: "Never breaks", v: "Every LLM step has a deterministic fallback; a circuit breaker drops to demo mode if the provider fails." },
  { icon: FlaskConical, k: "Reproducible", v: "Every trial is a pure function of (protocol, seed). Export CSV/JSON and re-analyze in Python." },
  { icon: Eye, k: "Honest", v: "Procedural phantoms and simplified physics, clearly labelled. Not NodMAISI, not validated, not for clinical use." },
];

const SKILLS = [
  {
    icon: Workflow,
    k: "LangGraph",
    v: "Typed Annotation state with reducers, conditional routing, a review→revise cycle, human-in-the-loop interrupt() with a checkpointer and Command(resume), map-reduce with Send, a RAG graph reused as a subgraph, custom streaming and graph introspection.",
  },
  {
    icon: Bot,
    k: "LangChain",
    v: "createAgent tool-calling agent with Zod tools, withStructuredOutput everywhere (including multimodal image input), LCEL prompt | model | parser with token streaming, custom Embeddings and VectorStore subclasses.",
  },
  {
    icon: Database,
    k: "RAG",
    v: "Hybrid dense + BM25 retrieval with Reciprocal Rank Fusion, multi-query planning, metadata routing, LLM relevance grading, a corrective re-query loop and a groundedness judge, over a source-verified corpus.",
  },
  {
    icon: ShieldCheck,
    k: "Evaluation rigor",
    v: "Seed-reproducible trials, Wilson and bootstrap CIs with n on every stratum, FROC/CPM, automatic failure-mode discovery, and a tested Python package (Docker) that re-analyzes exported results.",
  },
];

export default function Home() {
  return (
    <div>
      <section className="grid-bg border-b border-line">
        <div className="mx-auto max-w-7xl px-4 py-16 md:py-20">
          <p className="mb-4 inline-flex items-center gap-2 rounded-full border border-line bg-surface-1 px-3 py-1 font-mono text-[11px] uppercase tracking-widest text-ink-2">
            <GitBranch className="h-3.5 w-3.5 text-accent" /> Agentic prototype · programmable virtual lesion trials
          </p>
          <h1 className="max-w-4xl text-4xl font-semibold tracking-tight md:text-5xl">
            Specify a cohort in plain English. Let <span className="text-accent">agents</span> build, run and audit the virtual trial.
          </h1>
          <p className="mt-5 max-w-3xl text-base text-ink-2">
            A working prototype inspired by{" "}
            <a href="https://arxiv.org/abs/2605.05761" target="_blank" rel="noreferrer" className="text-ink underline decoration-line underline-offset-4 hover:decoration-accent">
              iTRIALSPACE
            </a>{" "}
            (Tushar et al., 2026), which treats evaluation of lung-CT models as a controlled experiment rather than a fixed benchmark. Here, LangGraph agents
            plan cohorts, design and power-check trial protocols with a human in the loop, synthesize virtual patients with custom nodules, and evaluate both
            detectors and a live vision-language model, stratified by size, lobe, type and scanner.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/trial" className="flex items-center gap-2 rounded-lg bg-accent px-5 py-2.5 text-sm font-medium text-white hover:brightness-110">
              Run a virtual trial <ArrowRight className="h-4 w-4" />
            </Link>
            <Link href="/vlm-bench" className="rounded-lg border border-line px-5 py-2.5 text-sm font-medium text-ink hover:border-accent">
              Benchmark a live VLM
            </Link>
            <Link href="/architecture" className="rounded-lg border border-line px-5 py-2.5 text-sm font-medium text-ink hover:border-accent">
              Architecture
            </Link>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 py-14">
        <div className="mb-6 flex flex-wrap items-baseline justify-between gap-2">
          <h2 className="text-lg font-semibold">One pipeline, four stages</h2>
          <span className="text-xs text-ink-3">Stage names follow the four-stage pipeline described in the iTRIALSPACE paper</span>
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {STAGES.map((m) => (
            <Link key={m.href} href={m.href} className="card group flex flex-col p-5 transition-colors hover:border-accent">
              <div className="mb-4 flex items-center justify-between">
                <m.icon className="h-6 w-6 text-accent" aria-hidden />
                <span className="font-mono text-xs text-ink-3">{m.n}</span>
              </div>
              <p className="text-[11px] uppercase tracking-wider text-ink-3">{m.stage}</p>
              <h3 className="mt-1 text-base font-semibold text-ink">{m.title}</h3>
              <p className="mt-1 font-mono text-[10px] text-accent">{m.tag}</p>
              <p className="mt-2 flex-1 text-sm text-ink-2">{m.body}</p>
              <span className="mt-4 inline-flex items-center gap-1 text-sm text-accent">
                Open <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-0.5" />
              </span>
            </Link>
          ))}
        </div>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <Link href="/copilot" className="card group flex gap-4 p-5 transition-colors hover:border-accent">
            <BrainCircuit className="h-6 w-6 shrink-0 text-accent" aria-hidden />
            <div>
              <h3 className="font-semibold text-ink">Research Copilot</h3>
              <p className="mt-1 text-sm text-ink-2">
                Corrective RAG over 53 source-verified documents (VICTRE, MAISI, NodMAISI, iTRIALSPACE, LIDC-IDRI, Lung-RADS, FDA…) with a full retrieval trace,
                citations and a grounding score.
              </p>
            </div>
          </Link>
          <div className="card flex gap-4 p-5">
            <Terminal className="h-6 w-6 shrink-0 text-accent" aria-hidden />
            <div>
              <h3 className="font-semibold text-ink">Python evaluation package</h3>
              <p className="mt-1 text-sm text-ink-2">
                <code className="font-mono text-xs">itrialspace-eval</code>: cluster-bootstrap CIs, FROC/CPM, failure-mode detection and synthetic-vs-real rank
                agreement (Spearman / Kendall) on any exported trial. Typed, tested, Dockerized. See <code className="font-mono text-xs">python/</code> in the repo.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pb-14">
        <div className="card grid gap-6 p-6 md:grid-cols-[1fr_2fr]">
          <div>
            <h2 className="text-lg font-semibold">Why controlled trials?</h2>
            <p className="mt-2 text-sm text-ink-2">
              Fixed benchmarks entangle lesion size, lobe, anatomy and acquisition, so a single accuracy number can&apos;t say what drives a model&apos;s
              behaviour. A virtual trial changes one factor while holding the rest fixed, with perfect ground truth.
            </p>
          </div>
          <ol className="grid gap-3 text-sm sm:grid-cols-2">
            {PRINCIPLES.map((p) => (
              <li key={p.k} className="rounded-lg border border-line p-3">
                <p.icon className="mb-1 h-4 w-4 text-accent" aria-hidden />
                <div className="font-medium text-ink">{p.k}</div>
                <div className="text-xs text-ink-2">{p.v}</div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="mx-auto max-w-7xl px-4 pb-20">
        <h2 className="mb-6 text-lg font-semibold">Engineering showcase</h2>
        <div className="grid gap-4 md:grid-cols-2">
          {SKILLS.map((s) => (
            <div key={s.k} className="card flex gap-4 p-5">
              <s.icon className="h-6 w-6 shrink-0 text-accent" aria-hidden />
              <div>
                <h3 className="font-semibold text-ink">{s.k}</h3>
                <p className="mt-1 text-sm text-ink-2">{s.v}</p>
              </div>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
