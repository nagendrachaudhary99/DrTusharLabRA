import type { Metadata } from "next";
import { readFile } from "node:fs/promises";
import path from "node:path";
import { trialGraph } from "@/lib/trial/graph";
import { ragGraph, ragRetrievalGraph } from "@/lib/rag/graph";
import { vlmGraph } from "@/lib/vlm/graph";
import { KNOWLEDGE_BASE } from "@/lib/rag";
import { Markdown } from "@/components/markdown";

export const metadata: Metadata = { title: "Architecture · iTRIALSPACE" };

const STACK = [
  ["Frontend", "Next.js 16 App Router, React 19, Tailwind CSS 4, NiiVue (WebGL2) 3D volume viewer, custom SVG charts & canvas CT renderer"],
  ["Orchestration", "LangGraph.js — trial graph (interrupt, Send, subgraph, cycle), corrective-RAG graph, VLM map-reduce graph; LangChain createAgent planner"],
  ["Human in the loop", "interrupt() + MemorySaver checkpointer + Command({ resume }); stateless replay fallback across serverless instances"],
  ["LLM layer", "LangChain.js ChatOpenAI → Vercel AI Gateway (any provider) or OpenAI; Zod structured output incl. multimodal image input"],
  ["Retrieval", "RecursiveCharacterTextSplitter · custom VectorStore (cosine) · BM25 · Reciprocal Rank Fusion · LLM grading · grounding judge"],
  ["Embeddings", "text-embedding-3-small via gateway, or offline 768-d feature-hashing embeddings"],
  ["Simulation", "Seeded cohort generator, Rose-model detectability, vendor/kernel/dose noise model, FROC/CPM, Wilson & bootstrap CIs"],
  ["Python", "itrialspace-eval package: cluster bootstrap, FROC/CPM, failure modes, synthetic-vs-real rank agreement; pytest, ruff, mypy, Docker"],
  ["Transport", "NDJSON streaming from Route Handlers (Node.js runtime, Fluid Compute) — node events + LLM tokens; per-IP rate limits"],
  ["Deploy", "Vercel; zero-config demo mode, live mode with AI_GATEWAY_API_KEY or OPENAI_API_KEY; optional LangSmith tracing"],
];

const ROADMAP = [
  [
    "NodMAISI synthesis",
    "Swap the procedural phantom for NodMAISI / MAISI-v2 (ControlNet-conditioned rectified flow) running in a GPU worker: the Studio's label map + nodule mask is already the conditioning input; the trial graph would call it per case.",
  ],
  [
    "Real nodule index",
    "Replace the synthetic index with real nodule profiles (LIDC-IDRI, LUNA16/25, LNDb, DLCS24 metadata) and donor matching into real host CTs with TotalSegmentator / VISTA3D lobe masks.",
  ],
  [
    "Medical VLMs under test",
    "Point the VLM Benchmark reader at MedGemma, LLaVA-Med or BiomedCLIP endpoints (same Zod answer schema), and run real detectors (MONAI, Tri-Reader) in the scanner-arm workers.",
  ],
  ["Durable runs", "PostgresSaver checkpointer (Neon) for multi-day, resumable trials and time-travel what-if re-runs; LangSmith datasets for regression-testing the agents."],
  ["Credibility", "Synthetic-vs-real agreement checks (Spearman ρ, as in the iTRIALSPACE paper) in CI, documented per FDA CM&S credibility framework and AAPM TG-387."],
];

async function getArchitectureData() {
  "use cache";
  const [trial, rag, retrieval, vlm] = await Promise.all([
    trialGraph.getGraphAsync(),
    ragGraph.getGraphAsync(),
    ragRetrievalGraph.getGraphAsync(),
    vlmGraph.getGraphAsync(),
  ]);
  const notes = await readFile(path.join(process.cwd(), "src/data/research-notes.md"), "utf8");
  return {
    trialMermaid: trial.drawMermaid(),
    ragMermaid: rag.drawMermaid(),
    retrievalMermaid: retrieval.drawMermaid(),
    vlmMermaid: vlm.drawMermaid(),
    notes,
  };
}

export default async function ArchitecturePage() {
  const { trialMermaid, ragMermaid, retrievalMermaid, vlmMermaid, notes } = await getArchitectureData();
  const categories = KNOWLEDGE_BASE.reduce<Record<string, number>>((acc, d) => ((acc[d.category] = (acc[d.category] ?? 0) + 1), acc), {});

  return (
    <div className="mx-auto max-w-7xl space-y-8 px-4 py-8">
      <header className="space-y-1">
        <p className="font-mono text-xs uppercase tracking-widest text-accent">System design</p>
        <h1 className="text-2xl font-semibold tracking-tight">Architecture</h1>
      </header>

      <section className="card overflow-x-auto p-5">
        <h2 className="mb-4 text-sm font-medium">System overview</h2>
        <div className="grid min-w-[900px] grid-cols-[1fr_auto_1.1fr_auto_1.6fr_auto_1fr] items-stretch gap-3 text-xs">
          <Tier
            title="Browser · React 19"
            items={[
              "Cohort Planner — faceted index, NoduleMap, agent tool trace",
              "Trial Designer — live agent graph, approval panel, forest plots, FROC",
              "Patient Studio — lesion insertion, imaging chain, NiiVue 3D (client-side)",
              "VLM Benchmark — slices rendered in-browser, live answers grid",
              "Research Copilot — chat + retrieval trace",
            ]}
          />
          <Arrow label="POST JSON" back="NDJSON stream" />
          <Tier
            title="Route Handlers · Node.js (Fluid Compute)"
            items={[
              "/api/cohort → createAgent(...).stream()",
              "/api/trial → trialGraph.stream() / Command(resume)",
              "/api/vlm → vlmGraph.stream()",
              "/api/rag → ragGraph.stream()",
              "/api/graph → getGraphAsync().drawMermaid()",
            ]}
          />
          <Arrow label="invoke" back="custom events + tokens" />
          <Tier
            title="LangGraph state machines"
            items={[
              "cohort planner: createAgent ⇄ tools (query_index · stratify · build_cohort · lung_rads)",
              "trialGraph: architect ⇄ reviewer → human_approval (interrupt) → (librarian[RAG subgraph] ∥ cohort → Send×arms → biostatistician) → report",
              "vlmGraph: plan → Send×cases → vlm_reader (multimodal) → score → analyst",
              "ragGraph: plan → retrieve → grade → (rewrite ↺ | generate) → grounding check",
              "Hybrid index: TextSplitter → VectorStore (cosine) + BM25 → RRF",
            ]}
            accent
          />
          <Arrow label="OpenAI-compatible" back="structured output / stream" />
          <Tier
            title="Model providers (optional)"
            items={["Vercel AI Gateway → OpenAI · Anthropic · Google …", "or OpenAI directly", "none → deterministic fallbacks (demo mode)"]}
          />
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <div className="card p-5">
          <h2 className="mb-3 text-sm font-medium">Technology stack</h2>
          <dl className="space-y-2 text-sm">
            {STACK.map(([k, v]) => (
              <div key={k} className="grid grid-cols-[120px_1fr] gap-3">
                <dt className="text-ink-3">{k}</dt>
                <dd className="text-ink-2">{v}</dd>
              </div>
            ))}
          </dl>
        </div>
        <div className="card p-5">
          <h2 className="mb-3 text-sm font-medium">Knowledge base ({KNOWLEDGE_BASE.length} source-verified documents)</h2>
          <table className="w-full text-sm">
            <tbody>
              {Object.entries(categories).map(([c, n]) => (
                <tr key={c} className="border-t border-line">
                  <td className="py-1.5 text-ink-2">{c}</td>
                  <td className="py-1.5 text-right font-mono text-ink">{n}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-3 text-xs text-ink-3">
            Curated by a research-analyst agent: each entry was checked against primary sources (arXiv, journals, TCIA, FDA, GitHub). Unverifiable details were
            omitted.
          </p>
        </div>
      </section>

      <section className="grid gap-4 lg:grid-cols-2">
        <MermaidCard
          title="trialGraph — introspected from the compiled LangGraph"
          note={
            <>
              <code>trialGraph.getGraphAsync().drawMermaid()</code>. human_approval routes with <code>Command(&#123; goto &#125;)</code>; cohort_synthesizer fans out
              with <code>Send</code>. Live JSON at <code>/api/graph</code>.
            </>
          }
          mermaid={trialMermaid}
        />
        <MermaidCard
          title="ragGraph — Corrective / Self-RAG"
          note="Conditional edge loops back to rewrite when fewer than 2 relevant chunks survive grading (max 2 attempts)."
          mermaid={ragMermaid}
        />
        <MermaidCard
          title="ragRetrievalGraph — subgraph used by the Evidence Librarian"
          note="Same nodes as the Copilot graph minus generation; invoked inside the trial with the parent config so its steps stream as nested events."
          mermaid={retrievalMermaid}
        />
        <MermaidCard
          title="vlmGraph — map-reduce VLM evaluation"
          note="One Send per synthetic slice to a multimodal structured-output reader (semaphore-limited), then deterministic scoring and an LLM analyst."
          mermaid={vlmMermaid}
        />
      </section>

      <section className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
        <div className="card p-5">
          <h2 className="mb-2 text-sm font-medium">Research notes: landscape & open-source components</h2>
          <Markdown>{notes.replace(/^# .*\n/, "")}</Markdown>
        </div>
        <div className="card p-5">
          <h2 className="mb-3 text-sm font-medium">Roadmap from prototype to research platform</h2>
          <ol className="space-y-3 text-sm">
            {ROADMAP.map(([k, v], i) => (
              <li key={k} className="flex gap-3">
                <span className="font-mono text-xs text-accent">0{i + 1}</span>
                <div>
                  <div className="font-medium text-ink">{k}</div>
                  <div className="text-xs text-ink-2">{v}</div>
                </div>
              </li>
            ))}
          </ol>
          <div className="mt-5 rounded-lg border border-line p-3 text-xs text-ink-3">
            <b className="text-ink-2">Honest limits.</b> The phantom, physics, detectors and nodule index are simplified or synthetic, built to demonstrate the
            workflow. They are not calibrated to specific scanners and make no regulatory-grade realism claims. The VLM Benchmark calls a real general-purpose
            model, not a medical VLM.
          </div>
        </div>
      </section>
    </div>
  );
}

function MermaidCard({ title, note, mermaid }: { title: string; note: React.ReactNode; mermaid: string }) {
  return (
    <div className="card p-5">
      <h2 className="mb-1 text-sm font-medium">{title}</h2>
      <p className="mb-3 text-xs text-ink-3">{note}</p>
      <pre className="max-h-96 overflow-auto rounded-lg bg-background p-3 font-mono text-[11px] text-ink-2">{mermaid}</pre>
    </div>
  );
}

function Tier({ title, items, accent }: { title: string; items: string[]; accent?: boolean }) {
  return (
    <div className={`rounded-lg border p-3 ${accent ? "border-accent/60 bg-accent/5" : "border-line bg-background"}`}>
      <div className="mb-2 font-medium text-ink">{title}</div>
      <ul className="space-y-1.5 text-ink-2">
        {items.map((i) => (
          <li key={i} className="leading-snug">
            {i}
          </li>
        ))}
      </ul>
    </div>
  );
}

function Arrow({ label, back }: { label: string; back: string }) {
  return (
    <div className="flex flex-col items-center justify-center gap-2 text-[10px] text-ink-3">
      <span>{label}</span>
      <span className="text-base leading-none text-accent">⇄</span>
      <span>{back}</span>
    </div>
  );
}
