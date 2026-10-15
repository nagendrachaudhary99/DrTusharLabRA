"use client";

import { useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import { CheckCircle2, Loader2, Send, ShieldAlert } from "lucide-react";
import { AgentGraph, type GraphEdge, type GraphNode, type NodeStatus } from "@/components/agent-graph";
import { Markdown } from "@/components/markdown";
import { streamAgent } from "@/lib/client-stream";
import type { AgentEvent } from "@/lib/stream";

const SUGGESTIONS = [
  "How does iTRIALSPACE build a virtual lesion trial, and how well do synthetic rankings transfer to real data?",
  "How does NodMAISI improve on MAISI-v2 for sub-centimeter nodules?",
  "What is a virtual imaging trial and how did VICTRE use one in place of a clinical trial?",
  "Can NVIDIA MAISI generate lung tumours in a specific lobe? What can't it condition on?",
  "What are the Lung-RADS v2022 size thresholds for solid nodules?",
  "How do reconstruction kernels and dose cause domain shift for deep-learning nodule detectors?",
  "How is FROC and the LUNA16 CPM metric computed?",
  "What does the FDA require for credibility of computational models used as evidence?",
];

const NODES: GraphNode[] = [
  { id: "plan_queries", title: "Query Planner", role: "multi-query + routing", x: 75, y: 50 },
  { id: "retrieve", title: "Hybrid Retriever", role: "dense + BM25 → RRF", x: 235, y: 50 },
  { id: "grade_documents", title: "Relevance Grader", role: "LLM-as-judge", x: 395, y: 50 },
  { id: "rewrite", title: "Query Rewriter", role: "corrective loop", x: 315, y: 170 },
  { id: "generate", title: "Generator", role: "cited, streamed", x: 555, y: 50 },
  { id: "check_grounding", title: "Grounding Check", role: "hallucination guard", x: 555, y: 170 },
];
const EDGES: GraphEdge[] = [
  { from: "plan_queries", to: "retrieve" },
  { from: "retrieve", to: "grade_documents" },
  { from: "grade_documents", to: "generate", kind: "conditional" },
  { from: "grade_documents", to: "rewrite", kind: "conditional" },
  { from: "rewrite", to: "retrieve" },
  { from: "generate", to: "check_grounding" },
];

type Retrieved = {
  chunkId: string;
  docId: string;
  title: string;
  source: string;
  url: string;
  category: string;
  dense: number | null;
  bm25: number | null;
  denseRank: number | null;
  bm25Rank: number | null;
  rrf: number;
};
type Trace = {
  queries?: string[];
  categories?: string[];
  retrieved?: Retrieved[];
  verdicts?: { chunkId: string; relevant: boolean; reason: string }[];
  rewrites?: string[][];
  citations?: { n: number; title: string; source: string; url: string; chunkId: string }[];
  grounding?: { score: number; verdict: string; unsupported: string[] };
  attempts: number;
};
type Turn = {
  q: string;
  answer: string;
  trace: Trace;
  status: Record<string, NodeStatus>;
  done: boolean;
  meta?: { mode: string; model: string; embeddingModel: string; chunks: number };
};

export function CopilotClient() {
  const sp = useSearchParams();
  const [input, setInput] = useState(sp.get("q") ?? "");
  const [turns, setTurns] = useState<Turn[]>([]);
  const [busy, setBusy] = useState(false);
  const [view, setView] = useState(0);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length]);

  async function ask(q: string) {
    if (!q.trim() || busy) return;
    setBusy(true);
    setInput("");
    const idx = turns.length;
    setView(idx);
    const history = turns.flatMap((t) => [
      { role: "user" as const, content: t.q },
      { role: "assistant" as const, content: t.answer.slice(0, 1500) },
    ]);
    setTurns((ts) => [...ts, { q, answer: "", trace: { attempts: 0 }, status: {}, done: false }]);
    const patch = (fn: (t: Turn) => Turn) => setTurns((ts) => ts.map((t, i) => (i === idx ? fn(t) : t)));
    try {
      await streamAgent("/api/rag", { question: q, history }, (e: AgentEvent) => {
        if (e.type === "meta") patch((t) => ({ ...t, meta: e.data as Turn["meta"] }));
        if (e.type === "token") patch((t) => ({ ...t, answer: t.answer + e.text }));
        if (e.type === "reset") patch((t) => ({ ...t, answer: "" }));
        if (e.type === "error") patch((t) => ({ ...t, answer: t.answer + `\n\n⚠ ${e.message}` }));
        if (e.type === "step")
          patch((t) => {
            const status = { ...t.status, [e.node]: (e.status === "start" ? "running" : "done") as NodeStatus };
            const tr = { ...t.trace };
            const d = (e.data ?? {}) as Record<string, unknown>;
            if (e.status === "done") {
              if (e.node === "plan_queries") Object.assign(tr, { queries: d.queries, categories: d.categories });
              if (e.node === "retrieve") Object.assign(tr, { retrieved: d.retrieved, attempts: tr.attempts + 1 });
              if (e.node === "grade_documents") tr.verdicts = d.verdicts as Trace["verdicts"];
              if (e.node === "rewrite") tr.rewrites = [...(tr.rewrites ?? []), d.queries as string[]];
              if (e.node === "generate") tr.citations = d.citations as Trace["citations"];
              if (e.node === "check_grounding") tr.grounding = d as Trace["grounding"];
            }
            return { ...t, status, trace: tr };
          });
      });
    } catch (err) {
      patch((t) => ({ ...t, answer: `⚠ ${(err as Error).message}` }));
    } finally {
      patch((t) => ({ ...t, done: true }));
      setBusy(false);
    }
  }

  const active = turns[view];

  return (
    <div className="mx-auto grid max-w-7xl gap-6 px-4 py-8 lg:grid-cols-[1fr_1fr]">
      <section className="flex min-h-[70vh] flex-col">
        <header className="mb-4 space-y-1">
          <p className="font-mono text-xs uppercase tracking-widest text-accent">Research Copilot · Agentic RAG (LangGraph + LangChain)</p>
          <h1 className="text-2xl font-semibold tracking-tight">Research Copilot</h1>
          <p className="text-sm text-ink-2">
            Ask about virtual imaging trials, generative CT, datasets, evaluation methods or regulation. Answers come only from a curated, source-verified
            knowledge base of 53 documents, with citations. The full retrieval trace is shown on the right.
          </p>
        </header>
        <div className="card flex-1 space-y-5 overflow-auto p-4">
          {!turns.length && (
            <div className="grid gap-2 sm:grid-cols-2">
              {SUGGESTIONS.map((s) => (
                <button
                  key={s}
                  onClick={() => ask(s)}
                  className="rounded-lg border border-line p-3 text-left text-xs text-ink-2 hover:border-accent hover:text-ink"
                >
                  {s}
                </button>
              ))}
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} className={`space-y-2 rounded-lg p-2 ${view === i ? "bg-surface-2/50" : ""}`} onClick={() => setView(i)}>
              <div className="ml-auto w-fit max-w-[85%] rounded-lg bg-accent/15 px-3 py-2 text-sm text-ink">{t.q}</div>
              <div className="text-sm">{t.answer ? <Markdown>{t.answer}</Markdown> : <Loader2 className="h-4 w-4 animate-spin text-accent" />}</div>
              {t.trace.citations && t.done && (
                <ol className="space-y-0.5 border-t border-line pt-2 text-[11px] text-ink-3">
                  {t.trace.citations.map((c) => (
                    <li key={c.chunkId}>
                      [{c.n}]{" "}
                      <a href={c.url} target="_blank" rel="noreferrer" className="text-ink-2 hover:text-accent">
                        {c.title}
                      </a>{" "}
                      — {c.source}
                    </li>
                  ))}
                </ol>
              )}
              {t.trace.grounding && (
                <div className="flex items-center gap-1.5 text-[11px]">
                  {t.trace.grounding.verdict === "grounded" ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-[var(--status-good)]" aria-hidden />
                  ) : (
                    <ShieldAlert className="h-3.5 w-3.5 text-[var(--status-warning)]" aria-hidden />
                  )}
                  <span className="text-ink-2">
                    {t.trace.grounding.verdict} · support score {(t.trace.grounding.score * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
          ))}
          <div ref={bottom} />
        </div>
        <form
          className="mt-3 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask a research question…"
            className="flex-1 rounded-lg border border-line bg-surface-1 px-3 py-2 text-sm outline-none focus:border-accent"
          />
          <button disabled={busy} className="flex items-center gap-1.5 rounded-lg bg-accent px-4 py-2 text-sm font-medium text-white disabled:opacity-50">
            {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />} Ask
          </button>
        </form>
      </section>

      <aside className="space-y-4">
        <div className="card p-4">
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-sm font-medium">RAG graph</h2>
            {active?.meta && (
              <span className="font-mono text-[10px] text-ink-3">
                {active.meta.mode === "live" ? active.meta.model : "demo mode"} · {active.meta.embeddingModel} · {active.meta.chunks} chunks
              </span>
            )}
          </div>
          <AgentGraph nodes={NODES} edges={EDGES} status={active?.status ?? {}} width={640} height={210} nodeWidth={140} nodeHeight={52} fontScale={1.1} />
        </div>
        {active ? (
          <TracePanel trace={active.trace} />
        ) : (
          <div className="card p-4 text-xs text-ink-3">The retrieval trace for the selected answer appears here.</div>
        )}
      </aside>
    </div>
  );
}

function TracePanel({ trace }: { trace: Trace }) {
  const verdict = new Map(trace.verdicts?.map((v) => [v.chunkId, v]));
  return (
    <div className="card space-y-4 p-4 text-xs">
      {trace.queries && (
        <div>
          <h3 className="mb-1 font-medium text-ink">1 · Query plan</h3>
          <ul className="space-y-1">
            {trace.queries.map((q, i) => (
              <li key={i} className="rounded bg-surface-2 px-2 py-1 font-mono text-[11px] text-ink-2">
                {q}
              </li>
            ))}
          </ul>
          {!!trace.categories?.length && <p className="mt-1 text-ink-3">Routed to: {trace.categories.join(", ")}</p>}
          {trace.rewrites?.map((r, i) => (
            <p key={i} className="mt-1 text-[var(--status-warning)]">
              ↻ Corrective rewrite: {r[0]}
            </p>
          ))}
        </div>
      )}
      {trace.retrieved && (
        <div>
          <h3 className="mb-1 font-medium text-ink">
            2 · Hybrid retrieval & grading <span className="font-normal text-ink-3">(attempt {trace.attempts})</span>
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-left">
              <thead className="text-ink-3">
                <tr>
                  <th className="py-1 font-normal">Chunk</th>
                  <th className="font-normal">Dense</th>
                  <th className="font-normal">BM25</th>
                  <th className="font-normal">RRF</th>
                  <th className="font-normal">Grade</th>
                </tr>
              </thead>
              <tbody>
                {trace.retrieved.map((r) => {
                  const v = verdict.get(r.chunkId);
                  return (
                    <tr key={r.chunkId} className="border-t border-line align-top">
                      <td className="max-w-[220px] py-1 pr-2">
                        <div className="truncate text-ink-2" title={r.title}>
                          {r.title}
                        </div>
                        <div className="text-[10px] text-ink-3">{r.category}</div>
                      </td>
                      <td className="font-mono text-ink-2">{r.dense !== null ? `${r.dense.toFixed(2)} #${r.denseRank}` : "–"}</td>
                      <td className="font-mono text-ink-2">{r.bm25 !== null ? `${r.bm25.toFixed(1)} #${r.bm25Rank}` : "–"}</td>
                      <td className="font-mono text-ink">{r.rrf.toFixed(4)}</td>
                      <td title={v?.reason}>
                        {v ? v.relevant ? <span className="text-[var(--status-good)]">✓ keep</span> : <span className="text-ink-3">✗ drop</span> : "…"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
      {trace.grounding && (
        <div>
          <h3 className="mb-1 font-medium text-ink">3 · Grounding check</h3>
          <p className="text-ink-2">
            Verdict <b>{trace.grounding.verdict}</b>, {(trace.grounding.score * 100).toFixed(0)}% of claims supported.
          </p>
          {trace.grounding.unsupported.length > 0 && (
            <ul className="mt-1 list-disc pl-4 text-[var(--status-warning)]">
              {trace.grounding.unsupported.map((u, i) => (
                <li key={i}>{u}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
