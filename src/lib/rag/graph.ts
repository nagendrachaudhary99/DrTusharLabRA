import { Annotation, END, START, StateGraph, type LangGraphRunnableConfig } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { z } from "zod";
import { withLLM } from "@/lib/llm";
import { emitter } from "@/lib/stream";
import { hybridSearch, type RetrievedChunk } from "./index";
import { tokenize } from "./embeddings";

/**
 * Agentic RAG (Corrective-RAG + Self-RAG style) as a LangGraph state machine:
 *
 *   plan_queries ─► retrieve ─► grade_documents ──(enough evidence)──► generate ─► check_grounding ─► END
 *                      ▲               │
 *                      └── rewrite ◄───┘ (too little relevant evidence, max 2 loops)
 */

const CATEGORIES = [
  "virtual-imaging-trials",
  "generative-models",
  "datasets",
  "open-source-tools",
  "clinical-guidelines",
  "evaluation-methodology",
  "ct-physics",
  "regulatory",
] as const;

export const RagState = Annotation.Root({
  question: Annotation<string>,
  history: Annotation<{ role: "user" | "assistant"; content: string }[]>({
    reducer: (_, b) => b,
    default: () => [],
  }),
  queries: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  categories: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  retrieved: Annotation<RetrievedChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  relevant: Annotation<RetrievedChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  attempts: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  answer: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
  grounding: Annotation<{ score: number; verdict: string; unsupported: string[] } | null>({
    reducer: (_, b) => b,
    default: () => null,
  }),
});
type S = typeof RagState.State;

const QueryPlan = z.object({
  queries: z.array(z.string()).min(1).max(3).describe("1-3 search queries: the standalone question plus focused rewrites"),
  categories: z.array(z.enum(CATEGORIES)).describe("Knowledge-base categories most likely to contain the answer; empty = all"),
});

const Grades = z.object({
  grades: z.array(z.object({ chunkId: z.string(), relevant: z.boolean(), reason: z.string() })),
});

const Grounding = z.object({
  score: z.number().min(0).max(1).describe("fraction of claims supported by the context"),
  verdict: z.enum(["grounded", "partially-grounded", "not-grounded"]),
  unsupported: z.array(z.string()).describe("claims not supported by the context"),
});

const context = (chunks: RetrievedChunk[]) => chunks.map((c, i) => `[${i + 1}] ${c.title} — ${c.source}\n${c.text}`).join("\n\n");

async function planQueries(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "plan_queries", status: "start" });
  const { queries, categories } = await withLLM(
    "plan_queries",
    send,
    async (llm) => {
      const hist = state.history
        .slice(-4)
        .map((m) => `${m.role}: ${m.content}`)
        .join("\n");
      const plan = await llm
        .withStructuredOutput(QueryPlan, { name: "query_plan" })
        .invoke(
          `You plan retrieval for a research assistant on virtual imaging trials, generative CT, lung nodule AI evaluation and regulation.\n` +
            `Conversation so far:\n${hist || "(none)"}\n\nUser question: ${state.question}\n` +
            `Rewrite into a standalone query and up to 2 focused sub-queries (different angles / terminology). Choose categories.`,
        );
      return { queries: plan.queries, categories: plan.categories.length >= CATEGORIES.length - 1 ? [] : (plan.categories as string[]) };
    },
    () => {
      // Deterministic multi-query: original + keyword-only variant.
      const kw = tokenize(state.question).join(" ");
      return { queries: kw && kw !== state.question.toLowerCase() ? [state.question, kw] : [state.question], categories: [] as string[] };
    },
  );
  send({ type: "step", node: "plan_queries", status: "done", data: { queries, categories } });
  return { queries, categories };
}

async function retrieve(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "retrieve", status: "start", detail: `attempt ${state.attempts + 1}` });
  // Category routing narrows the first attempt; later attempts search everything.
  const retrieved = await hybridSearch(state.queries, 8, state.attempts === 0 ? state.categories : undefined);
  send({ type: "step", node: "retrieve", status: "done", data: { retrieved } });
  return { retrieved, attempts: 1 };
}

async function gradeDocuments(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "grade_documents", status: "start" });
  const verdicts = await withLLM(
    "grade_documents",
    send,
    async (llm) =>
      (
        await llm
          .withStructuredOutput(Grades, { name: "grade_documents" })
          .invoke(
            `Grade each retrieved chunk for whether it contains information useful to answer the question.\n` +
              `Question: ${state.question}\n\n` +
              state.retrieved.map((c) => `chunkId=${c.chunkId}\n${c.text}`).join("\n---\n"),
          )
      ).grades,
    () => {
      const q = new Set(tokenize(state.question));
      return state.retrieved.map((c) => {
        const uniq = new Set(tokenize(c.text).filter((t) => q.has(t))).size;
        const relevant = uniq >= Math.min(2, q.size) || (c.denseRank === 1 && c.bm25Rank === 1);
        return { chunkId: c.chunkId, relevant, reason: `${uniq} query terms matched` };
      });
    },
  );
  const ok = new Set(verdicts.filter((v) => v.relevant).map((v) => v.chunkId));
  const relevant = state.retrieved.filter((c) => ok.has(c.chunkId)).slice(0, 6);
  send({ type: "step", node: "grade_documents", status: "done", data: { verdicts, kept: relevant.length } });
  return { relevant };
}

async function rewrite(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "rewrite", status: "start" });
  const queries = await withLLM(
    "rewrite",
    send,
    async (llm) => {
      const q = await ChatPromptTemplate.fromTemplate(
        "The search query below retrieved too little relevant evidence from a radiology / medical-AI knowledge base. " +
          "Write ONE broader alternative query using different technical terminology. Return only the query.\n\nQuery: {q}",
      )
        .pipe(llm)
        .pipe(new StringOutputParser())
        .invoke({ q: state.question });
      return [q.trim(), state.question];
    },
    () => [state.question, ...tokenize(state.question).filter((t) => t.length > 4)],
  );
  send({ type: "step", node: "rewrite", status: "done", data: { queries } });
  return { queries };
}

async function generate(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "generate", status: "start" });
  const docs = state.relevant.length ? state.relevant : state.retrieved.slice(0, 3);
  const answer = await withLLM(
    "generate",
    send,
    async (llm) => {
      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          "You are iTRIALSPACE Research Copilot, an expert in virtual imaging trials, CT physics, generative medical imaging and AI evaluation. " +
            "Answer ONLY from the numbered context. Cite every factual sentence with [n]. If the context is insufficient, say what is missing. " +
            "Be concise and technical; use short paragraphs or bullets.",
        ],
        ["human", "Context:\n{context}\n\nQuestion: {question}"],
      ]);
      const stream = await prompt
        .pipe(llm)
        .pipe(new StringOutputParser())
        .stream({ context: context(docs), question: state.question });
      let text = "";
      for await (const tok of stream) {
        text += tok;
        send({ type: "token", node: "generate", text: tok });
      }
      return text;
    },
    () => {
      send({ type: "reset", node: "generate" });
      // Extractive fallback: pick the sentences with the highest query-term overlap and cite them.
      const q = new Set(tokenize(state.question));
      const scored = docs.flatMap((d, i) =>
        d.text
          .split("\n")
          .slice(1)
          .join(" ")
          .split(/(?<=[.!?])\s+/)
          .map((s) => ({ s: s.trim(), i, score: tokenize(s).filter((t) => q.has(t)).length + (docs.length - i) * 0.1 })),
      );
      const top = scored
        .filter((x) => x.s.length > 40)
        .sort((a, b) => b.score - a.score)
        .slice(0, 5)
        .sort((a, b) => a.i - b.i);
      const text =
        `_Demo mode (no LLM configured): extractive answer assembled from the top-ranked evidence._\n\n` + top.map((x) => `- ${x.s} [${x.i + 1}]`).join("\n");
      for (const part of text.match(/[\s\S]{1,24}/g) ?? []) send({ type: "token", node: "generate", text: part });
      return text;
    },
    { temperature: 0.1 },
  );
  send({
    type: "step",
    node: "generate",
    status: "done",
    data: { citations: docs.map((d, i) => ({ n: i + 1, title: d.title, source: d.source, url: d.url, chunkId: d.chunkId })) },
  });
  return { answer, relevant: docs };
}

async function checkGrounding(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "check_grounding", status: "start" });
  const grounding = await withLLM<NonNullable<S["grounding"]>>(
    "check_grounding",
    send,
    (llm) =>
      llm
        .withStructuredOutput(Grounding, { name: "grounding_check" })
        .invoke(
          `You are a strict fact-checker. Compare the ANSWER to the CONTEXT and judge whether every claim is supported.\n\nCONTEXT:\n${context(state.relevant)}\n\nANSWER:\n${state.answer}`,
        ),
    () => {
      const sentences = state.answer.split("\n").filter((l) => l.startsWith("- "));
      const cited = sentences.filter((s) => /\[\d+\]/.test(s)).length;
      const score = sentences.length ? cited / sentences.length : 0;
      return { score, verdict: score > 0.9 ? "grounded" : "partially-grounded", unsupported: [] };
    },
  );
  send({ type: "step", node: "check_grounding", status: "done", data: grounding });
  return { grounding };
}

function routeAfterGrading(state: S) {
  return state.relevant.length < 2 && state.attempts < 2 ? "rewrite" : "generate";
}

export const ragGraph = new StateGraph(RagState)
  .addNode("plan_queries", planQueries)
  .addNode("retrieve", retrieve)
  .addNode("grade_documents", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addNode("generate", generate)
  .addNode("check_grounding", checkGrounding)
  .addEdge(START, "plan_queries")
  .addEdge("plan_queries", "retrieve")
  .addEdge("retrieve", "grade_documents")
  .addConditionalEdges("grade_documents", routeAfterGrading, ["rewrite", "generate"])
  .addEdge("rewrite", "retrieve")
  .addEdge("generate", "check_grounding")
  .addEdge("check_grounding", END)
  .compile();

/**
 * Retrieval-only variant (plan → retrieve → grade → rewrite loop, no answer generation), reused as a
 * subgraph by the trial's Evidence Librarian so trial reports cite graded, self-corrected evidence.
 */
export const ragRetrievalGraph = new StateGraph(RagState)
  .addNode("plan_queries", planQueries)
  .addNode("retrieve", retrieve)
  .addNode("grade_documents", gradeDocuments)
  .addNode("rewrite", rewrite)
  .addEdge(START, "plan_queries")
  .addEdge("plan_queries", "retrieve")
  .addEdge("retrieve", "grade_documents")
  .addConditionalEdges("grade_documents", (s: S) => (routeAfterGrading(s) === "rewrite" ? "rewrite" : END), ["rewrite", END])
  .addEdge("rewrite", "retrieve")
  .compile();
