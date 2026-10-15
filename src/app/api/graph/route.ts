import { trialGraph } from "@/lib/trial/graph";
import { ragGraph } from "@/lib/rag/graph";
import { KNOWLEDGE_BASE } from "@/lib/rag";
import { connection } from "next/server";
import { llmStatus } from "@/lib/llm";

/** Introspects the compiled LangGraph state machines and exports them as Mermaid. */
export async function GET() {
  await connection(); // report the runtime LLM configuration, not the build-time one
  const [trial, rag] = await Promise.all([trialGraph.getGraphAsync(), ragGraph.getGraphAsync()]);
  const categories = KNOWLEDGE_BASE.reduce<Record<string, number>>((acc, d) => ((acc[d.category] = (acc[d.category] ?? 0) + 1), acc), {});
  return Response.json({
    status: llmStatus(),
    trialMermaid: trial.drawMermaid(),
    ragMermaid: rag.drawMermaid(),
    knowledgeBase: { documents: KNOWLEDGE_BASE.length, categories },
  });
}
