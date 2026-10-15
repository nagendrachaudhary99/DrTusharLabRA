import { z } from "zod";
import { ragGraph } from "@/lib/rag/graph";
import { getIndex } from "@/lib/rag";
import { llmStatus } from "@/lib/llm";
import { rateLimit } from "@/lib/rate-limit";
import { ndjsonResponse, type AgentEvent } from "@/lib/stream";

export const maxDuration = 60;

const Body = z.object({
  question: z.string().min(2).max(1000),
  history: z
    .array(z.object({ role: z.enum(["user", "assistant"]), content: z.string().max(4000) }))
    .max(12)
    .optional(),
});

export async function POST(req: Request) {
  const limited = rateLimit(req, "rag", { limit: 60, windowMs: 60 * 60_000 });
  if (limited) return limited;
  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: parsed.error.issues }, { status: 400 });

  return ndjsonResponse(async (send) => {
    const index = await getIndex();
    send({ type: "meta", data: { ...llmStatus(), embeddingModel: index.embeddingModel, chunks: index.chunks.length } });
    const stream = await ragGraph.stream(
      { question: parsed.data.question, history: parsed.data.history ?? [] },
      { streamMode: "custom", recursionLimit: 20, signal: req.signal },
    );
    for await (const chunk of stream) send(chunk as AgentEvent);
  });
}
