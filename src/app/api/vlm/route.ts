import { z } from "zod";
import { vlmGraph } from "@/lib/vlm/graph";
import { MAX_CASES, VlmCase } from "@/lib/vlm/schema";
import { llmStatus } from "@/lib/llm";
import { rateLimit } from "@/lib/rate-limit";
import { ndjsonResponse, type AgentEvent } from "@/lib/stream";

export const maxDuration = 120;

// 48 × ~400 kB worst case, plus JSON overhead.
const MAX_BODY_BYTES = 20 * 1024 * 1024;

const Body = z.object({
  cases: z
    .array(VlmCase)
    .min(1)
    .max(MAX_CASES)
    .refine((cs) => new Set(cs.map((c) => c.id)).size === cs.length, "case ids must be unique"),
});

export async function POST(req: Request) {
  const limited = rateLimit(req, "vlm", { limit: 10, windowMs: 60 * 60_000 });
  if (limited) return limited;
  if (Number(req.headers.get("content-length") ?? 0) > MAX_BODY_BYTES) return Response.json({ error: "body too large" }, { status: 413 });
  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: parsed.error.issues.slice(0, 5) }, { status: 400 });
  const { cases } = parsed.data;

  return ndjsonResponse(async (send) => {
    const status = llmStatus();
    send({ type: "meta", data: { ...status, reader: status.mode === "live" ? `live VLM: ${status.model}` : "simulated reader (demo mode)" } });
    const final: Record<string, unknown> = {};
    const stream = await vlmGraph.stream({ cases }, { streamMode: ["custom", "updates"], recursionLimit: 20, signal: req.signal });
    for await (const [mode, chunk] of stream) {
      if (mode === "custom") send(chunk as AgentEvent);
      else
        for (const [node, update] of Object.entries(chunk as Record<string, Record<string, unknown> | null>)) {
          // vlm_reader updates are already streamed per case; keep only the reduced outputs.
          if (node !== "vlm_reader" && update) Object.assign(final, update);
        }
    }
    send({ type: "result", data: { metrics: final.metrics, report: final.report, model: status.model } });
  });
}
