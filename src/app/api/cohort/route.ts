import { z } from "zod";
import { runCohortPlanner } from "@/lib/cohort/agent";
import { llmStatus } from "@/lib/llm";
import { rateLimit } from "@/lib/rate-limit";
import { ndjsonResponse } from "@/lib/stream";

export const maxDuration = 120;

const Body = z.object({ request: z.string().min(3).max(2000) });

export async function POST(req: Request) {
  const limited = rateLimit(req, "cohort", { limit: 30, windowMs: 60 * 60_000 });
  if (limited) return limited;
  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: parsed.error.issues }, { status: 400 });
  return ndjsonResponse(async (send) => {
    send({ type: "meta", data: llmStatus() });
    const result = await runCohortPlanner(parsed.data.request, send, { signal: req.signal });
    send({ type: "result", data: result });
  });
}
