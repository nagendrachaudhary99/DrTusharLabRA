import { z } from "zod";
import { Command } from "@langchain/langgraph";
import { checkpointer, trialGraph, type ApprovalDecision } from "@/lib/trial/graph";
import { TrialSpec } from "@/lib/trial/spec";
import type { Cohort } from "@/lib/trial/simulate";
import type { RetrievedChunk } from "@/lib/rag";
import { llmStatus } from "@/lib/llm";
import { rateLimit } from "@/lib/rate-limit";
import { ndjsonResponse, type AgentEvent, type Writer } from "@/lib/stream";

export const maxDuration = 120;

const Start = z.object({
  request: z.string().min(3).max(2000),
  overrides: TrialSpec.partial().optional(),
  humanApproval: z.boolean().optional(),
});

const Resume = z.object({
  threadId: z.string().uuid(),
  request: z.string().min(3).max(2000),
  decision: z.union([
    z.object({ action: z.literal("approve"), spec: TrialSpec }),
    z.object({ action: z.literal("revise"), feedback: z.string().min(2).max(1000) }),
  ]),
});

const Body = z.union([Resume, Start]);

export async function POST(req: Request) {
  const limited = rateLimit(req, "trial", { limit: 30, windowMs: 60 * 60_000 });
  if (limited) return limited;
  const parsed = Body.safeParse(await req.json().catch(() => null));
  if (!parsed.success) return Response.json({ error: parsed.error.issues }, { status: 400 });
  const body = parsed.data;

  return ndjsonResponse(async (send) => {
    const status = llmStatus();
    send({ type: "meta", data: status });
    // Demo mode finishes in ~100 ms; pace node-start events so the orchestration is visible in the UI.
    const pace = status.mode === "demo" ? 250 : 0;

    let threadId: string;
    let input: Parameters<typeof trialGraph.stream>[0];
    if ("threadId" in body) {
      const paused = await trialGraph.getState({ configurable: { thread_id: body.threadId } }).catch(() => null);
      if (paused?.next.includes("human_approval")) {
        // Same instance still holds the checkpoint: resume the interrupted node.
        threadId = body.threadId;
        input = new Command({ resume: body.decision satisfies ApprovalDecision });
        send({ type: "log", node: "human_approval", message: `resuming thread ${threadId.slice(0, 8)} from checkpoint` });
      } else if (body.decision.action === "approve") {
        // Checkpoint lives on another instance (or expired): replay the approved protocol on a fresh thread.
        threadId = crypto.randomUUID();
        input = { request: body.request, spec: body.decision.spec, preApproved: true, approval: { by: "human", edited: true } };
        send({ type: "log", node: "human_approval", message: "checkpoint not on this instance; replaying approved protocol on a new thread" });
      } else {
        threadId = crypto.randomUUID();
        input = { request: `${body.request}\nResearcher feedback: ${body.decision.feedback}`, requireApproval: true };
      }
    } else {
      threadId = crypto.randomUUID();
      input = { request: body.request, overrides: body.overrides ?? {}, requireApproval: !!body.humanApproval };
    }

    const stream = await trialGraph.stream(input, {
      configurable: { thread_id: threadId },
      streamMode: ["custom", "updates"],
      recursionLimit: 40,
      signal: req.signal,
      runName: "itrialspace-trial",
      tags: ["trial"],
      metadata: { thread_id: threadId },
    });
    for await (const [mode, chunk] of stream) {
      if (mode === "custom") {
        const e = chunk as AgentEvent;
        if (pace && e.type === "step" && e.status === "start") await new Promise((r) => setTimeout(r, pace));
        send(e);
      } else if ("__interrupt__" in (chunk as object)) {
        const [first] = (chunk as { __interrupt__: { value: unknown }[] }).__interrupt__;
        send({ type: "interrupt", data: { threadId, ...(first.value as object) } });
        expireLater(threadId);
        return;
      }
    }
    await sendResult(send, threadId);
  });
}

async function sendResult(send: Writer, threadId: string) {
  const { values: v } = await trialGraph.getState({ configurable: { thread_id: threadId } });
  const cohort = v.cohort as Cohort;
  send({
    type: "result",
    data: {
      threadId,
      approval: v.approval,
      spec: v.spec,
      analysis: v.analysis,
      report: v.report,
      evidence: (v.evidence as RetrievedChunk[]).map(({ text, ...e }) => ({ ...e, snippet: text.split("\n").slice(1).join(" ").slice(0, 300) })),
      lesions: cohort.lesions.map(({ pos, ...l }) => ({ ...l, pos: pos.map((x) => +x.toFixed(3)) })),
      patients: cohort.patients.map((p) => ({ ...p, fpCandidates: p.falsePositiveScores?.length ?? 0 })),
    },
  });
  // Completed runs don't need their checkpoints; keep the in-memory saver small.
  await checkpointer.deleteThread(threadId);
}

/** Paused threads are dropped after 30 minutes if never resumed. */
function expireLater(threadId: string) {
  setTimeout(() => void checkpointer.deleteThread(threadId), 30 * 60_000).unref?.();
}
