import { Annotation, END, START, Send, StateGraph, type LangGraphRunnableConfig } from "@langchain/langgraph";
import { HumanMessage } from "@langchain/core/messages";
import { withLLM } from "@/lib/llm";
import { emitter } from "@/lib/stream";
import { gaussian, mulberry32 } from "@/lib/trial/physics";
import { LOBES } from "@/lib/trial/spec";
import { computeMetrics, TASKS, type VlmMetrics } from "./metrics";
import { READER_PROMPT, VlmAnswer, armKey, scoreAnswer, type CaseResult, type VlmCase } from "./schema";

/**
 * VLM benchmark on synthetic CT, orchestrated with LangGraph.
 *
 *   plan ──Send(case₁)──► vlm_reader ─┐
 *        ──Send(case₂)──► vlm_reader ─┼─► score ─► analyst ─► END       (map-reduce via the Send API)
 *        ──Send(caseₙ)──► vlm_reader ─┘
 *
 * Every vlm_reader branch makes ONE multimodal structured-output call answering all three tasks
 * (detection, lobe localization, diameter). Branches run in the same super-step; a semaphore caps
 * concurrent model calls. Results are merged by a concat reducer, scored deterministically, and an
 * analyst LLM summarises failure modes using only the computed numbers.
 */

export const VlmState = Annotation.Root({
  cases: Annotation<VlmCase[]>({ reducer: (_, b) => b, default: () => [] }),
  /** Payload of a single Send() branch. */
  item: Annotation<VlmCase | null>({ reducer: (_, b) => b, default: () => null }),
  results: Annotation<CaseResult[]>({ reducer: (a, b) => a.concat(b), default: () => [] }),
  metrics: Annotation<VlmMetrics | null>({ reducer: (_, b) => b, default: () => null }),
  report: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
});
type S = typeof VlmState.State;

/* Simple counting semaphore so a 48-way fan-out doesn't open 48 concurrent model requests. */
const MAX_CONCURRENT = 6;
let active = 0;
const waiters: (() => void)[] = [];
async function withSlot<T>(fn: () => Promise<T>): Promise<T> {
  if (active >= MAX_CONCURRENT) await new Promise<void>((r) => waiters.push(r));
  active++;
  try {
    return await fn();
  } finally {
    active--;
    waiters.shift()?.();
  }
}

/* 1 ─ Plan: describe the design, then fan out one Send per case. */
async function plan(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "plan", status: "start" });
  const pos = state.cases.filter((c) => c.groundTruth.hasNodule).length;
  const arms = [...new Set(state.cases.map((c) => armKey(c.groundTruth)))];
  send({
    type: "step",
    node: "plan",
    status: "done",
    detail: `${state.cases.length} cases (${pos} with nodule, ${state.cases.length - pos} negative controls) · ${arms.length} scanner arm(s)`,
    data: { prompt: READER_PROMPT, nCases: state.cases.length, positives: pos, arms, concurrency: MAX_CONCURRENT },
  });
  return {};
}

const fanOut = (state: S) => state.cases.map((c) => new Send("vlm_reader", { item: c }));

/**
 * Deterministic "simulated reader" used in demo mode or when a live call fails. Its detection
 * probability is a logistic function of diameter whose 50% point moves up with dose reduction
 * and for ground-glass nodules — a plausible-looking reader, clearly labelled as simulated.
 */
export function simulatedReader(c: VlmCase): VlmAnswer {
  const g = c.groundTruth;
  const rand = mulberry32([...c.id].reduce((h, ch) => (h * 31 + ch.charCodeAt(0)) >>> 0, 7));
  const doseShift = { standard: 0, low: 1.5, "ultra-low": 3.5 }[g.dose];
  if (!g.hasNodule) {
    const fp = rand() < 0.1 + doseShift * 0.03;
    return {
      nodulePresent: fp,
      confidence: +(fp ? 0.55 + rand() * 0.2 : 0.6 + rand() * 0.3).toFixed(2),
      lobe: fp ? LOBES[Math.floor(rand() * LOBES.length)] : "none",
      diameterMm: fp ? +(3 + rand() * 4).toFixed(1) : 0,
      rationale: fp ? "Simulated reader: false-positive call on a vessel cross-section." : "Simulated reader: no focal opacity.",
    };
  }
  const d50 = 5 + doseShift + (g.type === "ground-glass" ? 2.5 : g.type === "part-solid" ? 1 : 0);
  const pDetect = 1 / (1 + Math.exp(-(g.diameterMm - d50) / 1.5));
  const hit = rand() < pDetect;
  if (!hit) return { nodulePresent: false, confidence: +(0.5 + rand() * 0.3).toFixed(2), lobe: "none", diameterMm: 0, rationale: "Simulated reader: missed." };
  // Localization: mostly right; typical confusions are within the same lung (upper ↔ lower / middle).
  const sameLung = g.lobe.startsWith("R") ? (["RUL", "RML", "RLL"] as const) : (["LUL", "LLL"] as const);
  const lobe = rand() < 0.78 ? g.lobe : sameLung[Math.floor(rand() * sameLung.length)];
  return {
    nodulePresent: true,
    confidence: +Math.min(0.99, 0.55 + pDetect * 0.4).toFixed(2),
    lobe: lobe === "none" ? "RUL" : lobe,
    diameterMm: +Math.max(1, g.diameterMm * (1.1 + 0.25 * gaussian(rand))).toFixed(1),
    rationale: `Simulated reader: ${g.type} opacity in ${lobe}.`,
  };
}

/* 2 ─ VLM reader: one structured multimodal call per image (runs once per Send branch). */
async function vlmReader(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  const c = state.item!;
  send({ type: "step", node: "vlm_reader", status: "start", detail: c.id, data: { caseId: c.id } });
  const t0 = Date.now();
  let source: CaseResult["source"] = "live";
  const answer = await withLLM(
    "vlm_reader",
    send,
    (llm) =>
      withSlot(() =>
        llm.withStructuredOutput(VlmAnswer, { name: "ct_slice_read" }).invoke([
          new HumanMessage({
            content: [
              { type: "text", text: READER_PROMPT },
              { type: "image_url", image_url: { url: c.imageDataUrl, detail: "high" } },
            ],
          }),
        ]),
      ),
    () => {
      source = "simulated";
      return simulatedReader(c);
    },
  );
  // Models occasionally return a lobe while saying "no nodule" (or vice versa); keep the raw answer, score it as given.
  const result: CaseResult = { caseId: c.id, answer, source, latencyMs: Date.now() - t0, ...scoreAnswer(c.groundTruth, answer) };
  send({ type: "step", node: "vlm_reader", status: "done", detail: c.id, data: { caseId: c.id, result } });
  return { results: [result] };
}

/* 3 ─ Score: deterministic metrics with Wilson / bootstrap CIs. */
async function score(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "score", status: "start" });
  const metrics = computeMetrics(state.cases, state.results);
  send({
    type: "step",
    node: "score",
    status: "done",
    detail: TASKS.map((t) => `${t.short} ${metrics.overall[t.id] ? (metrics.overall[t.id]!.rate * 100).toFixed(0) + "%" : "–"}`).join(" · "),
    data: { metrics },
  });
  return { metrics };
}

/* 4 ─ Analyst: short markdown summary of where the reader fails (numbers only from `metrics`). */
async function analyst(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "analyst", status: "start" });
  const m = state.metrics!;
  const pct = (x: number) => `${(x * 100).toFixed(0)}%`;
  const ci = (r: { rate: number; lo: number; hi: number; k: number; n: number } | null) =>
    r ? `${pct(r.rate)} (${r.k}/${r.n}, 95% CI ${pct(r.lo)}–${pct(r.hi)})` : "n/a";
  const facts = {
    cases: m.n,
    positives: m.nPositive,
    negatives: m.nNegative,
    reader: m.sources.simulated === m.n ? "simulated reader (demo mode)" : m.sources.simulated ? `live VLM with ${m.sources.simulated} simulated fallbacks` : "live VLM",
    detectionAccuracy: ci(m.overall.detection),
    sensitivity: ci(m.sensitivity),
    specificity: ci(m.specificity),
    detectionAUC: m.auc?.toFixed(2) ?? "n/a",
    localizationAccuracy: ci(m.overall.localization),
    sizeAccuracy: ci(m.overall.size),
    sizeMAEmm: m.size ? `${m.size.mae.toFixed(1)} mm (bootstrap 95% CI ${m.size.lo.toFixed(1)}–${m.size.hi.toFixed(1)}), bias ${m.size.bias.toFixed(1)} mm` : "n/a",
    detectionBySize: m.bySize.map((r) => `${r.label}: ${ci(r.detection)}`),
    detectionByScanner: m.byArm.map((r) => `${r.label}: ${ci(r.detection)}`),
    detectionByType: m.byType.map((r) => `${r.label}: ${ci(r.detection)}`),
    confusion: m.confusion,
    weakSpots: m.weakSpots.map((w) => `${w.task} in ${w.factor}=${w.stratum}: ${pct(w.rate)} (n=${w.n}), ${w.gapPp} pp below overall`),
  };
  const fallback = () => {
    const lines = [
      `**Reader:** ${facts.reader} · ${m.n} cases (${m.nPositive} with a nodule, ${m.nNegative} negative controls).`,
      `- **Detection** accuracy ${facts.detectionAccuracy}; sensitivity ${facts.sensitivity}, specificity ${facts.specificity}, AUC ${facts.detectionAUC}.`,
      `- **Localization** (lobe) accuracy on positives ${facts.localizationAccuracy}.`,
      `- **Size** within tolerance ${facts.sizeAccuracy}; MAE ${facts.sizeMAEmm}.`,
      m.weakSpots.length ? `\n**Weak spots**\n${facts.weakSpots.map((w) => `- ${w}`).join("\n")}` : "\nNo stratum fell ≥15 pp below the overall accuracy (with n ≥ 3).",
      `\n_Template summary (no LLM). Small strata have wide CIs — treat subgroup gaps as hypotheses._`,
    ].join("\n");
    send({ type: "reset", node: "analyst" });
    send({ type: "token", node: "analyst", text: lines });
    return lines;
  };
  const report = await withLLM(
    "analyst",
    send,
    async (llm) => {
      send({ type: "reset", node: "analyst" });
      const stream = await llm.stream(
        "You are the biostatistics analyst for a benchmark of a vision-language model reading synthetic chest CT slices " +
          "(three tasks: nodule detection, lobe localization, diameter estimation). Write a concise markdown summary (≤180 words) with: " +
          "one headline sentence, a bullet list of where the reader fails (size, nodule type, scanner/dose, lobe confusions, size bias), " +
          "and one bullet on statistical caveats (CI width / small n). Use ONLY numbers present in METRICS — do not invent any. " +
          "No headings larger than ###.\n\nMETRICS:\n" +
          JSON.stringify(facts, null, 1),
      );
      let text = "";
      for await (const chunk of stream) {
        const t = typeof chunk.content === "string" ? chunk.content : "";
        if (!t) continue;
        text += t;
        send({ type: "token", node: "analyst", text: t });
      }
      if (!text.trim()) throw new Error("empty analyst response");
      return text;
    },
    fallback,
    { temperature: 0.2 },
  );
  send({ type: "step", node: "analyst", status: "done" });
  return { report };
}

export const vlmGraph = new StateGraph(VlmState)
  .addNode("plan", plan)
  .addNode("vlm_reader", vlmReader)
  .addNode("score", score)
  .addNode("analyst", analyst)
  .addEdge(START, "plan")
  .addConditionalEdges("plan", fanOut, ["vlm_reader"])
  .addEdge("vlm_reader", "score")
  .addEdge("score", "analyst")
  .addEdge("analyst", END)
  .compile();
