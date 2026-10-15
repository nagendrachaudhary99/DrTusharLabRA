import { Annotation, Command, END, MemorySaver, START, Send, StateGraph, interrupt, type LangGraphRunnableConfig } from "@langchain/langgraph";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { withLLM } from "@/lib/llm";
import { hybridSearch, type RetrievedChunk } from "@/lib/rag";
import { ragRetrievalGraph } from "@/lib/rag/graph";
import { emitter } from "@/lib/stream";
import { heuristicParse } from "./parse";
import { DEFAULT_SPEC, MODEL_PROFILE_INFO, TrialSpec } from "./spec";
import { analyze, armLabel, simulateArm, synthesizeCohort, type Cohort, type TrialAnalysis } from "./simulate";

/**
 * iTRIALSPACE multi-agent virtual imaging trial, orchestrated with LangGraph.
 *
 *  protocol_architect ─► protocol_reviewer ──(issues & revisions<2)──► protocol_architect      (cycle)
 *         ▲                     │ approved
 *         └── revise ── human_approval  ◄── interrupt(): researcher approves / edits / sends back (optional)
 *                               │
 *                ┌──────────────┴───────────────┐                                          (parallel fan-out)
 *        evidence_librarian            cohort_synthesizer ──Send×arms──► simulate_arm ×N    (map: one worker per scanner arm)
 *        (corrective-RAG subgraph)                                            │
 *                │                                                     biostatistician      (reduce)
 *                └──────────────┬───────────────┘                                          (fan-in / join)
 *                         report_writer ─► END
 *
 * Compiled with a checkpointer so a run can pause at human_approval and resume in a later request (thread_id).
 */

export const TrialState = Annotation.Root({
  request: Annotation<string>,
  overrides: Annotation<Partial<TrialSpec>>({ reducer: (_, b) => b, default: () => ({}) }),
  requireApproval: Annotation<boolean>({ reducer: (_, b) => b, default: () => false }),
  // Set when a resumed run has to be replayed on a fresh thread (e.g. a different serverless instance).
  preApproved: Annotation<boolean>({ reducer: (_, b) => b, default: () => false }),
  spec: Annotation<TrialSpec | null>({ reducer: (_, b) => b, default: () => null }),
  reviewIssues: Annotation<string[]>({ reducer: (_, b) => b, default: () => [] }),
  revisions: Annotation<number>({ reducer: (a, b) => a + b, default: () => 0 }),
  approval: Annotation<{ by: "human" | "auto"; edited: boolean } | null>({ reducer: (_, b) => b, default: () => null }),
  evidence: Annotation<RetrievedChunk[]>({ reducer: (_, b) => b, default: () => [] }),
  cohort: Annotation<Cohort | null>({ reducer: (_, b) => b, default: () => null }),
  armIndex: Annotation<number>({ reducer: (_, b) => b, default: () => 0 }),
  armResults: Annotation<Cohort[]>({ reducer: (a, b) => a.concat(b), default: () => [] }),
  analysis: Annotation<TrialAnalysis | null>({ reducer: (_, b) => b, default: () => null }),
  report: Annotation<string>({ reducer: (_, b) => b, default: () => "" }),
});
type S = typeof TrialState.State;

/* 1 ─ Protocol Architect: natural language → validated, typed trial protocol. */
async function protocolArchitect(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  const revising = state.reviewIssues.length > 0;
  const humanFeedback = state.reviewIssues.some((i) => i.startsWith("Researcher feedback"));
  send({ type: "step", node: "protocol_architect", status: "start", detail: revising ? "revising protocol" : "drafting protocol" });
  const base = state.spec ?? DEFAULT_SPEC;
  let spec = await withLLM(
    "protocol_architect",
    send,
    (llm) =>
      llm
        .withStructuredOutput(TrialSpec, { name: "trial_protocol" })
        .invoke(
          `You are the Protocol Architect for iTRIALSPACE, a virtual imaging trial platform that synthesizes virtual patients with ` +
            `inserted lung nodules and simulates CT acquisitions to stress-test AI nodule detectors.\n` +
            `Convert the researcher's request into a complete trial protocol. Use sensible defaults for anything unspecified ` +
            `(sizes ${JSON.stringify(DEFAULT_SPEC.sizesMm)} mm, all lobes, all nodule types, juxtavascularFraction 0.25, ` +
            `patientsPerArm 120, lesionsPerPatient 2, seed 42). Each scanner arm is one vendor+kernel+dose combination; ` +
            `when the user compares factors, create the cross product (max 8 arms). modelUnderTest: "vendor-overfit" if the user mentions a ` +
            `single-site/Siemens-trained model or generalisation, "lowdose-fragile" for a high/routine-dose model or screening, else "robust-baseline".\n` +
            (revising
              ? `\nThe protocol reviewer raised these issues; fix them:\n- ${state.reviewIssues.join("\n- ")}\nPrevious draft: ${JSON.stringify(base)}\n`
              : "") +
            `\nRequest: """${state.request}"""`,
        ),
    // Without an LLM, researcher feedback is folded back through the rule-based parser.
    () => (humanFeedback ? { ...base, ...heuristicParse(`${state.request} ${state.reviewIssues.join(" ")}`), seed: base.seed } : revising ? base : heuristicParse(state.request)),
  );
  // User-pinned settings win (unless the researcher just asked for a change); reviewer fixes are then applied
  // deterministically so a revision always converges.
  if (!humanFeedback) spec = { ...spec, ...state.overrides };
  if (revising) spec = autoFix(spec, state.reviewIssues);
  send({ type: "step", node: "protocol_architect", status: "done", data: { spec } });
  return { spec };
}

function autoFix(spec: TrialSpec, issues: string[]): TrialSpec {
  const s = structuredClone(spec);
  for (const i of issues) {
    if (i.includes("lesions per size stratum")) s.patientsPerArm = Math.min(600, Math.ceil(s.patientsPerArm * 1.8));
    if (i.includes("3 mm")) s.sizesMm = s.sizesMm.map((x) => Math.max(3, x));
    if (i.includes("30 mm")) s.sizesMm = s.sizesMm.map((x) => Math.min(30, x));
    if (i.includes("duplicate")) s.scanners = s.scanners.filter((a, idx, arr) => arr.findIndex((b) => armLabel(a) === armLabel(b)) === idx);
  }
  s.sizesMm = [...new Set(s.sizesMm)].sort((a, b) => a - b);
  return s;
}

/* 2 ─ Protocol Reviewer: statistical + clinical validity checks; may send the protocol back (cycle). */
async function protocolReviewer(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "protocol_reviewer", status: "start" });
  const s = state.spec!;
  const issues: string[] = [];
  const parsed = TrialSpec.safeParse(s);
  if (!parsed.success) issues.push(`schema: ${parsed.error.issues.map((i) => i.path.join(".") + " " + i.message).join("; ")}`);
  if (s.sizesMm.some((x) => x < 3)) issues.push("Sizes below 3 mm are below the reporting threshold of LIDC-IDRI style nodule annotation; clamp to ≥3 mm.");
  if (s.sizesMm.some((x) => x > 30)) issues.push("Lesions >30 mm are masses, not nodules; clamp to ≤30 mm.");
  const dup = s.scanners.length !== new Set(s.scanners.map(armLabel)).size;
  if (dup) issues.push("Protocol contains duplicate scanner arms.");
  // Expected lesions per (arm × size) cell → CI precision.
  const lesionsPerArm = s.patientsPerArm * ((1 + s.lesionsPerPatient) / 2);
  const perCell = lesionsPerArm / s.sizesMm.length;
  const halfWidth = 1.96 * Math.sqrt(0.25 / perCell);
  if (perCell < 25 && state.revisions < 2) {
    issues.push(
      `Only ~${perCell.toFixed(0)} lesions per size stratum per arm (95% CI half-width up to ±${(halfWidth * 100).toFixed(0)} pp); increase patientsPerArm.`,
    );
  }
  const approved = issues.length === 0 || state.revisions >= 2;
  send({
    type: "step",
    node: "protocol_reviewer",
    status: "done",
    detail: approved ? "approved" : `returned ${issues.length} issue(s) to architect`,
    data: { approved, issues, expectedLesionsPerCell: +perCell.toFixed(1) },
  });
  return { reviewIssues: approved ? [] : issues, revisions: approved ? 0 : 1 };
}

const FAN_OUT = ["evidence_librarian", "cohort_synthesizer"];
const routeReview = (s: S) => (s.reviewIssues.length ? "protocol_architect" : s.requireApproval ? "human_approval" : FAN_OUT);

export type ApprovalDecision = { action: "approve"; spec?: TrialSpec } | { action: "revise"; feedback: string };

/* 2b ─ Human approval: pauses the graph (interrupt) until the researcher approves, edits or sends the protocol back. */
async function humanApproval(state: S, config: LangGraphRunnableConfig) {
  const send = emitter(config);
  // Runs again from the top on resume, so everything before interrupt() must be side-effect free.
  const decision = interrupt<object, ApprovalDecision>({ spec: state.spec, revisions: state.revisions });
  if (decision.action === "revise") {
    send({ type: "step", node: "human_approval", status: "done", detail: "sent back to architect", data: { feedback: decision.feedback } });
    return new Command({ update: { reviewIssues: [`Researcher feedback: ${decision.feedback}`] }, goto: "protocol_architect" });
  }
  const parsed = TrialSpec.safeParse(decision.spec);
  const spec = parsed.success ? parsed.data : state.spec!;
  const edited = JSON.stringify(spec) !== JSON.stringify(state.spec);
  send({ type: "step", node: "human_approval", status: "done", detail: edited ? "approved with edits" : "approved", data: { spec } });
  return new Command({ update: { spec, approval: { by: "human", edited } }, goto: FAN_OUT });
}

/* 3 ─ Evidence Librarian: runs the corrective-RAG retrieval subgraph (plan → retrieve → grade → rewrite) in parallel with simulation. */
async function evidenceLibrarian(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "evidence_librarian", status: "start", detail: "invoking RAG subgraph" });
  const s = state.spec!;
  const question =
    `Evidence to design and interpret a virtual imaging trial of a lung nodule detector: Lung-RADS / Fleischner size thresholds for ` +
    `${s.noduleTypes.join(", ")} nodules of ${s.sizesMm.join(", ")} mm; CT vendor, reconstruction kernel and dose domain shift ` +
    `(${s.scanners.map(armLabel).join("; ")}); synthetic lesion insertion and generative CT for virtual trials; FROC / CPM evaluation.`;
  // The subgraph inherits this config, so its own step events stream to the browser as nested steps.
  const out = await ragRetrievalGraph.invoke({ question }, config);
  let evidence = out.relevant.length >= 2 ? out.relevant : out.retrieved.slice(0, 4);
  // Guarantee the report can cite the core methodology even if grading was strict.
  if (evidence.length < 4) {
    const extra = await hybridSearch([`virtual imaging trial lesion insertion FROC ${s.noduleTypes.join(" ")} Lung-RADS`], 6);
    evidence = [...evidence, ...extra.filter((e) => !evidence.some((x) => x.chunkId === e.chunkId))].slice(0, 6);
  }
  send({
    type: "step",
    node: "evidence_librarian",
    status: "done",
    detail: `${evidence.length} graded chunks · ${out.attempts} retrieval attempt(s)`,
    data: { queries: out.queries, evidence: evidence.map(({ text, ...e }) => ({ ...e, snippet: text.split("\n").slice(1).join(" ").slice(0, 220) })) },
  });
  return { evidence };
}

/* 4 ─ Virtual Cohort Synthesizer, then map: one Send per scanner arm. */
async function cohortSynthesizer(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "cohort_synthesizer", status: "start" });
  const cohort = synthesizeCohort(state.spec!);
  send({ type: "step", node: "cohort_synthesizer", status: "done", data: { patients: cohort.patients.length, lesions: cohort.lesions.length } });
  return { cohort };
}

const fanOutArms = (s: S) => s.spec!.scanners.map((_, armIndex) => new Send("simulate_arm", { spec: s.spec, cohort: s.cohort, armIndex }));

/* 5 ─ Scanner-arm worker (parallel): imaging chain (noise · kernel · dose) → model-under-test inference for one arm. */
async function simulateArmNode(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  const s = state.spec!;
  const a = state.armIndex;
  const label = armLabel(s.scanners[a]);
  send({ type: "step", node: "simulate_arm", status: "start", detail: label, data: { arm: a } });
  const out = simulateArm(s, state.cohort!, a);
  const hits = out.lesions.filter((l) => l.detected).length;
  send({
    type: "step",
    node: "simulate_arm",
    status: "done",
    detail: label,
    data: {
      arm: a,
      label,
      lesions: out.lesions.length,
      meanNoiseHU: +(out.lesions.reduce((t, l) => t + (l.sigmaHU ?? 0), 0) / Math.max(1, out.lesions.length)).toFixed(1),
      sensitivity: +(hits / Math.max(1, out.lesions.length)).toFixed(3),
    },
  });
  return { armResults: [out] };
}

/* 7 ─ Biostatistician: Wilson CIs, stratified sensitivity, FROC/CPM, automated failure-mode discovery. */
async function biostatistician(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "biostatistician", status: "start", detail: `reducing ${state.armResults.length} arm results` });
  // Reduce: merge the per-arm workers' outputs back into one cohort.
  const cohort: Cohort = {
    patients: state.armResults.flatMap((r) => r.patients),
    lesions: state.armResults.flatMap((r) => r.lesions),
  };
  const analysis = analyze(state.spec!, cohort);
  send({ type: "step", node: "biostatistician", status: "done", data: { cpm: analysis.froc.cpm, findings: analysis.findings.length } });
  return { analysis, cohort };
}

/* 8 ─ Report Writer: evidence-grounded trial report (streams tokens). */
async function reportWriter(state: S, config: LangGraphRunnableConfig): Promise<Partial<S>> {
  const send = emitter(config);
  send({ type: "step", node: "report_writer", status: "start" });
  const s = state.spec!;
  const a = state.analysis!;
  const pct = (x: number) => `${(x * 100).toFixed(1)}%`;
  const facts = {
    model: MODEL_PROFILE_INFO[s.modelUnderTest].name,
    patients: a.nPatients,
    lesions: a.nLesions,
    overallSensitivity: `${pct(a.overall.rate)} (95% CI ${pct(a.overall.lo)}–${pct(a.overall.hi)})`,
    fpPerScan: a.fpPerScanAtOperatingPoint,
    cpm: a.froc.cpm.toFixed(3),
    bySize: a.bySize.map((r) => `${r.label}: ${pct(r.rate)} [${pct(r.lo)}–${pct(r.hi)}] n=${r.n}`),
    byArm: a.byArm.map((r) => `${r.label}: ${pct(r.rate)} [${pct(r.lo)}–${pct(r.hi)}] n=${r.n}`),
    byType: a.byType.map((r) => `${r.label}: ${pct(r.rate)} n=${r.n}`),
    findings: a.findings.map((f) => `[${f.severity}] ${f.title} — ${f.detail}`),
  };
  const refs = state.evidence.map((e, i) => `[${i + 1}] ${e.title} — ${e.source}`).join("\n");
  const report = await withLLM(
    "report_writer",
    send,
    async (llm) => {
      const prompt = ChatPromptTemplate.fromMessages([
        [
          "system",
          "You are the Report Writer agent of a virtual imaging trial. Write a crisp markdown report for a radiology AI research lab with sections: " +
            "## Executive summary, ## Protocol, ## Key results, ## Failure modes, ## Recommendations, ## Limitations. " +
            "Use ONLY the numbers in RESULTS. Cite literature only from EVIDENCE using [n]. Note that results come from a simulated detector on synthetic data. Under 450 words.",
        ],
        ["human", "PROTOCOL:\n{spec}\n\nRESULTS:\n{facts}\n\nEVIDENCE:\n{refs}\n\nEVIDENCE TEXT:\n{evidence}"],
      ]);
      const stream = await prompt
        .pipe(llm)
        .pipe(new StringOutputParser())
        .stream({
          spec: JSON.stringify(s, null, 1),
          facts: JSON.stringify(facts, null, 1),
          refs,
          evidence: state.evidence.map((e, i) => `[${i + 1}] ${e.text}`).join("\n\n"),
        });
      let report = "";
      for await (const tok of stream) {
        report += tok;
        send({ type: "token", node: "report_writer", text: tok });
      }
      return report;
    },
    () => {
      send({ type: "reset", node: "report_writer" });
      const report = [
        `## Executive summary`,
        `**${facts.model}** was evaluated on **${a.nPatients} virtual patients** carrying **${a.nLesions} synthetic nodules** across ${s.scanners.length} simulated acquisition arm(s). Overall per-nodule sensitivity was **${facts.overallSensitivity}** at ${facts.fpPerScan} false positives per scan (CPM ${facts.cpm}).`,
        `## Key results`,
        `**By scanner arm**\n${facts.byArm.map((x) => `- ${x}`).join("\n")}`,
        `**By nodule size**\n${facts.bySize.map((x) => `- ${x}`).join("\n")}`,
        `## Failure modes`,
        a.findings.map((f) => `- **${f.title}.** ${f.detail}`).join("\n"),
        `## Recommendations`,
        `- Prioritise multi-vendor / multi-kernel training data or harmonization for arms flagged above.\n- Re-run this protocol with more patients per arm for tighter subgroup CIs (virtual patients are cheap).\n- Confirm findings on a held-out real-patient cohort (e.g. LIDC-IDRI / NLST subsets).`,
        `## Evidence consulted`,
        refs,
        `## Limitations`,
        `_Demo mode: report generated from a template (no LLM configured). Detector, phantoms and physics are simplified simulations intended to demonstrate the virtual-trial workflow._`,
      ].join("\n\n");
      for (const part of report.match(/[\s\S]{1,40}/g) ?? []) send({ type: "token", node: "report_writer", text: part });
      return report;
    },
    { temperature: 0.3 },
  );
  send({ type: "step", node: "report_writer", status: "done" });
  return { report };
}

// Per-process checkpointer: enough for pause/resume within one server instance. The API route replays an approved
// protocol on a fresh thread if the instance changed. Swap for PostgresSaver (@langchain/langgraph-checkpoint-postgres) to persist.
export const checkpointer = new MemorySaver();

export const trialGraph = new StateGraph(TrialState)
  .addNode("protocol_architect", protocolArchitect)
  .addNode("protocol_reviewer", protocolReviewer)
  .addNode("human_approval", humanApproval, { ends: ["protocol_architect", ...FAN_OUT] })
  .addNode("evidence_librarian", evidenceLibrarian)
  .addNode("cohort_synthesizer", cohortSynthesizer)
  .addNode("simulate_arm", simulateArmNode)
  .addNode("biostatistician", biostatistician)
  .addNode("report_writer", reportWriter)
  .addConditionalEdges(START, (s: S) => (s.preApproved ? FAN_OUT : "protocol_architect"), ["protocol_architect", ...FAN_OUT])
  .addEdge("protocol_architect", "protocol_reviewer")
  .addConditionalEdges("protocol_reviewer", routeReview, ["protocol_architect", "human_approval", ...FAN_OUT])
  .addConditionalEdges("cohort_synthesizer", fanOutArms, ["simulate_arm"])
  .addEdge("simulate_arm", "biostatistician")
  .addEdge(["evidence_librarian", "biostatistician"], "report_writer")
  .addEdge("report_writer", END)
  .compile({ checkpointer });
