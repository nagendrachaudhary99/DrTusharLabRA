import { createAgent, tool } from "langchain";
import { z } from "zod";
import { withLLM } from "@/lib/llm";
import type { Writer } from "@/lib/stream";
import { INDEX_DISCLAIMER } from "./index";
import { parseCohortRequest } from "./parse";
import {
  BuildCohortSchema,
  FacetsSchema,
  FactorEnum,
  LungRadsSchema,
  buildCohort,
  describeSpec,
  lungRads,
  queryIndex,
  stratify,
  summarizeForAgent,
  toFacets,
  type BuildCohortArgs,
  type CohortSummary,
  type ManifestRow,
} from "./tools";

/**
 * Cohort Planner: a LangChain v1 tool-calling agent (`createAgent`, itself a compiled
 * LangGraph ReAct graph: model_request ⇄ tools) over the synthetic nodule index.
 *
 * Every tool call and result is streamed to the browser as `step` events with
 * node = tool name, so the UI shows the agent's reasoning trace. The full case
 * manifest never goes back to the LLM; only a compact summary does.
 *
 * Demo mode (no LLM / failure): a rule-based parser calls the very same tool
 * functions in a fixed order and emits identical events.
 */

export type CohortResult = {
  mode: "live" | "demo";
  args: BuildCohortArgs;
  summary: CohortSummary;
  manifest: ManifestRow[];
  answer: string;
  trialRequest: string;
};

const SYSTEM_PROMPT = `You are the Cohort Planner of iTRIALSPACE, a virtual imaging trial platform for lung nodule AI.
You build "cohorts by specification" from a nodule index (here: a SYNTHETIC ${3000}-nodule index with distributions modelled on LIDC-IDRI, LUNA16, NLST and LNDb; plus 2,000 nodule-free host scans used as negative controls / insertion backgrounds).
A cohort case is a host scan; nodule-positive cases receive one donor nodule from the index. Prevalence = fraction of positive cases.

Workflow (always use the tools, never invent numbers):
1. If the request mentions size thresholds or Lung-RADS, call lung_rads on the relevant boundary sizes/types to ground the design (Lung-RADS v2022: solid <6 mm = 2, 6-<8 = 3, 8-<15 = 4A, >=15 = 4B).
2. Call query_index with the facets implied by the request to check the donor pool.
3. If the user wants balance across factors, call stratify with those factors and targetPositives = round(targetSize*prevalence) to find under-filled strata.
4. Call build_cohort exactly with the requested target size, prevalence (0-1), stratifyBy and allocation (balanced if the user asks for balance/equal strata, else natural), seed 42 unless given.
5. If build_cohort reports warnings (under-filled strata, capped size), you may rebuild ONCE with a justified adjustment, otherwise keep it.
6. Finish with a short answer (<120 words): the final spec, achieved counts/prevalence, per-stratum fill, and any caveats.
Facet hints: "upper lobes" = [RUL, LUL]; "lower lobes" = [RLL, LLL]; "subsolid" = part-solid + ground-glass. Defaults: targetSize 200, prevalence 0.3, seed 42.
Note: ${INDEX_DISCLAIMER}`;

const short = (o: unknown) => JSON.stringify(o).replace(/"/g, "").slice(0, 160);

function makeTools(send: Writer, state: { last: { args: BuildCohortArgs; summary: CohortSummary; manifest: ManifestRow[] } | null }) {
  const run = <A, R>(name: string, args: A, fn: () => R, detail: (r: R) => string, forLLM: (r: R) => unknown = (r) => r) => {
    send({ type: "step", node: name, status: "start", detail: short(args), data: { args } });
    const r = fn();
    send({ type: "step", node: name, status: "done", detail: detail(r), data: { args, result: forLLM(r) } });
    return JSON.stringify(forLLM(r));
  };

  const queryIndexTool = tool(
    (input: z.infer<typeof FacetsSchema>) =>
      run("query_index", input, () => queryIndex(toFacets(input)), (r) => `${r.matchingNodules} donor nodules · ${r.matchingNegativeHostScans} host scans`),
    {
      name: "query_index",
      description: "Faceted filter over the nodule index. Returns matching donor-nodule count, matching negative host scans and attribute histograms.",
      schema: FacetsSchema,
    },
  );

  const stratifyTool = tool(
    (input: { facets: z.infer<typeof FacetsSchema>; factors: z.infer<typeof FactorEnum>[]; targetPositives?: number }) =>
      run(
        "stratify",
        input,
        () => stratify(toFacets(input.facets), input.factors, input.targetPositives),
        (r) => `${r.nStrata} strata · quota ${r.balancedQuotaPerStratum ?? "–"} · ${r.underFilledStrata.length} under-filled`,
      ),
    {
      name: "stratify",
      description: "Count available donor nodules per stratum (cross product of factor levels) and flag strata that cannot meet a balanced quota.",
      schema: z.object({
        facets: FacetsSchema,
        factors: z.array(FactorEnum).min(1).max(3).describe("sizeBin uses Lung-RADS boundaries <6, 6-8, 8-15, >=15 mm"),
        targetPositives: z.number().int().min(1).optional().describe("Number of nodule-positive cases planned"),
      }),
    },
  );

  const buildTool = tool(
    (input: z.infer<typeof BuildCohortSchema>) => {
      const args: BuildCohortArgs = { ...input, facets: toFacets(input.facets) };
      return run(
        "build_cohort",
        input,
        () => {
          const r = buildCohort(args);
          state.last = { args, ...r };
          return r.summary;
        },
        (s) => `${s.nCases} cases · ${s.nPositive}+ / ${s.nNegative}− · prevalence ${(s.achievedPrevalence * 100).toFixed(1)}% · ${s.meetsSpec ? "meets spec" : `${s.warnings.length} warning(s)`}`,
        summarizeForAgent,
      );
    },
    {
      name: "build_cohort",
      description:
        "Sample a cohort: host scans for every case, donor nodules for round(targetSize*prevalence) positives, with per-stratum quotas (balanced water-filling or natural). Returns a manifest summary.",
      schema: BuildCohortSchema,
    },
  );

  const lungRadsTool = tool(
    (input: z.infer<typeof LungRadsSchema>) => run("lung_rads", input, () => lungRads(input), (r) => `category ${r.category} · ${r.rule}`),
    { name: "lung_rads", description: "Lung-RADS v2022 baseline category and management for a nodule size/type.", schema: LungRadsSchema },
  );

  return { qi: queryIndexTool, st: stratifyTool, bc: buildTool, lr: lungRadsTool, all: [queryIndexTool, stratifyTool, buildTool, lungRadsTool] };
}

function finish(mode: "live" | "demo", last: NonNullable<Parameters<typeof makeTools>[1]["last"]>, answer: string): CohortResult {
  return { mode, ...last, answer, trialRequest: describeSpec(last.args, last.summary) };
}

/* Demo mode: same tools, fixed plan. */
async function demoPlan(request: string, send: Writer, pace: number): Promise<CohortResult> {
  const state: Parameters<typeof makeTools>[1] = { last: null };
  const { qi, st, bc, lr } = makeTools(send, state);
  const wait = () => new Promise((r) => setTimeout(r, pace));
  const args = parseCohortRequest(request);
  const f = args.facets;
  const flat = {
    ...(f.diameterMm ? { minDiameterMm: f.diameterMm[0], maxDiameterMm: f.diameterMm[1] } : {}),
    ...(f.ageRange ? { minAge: f.ageRange[0], maxAge: f.ageRange[1] } : {}),
    ...Object.fromEntries(Object.entries(f).filter(([k]) => k !== "diameterMm" && k !== "ageRange")),
  } as z.infer<typeof FacetsSchema>;
  send({ type: "step", node: "planner", status: "done", detail: "demo mode · rule-based parser → tool plan", data: { text: `Parsed request into: ${describeSpec(args)}` } });
  await wait();
  if (f.diameterMm) {
    for (const d of f.diameterMm) {
      await lr.invoke({ diameterMm: d, type: f.types?.[0] ?? "solid" });
      await wait();
    }
  }
  await qi.invoke(flat);
  await wait();
  const nPos = Math.round(args.targetSize * args.prevalence);
  if (args.stratifyBy.length) {
    await st.invoke({ facets: flat, factors: args.stratifyBy, targetPositives: nPos });
    await wait();
  }
  await bc.invoke({ ...args, facets: flat });
  const s = state.last!.summary;
  const answer =
    `**Demo mode** (rule-based planner, no LLM). Built ${s.nCases} cases: ${s.nPositive} nodule-positive + ${s.nNegative} negative controls ` +
    `(prevalence ${(s.achievedPrevalence * 100).toFixed(1)}%, target ${(args.prevalence * 100).toFixed(0)}%). ` +
    (args.stratifyBy.length ? `Strata (${args.allocation}): ${s.strata.map((x) => `${x.stratum} ${x.selected}/${x.available}`).join(", ")}. ` : "") +
    (s.warnings.length ? `Caveats: ${s.warnings.join(" ")}` : "All quotas met.");
  return finish("demo", state.last!, answer);
}

type Msg = { type?: string; _getType?: () => string; content?: unknown; tool_calls?: { name: string; args: unknown }[] };
const textOf = (c: unknown) =>
  typeof c === "string"
    ? c
    : Array.isArray(c)
      ? c.map((p) => (typeof p === "string" ? p : (p as { type?: string; text?: string }).type === "text" ? (p as { text: string }).text : "")).join("")
      : "";

export async function runCohortPlanner(request: string, send: Writer, opts: { signal?: AbortSignal; pace?: number } = {}): Promise<CohortResult> {
  let emitted = false;
  const tracked: Writer = (e) => {
    emitted = true;
    send(e);
  };
  return withLLM(
    "planner",
    send,
    async (llm) => {
      const state: Parameters<typeof makeTools>[1] = { last: null };
      const agent = createAgent({ model: llm, tools: makeTools(tracked, state).all, systemPrompt: SYSTEM_PROMPT });
      const stream = await agent.stream({ messages: [{ role: "user", content: request }] }, { streamMode: "updates", recursionLimit: 20, signal: opts.signal });
      let answer = "";
      for await (const chunk of stream) {
        // "updates" yields { [nodeName]: partialState }; the ReAct loop's model node returns new AI messages.
        for (const [node, update] of Object.entries(chunk as Record<string, { messages?: Msg[] } | undefined>)) {
          for (const m of update?.messages ?? []) {
            const kind = m.type ?? m._getType?.();
            if (kind !== "ai") continue;
            const text = textOf(m.content).trim();
            const calls = m.tool_calls ?? [];
            if (text) {
              answer = text;
              if (calls.length) tracked({ type: "step", node: "planner", status: "done", detail: text.slice(0, 160), data: { text, graphNode: node } });
            }
          }
        }
      }
      if (!state.last) {
        tracked({ type: "log", node: "planner", message: "agent finished without build_cohort; running the rule-based plan to produce a manifest" });
        const demo = await demoPlan(request, tracked, 0);
        return { ...demo, mode: "live" as const, answer: answer || demo.answer };
      }
      return finish("live", state.last, answer);
    },
    async () => {
      if (emitted) send({ type: "reset", node: "trace" });
      return demoPlan(request, send, opts.pace ?? 250);
    },
  );
}
