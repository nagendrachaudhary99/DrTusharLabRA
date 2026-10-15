# iTRIALSPACE: Agentic Virtual Lesion Trials (prototype)

A working prototype inspired by **iTRIALSPACE** ([Tushar, Momy, Lo, Rubin, arXiv:2605.05761](https://arxiv.org/abs/2605.05761)), which
treats the evaluation of lung-CT models as a controlled experiment instead of a fixed benchmark. Here, **LangGraph** and **LangChain**
agents plan cohorts from a nodule index, design and power-check trial protocols with a **human in the loop**, synthesize virtual patients
with custom nodules, and evaluate detectors and a **live vision-language model**, stratified by size, lobe, type and scanner. A
**corrective-RAG** layer grounds every report in source-verified literature.

> Synthetic nodule index, procedural phantoms and simplified physics. This is a workflow demonstrator, not NodMAISI, not a validated
> simulator, and not for clinical use.

## Modules: one pipeline, four stages

Stage names follow the four-stage pipeline described in the iTRIALSPACE paper.

| # | Route | Stage | What it shows | Key tech |
|---|---|---|---|---|
| 1 | `/cohort` | Multidataset nodule profiling | **Cohort Planner**: faceted search over a 3,000-nodule index, NoduleMap-style embedding, and a tool-calling agent that builds a cohort by specification (size, prevalence, per-stratum quotas) | LangChain v1 `createAgent`, Zod `tool()`s, streamed tool trace |
| 2 | `/trial` | Explicit trial specification | **Trial Designer**: NL → typed protocol → power check → **human approval** → one parallel worker per scanner arm → stats (Wilson CI, FROC, CPM) → grounded report | LangGraph `interrupt()` + checkpointer + `Command`, `Send` map-reduce, RAG **subgraph**, review cycle |
| 3 | `/studio` | Anatomy-aware mask insertion → synthesis | **Virtual Patient Studio**: design a nodule, view the conditioning mask, the synthetic CT on 4 scanners, and a **3D volume in NiiVue**; export NIfTI | NiiVue (WebGL2), procedural phantom + imaging chain, NIfTI-1 writer |
| 4 | `/vlm-bench` | Controlled model evaluation | **VLM Benchmark**: a real multimodal LLM reads every synthetic slice (detection, lobe, size), scored against ground truth with stratified CIs | LangGraph `Send` fan-out, multimodal `withStructuredOutput` |
| | `/copilot` | | **Research Copilot**: corrective RAG over 53 source-verified documents, including the iTRIALSPACE, NodMAISI and Tri-Reader papers | Hybrid dense + BM25 + RRF, LLM grading, rewrite loop, grounding judge |
| | `/architecture` | | System design, graphs introspected from the compiled LangGraphs (Mermaid), roadmap | `getGraphAsync().drawMermaid()` |
| | `python/` | | **`itrialspace-eval`**: re-analyze exported trials (cluster bootstrap, FROC/CPM, failure modes, synthetic-vs-real rank agreement) | numpy/pandas/scipy, pytest, ruff, mypy --strict, Docker |

## The LangGraph trial pipeline

```
protocol_architect ─► protocol_reviewer ──(issues, ≤2 revisions)──► protocol_architect                ← cycle
        ▲                     │ approved
        └── send back ── human_approval   ← interrupt(): approve · edit protocol · send back with feedback
                              │ Command(goto)
               ┌──────────────┴──────────────┐                                                        ← parallel fan-out
      evidence_librarian        cohort_synthesizer ──Send × scanner arms──► simulate_arm ×N           ← map
   (corrective-RAG subgraph)                                                     │
               │                                                          biostatistician             ← reduce
               └──────────────┬──────────────┘                                                        ← fan-in (join)
                        report_writer ─► END
```

- **Human in the loop.** With approval on, the graph pauses after review. The UI shows an editable protocol; *Approve* resumes the same
  thread with `Command({ resume })`, *Send back* routes to the architect with the researcher's feedback.
- **Serverless-safe resume.** The checkpointer (`MemorySaver`) is per instance. If a resume lands on another instance, the route replays
  the approved protocol on a fresh thread (`preApproved` entry route), so it never fails. For durable runs, swap in
  `PostgresSaver` from `@langchain/langgraph-checkpoint-postgres`.
- **Map-reduce.** `cohort_synthesizer` emits one `Send` per scanner arm; each worker simulates the imaging chain and runs the model under
  test for its arm (seeded per arm). The biostatistician reduces the results via a concat reducer.
- **Subgraph.** The Evidence Librarian invokes `ragRetrievalGraph` (plan → retrieve → grade → rewrite loop) with the parent config, so its
  steps stream to the UI as nested events.

## Quick start

```bash
npm install
npm run dev            # http://localhost:3000, works with zero config (demo mode)
npm run smoke:trial    # run the trial graph headless
npm run smoke:rag      # run the RAG graph headless
```

**Live LLM mode:** copy `.env.example` to `.env.local` and set `AI_GATEWAY_API_KEY` or `OPENAI_API_KEY` there.
**Never put keys in `.env.example`**: it is committed. If any LLM call fails (auth, quota, timeout), that agent falls back to its
deterministic implementation and a circuit breaker switches to demo mode for 10 minutes. The demo never breaks.

**Re-analyze a trial in Python:**

```bash
cd python && python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/itrialspace-eval analyze itrialspace-lesions.csv --patients itrialspace-patients.csv --by arm --bootstrap 2000 --seed 42
# or: docker build -t itrialspace-eval python && docker run --rm -v "$PWD:/data" itrialspace-eval analyze /data/itrialspace-lesions.csv
```

The Python CLI reproduces the web app's sensitivity and CPM exactly on the same export.

## Deploy to Vercel

```bash
npm i -g vercel@latest
vercel                                # preview
vercel env add OPENAI_API_KEY         # or AI_GATEWAY_API_KEY
vercel --prod                         # production
```

Before sharing a public URL: set a **monthly spending cap** on the OpenAI account. The API routes also have per-IP rate limits
(`src/lib/rate-limit.ts`, best-effort per instance).

Optional LangSmith tracing: `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, `LANGSMITH_PROJECT`, and `LANGCHAIN_CALLBACKS_BACKGROUND=false`
on Vercel.

## Design decisions worth discussing

- **Typed state and structured output everywhere.** Agents exchange a Zod-validated `TrialSpec`, not free text; the VLM reader answers in
  a Zod schema; human edits are re-validated before the graph resumes.
- **Cycles, interrupts and fan-out are why LangGraph is used.** A linear chain can't express review → revise loops, pausing for a human,
  or a dynamic number of parallel workers.
- **Agents call tools; tools are plain functions.** The cohort planner's tools are ordinary TypeScript functions, so demo mode calls the
  exact same code in a fixed sequence and the UI can rebuild cohorts in the browser.
- **Hybrid retrieval.** Dense embeddings miss exact identifiers ("Lung-RADS 4B", "LIDC-IDRI"); BM25 catches them. Reciprocal Rank Fusion
  merges the rankings without score calibration.
- **Corrective RAG.** Chunks are graded; if fewer than 2 survive, the query is rewritten and retrieval repeats. Answers are checked for
  groundedness by a separate judge call.
- **Honest evaluation.** Every rate comes with n and a 95% CI. The VLM benchmark reports real failures of a general-purpose model on
  phantom slices instead of hiding them.
- **Reproducibility.** Every trial is a pure function of (protocol, seed). Exports re-analyze identically in Python.

## Path to a research-grade system

1. **NodMAISI synthesis** in a GPU worker, conditioned on the Studio's label map + nodule mask, instead of the procedural phantom.
2. **Real nodule index** (LIDC-IDRI, LUNA16/25, LNDb, DLCS24 metadata) with donor matching into real host CTs and TotalSegmentator /
   VISTA3D lobe masks.
3. **Medical VLMs and real detectors under test**: MedGemma / LLaVA-Med / BiomedCLIP in the VLM reader (same answer schema), MONAI or
   Tri-Reader in the scanner-arm workers.
4. **Durable runs**: PostgresSaver checkpointer, time-travel what-if re-runs, LangSmith datasets to regression-test the agents.
5. **Credibility**: synthetic-vs-real agreement checks (`itrialspace-eval agreement`) in CI.

## Project layout

```
src/lib/trial/   spec.ts (Zod schema) · parse.ts · physics.ts · simulate.ts (cohort, per-arm simulation, stats) · graph.ts (LangGraph)
src/lib/cohort/  index.ts (synthetic nodule index, Lung-RADS, PCA map) · tools.ts · agent.ts (createAgent) · parse.ts (demo planner)
src/lib/vlm/     schema.ts · design.ts · metrics.ts · graph.ts (Send map-reduce VLM evaluation)
src/lib/rag/     embeddings.ts · vectorstore.ts · index.ts (chunking, BM25, RRF) · graph.ts (agentic RAG + retrieval subgraph)
src/lib/phantom/ render.ts (label map, lesion insertion, imaging chain) · volume.ts (3D volume, NIfTI writer)
src/lib/llm.ts   provider resolution (AI Gateway / OpenAI / OIDC), withLLM fallback + circuit breaker
src/lib/rate-limit.ts
src/app/api/     cohort · trial · vlm · rag · graph route handlers (NDJSON streaming)
src/data/        knowledge-base.json (53 verified docs) · research-notes.md
python/          itrialspace-eval package, tests, Dockerfile, CI workflow
```
