import { ragGraph } from "../src/lib/rag/graph.ts";
for (const q of ["What is MAISI and can it condition on lung lobes?", "What are the Lung-RADS size thresholds for solid nodules?", "How does reconstruction kernel affect deep learning nodule detection?"]) {
  console.log("\nQ:", q);
  for await (const e of await ragGraph.stream({ question: q }, { streamMode: "custom" })) {
    const ev = e as any;
    if (ev.type === "step" && ev.status === "done" && ev.node === "retrieve") console.log(" retrieved:", ev.data.retrieved.slice(0, 4).map((r: any) => r.docId).join(", "));
    if (ev.type === "step" && ev.status === "done" && ev.node === "grade_documents") console.log(" kept:", ev.data.kept);
    if (ev.type === "step" && ev.status === "done" && ev.node === "check_grounding") console.log(" grounding:", ev.data.score);
  }
}
