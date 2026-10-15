import { trialGraph } from "../src/lib/trial/graph.ts";
const prompt = process.argv[2] ?? "Does the single-site model generalise? Compare Siemens, GE, Philips and Canon at standard dose, 150 patients per arm.";
const t0 = Date.now();
let final: any = {};
for await (const [mode, chunk] of await trialGraph.stream({ request: prompt }, { streamMode: ["custom", "updates"], configurable: { thread_id: crypto.randomUUID() } })) {
  if (mode === "custom" && (chunk as any).type === "step") console.log((chunk as any).node, (chunk as any).status, (chunk as any).detail ?? "");
  if (mode === "updates") for (const u of Object.values(chunk as any)) if (u && !Array.isArray(u)) Object.assign(final, u);
}
const a = final.analysis;
console.log("spec", JSON.stringify(final.spec));
console.log("overall", a.overall, "fp/scan", a.fpPerScanAtOperatingPoint, "cpm", a.froc.cpm.toFixed(3));
console.log("byArm", a.byArm.map((r: any) => `${r.label} ${(r.rate*100).toFixed(1)}`));
console.log("bySize", a.bySize.map((r: any) => `${r.label} ${(r.rate*100).toFixed(1)}`));
console.log("byType", a.byType.map((r: any) => `${r.label} ${(r.rate*100).toFixed(1)}`));
console.log("findings", a.findings.map((f: any) => f.severity + ": " + f.title));
console.log("ms", Date.now() - t0);
