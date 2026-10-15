import type { Metadata } from "next";
import { VlmBenchClient } from "./vlm-bench-client";

export const metadata: Metadata = { title: "VLM Benchmark · iTRIALSPACE" };

export default function VlmBenchPage() {
  return <VlmBenchClient />;
}
