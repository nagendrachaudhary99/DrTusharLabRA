import type { Metadata } from "next";
import { Suspense } from "react";
import { CopilotClient } from "./copilot-client";

export const metadata: Metadata = { title: "Research Copilot · iTRIALSPACE" };

export default function CopilotPage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-ink-3">Loading…</div>}>
      <CopilotClient />
    </Suspense>
  );
}
