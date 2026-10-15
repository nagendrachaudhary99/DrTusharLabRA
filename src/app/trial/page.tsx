import type { Metadata } from "next";
import { Suspense } from "react";
import { TrialClient } from "./trial-client";

export const metadata: Metadata = { title: "Trial Designer · iTRIALSPACE" };

export default function TrialPage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-ink-3">Loading…</div>}>
      <TrialClient />
    </Suspense>
  );
}
