import type { Metadata } from "next";
import { Suspense } from "react";
import { StudioClient } from "./studio-client";

export const metadata: Metadata = { title: "Virtual Patient Studio · iTRIALSPACE" };

export default function StudioPage() {
  return (
    <Suspense fallback={<div className="p-8 text-sm text-ink-3">Loading studio…</div>}>
      <StudioClient />
    </Suspense>
  );
}
