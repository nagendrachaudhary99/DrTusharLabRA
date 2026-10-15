import type { Metadata } from "next";
import { CohortClient } from "./cohort-client";

export const metadata: Metadata = { title: "Cohort Planner · iTRIALSPACE" };

export default function CohortPage() {
  return <CohortClient />;
}
