"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { Activity } from "lucide-react";

const LINKS = [
  { href: "/", label: "Overview" },
  { href: "/cohort", label: "1 · Cohort Planner" },
  { href: "/trial", label: "2 · Trial Designer" },
  { href: "/studio", label: "3 · Patient Studio" },
  { href: "/vlm-bench", label: "4 · VLM Benchmark" },
  { href: "/copilot", label: "Research Copilot" },
  { href: "/architecture", label: "Architecture" },
];

export function Nav() {
  const path = usePathname();
  return (
    <header className="sticky top-0 z-30 border-b border-line bg-background/85 backdrop-blur">
      <nav className="mx-auto flex max-w-7xl items-center gap-6 px-4 py-3">
        <Link href="/" className="flex items-center gap-2 font-semibold tracking-tight">
          <Activity className="h-5 w-5 text-accent" aria-hidden />
          <span>
            i<span className="text-accent">TRIAL</span>SPACE
          </span>
        </Link>
        <div className="flex flex-1 items-center gap-1 overflow-x-auto text-sm">
          {LINKS.map((l) => {
            const active = l.href === "/" ? path === "/" : path.startsWith(l.href);
            return (
              <Link
                key={l.href}
                href={l.href}
                className={`whitespace-nowrap rounded-md px-3 py-1.5 transition-colors ${active ? "bg-surface-2 text-ink" : "text-ink-3 hover:text-ink"}`}
              >
                {l.label}
              </Link>
            );
          })}
        </div>
      </nav>
    </header>
  );
}
