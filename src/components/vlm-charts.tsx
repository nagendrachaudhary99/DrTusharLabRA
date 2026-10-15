"use client";

import { useState } from "react";
import { seriesColor } from "@/components/charts";
import { TASKS, type StratumRow, type VlmMetrics } from "@/lib/vlm/metrics";
import { LOBE_OR_NONE, sizeTolerance } from "@/lib/vlm/schema";

const pct = (x: number) => `${(x * 100).toFixed(0)}%`;

/** Forest plot with one CI line per task (detection / localization / size) for every stratum. */
export function TaskForest({ title, rows, note }: { title: string; rows: StratumRow[]; note?: string }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 520;
  const labelW = 170;
  const rowH = 34;
  const H = rows.length * rowH + 26;
  const x = (v: number) => labelW + v * (W - labelW - 40);
  return (
    <figure className="card p-4">
      <figcaption className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-ink">{title}</span>
        <span className="flex gap-3 text-[11px] text-ink-2">
          {TASKS.map((t, i) => (
            <span key={t.id} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: seriesColor(i) }} />
              {t.short}
            </span>
          ))}
        </span>
      </figcaption>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={title}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={H - 20} stroke="var(--border)" />
              <text x={x(t)} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
                {t * 100}%
              </text>
            </g>
          ))}
          {rows.map((r, i) => {
            const cy = i * rowH + rowH / 2 + 2;
            return (
              <g key={r.key} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill={hover === i ? "var(--surface-2)" : "transparent"} />
                <text x={8} y={cy + 4} fontSize={11} fill="var(--text-secondary)">
                  {r.label.length > 26 ? r.label.slice(0, 25) + "…" : r.label}
                </text>
                <text x={W - 4} y={cy + 4} fontSize={10} textAnchor="end" fill="var(--text-muted)" fontFamily="var(--font-geist-mono)">
                  n={r.n}
                </text>
                {TASKS.map((t, ti) => {
                  const v = r[t.id];
                  if (!v) return null;
                  const y = cy + (ti - 1) * 8;
                  return (
                    <g key={t.id}>
                      <line x1={x(v.lo)} x2={x(v.hi)} y1={y} y2={y} stroke={seriesColor(ti)} strokeWidth={2} strokeLinecap="round" opacity={0.85} />
                      <circle cx={x(v.rate)} cy={y} r={3.5} fill={seriesColor(ti)} stroke="var(--surface-1)" strokeWidth={1.5} />
                    </g>
                  );
                })}
              </g>
            );
          })}
        </svg>
        {hover !== null && (
          <div className="pointer-events-none absolute right-2 top-0 rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-xs shadow-lg">
            <div className="mb-0.5 font-medium text-ink">{rows[hover].label}</div>
            {TASKS.map((t, i) => {
              const v = rows[hover][t.id];
              return (
                <div key={t.id} className="flex items-center gap-1.5 text-ink-2">
                  <span className="inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(i) }} />
                  <span className="w-10">{t.short}</span>
                  <span className="font-mono text-ink">{v ? `${pct(v.rate)} (${v.k}/${v.n})` : "n/a"}</span>
                  {v && <span className="text-ink-3">CI {pct(v.lo)}–{pct(v.hi)}</span>}
                </div>
              );
            })}
          </div>
        )}
      </div>
      <p className="mt-1 text-[11px] text-ink-3">{note ?? "Accuracy per task with 95% Wilson CIs. Localization & size are scored on nodule-positive cases only."}</p>
    </figure>
  );
}

/** Predicted vs true diameter with the tolerance band used for the size task. */
export function SizeScatter({ points, onSelect }: { points: VlmMetrics["sizeScatter"]; onSelect?: (id: string) => void }) {
  const W = 360;
  const H = 300;
  const m = { l: 40, r: 12, t: 12, b: 34 };
  const max = Math.max(25, ...points.map((p) => Math.max(p.truth, p.pred))) * 1.05;
  const x = (v: number) => m.l + (v / max) * (W - m.l - m.r);
  const y = (v: number) => H - m.b - (v / max) * (H - m.t - m.b);
  const band = Array.from({ length: 41 }, (_, i) => (i / 40) * max);
  const upper = band.map((t) => `${x(t)},${y(Math.min(max, t + sizeTolerance(t)))}`);
  const lower = band.map((t) => `${x(t)},${y(Math.max(0, t - sizeTolerance(t)))}`).reverse();
  const ticks = [0, 5, 10, 15, 20, 25, 30, 40, 50].filter((t) => t <= max);
  return (
    <figure className="card p-4">
      <figcaption className="mb-2 text-sm font-medium text-ink">Size task · predicted vs true diameter</figcaption>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Predicted vs true diameter">
        {ticks.map((t) => (
          <g key={t}>
            <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke="var(--border)" />
            <text x={m.l - 5} y={y(t) + 3} fontSize={10} textAnchor="end" fill="var(--text-muted)">
              {t}
            </text>
            <text x={x(t)} y={H - m.b + 14} fontSize={10} textAnchor="middle" fill="var(--text-muted)">
              {t}
            </text>
          </g>
        ))}
        <polygon points={[...upper, ...lower].join(" ")} fill="var(--series-3)" opacity={0.12} />
        <line x1={x(0)} y1={y(0)} x2={x(max)} y2={y(max)} stroke="var(--text-muted)" strokeDasharray="3 3" />
        <text x={(W + m.l) / 2} y={H - 4} fontSize={10} textAnchor="middle" fill="var(--text-muted)">
          True diameter (mm)
        </text>
        <text x={10} y={m.t + 4} fontSize={10} fill="var(--text-muted)">
          Pred. (mm)
        </text>
        {points.map((p, i) => {
          const ok = p.pred > 0 && Math.abs(p.pred - p.truth) <= sizeTolerance(p.truth);
          const jitter = ((i * 37) % 7) - 3;
          return (
            <circle
              key={p.caseId}
              cx={x(p.truth) + jitter}
              cy={y(p.pred)}
              r={4.5}
              fill={p.pred === 0 ? "transparent" : ok ? "var(--series-3)" : "var(--series-2)"}
              stroke={p.pred === 0 ? "var(--status-critical)" : "var(--surface-1)"}
              strokeWidth={1.5}
              className="cursor-pointer"
              onClick={() => onSelect?.(p.caseId)}
            >
              <title>{`${p.caseId}: true ${p.truth} mm → predicted ${p.pred} mm`}</title>
            </circle>
          );
        })}
      </svg>
      <p className="mt-1 text-[11px] text-ink-3">Shaded: tolerance ±max(2 mm, 30%). Hollow red circles at 0 = missed nodules. Click a point to inspect.</p>
    </figure>
  );
}

/** 2×2 detection table + truth-lobe × predicted-lobe matrix. */
export function ConfusionSummary({ m }: { m: VlmMetrics }) {
  const { tp, fp, fn, tn } = m.confusion;
  const cell = (v: number, good: boolean, label: string) => (
    <td className="rounded p-2 text-center" style={{ background: v ? (good ? "rgba(25,158,112,0.18)" : "rgba(208,59,59,0.18)") : "var(--surface-2)" }}>
      <div className="font-mono text-lg text-ink">{v}</div>
      <div className="text-[10px] text-ink-3">{label}</div>
    </td>
  );
  const rowMax = Math.max(1, ...m.lobeMatrix.flatMap((r) => r.counts));
  return (
    <figure className="card space-y-4 p-4">
      <figcaption className="text-sm font-medium text-ink">Confusion summary</figcaption>
      <div className="grid gap-4 sm:grid-cols-[auto_1fr]">
        <table className="border-separate text-xs" style={{ borderSpacing: 3 }}>
          <thead>
            <tr>
              <th />
              <th className="px-2 font-normal text-ink-3">VLM: nodule</th>
              <th className="px-2 font-normal text-ink-3">VLM: none</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <th className="pr-2 text-left font-normal text-ink-3">Truth: nodule</th>
              {cell(tp, true, "TP")}
              {cell(fn, false, "FN")}
            </tr>
            <tr>
              <th className="pr-2 text-left font-normal text-ink-3">Truth: none</th>
              {cell(fp, false, "FP")}
              {cell(tn, true, "TN")}
            </tr>
          </tbody>
        </table>
        <div className="overflow-x-auto">
          <table className="w-full border-separate text-[11px]" style={{ borderSpacing: 2 }}>
            <thead>
              <tr>
                <th className="text-left font-normal text-ink-3">truth ↓ / VLM →</th>
                {LOBE_OR_NONE.map((l) => (
                  <th key={l} className="px-1 font-normal text-ink-3">
                    {l}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {m.lobeMatrix.map((r) => (
                <tr key={r.truth}>
                  <th className="pr-2 text-left font-normal text-ink-2">{r.truth}</th>
                  {r.counts.map((v, i) => {
                    const diag = LOBE_OR_NONE[i] === r.truth;
                    return (
                      <td
                        key={i}
                        className="h-7 min-w-8 rounded text-center font-mono"
                        style={{
                          background: v ? (diag ? `rgba(25,158,112,${0.15 + (0.6 * v) / rowMax})` : `rgba(217,89,38,${0.15 + (0.6 * v) / rowMax})`) : "var(--surface-2)",
                          color: "var(--text-primary)",
                        }}
                      >
                        {v || ""}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-1 text-[10px] text-ink-3">Green diagonal = correct lobe; orange = wrong lobe; &quot;none&quot; column = missed. Left↔right swaps indicate laterality-convention errors.</p>
        </div>
      </div>
    </figure>
  );
}
