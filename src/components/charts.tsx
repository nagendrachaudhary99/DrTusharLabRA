"use client";

import { useState } from "react";

const SERIES = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"].map((v) => `var(${v})`);
export const seriesColor = (i: number) => SERIES[i % SERIES.length];
const pct = (x: number) => `${(x * 100).toFixed(1)}%`;

type Rate = { key: string; label: string; n: number; k: number; rate: number; lo: number; hi: number };

/** Forest plot: point estimate + 95% Wilson CI per stratum, with the overall estimate as a reference line. */
export function ForestPlot({ title, rows, overall }: { title: string; rows: Rate[]; overall?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 520;
  const labelW = 200;
  const rowH = 26;
  const H = rows.length * rowH + 28;
  const x = (v: number) => labelW + v * (W - labelW - 56);
  return (
    <figure className="card p-4">
      <figcaption className="mb-2 text-sm font-medium text-ink">{title}</figcaption>
      <div className="relative">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label={title}>
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={x(t)} x2={x(t)} y1={0} y2={H - 20} stroke="var(--border)" strokeWidth={1} />
              <text x={x(t)} y={H - 6} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
                {t * 100}%
              </text>
            </g>
          ))}
          {overall !== undefined && (
            <line x1={x(overall)} x2={x(overall)} y1={0} y2={H - 20} stroke="var(--text-secondary)" strokeDasharray="3 3" strokeWidth={1} />
          )}
          {rows.map((r, i) => {
            const cy = i * rowH + rowH / 2 + 2;
            const active = hover === i;
            return (
              <g key={r.key} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
                <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill={active ? "var(--surface-2)" : "transparent"} />
                <text x={8} y={cy + 4} fontSize={11} fill="var(--text-secondary)">
                  {r.label.length > 32 ? r.label.slice(0, 31) + "…" : r.label}
                </text>
                <line x1={x(r.lo)} x2={x(r.hi)} y1={cy} y2={cy} stroke={seriesColor(0)} strokeWidth={2} strokeLinecap="round" />
                <circle cx={x(r.rate)} cy={cy} r={5} fill={seriesColor(0)} stroke="var(--surface-1)" strokeWidth={2} />
                <text x={W - 4} y={cy + 4} fontSize={11} textAnchor="end" fill="var(--text-primary)" fontFamily="var(--font-geist-mono)">
                  {pct(r.rate)}
                </text>
              </g>
            );
          })}
        </svg>
        {hover !== null && (
          <div className="pointer-events-none absolute right-2 top-0 rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-xs shadow-lg">
            <div className="font-medium text-ink">{rows[hover].label}</div>
            <div className="text-ink-2">
              Sensitivity {pct(rows[hover].rate)} · 95% CI {pct(rows[hover].lo)}–{pct(rows[hover].hi)}
            </div>
            <div className="text-ink-3">
              {rows[hover].k}/{rows[hover].n} nodules detected
            </div>
          </div>
        )}
      </div>
      {overall !== undefined && <p className="mt-1 text-[11px] text-ink-3">Dashed line: overall sensitivity {pct(overall)}. Bars: 95% Wilson CI.</p>}
    </figure>
  );
}

// Sequential blue ramp (steps 100 → 700).
const RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"];

type HeatRow = { arm: string; cells: { size: number; n: number; rate: number; lo: number; hi: number }[] };

/** Sensitivity heat map: scanner arm × nodule size. */
export function Heatmap({ rows }: { rows: HeatRow[] }) {
  const [hover, setHover] = useState<{ r: number; c: number } | null>(null);
  const sizes = rows[0]?.cells.map((c) => c.size) ?? [];
  return (
    <figure className="card p-4">
      <figcaption className="mb-3 text-sm font-medium text-ink">Sensitivity by scanner arm × nodule size</figcaption>
      <div className="overflow-x-auto">
        <table className="w-full border-separate" style={{ borderSpacing: 2 }}>
          <thead>
            <tr>
              <th />
              {sizes.map((s) => (
                <th key={s} className="px-1 pb-1 text-center text-[11px] font-normal text-ink-3">
                  {s} mm
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, r) => (
              <tr key={row.arm}>
                <th className="whitespace-nowrap pr-2 text-left text-[11px] font-normal text-ink-2">{row.arm}</th>
                {row.cells.map((cell, c) => {
                  const idx = Math.min(RAMP.length - 1, Math.round(cell.rate * (RAMP.length - 1)));
                  const dark = idx >= 6;
                  return (
                    <td
                      key={c}
                      onMouseEnter={() => setHover({ r, c })}
                      onMouseLeave={() => setHover(null)}
                      className="h-9 min-w-12 rounded text-center font-mono text-[11px]"
                      style={{
                        background: cell.n ? RAMP[idx] : "var(--surface-2)",
                        color: dark ? "#f0efec" : "#0b0b0b",
                        outline: hover?.r === r && hover?.c === c ? "2px solid var(--text-primary)" : "none",
                      }}
                    >
                      {cell.n ? Math.round(cell.rate * 100) : "–"}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex items-center justify-between text-[11px] text-ink-3">
        <span>
          {hover
            ? (() => {
                const cell = rows[hover.r].cells[hover.c];
                return `${rows[hover.r].arm} · ${cell.size} mm → ${pct(cell.rate)} (95% CI ${pct(cell.lo)}–${pct(cell.hi)}, n=${cell.n})`;
              })()
            : "Hover a cell for CI and n. Values are % sensitivity."}
        </span>
        <span className="flex items-center gap-1">
          0%
          <span className="inline-block h-2 w-24 rounded" style={{ background: `linear-gradient(90deg, ${RAMP.join(",")})` }} />
          100%
        </span>
      </div>
    </figure>
  );
}

type FrocSeries = { arm: string; cpm: number; curve: { fpPerScan: number; sensitivity: number }[]; points: { fpPerScan: number; sensitivity: number }[] };

/** FROC curves per scanner arm on a log2 FP/scan axis. */
export function FrocChart({ series }: { series: FrocSeries[] }) {
  const [hoverFp, setHoverFp] = useState<number | null>(null);
  const W = 560;
  const H = 300;
  const m = { l: 44, r: 16, t: 12, b: 36 };
  const FPS = [0.125, 0.25, 0.5, 1, 2, 4, 8];
  const x = (fp: number) => m.l + ((Math.log2(Math.max(fp, 0.125)) + 3) / 6) * (W - m.l - m.r);
  const y = (s: number) => m.t + (1 - s) * (H - m.t - m.b);
  const path = (pts: { fpPerScan: number; sensitivity: number }[]) => {
    const p = pts.filter((q) => q.fpPerScan >= 0.125 && q.fpPerScan <= 8);
    return p.map((q, i) => `${i ? "L" : "M"}${x(q.fpPerScan).toFixed(1)},${y(q.sensitivity).toFixed(1)}`).join("");
  };
  const step = (pts: FrocSeries["curve"]) => {
    // Monotone step curve sampled on a fine grid.
    const out: { fpPerScan: number; sensitivity: number }[] = [];
    for (let e = -3; e <= 3.001; e += 0.05) {
      const fp = 2 ** e;
      let best = 0;
      for (const c of pts) if (c.fpPerScan <= fp) best = Math.max(best, c.sensitivity);
      out.push({ fpPerScan: fp, sensitivity: best });
    }
    return out;
  };
  const curves = series.map((s) => step(s.curve));
  const at = (ci: number, fp: number) => curves[ci].reduce((b, q) => (q.fpPerScan <= fp * 1.0001 ? q.sensitivity : b), 0);
  return (
    <figure className="card p-4">
      <figcaption className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
        <span className="text-sm font-medium text-ink">FROC — sensitivity vs false positives per scan</span>
        <span className="flex flex-wrap gap-3 text-[11px] text-ink-2">
          {series.map((s, i) => (
            <span key={s.arm} className="flex items-center gap-1.5">
              <span className="inline-block h-0.5 w-4 rounded" style={{ background: seriesColor(i) }} />
              {s.arm} <span className="font-mono text-ink-3">CPM {s.cpm.toFixed(3)}</span>
            </span>
          ))}
        </span>
      </figcaption>
      <div className="relative">
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="w-full"
          role="img"
          aria-label="FROC curves"
          onMouseMove={(e) => {
            const rect = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
            const px = ((e.clientX - rect.left) / rect.width) * W;
            const e2 = ((px - m.l) / (W - m.l - m.r)) * 6 - 3;
            setHoverFp(e2 < -3 || e2 > 3 ? null : FPS.reduce((b, f) => (Math.abs(Math.log2(f) - e2) < Math.abs(Math.log2(b) - e2) ? f : b)));
          }}
          onMouseLeave={() => setHoverFp(null)}
        >
          {[0, 0.25, 0.5, 0.75, 1].map((t) => (
            <g key={t}>
              <line x1={m.l} x2={W - m.r} y1={y(t)} y2={y(t)} stroke="var(--border)" />
              <text x={m.l - 6} y={y(t) + 3} textAnchor="end" fontSize={10} fill="var(--text-muted)">
                {t * 100}%
              </text>
            </g>
          ))}
          {FPS.map((f) => (
            <text key={f} x={x(f)} y={H - m.b + 16} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
              {f < 1 ? `1/${1 / f}` : f}
            </text>
          ))}
          <text x={(W + m.l) / 2} y={H - 4} textAnchor="middle" fontSize={10} fill="var(--text-muted)">
            False positives per scan (log scale)
          </text>
          {hoverFp !== null && <line x1={x(hoverFp)} x2={x(hoverFp)} y1={m.t} y2={H - m.b} stroke="var(--text-muted)" strokeDasharray="2 3" />}
          {curves.map((c, i) => (
            <path key={i} d={path(c)} fill="none" stroke={seriesColor(i)} strokeWidth={2} strokeLinejoin="round" />
          ))}
          {hoverFp !== null &&
            curves.map((_, i) => (
              <circle key={i} cx={x(hoverFp)} cy={y(at(i, hoverFp))} r={4} fill={seriesColor(i)} stroke="var(--surface-1)" strokeWidth={2} />
            ))}
        </svg>
        {hoverFp !== null && (
          <div className="pointer-events-none absolute right-2 bottom-12 rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-xs shadow-lg">
            <div className="mb-0.5 font-medium text-ink">{hoverFp < 1 ? `1/${1 / hoverFp}` : hoverFp} FP / scan</div>
            {series.map((s, i) => (
              <div key={s.arm} className="flex items-center gap-1.5 text-ink-2">
                <span className="inline-block h-2 w-2 rounded-full" style={{ background: seriesColor(i) }} />
                <span className="flex-1">{s.arm}</span>
                <span className="font-mono text-ink">{pct(at(i, hoverFp))}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </figure>
  );
}
