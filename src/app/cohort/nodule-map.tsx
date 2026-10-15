"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type { Nodule } from "@/lib/cohort/index";

const SERIES_VARS = ["--series-1", "--series-2", "--series-3", "--series-4", "--series-5", "--series-6", "--series-7", "--series-8"];

/**
 * NoduleMap-style scatter of the 2-D PCA embedding. Canvas (3,000 points), colour by a
 * categorical attribute, facet-matching points opaque, cohort donors ringed, hover tooltip.
 */
export function NoduleMap({
  nodules,
  embedding,
  colorKey,
  levels,
  matches,
  cohort,
}: {
  nodules: Nodule[];
  embedding: Float32Array;
  colorKey: (n: Nodule) => string;
  levels: readonly string[];
  matches: Uint8Array;
  cohort: Set<string>;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const wrap = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<{ i: number; x: number; y: number } | null>(null);
  const H = 420;

  useEffect(() => {
    const el = wrap.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => setWidth(Math.max(280, Math.floor(e.contentRect.width))));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const proj = useMemo(() => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < nodules.length; i++) {
      x0 = Math.min(x0, embedding[2 * i]);
      x1 = Math.max(x1, embedding[2 * i]);
      y0 = Math.min(y0, embedding[2 * i + 1]);
      y1 = Math.max(y1, embedding[2 * i + 1]);
    }
    const pad = 14;
    const pts = new Float32Array(nodules.length * 2);
    for (let i = 0; i < nodules.length; i++) {
      pts[2 * i] = pad + ((embedding[2 * i] - x0) / (x1 - x0 || 1)) * (width - 2 * pad);
      pts[2 * i + 1] = pad + (1 - (embedding[2 * i + 1] - y0) / (y1 - y0 || 1)) * (H - 2 * pad);
    }
    return pts;
  }, [nodules.length, embedding, width]);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const dpr = window.devicePixelRatio || 1;
    c.width = width * dpr;
    c.height = H * dpr;
    const ctx = c.getContext("2d")!;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const css = getComputedStyle(document.documentElement);
    const palette = SERIES_VARS.map((v) => css.getPropertyValue(v).trim() || "#888");
    const ink = css.getPropertyValue("--text-primary").trim() || "#fff";
    const colorOf = new Map(levels.map((l, i) => [l, palette[i % palette.length]]));
    ctx.clearRect(0, 0, width, H);
    const hasCohort = cohort.size > 0;
    // Pass 1: non-matching (dim). Pass 2: matching. Pass 3: cohort donors with a ring.
    for (const pass of [0, 1, 2]) {
      for (let i = 0; i < nodules.length; i++) {
        const n = nodules[i];
        const inCohort = hasCohort && cohort.has(n.id);
        const m = matches[i] === 1;
        if ((pass === 0 && m) || (pass === 1 && (!m || inCohort)) || (pass === 2 && !inCohort)) continue;
        if (pass === 0 && inCohort) continue;
        ctx.globalAlpha = pass === 0 ? 0.12 : pass === 1 ? (hasCohort ? 0.45 : 0.8) : 1;
        ctx.fillStyle = colorOf.get(colorKey(n)) ?? "#888";
        ctx.beginPath();
        ctx.arc(proj[2 * i], proj[2 * i + 1], pass === 2 ? 3.4 : 2.2, 0, Math.PI * 2);
        ctx.fill();
        if (pass === 2) {
          ctx.globalAlpha = 0.9;
          ctx.strokeStyle = ink;
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
    if (hover) {
      ctx.globalAlpha = 1;
      ctx.strokeStyle = ink;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(proj[2 * hover.i], proj[2 * hover.i + 1], 6, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [proj, nodules, colorKey, levels, matches, cohort, width, hover]);

  function onMove(e: React.MouseEvent) {
    const r = canvas.current!.getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    let best = -1;
    let bd = 64; // 8 px radius
    for (let i = 0; i < nodules.length; i++) {
      const d = (proj[2 * i] - x) ** 2 + (proj[2 * i + 1] - y) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    setHover(best >= 0 ? { i: best, x: proj[2 * best], y: proj[2 * best + 1] } : null);
  }

  const h = hover ? nodules[hover.i] : null;
  return (
    <div ref={wrap} className="relative">
      <canvas
        ref={canvas}
        style={{ width, height: H }}
        className="block rounded-lg border border-line bg-background"
        onMouseMove={onMove}
        onMouseLeave={() => setHover(null)}
        role="img"
        aria-label="NoduleMap: 2-D embedding of the synthetic nodule index"
      />
      {h && hover && (
        <div
          className="pointer-events-none absolute z-10 w-56 rounded-md border border-line bg-surface-2 px-2.5 py-1.5 text-[11px] shadow-lg"
          style={{ left: Math.min(hover.x + 12, width - 230), top: Math.min(hover.y + 12, H - 150) }}
        >
          <div className="flex justify-between font-mono text-ink">
            <span>{h.id}</span>
            <span className="text-ink-3">{h.sourceDataset}</span>
          </div>
          <div className="text-ink-2">
            {h.diameterMm} mm {h.type} · {h.lobe} · {h.location}
          </div>
          <div className="text-ink-3">
            Lung-RADS {h.lungRads} · malignancy {h.malignancy}/5 · spiculation {h.spiculation}/5
          </div>
          <div className="text-ink-3">
            {h.age} y {h.sex} · {h.smokingPackYears} pack-yr · {h.vendor} {h.kernel} {h.sliceThicknessMm} mm
          </div>
          {h.calcification !== "none" && <div className="text-ink-3">calcification: {h.calcification}</div>}
          {cohort.has(h.id) && <div className="mt-0.5 text-accent">in selected cohort</div>}
        </div>
      )}
    </div>
  );
}
