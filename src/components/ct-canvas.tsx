"use client";

import { useEffect, useRef } from "react";
import { toImageData, type RenderResult } from "@/lib/phantom/render";

export function CTCanvas({
  result,
  mode,
  window: win,
  className,
  label,
  showMarker,
}: {
  result: RenderResult | null;
  mode: "ct" | "mask" | "overlay" | "clean";
  window: { level: number; width: number };
  className?: string;
  label?: string;
  showMarker?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !result) return;
    c.width = result.size;
    c.height = result.size;
    const ctx = c.getContext("2d")!;
    ctx.putImageData(toImageData(result, mode, win), 0, 0);
    if (showMarker && result.noduleCenterPx && result.noduleRadiusPxInSlice > 0) {
      const [x, y] = result.noduleCenterPx;
      const r = Math.max(6, result.noduleRadiusPxInSlice * 1.8);
      ctx.strokeStyle = "rgba(250,178,25,0.9)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.arc(x, y, r, 0, Math.PI * 2);
      ctx.stroke();
    }
  }, [result, mode, win, showMarker]);
  return (
    <figure className={`relative overflow-hidden rounded-lg border border-line bg-black ${className ?? ""}`}>
      <canvas ref={ref} className="block h-auto w-full [image-rendering:auto]" aria-label={label} role="img" />
      {label && (
        <figcaption className="absolute left-2 top-2 rounded bg-black/60 px-1.5 py-0.5 font-mono text-[10px] uppercase tracking-wide text-ink-2">
          {label}
        </figcaption>
      )}
    </figure>
  );
}
