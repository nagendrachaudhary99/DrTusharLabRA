"use client";

import dynamic from "next/dynamic";

/** NiiVue touches window / WebGL2 at import time, so it is only ever loaded in the browser. */
export const VolumeViewer = dynamic(() => import("./volume-viewer"), {
  ssr: false,
  loading: () => <div className="flex h-[520px] items-center justify-center rounded-lg border border-line bg-black text-xs text-ink-3">Loading NiiVue…</div>,
});
