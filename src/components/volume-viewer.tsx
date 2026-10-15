"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Niivue, NVImage, SHOW_RENDER } from "@niivue/niivue";
import { Crosshair } from "lucide-react";
import { LABEL_COLORS, LABELS, WINDOWS } from "@/lib/phantom/render";
import type { PhantomVolume } from "@/lib/phantom/volume";

/**
 * NiiVue (WebGL2) viewer for the stacked phantom volume: multiplanar + 3D render,
 * with the nodule mask and the anatomy label map ("conditioning mask") as overlays.
 * Must be loaded with next/dynamic({ ssr: false }) — see volume-viewer-lazy.tsx.
 */

const LABEL_NAMES: Record<number, string> = Object.fromEntries(Object.entries(LABELS).map(([k, v]) => [v, k]));

type LUT = NonNullable<NVImage["colormapLabel"]>;
type Layout = "multi" | "render" | "axial";
type Win = "lung" | "mediastinum";

const DT_UINT8 = 2;
const DT_INT16 = 4;

/** Discrete LUT for the anatomy labels: lobes in their categorical colours, everything else transparent. */
function anatomyLut(): LUT {
  const max = LABELS.nodule;
  const lut = new Uint8ClampedArray((max + 1) * 4);
  const show: number[] = [LABELS.RUL, LABELS.RML, LABELS.RLL, LABELS.LUL, LABELS.LLL, LABELS.nodule, LABELS.heart, LABELS.bone];
  for (const l of show) {
    const [r, g, b] = l === LABELS.nodule ? [255, 60, 60] : LABEL_COLORS[l];
    lut.set([r, g, b, 255], l * 4);
  }
  return { lut, min: 0, max };
}

async function toNVImage(vol: PhantomVolume, data: Int16Array | Uint8Array, name: string, opts: Partial<Parameters<typeof NVImage.loadFromUrl>[0]>) {
  const bytes = NVImage.createNiftiArray(vol.dims, vol.spacingMm, vol.affine, data instanceof Int16Array ? DT_INT16 : DT_UINT8, data);
  return NVImage.loadFromUrl({ url: name, name, buffer: bytes.buffer as ArrayBuffer, ...opts });
}

function voxToMm(vol: PhantomVolume, [i, j, k]: [number, number, number]): [number, number, number] {
  const a = vol.affine;
  return [a[0] * i + a[1] * j + a[2] * k + a[3], a[4] * i + a[5] * j + a[6] * k + a[7], a[8] * i + a[9] * j + a[10] * k + a[11]];
}

export default function VolumeViewer({ volume }: { volume: PhantomVolume | null }) {
  const hostRef = useRef<HTMLDivElement>(null);
  const nvRef = useRef<Niivue | null>(null);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(() =>
    typeof WebGL2RenderingContext === "undefined" ? "This browser does not support WebGL2, which NiiVue needs for the 3D view." : null,
  );
  const [loaded, setLoaded] = useState<PhantomVolume | null>(null);
  const [layout, setLayout] = useState<Layout>("multi");
  const [win, setWin] = useState<Win>("lung");
  const [showMask, setShowMask] = useState(true);
  const [showLabels, setShowLabels] = useState(false);
  const [cutaway, setCutaway] = useState(true);
  const [location, setLocation] = useState("");

  // Create the canvas imperatively so a StrictMode remount never re-attaches to a context we have already lost.
  useEffect(() => {
    const host = hostRef.current;
    if (!host || error) return;
    let disposed = false;
    const canvas = document.createElement("canvas");
    canvas.className = "block h-full w-full outline-none";
    host.appendChild(canvas);
    const nv = new Niivue({
      backColor: [0.02, 0.03, 0.05, 1],
      crosshairColor: [0.98, 0.7, 0.1, 0.9],
      show3Dcrosshair: true,
      isColorbar: false,
      isOrientCube: false,
      isRadiologicalConvention: true, // match the 2D studio view (patient right on screen left)
      multiplanarShowRender: SHOW_RENDER.ALWAYS,
      loadingText: "",
    });
    nv.onLocationChange = (loc) => {
      const l = loc as { mm?: ArrayLike<number>; values?: { value: number }[] } | null;
      if (!l?.mm || !l.values?.length) return;
      const [hu, lab, nod] = l.values.map((v) => Math.round(v.value));
      const mm = Array.from(l.mm.length > 3 ? Array.from(l.mm).slice(0, 3) : l.mm, (v) => v.toFixed(0)).join(", ");
      setLocation(`RAS (${mm}) mm · ${hu} HU · ${LABEL_NAMES[lab] ?? "—"}${nod ? " · inside nodule mask" : ""}`);
    };
    nv.attachToCanvas(canvas)
      .then(() => {
        if (disposed) return;
        nvRef.current = nv;
        setReady(true);
      })
      .catch((e: unknown) => {
        if (!disposed) setError(`Could not start WebGL2 viewer: ${e instanceof Error ? e.message : String(e)}`);
      });
    return () => {
      disposed = true;
      nvRef.current = null;
      setReady(false);
      try {
        nv.cleanup();
        nv.gl?.getExtension("WEBGL_lose_context")?.loseContext();
      } catch {
        /* never attached */
      }
      canvas.remove();
    };
  }, [error]);

  const jumpToNodule = useCallback(() => {
    const nv = nvRef.current;
    if (!nv || !loaded?.noduleCenterVox || !nv.volumes.length) return;
    const mm = voxToMm(loaded, loaded.noduleCenterVox);
    nv.scene.crosshairPos = nv.mm2frac([mm[0], mm[1], mm[2]]);
    nv.drawScene();
    nv.createOnLocationChange();
  }, [loaded]);

  // (Re)load the three volumes whenever a new phantom volume arrives.
  useEffect(() => {
    const nv = nvRef.current;
    if (!ready || !nv || !volume) return;
    let stale = false;
    (async () => {
      const w = WINDOWS[win];
      const [ct, labels, mask] = await Promise.all([
        toNVImage(volume, volume.ct, "ct.nii", { colormap: "gray", cal_min: w.level - w.width / 2, cal_max: w.level + w.width / 2 }),
        toNVImage(volume, volume.labels, "labels.nii", { colormap: "gray", opacity: showLabels ? 0.45 : 0 }),
        toNVImage(volume, volume.noduleMask, "nodule.nii", { colormap: "red", cal_min: 0, cal_max: 1, opacity: showMask ? 0.85 : 0 }),
      ]);
      if (stale || nvRef.current !== nv) return;
      labels.colormapLabel = anatomyLut(); // loadFromUrl ignores its colormapLabel option in 0.69
      for (const v of [...nv.volumes].reverse()) nv.removeVolume(v);
      nv.addVolume(ct);
      nv.addVolume(labels);
      nv.addVolume(mask);
      setLoaded(volume);
    })().catch((e: unknown) => setError(`NiiVue failed to load the volume: ${e instanceof Error ? e.message : String(e)}`));
    return () => {
      stale = true;
    };
    // Display state is applied by the effects below; only a new volume should trigger a reload.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ready, volume]);

  // Centre the crosshair on the nodule each time a new volume finishes loading.
  useEffect(() => {
    jumpToNodule();
  }, [jumpToNodule]);

  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !loaded || !nv.volumes[0]) return;
    const w = WINDOWS[win];
    nv.volumes[0].cal_min = w.level - w.width / 2;
    nv.volumes[0].cal_max = w.level + w.width / 2;
    nv.updateGLVolume();
  }, [win, loaded]);

  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !loaded || nv.volumes.length < 3) return;
    nv.setOpacity(1, showLabels ? 0.45 : 0);
    nv.setOpacity(2, showMask ? 0.85 : 0);
  }, [showLabels, showMask, loaded]);

  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !ready) return;
    if (layout === "multi") {
      nv.setMultiplanarLayout(2); // 2×2 grid: axial / coronal / sagittal / 3D
      nv.setSliceType(nv.sliceTypeMultiplanar);
    } else if (layout === "render") nv.setSliceType(nv.sliceTypeRender);
    else nv.setSliceType(nv.sliceTypeAxial);
  }, [layout, ready]);

  useEffect(() => {
    const nv = nvRef.current;
    if (!nv || !ready) return;
    // depth, azimuth, elevation — a coronal cut through the chest exposes the lungs and the lesion; depth > 1.8 disables.
    nv.setClipPlane(cutaway ? [0.05, 180, 0] : [2, 0, 0]);
    nv.setRenderAzimuthElevation(cutaway ? 200 : 120, 15);
  }, [cutaway, ready]);

  if (error)
    return (
      <div className="flex h-[480px] items-center justify-center rounded-lg border border-line bg-black/40 p-6 text-center text-xs text-ink-2">
        {error} The NIfTI downloads below still work — open them in 3D Slicer or ITK-SNAP.
      </div>
    );

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-center gap-2 text-[11px]">
        <Seg
          value={layout}
          onChange={setLayout}
          options={[
            ["multi", "Multiplanar + 3D"],
            ["render", "3D only"],
            ["axial", "Axial only"],
          ]}
        />
        <Seg
          value={win}
          onChange={setWin}
          options={[
            ["lung", "Lung W1500/L-600"],
            ["mediastinum", "Mediastinal W400/L40"],
          ]}
        />
        <Check label="Nodule mask" value={showMask} onChange={setShowMask} />
        <Check label="Anatomy labels" value={showLabels} onChange={setShowLabels} />
        <Check label="3D cut-away" value={cutaway} onChange={setCutaway} />
        <button
          onClick={jumpToNodule}
          disabled={!loaded?.noduleCenterVox}
          className="ml-auto flex items-center gap-1 rounded-md border border-line px-2 py-1 hover:border-accent disabled:opacity-40"
        >
          <Crosshair className="h-3.5 w-3.5" /> Nodule centre
        </button>
      </div>
      <div className="relative h-[520px] overflow-hidden rounded-lg border border-line bg-black">
        <div ref={hostRef} className="absolute inset-0" />
        {(!ready || !loaded) && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-xs text-ink-3">
            {ready ? "Uploading volume to GPU…" : "Starting WebGL2…"}
          </div>
        )}
      </div>
      <div className="flex justify-between gap-3 font-mono text-[10px] text-ink-3">
        <span className="truncate">{location || "Click / drag in any view to move the crosshair · right-drag in 3D to rotate · scroll to change slice"}</span>
        {loaded && (
          <span className="shrink-0">
            {loaded.dims.join("×")} vox · {loaded.spacingMm.map((s) => s.toFixed(2)).join("×")} mm
          </span>
        )}
      </div>
    </div>
  );
}

function Seg<T extends string>({ value, onChange, options }: { value: T; onChange: (v: T) => void; options: [T, string][] }) {
  return (
    <div className="flex overflow-hidden rounded-md border border-line">
      {options.map(([k, label]) => (
        <button
          key={k}
          onClick={() => onChange(k)}
          className={`px-2 py-1 ${value === k ? "bg-accent/20 text-ink" : "text-ink-2 hover:bg-surface-2"}`}
          aria-pressed={value === k}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function Check({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-1.5 text-ink-2">
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} className="accent-[var(--accent)]" />
      {label}
    </label>
  );
}
