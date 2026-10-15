"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Box, Download } from "lucide-react";
import { VolumeViewer } from "@/components/volume-viewer-lazy";
import { buildVolume, toNifti1, type PhantomVolume } from "@/lib/phantom/volume";
import type { PhantomParams } from "@/lib/phantom/render";

const RESOLUTIONS = {
  fast: { size: 128, sliceMm: 3, label: "Fast · 128² × 87 @ 3 mm" },
  standard: { size: 160, sliceMm: 2.5, label: "Standard · 160² × 104 @ 2.5 mm" },
  fine: { size: 192, sliceMm: 2, label: "Fine · 192² × 130 @ 2 mm" },
} as const;
type Res = keyof typeof RESOLUTIONS;

type Props = Pick<PhantomParams, "seed" | "bmi" | "nodule" | "scanner">;

export function VolumePanel({ seed, bmi, nodule, scanner }: Props) {
  const [res, setRes] = useState<Res>("standard");
  const [auto, setAuto] = useState(true);
  const [volume, setVolume] = useState<PhantomVolume | null>(null);
  const [builtKey, setBuiltKey] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const key = useMemo(() => JSON.stringify({ seed, bmi, nodule, scanner, res }), [seed, bmi, nodule, scanner, res]);

  const build = useCallback(async () => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setProgress(0);
    try {
      const v = await buildVolume({ seed, bmi, nodule, scanner }, { ...RESOLUTIONS[res], signal: ac.signal, onProgress: (f) => setProgress(f) });
      if (ac.signal.aborted) return;
      setVolume(v);
      setBuiltKey(key);
    } catch (e) {
      if ((e as Error).name !== "AbortError") throw e;
    } finally {
      if (abortRef.current === ac) setProgress(null);
    }
  }, [seed, bmi, nodule, scanner, res, key]);

  // Debounced auto-rebuild while sliders move.
  useEffect(() => {
    if (!auto) return;
    const t = setTimeout(() => void build(), 450);
    return () => clearTimeout(t);
  }, [auto, build]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const download = (which: "ct" | "labels" | "nodule") => {
    if (!volume) return;
    const url = URL.createObjectURL(toNifti1(volume, which));
    const a = document.createElement("a");
    a.href = url;
    a.download = `itrialspace_seed${seed}_${which === "ct" ? `ct_${scanner.vendor}_${scanner.kernel}_${scanner.dose}` : which === "nodule" ? "nodule_mask" : "anatomy_labels"}.nii`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 2000);
  };

  const voxMm3 = volume ? volume.spacingMm[0] * volume.spacingMm[1] * volume.spacingMm[2] : 0;
  const sphereMm3 = nodule ? (4 / 3) * Math.PI * (nodule.diameterMm / 2) ** 3 : 0;
  const stale = builtKey !== key;

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs">
        <label className="flex items-center gap-1.5 text-ink-2">
          Grid
          <select
            value={res}
            onChange={(e) => setRes(e.target.value as Res)}
            className="rounded-md border border-line bg-background px-2 py-1 text-ink outline-none focus:border-accent"
          >
            {(Object.keys(RESOLUTIONS) as Res[]).map((r) => (
              <option key={r} value={r}>
                {RESOLUTIONS[r].label}
              </option>
            ))}
          </select>
        </label>
        <label className="flex cursor-pointer items-center gap-1.5 text-ink-2">
          <input type="checkbox" checked={auto} onChange={(e) => setAuto(e.target.checked)} className="accent-[var(--accent)]" />
          Auto-rebuild on change
        </label>
        <button
          onClick={() => void build()}
          disabled={progress !== null}
          className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1 hover:border-accent disabled:opacity-50"
        >
          <Box className="h-3.5 w-3.5" /> Build 3D volume
        </button>
        {progress !== null ? (
          <span className="flex items-center gap-2 text-ink-3">
            <span className="h-1.5 w-24 rounded bg-surface-2">
              <span className="block h-1.5 rounded" style={{ width: `${progress * 100}%`, background: "var(--series-1)" }} />
            </span>
            stacking slices…
          </span>
        ) : (
          volume && (
            <span className={stale ? "text-amber-400" : "text-ink-3"}>
              {stale ? "Parameters changed — rebuild to update" : `built in ${volume.buildMs.toFixed(0)} ms`}
            </span>
          )
        )}
      </div>

      <VolumeViewer volume={volume} />

      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="text-ink-2">Download NIfTI-1 (.nii) for 3D Slicer / ITK-SNAP:</span>
        {(
          [
            ["ct", "Synthetic CT (HU, int16)"],
            ["nodule", "Nodule mask (uint8)"],
            ["labels", "Anatomy label map (uint8)"],
          ] as const
        ).map(([k, label]) => (
          <button
            key={k}
            onClick={() => download(k)}
            disabled={!volume}
            className="flex items-center gap-1 rounded-md border border-line px-2 py-1 hover:border-accent disabled:opacity-40"
          >
            <Download className="h-3.5 w-3.5" /> {label}
          </button>
        ))}
      </div>
      {volume && (
        <p className="text-[11px] text-ink-3">
          The volume is {volume.dims[2]} independently rendered axial slices stacked along z — the same pipeline as the 2D view, so the label map is the 3D
          conditioning mask a MAISI/ControlNet generator would consume. Nodule mask: {volume.noduleVoxels} voxels ≈ {(volume.noduleVoxels * voxMm3).toFixed(0)}{" "}
          mm³
          {nodule && ` vs ${sphereMm3.toFixed(0)} mm³ for an ideal ${nodule.diameterMm} mm sphere${nodule.spiculated ? " (spicules add volume)" : ""}`}. Label
          values: 1 body, 2 fat, 3 bone, 4 heart, 10–14 RUL/RML/RLL/LUL/LLL, 20 nodule.
        </p>
      )}
    </div>
  );
}
