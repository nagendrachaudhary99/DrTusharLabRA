"use client";

import { useMemo, useState } from "react";
import { useSearchParams } from "next/navigation";
import { Crosshair, Shuffle } from "lucide-react";
import { CTCanvas } from "@/components/ct-canvas";
import { VolumePanel } from "./volume-panel";
import { renderSlice, placeNodule, WINDOWS, Z_EXTENT_MM, LABEL_COLORS, LABELS } from "@/lib/phantom/render";
import { detectability, noiseSigmaHU, NODULE_HU, LUNG_HU } from "@/lib/trial/physics";
import { detectionLogit } from "@/lib/trial/simulate";
import {
  DOSES,
  KERNELS,
  LOBES,
  LOBE_NAMES,
  MODEL_PROFILES,
  MODEL_PROFILE_INFO,
  NODULE_TYPES,
  VENDORS,
  type ScannerArm,
  type TrialSpec,
} from "@/lib/trial/spec";

type Lobe = TrialSpec["lobes"][number];
type NType = TrialSpec["noduleTypes"][number];

export function StudioClient() {
  const sp = useSearchParams();
  const [diameter, setDiameter] = useState(+(sp.get("d") ?? 10));
  const [type, setType] = useState<NType>((sp.get("type") as NType) ?? "part-solid");
  const [lobe, setLobe] = useState<Lobe>((sp.get("lobe") as Lobe) ?? "RUL");
  const [juxta, setJuxta] = useState(sp.get("jv") === "1");
  const [spiculated, setSpiculated] = useState(sp.get("spic") === "1");
  const [pos, setPos] = useState<[number, number, number]>((sp.get("pos")?.split(",").map(Number) as [number, number, number]) ?? [0.42, 0.6, 0.5]);
  const [seed, setSeed] = useState(+(sp.get("seed") ?? 7));
  const [bmi, setBmi] = useState(+(sp.get("bmi") ?? 26));
  const [scanner, setScanner] = useState<ScannerArm>({
    vendor: (sp.get("vendor") as ScannerArm["vendor"]) ?? "Siemens",
    kernel: (sp.get("kernel") as ScannerArm["kernel"]) ?? "standard",
    dose: (sp.get("dose") as ScannerArm["dose"]) ?? "standard",
  });
  const [win, setWin] = useState<keyof typeof WINDOWS>("lung");
  const [view, setView] = useState<"2d" | "3d">(sp.get("view") === "3d" ? "3d" : "2d");

  const nodule = useMemo(() => ({ diameterMm: diameter, type, lobe, juxtavascular: juxta, spiculated, pos }), [diameter, type, lobe, juxta, spiculated, pos]);
  const nominalZ = useMemo(() => placeNodule(lobe, pos)[2] * Z_EXTENT_MM, [lobe, pos]);

  // Find the true centre (juxtavascular placement may shift it) and jump to it whenever the lesion moves.
  const centerZ = useMemo(
    () => renderSlice({ size: 64, fovMm: 380, zMm: nominalZ, seed, bmi, nodule, scanner }).noduleCenterZmm ?? nominalZ,
    [nominalZ, seed, bmi, nodule, scanner],
  );
  // Manual slice choice is remembered per lesion position; moving the lesion re-centres on it.
  const [manualZ, setManualZ] = useState<{ center: number; z: number } | null>(null);
  const zMm = manualZ && manualZ.center === centerZ ? manualZ.z : Math.round(centerZ * 2) / 2;
  const setZ = (z: number) => setManualZ({ center: centerZ, z });

  const result = useMemo(() => renderSlice({ size: 384, fovMm: 380, zMm, seed, bmi, nodule, scanner }), [zMm, seed, bmi, nodule, scanner]);

  const compareArms: ScannerArm[] = [
    { vendor: "Siemens", kernel: "standard", dose: "standard" },
    { vendor: "GE", kernel: "soft", dose: "low" },
    { vendor: "Philips", kernel: "sharp", dose: "standard" },
    { vendor: "Canon", kernel: "sharp", dose: "ultra-low" },
  ];
  const compare = useMemo(
    () => compareArms.map((arm) => ({ arm, r: renderSlice({ size: 224, fovMm: 380, zMm, seed, bmi, nodule, scanner: arm }) })),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [zMm, seed, bmi, nodule],
  );

  const metrics = (arm: ScannerArm) => {
    const sigma = noiseSigmaHU(arm, bmi);
    const contrast = NODULE_HU[type] - LUNG_HU;
    const { cnr, dprime } = detectability({ diameterMm: diameter, contrastHU: contrast, sigmaHU: sigma, kernel: arm.kernel });
    const probs = MODEL_PROFILES.map((m) => ({
      m,
      p: 1 / (1 + Math.exp(-detectionLogit(m, arm, { dprime, type, juxtavascular: juxta, diameterMm: diameter }))),
    }));
    return { sigma, cnr, dprime, probs };
  };
  const m = metrics(scanner);

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      <header className="space-y-1">
        <p className="font-mono text-xs uppercase tracking-widest text-accent">Stage 3 · Anatomy-aware mask insertion → synthetic CT</p>
        <h1 className="text-2xl font-semibold tracking-tight">Virtual Patient Studio</h1>
        <p className="max-w-3xl text-sm text-ink-2">
          Design a lung tumour to the millimetre, place it in a specific lobe, and image the same virtual patient on different scanners. The anatomy label map
          is exactly the kind of conditioning input a mask-conditioned generator (ControlNet / NVIDIA MAISI) consumes. Here a procedural phantom stands in for
          the diffusion model so everything runs live in the browser.
        </p>
      </header>

      <section className="grid gap-4 lg:grid-cols-[300px_1fr]">
        <aside className="card space-y-4 p-4 text-xs">
          <h2 className="text-sm font-medium">Lesion design</h2>
          <Slider label="Diameter" unit="mm" min={3} max={30} step={0.5} value={diameter} onChange={setDiameter} />
          <Select label="Nodule type" value={type} options={NODULE_TYPES} onChange={(v) => setType(v as NType)} />
          <Select label="Lobe" value={lobe} options={LOBES} format={(l) => `${l} — ${LOBE_NAMES[l as Lobe]}`} onChange={(v) => setLobe(v as Lobe)} />
          <div className="flex gap-4">
            <Toggle label="Juxtavascular" value={juxta} onChange={setJuxta} />
            <Toggle label="Spiculated" value={spiculated} onChange={setSpiculated} />
          </div>
          <button
            onClick={() => setPos([Math.random(), Math.random(), Math.random()])}
            className="flex items-center gap-1.5 rounded-md border border-line px-2.5 py-1 hover:border-accent"
          >
            <Shuffle className="h-3.5 w-3.5" /> Random position in lobe
          </button>

          <h2 className="pt-2 text-sm font-medium">Virtual patient</h2>
          <Slider label="BMI" unit="" min={17} max={42} step={0.5} value={bmi} onChange={setBmi} />
          <Slider label="Anatomy seed" unit="" min={1} max={200} step={1} value={seed} onChange={setSeed} />

          <h2 className="pt-2 text-sm font-medium">CT acquisition</h2>
          <Select label="Vendor" value={scanner.vendor} options={VENDORS} onChange={(v) => setScanner({ ...scanner, vendor: v as ScannerArm["vendor"] })} />
          <Select
            label="Reconstruction kernel"
            value={scanner.kernel}
            options={KERNELS}
            onChange={(v) => setScanner({ ...scanner, kernel: v as ScannerArm["kernel"] })}
          />
          <Select label="Dose" value={scanner.dose} options={DOSES} onChange={(v) => setScanner({ ...scanner, dose: v as ScannerArm["dose"] })} />
          <Select
            label="Display window"
            value={win}
            options={Object.keys(WINDOWS) as (keyof typeof WINDOWS)[]}
            format={(k) => WINDOWS[k as keyof typeof WINDOWS].label}
            onChange={(v) => setWin(v as keyof typeof WINDOWS)}
          />
        </aside>

        <div className="space-y-4">
          <div className="card p-4">
            <div role="tablist" className="mb-3 flex gap-1 border-b border-line text-xs">
              {(
                [
                  ["2d", "2D slice + conditioning mask"],
                  ["3d", "3D volume (NiiVue)"],
                ] as const
              ).map(([k, label]) => (
                <button
                  key={k}
                  role="tab"
                  aria-selected={view === k}
                  onClick={() => setView(k)}
                  className={`-mb-px border-b-2 px-3 py-1.5 ${view === k ? "border-accent text-ink" : "border-transparent text-ink-2 hover:text-ink"}`}
                >
                  {label}
                </button>
              ))}
            </div>
            {view === "3d" ? (
              <VolumePanel seed={seed} bmi={bmi} nodule={nodule} scanner={scanner} />
            ) : (
              <>
                <div className="mb-3 flex flex-wrap items-center gap-3 text-xs">
                  <span className="text-ink-2">Slice z</span>
                  <input
                    type="range"
                    min={0}
                    max={Z_EXTENT_MM}
                    step={0.5}
                    value={zMm}
                    onChange={(e) => setZ(+e.target.value)}
                    className="flex-1"
                    aria-label="Slice position"
                  />
                  <span className="w-16 font-mono text-ink">{zMm.toFixed(1)} mm</span>
                  <button
                    onClick={() => setZ(Math.round(centerZ * 2) / 2)}
                    className="flex items-center gap-1 rounded-md border border-line px-2 py-1 hover:border-accent"
                  >
                    <Crosshair className="h-3.5 w-3.5" /> Nodule centre
                  </button>
                </div>
                <div className="grid gap-3 md:grid-cols-3">
                  <CTCanvas result={result} mode="mask" window={WINDOWS[win]} label="① Conditioning mask" />
                  <CTCanvas result={result} mode="ct" window={WINDOWS[win]} label="② Synthetic CT" />
                  <CTCanvas result={result} mode="overlay" window={WINDOWS[win]} label="③ Ground-truth overlay" />
                </div>
                <div className="mt-3 flex flex-wrap gap-3 text-[11px] text-ink-2">
                  {(["RUL", "RML", "RLL", "LUL", "LLL"] as const).map((l) => (
                    <span key={l} className="flex items-center gap-1.5">
                      <span className="inline-block h-2.5 w-2.5 rounded-sm" style={{ background: `rgb(${LABEL_COLORS[LABELS[l]].join(",")})` }} />
                      {l}
                    </span>
                  ))}
                  <span className="flex items-center gap-1.5">
                    <span className="inline-block h-2.5 w-2.5 rounded-sm bg-white" /> inserted nodule
                  </span>
                  <span className="text-ink-3">
                    · in-slice radius {(result.noduleRadiusPxInSlice * (380 / 384)).toFixed(1)} mm · centre z {result.noduleCenterZmm?.toFixed(1)} mm
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="grid gap-4 md:grid-cols-2">
            <div className="card p-4 text-xs">
              <h2 className="mb-3 text-sm font-medium">Imaging physics (this acquisition)</h2>
              <dl className="grid grid-cols-3 gap-3">
                <Metric k="Noise σ" v={`${m.sigma.toFixed(1)} HU`} />
                <Metric k="CNR (vs clutter)" v={m.cnr.toFixed(2)} />
                <Metric k="Detectability d′" v={m.dprime.toFixed(1)} />
              </dl>
              <p className="mt-3 text-ink-3">
                σ ∝ vendor baseline × kernel factor × 1/√dose × (BMI/25)^1.2. d′ follows a Rose-model with partial-volume loss and anatomical clutter.
              </p>
            </div>
            <div className="card p-4 text-xs">
              <h2 className="mb-3 text-sm font-medium">Predicted detection probability by AI model</h2>
              <ul className="space-y-2">
                {m.probs.map(({ m: k, p }) => (
                  <li key={k}>
                    <div className="mb-1 flex justify-between">
                      <span className="text-ink-2">{MODEL_PROFILE_INFO[k].name}</span>
                      <span className="font-mono text-ink">{(p * 100).toFixed(0)}%</span>
                    </div>
                    <div className="h-1.5 rounded bg-surface-2">
                      <div className="h-1.5 rounded" style={{ width: `${p * 100}%`, background: "var(--series-1)" }} />
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </div>

          <div className="card p-4">
            <h2 className="mb-1 text-sm font-medium">Same patient, same tumour, four scanners</h2>
            <p className="mb-3 text-xs text-ink-3">Exactly the controlled comparison that real retrospective data cannot give you.</p>
            <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
              {compare.map(({ arm, r }) => {
                const mm = metrics(arm);
                return (
                  <div key={arm.vendor + arm.kernel + arm.dose} className="space-y-1">
                    <CTCanvas result={r} mode="ct" window={WINDOWS[win]} label={arm.vendor} />
                    <div className="text-[11px] text-ink-2">
                      {arm.kernel} kernel · {arm.dose} dose
                    </div>
                    <div className="font-mono text-[10px] text-ink-3">
                      σ {mm.sigma.toFixed(0)} HU · d′ {mm.dprime.toFixed(0)} · P(det | single-site) {(mm.probs[1].p * 100).toFixed(0)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}

function Slider({
  label,
  unit,
  min,
  max,
  step,
  value,
  onChange,
}: {
  label: string;
  unit: string;
  min: number;
  max: number;
  step: number;
  value: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="block space-y-1">
      <span className="flex justify-between text-ink-2">
        {label}
        <span className="font-mono text-ink">
          {value}
          {unit && ` ${unit}`}
        </span>
      </span>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(+e.target.value)} className="w-full" />
    </label>
  );
}

function Select<T extends string>({
  label,
  value,
  options,
  onChange,
  format,
}: {
  label: string;
  value: T;
  options: readonly T[];
  onChange: (v: string) => void;
  format?: (v: T) => string;
}) {
  return (
    <label className="block space-y-1">
      <span className="text-ink-2">{label}</span>
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full rounded-md border border-line bg-background px-2 py-1.5 text-ink outline-none focus:border-accent"
      >
        {options.map((o) => (
          <option key={o} value={o}>
            {format ? format(o) : o}
          </option>
        ))}
      </select>
    </label>
  );
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <label className="flex cursor-pointer items-center gap-2 text-ink-2">
      <input type="checkbox" checked={value} onChange={(e) => onChange(e.target.checked)} className="accent-[var(--accent)]" />
      {label}
    </label>
  );
}

const Metric = ({ k, v }: { k: string; v: string }) => (
  <div>
    <dt className="text-ink-3">{k}</dt>
    <dd className="font-mono text-base text-ink">{v}</dd>
  </div>
);
