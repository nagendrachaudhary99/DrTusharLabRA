import { LABELS, Z_EXTENT_MM, renderSlice, type PhantomParams } from "@/lib/phantom/render";

/**
 * 3D virtual-patient volume built by stacking renderSlice() outputs along z.
 *
 * Voxel ordering is x-fastest (i = column, j = row, k = slice), i.e. standard NIfTI order.
 * Image orientation matches the 2D studio view (radiological): column → patient left,
 * row → posterior, slice → inferior (k = 0 is the apex). The affine below encodes that in
 * RAS+ world space so 3D Slicer / ITK-SNAP / NiiVue display it correctly.
 */
export type PhantomVolume = {
  dims: [number, number, number];
  spacingMm: [number, number, number];
  /** Row-major 4×4 voxel → RAS mm affine (16 numbers, as NiiVue's createNiftiArray expects). */
  affine: number[];
  ct: Int16Array; // simulated CT, HU
  labels: Uint8Array; // anatomy label map incl. inserted nodule (LABELS)
  noduleMask: Uint8Array; // 1 = nodule voxel
  noduleVoxels: number;
  /** Nodule centre in (continuous) voxel coordinates, or null when no nodule. */
  noduleCenterVox: [number, number, number] | null;
  huRange: [number, number];
  buildMs: number;
};

export type VolumeOptions = {
  size?: number; // in-plane matrix (square)
  sliceMm?: number; // slice spacing
  fovMm?: number;
  /** Called after each chunk with progress 0..1. */
  onProgress?: (frac: number) => void;
  /** Abort a stale build (params changed). */
  signal?: AbortSignal;
  /** Slices rendered between yields to the event loop. */
  chunk?: number;
};

const yieldToEventLoop = () =>
  new Promise<void>((res) => {
    if (typeof requestIdleCallback === "function") requestIdleCallback(() => res(), { timeout: 50 });
    else setTimeout(res, 0);
  });

export async function buildVolume(params: Omit<PhantomParams, "size" | "zMm" | "fovMm">, opts: VolumeOptions = {}): Promise<PhantomVolume> {
  const t0 = performance.now();
  const n = opts.size ?? 128;
  const fovMm = opts.fovMm ?? 380;
  const dz = opts.sliceMm ?? 3;
  const nz = Math.round(Z_EXTENT_MM / dz);
  const dxy = fovMm / n;
  const plane = n * n;
  const ct = new Int16Array(plane * nz);
  const labels = new Uint8Array(plane * nz);
  const noduleMask = new Uint8Array(plane * nz);
  let noduleVoxels = 0;
  let lo = Infinity;
  let hi = -Infinity;
  let center: [number, number, number] | null = null;
  const chunk = opts.chunk ?? 8;

  for (let k = 0; k < nz; k++) {
    if (opts.signal?.aborted) throw new DOMException("Volume build aborted", "AbortError");
    const r = renderSlice({ ...params, size: n, fovMm, zMm: (k + 0.5) * dz });
    const off = k * plane;
    for (let p = 0; p < plane; p++) {
      const v = Math.max(-1024, Math.min(3071, Math.round(r.hu[p])));
      ct[off + p] = v;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      const l = r.labels[p];
      labels[off + p] = l;
      if (l === LABELS.nodule) {
        noduleMask[off + p] = 1;
        noduleVoxels++;
      }
    }
    if (!center && r.noduleCenterPx && r.noduleCenterZmm != null) center = [r.noduleCenterPx[0] - 0.5, r.noduleCenterPx[1] - 0.5, r.noduleCenterZmm / dz - 0.5];
    if ((k + 1) % chunk === 0 && k + 1 < nz) {
      opts.onProgress?.((k + 1) / nz);
      await yieldToEventLoop();
    }
  }
  opts.onProgress?.(1);

  // voxel (i,j,k) → RAS mm. i → patient left (−x), j → posterior (−y), k → inferior (−z); centred at the FOV centre.
  const affine = [-dxy, 0, 0, (dxy * (n - 1)) / 2, 0, -dxy, 0, (dxy * (n - 1)) / 2, 0, 0, -dz, (dz * (nz - 1)) / 2, 0, 0, 0, 1];

  return {
    dims: [n, n, nz],
    spacingMm: [dxy, dxy, dz],
    affine,
    ct,
    labels,
    noduleMask,
    noduleVoxels,
    noduleCenterVox: center,
    huRange: [lo, hi],
    buildMs: performance.now() - t0,
  };
}

/**
 * Minimal NIfTI-1 (.nii, single file) writer — no WebGL / NiiVue dependency, so downloads work everywhere.
 * Writes both qform (code 1) and sform (code 1) so 3D Slicer, ITK-SNAP, FSL and nibabel agree on orientation.
 */
export function toNifti1(vol: PhantomVolume, which: "ct" | "labels" | "nodule"): Blob {
  const data = which === "ct" ? vol.ct : which === "labels" ? vol.labels : vol.noduleMask;
  const [nx, ny, nz] = vol.dims;
  const [dx, dy, dz] = vol.spacingMm;
  const hdr = new ArrayBuffer(352);
  const v = new DataView(hdr);
  const le = true;
  v.setInt32(0, 348, le); // sizeof_hdr
  [3, nx, ny, nz, 1, 1, 1, 1].forEach((d, i) => v.setInt16(40 + 2 * i, d, le));
  const isCT = which === "ct";
  v.setInt16(70, isCT ? 4 : 2, le); // datatype: INT16 / UINT8
  v.setInt16(72, isCT ? 16 : 8, le); // bitpix
  // pixdim[0] = qfac. Our rotation is diag(-1,-1,-1) (det −1) → qfac −1, quaternion (0,0,1) = 180° about z.
  [-1, dx, dy, dz, 1, 1, 1, 1].forEach((d, i) => v.setFloat32(76 + 4 * i, d, le));
  v.setFloat32(108, 352, le); // vox_offset
  v.setFloat32(112, 1, le); // scl_slope
  v.setFloat32(116, 0, le); // scl_inter
  v.setUint8(123, 2); // xyzt_units: mm
  if (isCT) {
    v.setFloat32(124, 150, le); // cal_max (lung window)
    v.setFloat32(128, -1350, le); // cal_min
  } else {
    v.setFloat32(124, which === "labels" ? LABELS.nodule : 1, le);
  }
  const desc = `iTRIALSPACE phantom ${which} (procedural, not patient data)`.slice(0, 79);
  for (let i = 0; i < desc.length; i++) v.setUint8(148 + i, desc.charCodeAt(i));
  if (!isCT) v.setInt16(68, 1002, le); // intent_code NIFTI_INTENT_LABEL
  v.setInt16(252, 1, le); // qform_code: scanner
  v.setInt16(254, 1, le); // sform_code: scanner
  v.setFloat32(256, 0, le); // quatern_b
  v.setFloat32(260, 0, le); // quatern_c
  v.setFloat32(264, 1, le); // quatern_d
  const a = vol.affine;
  v.setFloat32(268, a[3], le);
  v.setFloat32(272, a[7], le);
  v.setFloat32(276, a[11], le);
  for (let r = 0; r < 3; r++) for (let c = 0; c < 4; c++) v.setFloat32(280 + 16 * r + 4 * c, a[r * 4 + c], le);
  "n+1\0".split("").forEach((ch, i) => v.setUint8(344 + i, ch.charCodeAt(0)));
  // bytes 348–351: extension flag = 0. Typed arrays are little-endian on every browser platform.
  return new Blob([hdr, new Uint8Array(data.buffer as ArrayBuffer, data.byteOffset, data.byteLength)], { type: "application/octet-stream" });
}
