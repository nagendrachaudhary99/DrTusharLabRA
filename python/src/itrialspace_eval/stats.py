"""Core statistics: Wilson CIs, patient-level cluster bootstrap, stratified sensitivity, FROC/CPM, DeLong AUC.

All randomness goes through :func:`numpy.random.default_rng` with an explicit seed, so every
result is bit-for-bit reproducible for a given (input, seed, n_boot).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import stats as sps

FloatArray = npt.NDArray[np.float64]

#: FP-per-scan operating points of the LUNA16 Competition Performance Metric.
CPM_FP_RATES: tuple[float, ...] = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

#: Default bin edges (mm) for the continuous ``diameterMm`` column; centred on the web app's
#: default size strata 4/6/8/10/15/20 mm.
DEFAULT_BINS: dict[str, list[float]] = {"diameterMm": [0.0, 5.0, 7.0, 9.0, 12.5, 17.5, float("inf")]}


def _z(alpha: float) -> float:
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return float(sps.norm.ppf(1.0 - alpha / 2.0))


# ───────────────────────────── Wilson ─────────────────────────────


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion k/n. Returns (lo, hi); (0, 0) if n == 0."""
    if n < 0 or k < 0 or k > n:
        raise ValueError(f"invalid counts k={k}, n={n}")
    if n == 0:
        return (0.0, 0.0)
    z = _z(alpha)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    lo = 0.0 if k == 0 else max(0.0, float(centre - half))
    hi = 1.0 if k == n else min(1.0, float(centre + half))
    return (lo, hi)


# ───────────────────────────── cluster bootstrap ─────────────────────────────


@dataclass(frozen=True)
class BootstrapCI:
    estimate: float
    lo: float
    hi: float
    n_boot: int
    seed: int


def cluster_bootstrap_sensitivity(
    detected: npt.ArrayLike,
    patient_ids: npt.ArrayLike,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> BootstrapCI:
    """Percentile CI for lesion-level sensitivity, resampling *patients* (clusters) with replacement.

    Lesions from the same patient share anatomy and acquisition, so treating them as independent
    (as the Wilson interval does) understates uncertainty; resampling whole patients does not.
    """
    det = np.asarray(detected, dtype=bool)
    pids = np.asarray(patient_ids)
    if det.shape != pids.shape:
        raise ValueError("detected and patient_ids must have the same length")
    if det.size == 0:
        return BootstrapCI(0.0, 0.0, 0.0, n_boot, seed)
    _, inv = np.unique(pids, return_inverse=True)
    n_per = np.bincount(inv).astype(np.float64)
    k_per = np.bincount(inv, weights=det.astype(np.float64))
    n_pat = n_per.size
    rng = np.random.default_rng(seed)
    # Each bootstrap replicate = multinomial counts of how often each patient is drawn.
    counts = rng.multinomial(n_pat, np.full(n_pat, 1.0 / n_pat), size=n_boot).astype(np.float64)
    sens = (counts @ k_per) / (counts @ n_per)
    lo, hi = np.quantile(sens, [alpha / 2, 1 - alpha / 2])
    return BootstrapCI(float(det.mean()), float(lo), float(hi), n_boot, seed)


# ───────────────────────────── stratification ─────────────────────────────


def _bin_labels(edges: Sequence[float]) -> list[str]:
    def fmt(x: float) -> str:
        return "inf" if np.isinf(x) else f"{x:g}"

    return [f"[{fmt(a)}, {fmt(b)})" for a, b in itertools.pairwise(edges)]


def stratum_column(
    df: pd.DataFrame, col: str, bins: Mapping[str, Sequence[float]] | None = None
) -> pd.Series:
    """Return a stratum label for ``col``; continuous columns are binned (ordered categorical).

    Bins come from ``bins`` (user), then :data:`DEFAULT_BINS`, then quartiles if the column is
    numeric with more than 12 distinct values; otherwise values are used as-is.
    """
    if col not in df.columns:
        raise KeyError(f"column {col!r} not in data (available: {list(df.columns)})")
    s = df[col]
    edges: Sequence[float] | None = None
    if bins and col in bins:
        edges = list(bins[col])
    elif col in DEFAULT_BINS:
        edges = DEFAULT_BINS[col]
    elif pd.api.types.is_numeric_dtype(s) and s.nunique() > 12 and s.dtype != bool:
        q = np.unique(np.quantile(s.dropna(), [0, 0.25, 0.5, 0.75, 1.0]))
        edges = [-np.inf, *q[1:-1].tolist(), np.inf]
    if edges is None:
        return s.astype(str)
    labels = _bin_labels(edges)
    # Ordered categorical so tables list bins numerically, not lexicographically.
    return pd.cut(s, bins=list(edges), labels=labels, right=False, ordered=True)


def stratified_sensitivity(
    df: pd.DataFrame,
    by: str | Sequence[str],
    *,
    bins: Mapping[str, Sequence[float]] | None = None,
    n_boot: int = 0,
    seed: int = 42,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Sensitivity per stratum of one or more columns (cross-classified if several).

    Columns: one per ``by`` column, ``n``, ``k``, ``n_patients``, ``sensitivity``, ``wilson_lo``,
    ``wilson_hi`` and, if ``n_boot > 0``, ``boot_lo``/``boot_hi`` (patient-cluster bootstrap,
    same seed for every stratum so results do not depend on stratum order).
    """
    cols = [by] if isinstance(by, str) else list(by)
    keyed = pd.DataFrame({c: stratum_column(df, c, bins) for c in cols})
    keyed["detected"] = df["detected"].to_numpy(dtype=bool)
    keyed["patientId"] = df["patientId"].to_numpy()
    rows: list[dict[str, Any]] = []
    for key, g in keyed.groupby(cols, sort=True, observed=True):
        key_t = key if isinstance(key, tuple) else (key,)
        n, k = len(g), int(g["detected"].sum())
        lo, hi = wilson_ci(k, n, alpha)
        row: dict[str, Any] = dict(zip(cols, key_t, strict=True))
        row.update(
            n=n, k=k, n_patients=int(g["patientId"].nunique()), sensitivity=k / n, wilson_lo=lo, wilson_hi=hi
        )
        if n_boot > 0:
            b = cluster_bootstrap_sensitivity(
                g["detected"].to_numpy(), g["patientId"].to_numpy(), n_boot=n_boot, seed=seed, alpha=alpha
            )
            row.update(boot_lo=b.lo, boot_hi=b.hi)
        rows.append(row)
    table: pd.DataFrame = pd.DataFrame(rows)
    return table


# ───────────────────────────── FROC / CPM ─────────────────────────────


@dataclass
class FrocResult:
    cpm: float
    points: dict[float, float]
    """Sensitivity at each CPM FP/scan rate."""
    curve_fp_per_scan: list[float] = field(repr=False)
    curve_sensitivity: list[float] = field(repr=False)
    n_lesions: int = 0
    n_scans: int = 0
    cpm_ci: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["points"] = {str(k): v for k, v in self.points.items()}
        return d


def _froc_core(
    lesion_scores: FloatArray, fp_scores: FloatArray, n_scans: int, fp_rates: Sequence[float]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    n_les = lesion_scores.size
    tp = np.sort(lesion_scores[~np.isnan(lesion_scores)])[::-1]
    fp = np.sort(fp_scores)[::-1]
    thr = np.unique(np.concatenate([tp, fp]))[::-1]
    # number of scores >= t, via searchsorted on ascending arrays
    tp_asc, fp_asc = tp[::-1], fp[::-1]
    n_tp = tp_asc.size - np.searchsorted(tp_asc, thr, side="left")
    n_fp = fp_asc.size - np.searchsorted(fp_asc, thr, side="left")
    sens = n_tp / n_les if n_les else np.zeros_like(thr)
    fpps = n_fp / n_scans
    pts = np.array([float(np.max(sens[fpps <= r])) if np.any(fpps <= r) else 0.0 for r in fp_rates])
    return fpps.astype(np.float64), np.asarray(sens, dtype=np.float64), pts


def froc(
    lesion_scores: Sequence[float] | FloatArray,
    fp_scores_per_scan: Mapping[str, Sequence[float]],
    *,
    lesion_patient_ids: Sequence[str] | None = None,
    fp_rates: Sequence[float] = CPM_FP_RATES,
    n_boot: int = 0,
    seed: int = 42,
    alpha: float = 0.05,
) -> FrocResult:
    """FROC curve and Competition Performance Metric (LUNA16 definition).

    Sensitivity at a target FP/scan rate is the highest sensitivity reached at any threshold whose
    FP/scan does not exceed the target (step / conservative reading, identical to the web app).
    NaN lesion scores count as never detected. ``fp_scores_per_scan`` maps every scan (patient)
    to its false-positive candidate scores -- scans with no candidates must still be present with
    an empty list, because they contribute to the FP/scan denominator.

    If ``n_boot > 0`` and ``lesion_patient_ids`` is given, a patient-level bootstrap percentile CI for
    the CPM is added (lesions and FP candidates of a resampled patient move together).
    """
    les = np.asarray(lesion_scores, dtype=np.float64)
    scans = list(fp_scores_per_scan.keys())
    if not scans:
        raise ValueError("need at least one scan for FP/scan normalisation")
    fp_all = (
        np.concatenate([np.asarray(fp_scores_per_scan[s], dtype=np.float64) for s in scans])
        if any(len(v) for v in fp_scores_per_scan.values())
        else np.zeros(0)
    )
    fpps, sens, pts = _froc_core(les, fp_all, len(scans), fp_rates)
    res = FrocResult(
        cpm=float(pts.mean()),
        points={float(r): float(p) for r, p in zip(fp_rates, pts, strict=True)},
        curve_fp_per_scan=fpps.tolist(),
        curve_sensitivity=sens.tolist(),
        n_lesions=int(les.size),
        n_scans=len(scans),
    )
    if n_boot > 0:
        if lesion_patient_ids is None:
            raise ValueError("lesion_patient_ids is required for the CPM bootstrap")
        pids = np.asarray(lesion_patient_ids)
        unknown = set(pids.tolist()) - set(scans)
        if unknown:
            raise ValueError(
                f"{len(unknown)} lesion patientId(s) missing from the FP table, e.g. {sorted(unknown)[:3]}"
            )
        les_by = {s: les[pids == s] for s in scans}
        fp_by = {s: np.asarray(fp_scores_per_scan[s], dtype=np.float64) for s in scans}
        rng = np.random.default_rng(seed)
        cpms = np.empty(n_boot)
        for b in range(n_boot):
            draw = [scans[i] for i in rng.integers(0, len(scans), size=len(scans))]
            bl = np.concatenate([les_by[s] for s in draw])
            bf = np.concatenate([fp_by[s] for s in draw])
            cpms[b] = _froc_core(bl, bf, len(draw), fp_rates)[2].mean() if bl.size else 0.0
        lo, hi = np.quantile(cpms, [alpha / 2, 1 - alpha / 2])
        res.cpm_ci = (float(lo), float(hi))
    return res


# ───────────────────────────── DeLong AUC ─────────────────────────────


@dataclass(frozen=True)
class AucResult:
    auc: float
    se: float
    lo: float
    hi: float


def auc_delong(
    positives: Sequence[float] | FloatArray, negatives: Sequence[float] | FloatArray, alpha: float = 0.05
) -> AucResult:
    """AUC (Mann-Whitney) with DeLong standard error and a normal-approximation CI clipped to [0, 1].

    In a detection setting positives are lesion scores and negatives are false-positive candidate
    scores; this is a candidate-level AUC and does not replace FROC.
    """
    x = np.asarray(positives, dtype=np.float64)
    y = np.asarray(negatives, dtype=np.float64)
    m, n = x.size, y.size
    if m < 2 or n < 2:
        raise ValueError("need at least 2 positives and 2 negatives")
    r_all = sps.rankdata(np.concatenate([x, y]))
    r_x, r_y = sps.rankdata(x), sps.rankdata(y)
    v10 = (r_all[:m] - r_x) / n
    v01 = 1.0 - (r_all[m:] - r_y) / m
    auc = float(v10.mean())
    se = float(np.sqrt(v10.var(ddof=1) / m + v01.var(ddof=1) / n))
    z = _z(alpha)
    return AucResult(auc, se, max(0.0, auc - z * se), min(1.0, auc + z * se))
