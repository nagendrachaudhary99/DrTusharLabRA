"""Synthetic-vs-real benchmark agreement.

A virtual trial is only useful for model selection if it ranks models the way a real-patient
benchmark would. The iTRIALSPACE paper reports a Spearman correlation of about 0.93 between
model rankings on its synthetic trials and on real data; this module computes the same kind of
check for your own models: rank correlations (Spearman rho, Kendall tau) with model-level
bootstrap CIs, plus Bland-Altman bias / limits of agreement on the metric scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps


@dataclass(frozen=True)
class BlandAltman:
    bias: float
    sd: float
    loa_lo: float
    loa_hi: float
    n: int


@dataclass(frozen=True)
class AgreementResult:
    models: list[str]
    spearman_rho: float
    spearman_ci: tuple[float, float]
    kendall_tau: float
    kendall_ci: tuple[float, float]
    bland_altman: BlandAltman
    n_boot: int
    seed: int
    n_degenerate_boot: int
    """Bootstrap replicates dropped because a resample had constant values (correlation undefined)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def bland_altman(synthetic: np.ndarray, real: np.ndarray, z: float = 1.96) -> BlandAltman:
    """Bias = mean(synthetic - real); limits of agreement = bias +/- z * SD(differences)."""
    s = np.asarray(synthetic, dtype=np.float64)
    r = np.asarray(real, dtype=np.float64)
    if s.shape != r.shape or s.size < 2:
        raise ValueError("need two equal-length arrays with at least 2 values")
    d = s - r
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return BlandAltman(bias, sd, bias - z * sd, bias + z * sd, int(d.size))


def _corr(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    rho = sps.spearmanr(a, b).statistic
    tau = sps.kendalltau(a, b).statistic
    return float(rho), float(tau)


def synthetic_real_agreement(
    synthetic: pd.Series,
    real: pd.Series,
    *,
    n_boot: int = 2000,
    seed: int = 42,
    alpha: float = 0.05,
) -> AgreementResult:
    """Compare per-model metrics (Series indexed by model name) on synthetic vs real benchmarks.

    Models present in only one Series are ignored; at least 3 shared models are required.
    CIs resample *models* with replacement (percentile method). With few models the CIs are wide
    and the bootstrap distribution is lumpy -- report them as such.
    """
    shared = sorted(set(synthetic.index) & set(real.index))
    if len(shared) < 3:
        raise ValueError(f"need >= 3 models present in both tables, got {len(shared)}")
    s = synthetic.loc[shared].to_numpy(dtype=np.float64)
    r = real.loc[shared].to_numpy(dtype=np.float64)
    rho, tau = _corr(s, r)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    rhos: list[float] = []
    taus: list[float] = []
    for row in idx:
        bs, br = s[row], r[row]
        if np.ptp(bs) == 0 or np.ptp(br) == 0:
            continue
        a, b = _corr(bs, br)
        if np.isfinite(a) and np.isfinite(b):
            rhos.append(a)
            taus.append(b)
    q = [alpha / 2, 1 - alpha / 2]
    rho_ci = tuple(float(v) for v in np.quantile(rhos, q)) if rhos else (float("nan"), float("nan"))
    tau_ci = tuple(float(v) for v in np.quantile(taus, q)) if taus else (float("nan"), float("nan"))
    return AgreementResult(
        models=shared,
        spearman_rho=rho,
        spearman_ci=(rho_ci[0], rho_ci[1]),
        kendall_tau=tau,
        kendall_ci=(tau_ci[0], tau_ci[1]),
        bland_altman=bland_altman(s, r),
        n_boot=n_boot,
        seed=seed,
        n_degenerate_boot=n_boot - len(rhos),
    )
