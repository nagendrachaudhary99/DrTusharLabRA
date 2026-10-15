"""Flag subgroups (strata) where a model is credibly worse than its overall performance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class FailureMode:
    column: str
    stratum: str
    n: int
    sensitivity: float
    ci_lo: float
    ci_hi: float
    reference: float
    effect: float
    """reference - sensitivity (absolute sensitivity deficit)."""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def find_failure_modes(
    tables: dict[str, pd.DataFrame],
    reference: float,
    *,
    min_n: int = 10,
    use_bootstrap: bool = True,
) -> list[FailureMode]:
    """Return strata whose CI upper bound lies below ``reference``, largest deficit first.

    ``tables`` maps a label (e.g. ``"arm"``) to a table from
    :func:`itrialspace_eval.stats.stratified_sensitivity`. ``reference`` is the overall
    sensitivity point estimate or a user threshold. Strata with fewer than ``min_n`` lesions are
    skipped (their CIs are too wide to be informative and small cells invite false alarms).
    When a patient-cluster bootstrap CI is available the *more conservative* of the two upper
    bounds (max of Wilson and bootstrap) is used: the percentile bootstrap collapses to a point
    when a stratum has 0 % or 100 % sensitivity, which would otherwise create spurious flags.

    Note: this is a screening heuristic across many strata, not a multiplicity-corrected test.
    """
    out: list[FailureMode] = []
    for label, t in tables.items():
        if t.empty:
            continue
        boot = use_bootstrap and "boot_hi" in t.columns
        key_cols = [
            c
            for c in t.columns
            if c
            not in {"n", "k", "n_patients", "sensitivity", "wilson_lo", "wilson_hi", "boot_lo", "boot_hi"}
        ]
        for _, row in t.iterrows():
            hi = max(float(row["wilson_hi"]), float(row["boot_hi"])) if boot else float(row["wilson_hi"])
            lo = min(float(row["wilson_lo"]), float(row["boot_lo"])) if boot else float(row["wilson_lo"])
            if int(row["n"]) < min_n or hi >= reference:
                continue
            out.append(
                FailureMode(
                    column=label,
                    stratum=" / ".join(str(row[c]) for c in key_cols),
                    n=int(row["n"]),
                    sensitivity=float(row["sensitivity"]),
                    ci_lo=lo,
                    ci_hi=hi,
                    reference=float(reference),
                    effect=float(reference - row["sensitivity"]),
                )
            )
    return sorted(out, key=lambda f: (-f.effect, f.column, f.stratum))
