"""Command-line interface: ``itrialspace-eval analyze`` and ``itrialspace-eval agreement``."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from itrialspace_eval import __version__
from itrialspace_eval.agreement import synthetic_real_agreement
from itrialspace_eval.failure_modes import find_failure_modes
from itrialspace_eval.io import load_lesions, load_model_metrics, load_patients
from itrialspace_eval.stats import (
    auc_delong,
    cluster_bootstrap_sensitivity,
    froc,
    stratified_sensitivity,
    wilson_ci,
)

DEFAULT_BY = ("arm", "diameterMm", "type", "juxtavascular", "lobe")


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _clean(o: Any) -> Any:
    """Make an object JSON-safe (numpy scalars, NaN -> null, tuples -> lists)."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_clean(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def _fmt_table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows, strict=True)]
    line = "  ".join(h.ljust(w) for h, w in zip(header, widths, strict=True))
    sep = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(c).ljust(w) for c, w in zip(r, widths, strict=True)) for r in rows]
    return "\n".join([line, sep, *body])


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


def _parse_bins(specs: Sequence[str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for spec in specs:
        col, _, edges = spec.partition("=")
        if not edges:
            raise SystemExit(f"--bin expects COL=e0,e1,...; got {spec!r}")
        vals = [float(e) for e in edges.split(",")]
        if vals != sorted(vals) or len(vals) < 2:
            raise SystemExit(f"--bin edges must be increasing, got {vals}")
        out[col.strip()] = vals
    return out


def cmd_analyze(args: argparse.Namespace) -> int:
    df = load_lesions(args.lesions)
    by_specs: list[str] = args.by or [c for c in DEFAULT_BY if c in df.columns]
    bins = _parse_bins(args.bin or [])
    n, k = len(df), int(df["detected"].sum())
    lo, hi = wilson_ci(k, n, args.alpha)
    boot = (
        cluster_bootstrap_sensitivity(
            df["detected"].to_numpy(),
            df["patientId"].to_numpy(),
            n_boot=args.bootstrap,
            seed=args.seed,
            alpha=args.alpha,
        )
        if args.bootstrap > 0
        else None
    )
    overall: dict[str, Any] = {
        "n_lesions": n,
        "n_patients": int(df["patientId"].nunique()),
        "detected": k,
        "sensitivity": k / n if n else 0.0,
        "wilson_ci": [lo, hi],
    }
    if boot:
        overall["cluster_bootstrap_ci"] = [boot.lo, boot.hi]

    print(
        f"itrialspace-eval {__version__}  |  {args.lesions}  |  seed={args.seed}  bootstrap={args.bootstrap}"
    )
    ci_txt = f"Wilson {_pct(lo)}-{_pct(hi)}"
    if boot:
        ci_txt += f"; patient bootstrap {_pct(boot.lo)}-{_pct(boot.hi)}"
    print(
        f"\nOverall sensitivity: {k}/{n} = {_pct(overall['sensitivity'])}  ({ci_txt})  "
        f"[{overall['n_patients']} patients]"
    )

    tables: dict[str, pd.DataFrame] = {}
    for spec in by_specs:
        cols = [c.strip() for c in spec.split(",") if c.strip()]
        t = stratified_sensitivity(
            df, cols, bins=bins, n_boot=args.bootstrap, seed=args.seed, alpha=args.alpha
        )
        label = " x ".join(cols)
        tables[label] = t
        rows = []
        for _, r in t.iterrows():
            row = [
                " / ".join(str(r[c]) for c in cols),
                str(r["n"]),
                str(r["k"]),
                _pct(r["sensitivity"]),
                f"{_pct(r['wilson_lo'])}-{_pct(r['wilson_hi'])}",
            ]
            if "boot_lo" in t.columns:
                row.append(f"{_pct(r['boot_lo'])}-{_pct(r['boot_hi'])}")
            rows.append(row)
        header = [label, "n", "k", "sens", "Wilson 95% CI"] + (
            ["bootstrap 95% CI"] if "boot_lo" in t.columns else []
        )
        print(f"\nSensitivity by {label}\n" + _fmt_table(rows, header))

    reference = args.threshold if args.threshold is not None else overall["sensitivity"]
    fms = find_failure_modes(tables, reference, min_n=args.min_n)
    ref_txt = f"threshold {_pct(reference)}" if args.threshold is not None else f"overall {_pct(reference)}"
    print(f"\nFailure modes (CI upper bound < {ref_txt}, n >= {args.min_n}):")
    if fms:
        print(
            _fmt_table(
                [
                    [
                        f.column,
                        f.stratum,
                        str(f.n),
                        _pct(f.sensitivity),
                        f"{_pct(f.ci_lo)}-{_pct(f.ci_hi)}",
                        f"-{100 * f.effect:.1f} pp",
                    ]
                    for f in fms
                ],
                ["factor", "stratum", "n", "sens", "95% CI", "deficit"],
            )
        )
    else:
        print("  none flagged")

    report: dict[str, Any] = {
        "tool": {"name": "itrialspace-eval", "version": __version__},
        "inputs": {"lesions": str(args.lesions), "lesions_sha256": _sha256(args.lesions)},
        "params": {
            "by": by_specs,
            "bins": bins,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
            "alpha": args.alpha,
            "min_n": args.min_n,
            "threshold": args.threshold,
        },
        "overall": overall,
        "strata": {lbl: t.to_dict(orient="records") for lbl, t in tables.items()},
        "failure_modes": [f.to_dict() for f in fms],
        "froc": None,
        "candidate_auc": None,
    }

    if args.patients:
        fp = load_patients(args.patients)
        for pid in df["patientId"].unique():  # scans with lesions but no listed FPs still count
            fp.setdefault(str(pid), [])
        report["inputs"].update(patients=str(args.patients), patients_sha256=_sha256(args.patients))
        scores = df["score"].to_numpy(dtype=np.float64) if "score" in df.columns else np.full(n, np.nan)
        fr = froc(
            scores,
            fp,
            lesion_patient_ids=df["patientId"].astype(str).tolist(),
            n_boot=args.bootstrap,
            seed=args.seed,
            alpha=args.alpha,
        )
        report["froc"] = fr.to_dict()
        ci = f"  (patient bootstrap {fr.cpm_ci[0]:.3f}-{fr.cpm_ci[1]:.3f})" if fr.cpm_ci else ""
        print(f"\nFROC ({fr.n_lesions} lesions, {fr.n_scans} scans): CPM = {fr.cpm:.3f}{ci}")
        print(_fmt_table([[f"{r:g}", _pct(s)] for r, s in fr.points.items()], ["FP/scan", "sensitivity"]))
        neg = [s for v in fp.values() for s in v]
        pos = scores[~np.isnan(scores)]
        if pos.size >= 2 and len(neg) >= 2:
            a = auc_delong(pos, neg, args.alpha)
            report["candidate_auc"] = {"auc": a.auc, "se": a.se, "ci": [a.lo, a.hi], "method": "DeLong"}
            print(
                f"Candidate-level AUC (lesions vs FP candidates): {a.auc:.3f} (DeLong 95% CI {a.lo:.3f}-{a.hi:.3f})"
            )
    else:
        print("\nFROC/CPM skipped: pass --patients with per-scan false-positive scores (see README).")

    if args.out:
        Path(args.out).write_text(json.dumps(_clean(report), indent=2) + "\n")
        print(f"\nWrote {args.out}")
    return 0


def cmd_agreement(args: argparse.Namespace) -> int:
    syn = load_model_metrics(args.synthetic, args.metric)
    real = load_model_metrics(args.real, args.metric)
    res = synthetic_real_agreement(syn, real, n_boot=args.bootstrap, seed=args.seed, alpha=args.alpha)
    print(
        f"itrialspace-eval {__version__}  |  synthetic={args.synthetic}  real={args.real}  "
        f"metric={syn.name}  seed={args.seed}  bootstrap={args.bootstrap}"
    )
    rows = [[m, f"{syn[m]:.3f}", f"{real[m]:.3f}", f"{syn[m] - real[m]:+.3f}"] for m in res.models]
    print("\n" + _fmt_table(rows, ["model", "synthetic", "real", "diff"]))
    ba = res.bland_altman
    print(
        f"\nSpearman rho = {res.spearman_rho:.3f}  (95% CI {res.spearman_ci[0]:.3f} to {res.spearman_ci[1]:.3f})"
    )
    print(
        f"Kendall tau  = {res.kendall_tau:.3f}  (95% CI {res.kendall_ci[0]:.3f} to {res.kendall_ci[1]:.3f})"
    )
    print(
        f"Bland-Altman bias = {ba.bias:+.3f}, limits of agreement {ba.loa_lo:+.3f} to {ba.loa_hi:+.3f} (n={ba.n})"
    )
    if res.n_degenerate_boot:
        print(f"({res.n_degenerate_boot} degenerate bootstrap resamples dropped)")
    print("Reference: the iTRIALSPACE paper reports Spearman rho = 0.93 between synthetic and real rankings.")
    if args.out:
        out = res.to_dict() | {
            "metric": syn.name,
            "tool": {"name": "itrialspace-eval", "version": __version__},
        }
        Path(args.out).write_text(json.dumps(_clean(out), indent=2) + "\n")
        print(f"\nWrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="itrialspace-eval", description=__doc__)
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--bootstrap", type=int, default=2000, help="bootstrap replicates (0 disables)")
        sp.add_argument("--seed", type=int, default=42)
        sp.add_argument("--alpha", type=float, default=0.05, help="1 - confidence level")
        sp.add_argument("--out", type=Path, help="write JSON report here")

    a = sub.add_parser("analyze", help="sensitivity, strata, failure modes, FROC/CPM from a lesions CSV")
    a.add_argument("lesions", type=Path)
    a.add_argument("--patients", type=Path, help="per-scan false-positive scores (enables FROC/CPM)")
    a.add_argument("--by", action="append", help="stratify by column; repeatable; 'a,b' cross-classifies")
    a.add_argument("--bin", action="append", metavar="COL=E0,E1,...", help="bin edges for a numeric column")
    a.add_argument("--min-n", type=int, default=10, help="minimum stratum size for failure-mode flags")
    a.add_argument("--threshold", type=float, help="flag strata below this sensitivity instead of overall")
    common(a)
    a.set_defaults(func=cmd_analyze)

    g = sub.add_parser("agreement", help="synthetic-vs-real ranking agreement across models")
    g.add_argument("synthetic", type=Path, help="CSV with columns model,<metric>")
    g.add_argument("real", type=Path, help="CSV with columns model,<metric>")
    g.add_argument("--metric", help="metric column (default: the single numeric column)")
    common(g)
    g.set_defaults(func=cmd_agreement)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
