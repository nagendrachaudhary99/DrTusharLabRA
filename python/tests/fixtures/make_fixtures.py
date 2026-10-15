"""Regenerate the committed test fixtures (deterministic; run from anywhere).

Mimics the web app's simulator (src/lib/trial/simulate.ts) closely enough to exercise the
pipeline: 3 scanner arms x 12 virtual patients, 1-2 lesions each, logistic detector scores,
Poisson false-positive candidates per scan.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARMS = [
    ("Siemens", "standard", "standard", 12.0),
    ("GE", "sharp", "low", 22.0),
    ("Philips", "standard", "ultra-low", 34.0),
]
SIZES = [4, 6, 8, 10, 15, 20]
TYPES = ["solid", "part-solid", "ground-glass"]
LOBES = ["RUL", "RML", "RLL", "LUL", "LLL"]
CONTRAST = {"solid": 850.0, "part-solid": 400.0, "ground-glass": 200.0}


def js(x: float, nd: int) -> str:
    """Format like the web app: rounded with toFixed(nd), then String(Number) drops trailing zeros."""
    return f"{round(x, nd):.{nd}f}".rstrip("0").rstrip(".")


def main() -> None:
    rng = np.random.default_rng(20260513)
    lesions: list[list[str]] = []
    patients: list[list[str]] = []
    for a, (vendor, kernel, dose, sigma0) in enumerate(ARMS):
        for i in range(12):
            pid = f"VP-{a + 1}-{i + 1:04d}"
            bmi = float(np.clip(np.exp(np.log(26.5) + 0.17 * rng.normal()), 17, 45))
            sigma = sigma0 * (bmi / 26.5) ** 0.9
            for j in range(1 + int(rng.random() < 0.67)):
                size = int(rng.choice(SIZES))
                typ = str(rng.choice(TYPES, p=[0.6, 0.2, 0.2]))
                diam = max(2.0, size * (1 + 0.08 * rng.normal()))
                juxta = bool(rng.random() < 0.25)
                contrast = CONTRAST[typ] + 25 * rng.normal()
                cnr = contrast / sigma * (0.6 if kernel == "sharp" else 1.0)
                dprime = cnr * math.sqrt(diam) * 0.35
                logit = 1.9 * math.log(max(dprime, 1e-3)) - 5.7 - 0.6 * a
                logit -= 0.4 if typ == "ground-glass" else 0.3 if typ == "part-solid" else 0.0
                logit -= 0.8 if juxta else 0.0
                logit -= 0.6 if diam < 5 else 0.0
                score = 1 / (1 + math.exp(-(logit + 0.7 * rng.normal())))
                lesions.append(
                    [
                        f"{pid}-L{j + 1}",
                        pid,
                        f'"{vendor} {kernel} {dose}"',
                        js(diam, 1),
                        typ,
                        str(rng.choice(LOBES)),
                        str(juxta).lower(),
                        js(sigma, 1),
                        js(cnr, 2),
                        js(dprime, 2),
                        js(score, 4),
                        str(score >= 0.5).lower(),
                    ]
                )
            n_fp = int(rng.poisson(1.2 * (sigma / 12) ** 0.8 * 3))
            fps = [1 / (1 + math.exp(-(-1.7 + 1.1 * rng.normal()))) for _ in range(n_fp)]
            patients.append([pid, ";".join(f"{s:.4f}" for s in fps)])

    # Written by hand (not csv.writer) to reproduce the web app's Results.csv() byte format,
    # which always double-quotes the arm label.
    with (HERE / "lesions_sample.csv").open("w", newline="") as f:
        f.write(
            ",".join(
                [
                    "lesionId",
                    "patientId",
                    "arm",
                    "diameterMm",
                    "type",
                    "lobe",
                    "juxtavascular",
                    "sigmaHU",
                    "cnr",
                    "dprime",
                    "score",
                    "detected",
                ]
            )
            + "\n"
        )
        f.writelines(",".join(row) + "\n" for row in lesions)
    with (HERE / "patients_sample.csv").open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["patientId", "falsePositiveScores"])
        w.writerows(patients)
    print(f"wrote {len(lesions)} lesions, {len(patients)} patients")


if __name__ == "__main__":
    main()
