# itrialspace-eval

Reproducible statistics for **virtual-lesion trials** of lung-CT nodule detectors, packaged as a
typed, tested Python library + CLI. It is the offline/analysis companion to the iTRIALSPACE demo web
app in the parent directory, which is inspired by *iTRIALSPACE* (Tushar et al., arXiv 2605.05761,
University of Arizona) — a programmable virtual-lesion trial framework for controlled evaluation of
lung CT models.

The web app runs a trial in the browser and lets you download per-lesion results; this package
re-analyses that export with patient-level uncertainty, subgroup failure-mode screening, FROC/CPM
and synthetic-vs-real agreement — deterministically, from the command line or in Docker.

## What it computes

| Module | Contents |
| --- | --- |
| `stats` | Wilson score CI; **patient-level cluster bootstrap** CI for sensitivity (resamples patients, seeded); stratified sensitivity tables by any column(s), with numeric binning; **FROC curve + CPM** (mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan, LUNA16 definition) with optional patient-bootstrap CI; candidate-level AUC with **DeLong** SE. |
| `agreement` | Synthetic-vs-real benchmark agreement across models: **Spearman ρ** and **Kendall τ** with model-level bootstrap CIs, **Bland–Altman** bias and limits of agreement. The iTRIALSPACE paper reports a synthetic-vs-real ranking correlation of ρ = 0.93; this lets you run the same check on your own models. |
| `failure_modes` | Flags strata whose CI upper bound is below the overall sensitivity (or a user threshold), sorted by deficit, with a minimum-n guard. |
| `cli` | `itrialspace-eval analyze …` and `itrialspace-eval agreement …`; plain-text tables + JSON report. |

Everything random takes an explicit `--seed` (default 42) and uses `numpy.random.default_rng`, so a
given input + seed + `--bootstrap` gives identical output (the test suite checks this, and the
Docker image on Python 3.12 reproduces the local Python 3.14 numbers exactly).

## Install

```bash
cd python
python3 -m venv .venv
.venv/bin/pip install -e ".[dev]"     # or: pip install .   (runtime deps only: numpy, pandas, scipy)
```

Python ≥ 3.10. `make venv`, `make check` (ruff + mypy --strict + pytest) and `make demo` wrap the
commands below.

## Usage

### 1. Analyse a lesions CSV exported from the web app

```bash
itrialspace-eval analyze lesions.csv --patients patients.csv \
    --by diameterMm --by arm --bootstrap 2000 --seed 42 --out report.json
```

Sample output on the bundled fixture (`tests/fixtures/lesions_sample.csv`, 59 lesions / 36 patients):

```
Overall sensitivity: 22/59 =  37.3%  (Wilson  26.1%- 50.0%; patient bootstrap  22.2%- 53.2%)  [36 patients]

Sensitivity by diameterMm
diameterMm    n   k  sens    Wilson 95% CI  bootstrap 95% CI
------------  --  -  ------  -------------  ----------------
[0, 5)        9   1   11.1%    2.0%- 43.5%    0.0%- 37.5%
[5, 7)        10  3   30.0%   10.8%- 60.3%    0.0%- 66.7%
...
Sensitivity by arm
arm                         n   k   sens    Wilson 95% CI  bootstrap 95% CI
--------------------------  --  --  ------  -------------  ----------------
GE sharp low                19  3    15.8%    5.5%- 37.6%    0.0%- 33.3%
Philips standard ultra-low  19  1     5.3%    0.9%- 24.6%    0.0%- 18.8%
Siemens standard standard   21  18   85.7%   65.4%- 95.0%   65.0%-100.0%

Failure modes (CI upper bound < overall  37.3%, n >= 10):
factor  stratum                     n   sens    95% CI         deficit
------  --------------------------  --  ------  -------------  --------
arm     Philips standard ultra-low  19    5.3%    0.0%- 24.6%  -32.0 pp

FROC (59 lesions, 36 scans): CPM = 0.547  (patient bootstrap 0.438-0.675)
FP/scan  sensitivity
-------  -----------
0.125     28.8%
...
8        100.0%
Candidate-level AUC (lesions vs FP candidates): 0.647 (DeLong 95% CI 0.553-0.742)
```

Options:

* `--by COL` (repeatable) — one table per flag; `--by type,juxtavascular` cross-classifies.
  Without `--by`, defaults to `arm, diameterMm, type, juxtavascular, lobe`.
* `--bin COL=e0,e1,...` — custom bin edges (left-closed). `diameterMm` defaults to
  `[0,5,7,9,12.5,17.5,inf)` mm, i.e. bins around the app's default 4/6/8/10/15/20 mm strata; other
  continuous columns with > 12 distinct values fall back to quartile bins.
* `--min-n 10`, `--threshold 0.8` — failure-mode guard and fixed reference instead of overall.
* `--bootstrap 0` disables the bootstrap (Wilson only). `--alpha` sets 1 − confidence level.

The JSON report contains tool version, SHA-256 of every input file, all parameters, overall and
per-stratum tables, failure modes, FROC points/curve and AUC — enough to audit or re-run a result.

### 2. Synthetic-vs-real agreement across models

```bash
itrialspace-eval agreement synthetic.csv real.csv [--metric cpm] --bootstrap 2000 --seed 42 --out agreement.json
```

Each CSV has a `model` column plus one or more metric columns (pass `--metric` if more than one).
Sample output (`tests/fixtures/{synthetic,real}_models.csv`, 7 illustrative models):

```
Spearman rho = 0.929  (95% CI 0.412 to 1.000)
Kendall tau  = 0.810  (95% CI 0.294 to 1.000)
Bland-Altman bias = -0.029, limits of agreement -0.060 to +0.001 (n=7)
```

The fixture numbers are illustrative, not results from the paper.

### 3. Library use

```python
from itrialspace_eval import load_lesions, stratified_sensitivity, cluster_bootstrap_sensitivity

df = load_lesions("itrialspace-lesions.csv")
stratified_sensitivity(df, ["arm", "diameterMm"], n_boot=2000, seed=42)
```

### Docker

```bash
docker build -t itrialspace-eval .
docker run --rm -v "$PWD/tests/fixtures:/data:ro" itrialspace-eval \
    analyze /data/lesions_sample.csv --patients /data/patients_sample.csv --by arm --seed 42
# write the JSON report to the host:
docker run --rm -v "$PWD:/work" -v "$PWD/tests/fixtures:/data:ro" itrialspace-eval \
    analyze /data/lesions_sample.csv --out /work/report.json
```

`python:3.12-slim`, non-root user (uid 10001), entrypoint `itrialspace-eval`.

## How it plugs into the web app

In the web app, run a trial and on the *Results* panel click **Lesions CSV** → this downloads
`itrialspace-lesions.csv` with exactly the columns

```
lesionId,patientId,arm,diameterMm,type,lobe,juxtavascular,sigmaHU,cnr,dprime,score,detected
```

(`arm` is a quoted `"vendor kernel dose"` label, booleans are `true`/`false`; see
`Results.csv()` in `src/app/trial/trial-client.tsx`). Pass that file straight to `analyze`.

**False positives / FROC.** The lesions export contains only true lesions. The simulator stores
per-scan false-positive candidate scores on each virtual patient (`falsePositiveScores` in
`src/lib/trial/simulate.ts`) but they are not in the CSV, so FROC/CPM needs an extra
`--patients` file in either layout:

```
patientId,falsePositiveScores          # wide: ';'- or space-separated, empty = no candidates
VP-1-0001,0.7698;0.4196;0.1201
VP-1-0003,

patientId,fpScore                      # long: one row per candidate (empty fpScore = scan with none)
VP-1-0001,0.7698
```

Every scan must be represented (patients that appear in the lesions file are added automatically
with zero FPs), because the FP/scan denominator is the number of scans. Without `--patients`,
FROC is skipped and everything else still runs. A small addition to the web app (a "Patients CSV"
button serialising `patientId` + `falsePositiveScores.join(";")`) would close this gap.

The CPM uses the same step definition as the web app (best sensitivity at any threshold whose
FP/scan ≤ target), so the two should agree on the same cohort.

## Tests

```bash
.venv/bin/pytest          # 22 tests
.venv/bin/ruff check . && .venv/bin/ruff format --check .
.venv/bin/mypy            # --strict over src/ and tests/
```

Covers Wilson CIs against Newcombe (1998) published values, bootstrap determinism, clustered vs
independent CI width, CPM on a hand-worked toy case, Spearman/Kendall on known rankings, DeLong vs
brute-force AUC, failure-mode ranking and min-n guard, and CLI smoke tests (fixture + CSV generated
in the test). Fixtures are regenerated deterministically by `tests/fixtures/make_fixtures.py`.

CI: `ci/python-ci.yml` is a GitHub Actions workflow (lint, mypy, pytest on 3.10/3.12/3.13, CLI
smoke test, Docker build + run). Move it to `.github/workflows/` at the repository root to enable it.

## Limitations (honest)

* **Simulated data.** The fixtures and the web app's cohorts come from a lightweight analytic
  simulator (noise/CNR/d′ model + logistic detector), not from the iTRIALSPACE paper's lesion
  insertion pipeline or real CT. Numbers here say nothing about any real model.
* **Lesion-level only.** No lesion-to-candidate matching (e.g. LUNA16's radius criterion) — the
  export already says whether each lesion was hit, and its `score` is taken as the score of the
  matching candidate.
* **CPM** uses the conservative step reading of the FROC curve; LUNA16's official script
  interpolates, which can give slightly higher values.
* **Failure-mode screening** compares many strata without multiplicity correction; treat flags as
  hypotheses. Percentile bootstrap CIs collapse for 0 % / 100 % strata — the screener therefore
  uses the wider of the Wilson and bootstrap upper bounds.
* **Agreement CIs** resample models; with fewer than ~10 models they are wide and lumpy, and
  degenerate resamples (constant values) are dropped and counted in the output.
* DeLong AUC here is candidate-level (lesions vs FP candidates), not a patient-level ROC.
