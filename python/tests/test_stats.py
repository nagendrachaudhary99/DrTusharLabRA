from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from itrialspace_eval.stats import (
    auc_delong,
    cluster_bootstrap_sensitivity,
    froc,
    stratified_sensitivity,
    wilson_ci,
)


# Newcombe (1998), Stat Med 17:857-872, Table I, method 3 (Wilson score interval).
@pytest.mark.parametrize(
    ("k", "n", "lo", "hi"),
    [(81, 263, 0.2553, 0.3662), (15, 148, 0.0624, 0.1605), (0, 20, 0.0, 0.1611), (1, 29, 0.0061, 0.1718)],
)
def test_wilson_matches_newcombe(k: int, n: int, lo: float, hi: float) -> None:
    got = wilson_ci(k, n)
    assert got[0] == pytest.approx(lo, abs=1e-4)
    assert got[1] == pytest.approx(hi, abs=1e-4)


def test_wilson_symmetric_and_edges() -> None:
    lo, hi = wilson_ci(5, 10)
    assert lo == pytest.approx(1 - hi)
    assert wilson_ci(0, 0) == (0.0, 0.0)
    assert wilson_ci(10, 10)[1] == 1.0
    assert wilson_ci(0, 10)[0] == 0.0
    with pytest.raises(ValueError):
        wilson_ci(11, 10)


def _toy_lesions(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pids = np.repeat([f"P{i}" for i in range(40)], 2)
    df: pd.DataFrame = pd.DataFrame(
        {
            "patientId": pids,
            "detected": rng.random(pids.size) < 0.7,
            "arm": np.where(np.arange(pids.size) < 40, "A", "B"),
            "diameterMm": rng.uniform(3, 22, pids.size),
        }
    )
    return df


def test_bootstrap_deterministic_with_seed() -> None:
    df = _toy_lesions()
    a = cluster_bootstrap_sensitivity(df["detected"], df["patientId"], n_boot=500, seed=7)
    b = cluster_bootstrap_sensitivity(df["detected"], df["patientId"], n_boot=500, seed=7)
    assert (a.lo, a.hi, a.seed) == (b.lo, b.hi, b.seed)
    assert a.lo <= a.estimate <= a.hi


def test_cluster_bootstrap_wider_than_wilson_when_clustered() -> None:
    # Perfectly clustered: every patient's 5 lesions are all detected or all missed.
    pids = np.repeat([f"P{i}" for i in range(30)], 5)
    det = np.repeat(np.arange(30) % 2 == 0, 5)
    b = cluster_bootstrap_sensitivity(det, pids, n_boot=2000, seed=1)
    w = wilson_ci(int(det.sum()), det.size)
    assert (b.hi - b.lo) > (w[1] - w[0])


def test_stratified_table_and_bins() -> None:
    df = _toy_lesions()
    t = stratified_sensitivity(df, "arm", n_boot=200, seed=3)
    assert list(t["arm"]) == ["A", "B"]
    assert t["n"].sum() == len(df)
    assert {"wilson_lo", "wilson_hi", "boot_lo", "boot_hi", "n_patients"} <= set(t.columns)
    sizes = stratified_sensitivity(df, "diameterMm")
    assert sizes["diameterMm"].iloc[0].startswith("[0, 5)")  # default bins, numeric order
    cross = stratified_sensitivity(df, ["arm", "diameterMm"], bins={"diameterMm": [0, 10, 100]})
    assert len(cross) == 4


def test_cpm_hand_constructed() -> None:
    # 4 lesions on 2 scans; one lesion never produced a candidate (NaN).
    lesions = [0.9, 0.8, 0.3, math.nan]
    fps = {"S1": [0.85, 0.2], "S2": [0.5]}
    r = froc(lesions, fps)
    # thresholds: .9 (sens .25, fp 0) .85 (.25, .5) .8 (.5, .5) .5 (.5, 1) .3 (.75, 1) .2 (.75, 1.5)
    assert r.points == {0.125: 0.25, 0.25: 0.25, 0.5: 0.5, 1.0: 0.75, 2.0: 0.75, 4.0: 0.75, 8.0: 0.75}
    assert r.cpm == pytest.approx(4 / 7)
    assert r.n_scans == 2 and r.n_lesions == 4


def test_cpm_perfect_and_empty_scans_count() -> None:
    r = froc([0.9, 0.8], {"S1": [], "S2": [], "S3": [0.1]})
    assert r.cpm == 1.0
    assert r.n_scans == 3


def test_cpm_bootstrap_deterministic() -> None:
    les = [0.9, 0.7, 0.4, 0.2, 0.6, 0.95]
    pids = ["A", "A", "B", "C", "C", "D"]
    fps = {"A": [0.3], "B": [0.8, 0.1], "C": [], "D": [0.5]}
    r1 = froc(les, fps, lesion_patient_ids=pids, n_boot=300, seed=5)
    r2 = froc(les, fps, lesion_patient_ids=pids, n_boot=300, seed=5)
    assert r1.cpm_ci == r2.cpm_ci and r1.cpm_ci is not None


def test_auc_delong() -> None:
    r = auc_delong([0.9, 0.8, 0.7, 0.6], [0.5, 0.4, 0.3])
    assert r.auc == 1.0
    r = auc_delong([1, 2, 3], [1, 2, 3])
    assert r.auc == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    pos, neg = rng.normal(1, 1, 200), rng.normal(0, 1, 300)
    r = auc_delong(pos, neg)
    brute = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
    assert r.auc == pytest.approx(brute)
    assert r.lo < r.auc < r.hi
