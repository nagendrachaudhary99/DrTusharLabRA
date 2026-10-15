from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from itrialspace_eval.agreement import bland_altman, synthetic_real_agreement
from itrialspace_eval.failure_modes import find_failure_modes
from itrialspace_eval.stats import stratified_sensitivity


def test_spearman_identical_and_reversed_rankings() -> None:
    models = list("ABCDEF")
    syn = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], index=models)
    same = synthetic_real_agreement(syn, syn + 0.05, n_boot=200, seed=0)
    assert same.spearman_rho == pytest.approx(1.0)
    assert same.kendall_tau == pytest.approx(1.0)
    assert same.bland_altman.bias == pytest.approx(-0.05)
    rev = synthetic_real_agreement(syn, pd.Series(syn.to_numpy()[::-1], index=models), n_boot=200, seed=0)
    assert rev.spearman_rho == pytest.approx(-1.0)


def test_spearman_known_value() -> None:
    # One adjacent swap among 5 items: rho = 1 - 6*2/(5*24) = 0.9, tau = (9-1)/10 = 0.8
    syn = pd.Series([1, 2, 3, 4, 5], index=list("abcde"), dtype=float)
    real = pd.Series([1, 2, 4, 3, 5], index=list("abcde"), dtype=float)
    r = synthetic_real_agreement(syn, real, n_boot=500, seed=1)
    assert r.spearman_rho == pytest.approx(0.9)
    assert r.kendall_tau == pytest.approx(0.8)
    assert r.spearman_ci[0] <= r.spearman_ci[1]
    r2 = synthetic_real_agreement(syn, real, n_boot=500, seed=1)
    assert r.spearman_ci == r2.spearman_ci


def test_agreement_requires_three_shared_models() -> None:
    with pytest.raises(ValueError):
        synthetic_real_agreement(
            pd.Series([1.0, 2.0], index=["a", "b"]), pd.Series([1.0, 2.0], index=["a", "b"])
        )


def test_bland_altman_values() -> None:
    ba = bland_altman(np.array([1.0, 2.0, 3.0]), np.array([0.0, 2.0, 2.0]))
    assert ba.bias == pytest.approx(2 / 3)
    assert ba.loa_hi - ba.loa_lo == pytest.approx(2 * 1.96 * np.std([1, 0, 1], ddof=1))


def test_failure_modes_flags_bad_stratum_and_min_n() -> None:
    df = pd.DataFrame(
        {
            "patientId": [f"P{i}" for i in range(100)],
            "arm": ["good"] * 60 + ["bad"] * 30 + ["tiny"] * 10,
            "detected": [True] * 55 + [False] * 5 + [True] * 5 + [False] * 25 + [False] * 10,
        }
    )
    t = {"arm": stratified_sensitivity(df, "arm", n_boot=300, seed=0)}
    overall = df["detected"].mean()
    fm = find_failure_modes(t, overall, min_n=20)
    assert [f.stratum for f in fm] == ["bad"]
    fm_all = find_failure_modes(t, overall, min_n=5)
    assert [f.stratum for f in fm_all] == ["tiny", "bad"]  # sorted by effect size
    assert find_failure_modes(t, 0.05, min_n=5) == []
