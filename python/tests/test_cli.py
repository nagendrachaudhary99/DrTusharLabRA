from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from itrialspace_eval.cli import main
from itrialspace_eval.io import load_lesions, load_patients


def test_fixture_matches_web_app_format(fixtures: Path) -> None:
    df = load_lesions(fixtures / "lesions_sample.csv")
    assert list(df.columns) == [
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
    assert 50 <= len(df) <= 70
    assert df["detected"].dtype == bool
    fp = load_patients(fixtures / "patients_sample.csv")
    assert set(df["patientId"]) <= set(fp)


def test_analyze_smoke_and_deterministic(
    fixtures: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    args = [
        "analyze",
        str(fixtures / "lesions_sample.csv"),
        "--patients",
        str(fixtures / "patients_sample.csv"),
        "--by",
        "diameterMm",
        "--by",
        "arm",
        "--bootstrap",
        "300",
        "--seed",
        "42",
    ]
    assert main([*args, "--out", str(tmp_path / "a.json")]) == 0
    out = capsys.readouterr().out
    assert "Overall sensitivity" in out and "Sensitivity by arm" in out and "CPM =" in out
    assert main([*args, "--out", str(tmp_path / "b.json")]) == 0
    a = json.loads((tmp_path / "a.json").read_text())
    b = json.loads((tmp_path / "b.json").read_text())
    assert a == b
    assert set(a["strata"]) == {"diameterMm", "arm"}
    assert 0 <= a["froc"]["cpm"] <= 1 and a["froc"]["cpm_ci"] is not None


def test_analyze_generated_csv_without_patients(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rng = np.random.default_rng(0)
    n = 40
    pd.DataFrame(
        {
            "lesionId": [f"L{i}" for i in range(n)],
            "patientId": [f"P{i // 2}" for i in range(n)],
            "arm": ["GE standard low"] * 20 + ["Siemens sharp standard"] * 20,
            "diameterMm": rng.uniform(3, 20, n).round(1),
            "type": "solid",
            "lobe": "RUL",
            "juxtavascular": "false",
            "sigmaHU": 15.0,
            "cnr": 3.0,
            "dprime": 4.0,
            "score": rng.random(n).round(4),
            "detected": ["true"] * 30 + ["false"] * 10,
        }
    ).to_csv(tmp_path / "l.csv", index=False)
    assert (
        main(
            [
                "analyze",
                str(tmp_path / "l.csv"),
                "--bootstrap",
                "0",
                "--threshold",
                "0.9",
                "--out",
                str(tmp_path / "r.json"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "FROC/CPM skipped" in out
    rep = json.loads((tmp_path / "r.json").read_text())
    assert rep["overall"]["sensitivity"] == 0.75
    assert rep["froc"] is None


def test_agreement_cli(fixtures: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        main(
            [
                "agreement",
                str(fixtures / "synthetic_models.csv"),
                str(fixtures / "real_models.csv"),
                "--bootstrap",
                "500",
                "--out",
                str(tmp_path / "ag.json"),
            ]
        )
        == 0
    )
    assert "Spearman rho" in capsys.readouterr().out
    rep = json.loads((tmp_path / "ag.json").read_text())
    assert rep["spearman_rho"] > 0.9


def test_bad_input_returns_error(tmp_path: Path) -> None:
    (tmp_path / "bad.csv").write_text("foo,bar\n1,2\n")
    assert main(["analyze", str(tmp_path / "bad.csv")]) == 2
