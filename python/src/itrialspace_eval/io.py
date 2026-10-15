"""Readers for the CSV files exported by the iTRIALSPACE web app.

Lesions CSV (``Results -> Lesions CSV`` in the web app)::

    lesionId,patientId,arm,diameterMm,type,lobe,juxtavascular,sigmaHU,cnr,dprime,score,detected
    VP-1-0001-L1,VP-1-0001,"Siemens sharp low",6.2,solid,RUL,false,21.4,3.1,4.2,0.7312,true

The web app does **not** export per-patient false-positive candidates, so FROC analysis
needs an optional patients CSV in one of two layouts:

* wide: ``patientId,falsePositiveScores`` where scores are ``;``- or space-separated
  (an empty cell means the scan had no false-positive candidates);
* long: ``patientId,fpScore`` with one row per candidate (a row with an empty ``fpScore``
  registers a scan without candidates).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

LESION_COLUMNS: tuple[str, ...] = (
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
)
REQUIRED_LESION_COLUMNS: tuple[str, ...] = ("lesionId", "patientId", "detected")

_TRUE = {"true", "1", "yes", "y", "t"}
_FALSE = {"false", "0", "no", "n", "f", ""}


def _to_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    s = series.astype(str).str.strip().str.lower()
    bad = ~s.isin(_TRUE | _FALSE)
    if bad.any():
        raise ValueError(f"column {series.name!r} has non-boolean values: {sorted(s[bad].unique())[:5]}")
    return s.isin(_TRUE)


def load_lesions(path: str | Path) -> pd.DataFrame:
    """Read and validate a lesions CSV. Returns a DataFrame with typed columns."""
    df = pd.read_csv(path, dtype={"lesionId": str, "patientId": str, "arm": str})
    missing = [c for c in REQUIRED_LESION_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required column(s) {missing}")
    if df["lesionId"].duplicated().any():
        raise ValueError(f"{path}: duplicated lesionId values")
    df["detected"] = _to_bool(df["detected"])
    if "juxtavascular" in df.columns:
        df["juxtavascular"] = _to_bool(df["juxtavascular"])
    for c in ("diameterMm", "sigmaHU", "cnr", "dprime", "score"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _parse_scores(cell: object) -> list[float]:
    if cell is None or (isinstance(cell, float) and np.isnan(cell)):
        return []
    text = str(cell).replace(";", " ").replace(",", " ").strip()
    return [float(t) for t in text.split()] if text else []


def load_patients(path: str | Path) -> dict[str, list[float]]:
    """Read false-positive candidate scores per scan (patient). See module docstring for layouts."""
    df = pd.read_csv(path, dtype={"patientId": str}, keep_default_na=True)
    if "patientId" not in df.columns:
        raise ValueError(f"{path}: missing 'patientId' column")
    out: dict[str, list[float]] = {}
    if "falsePositiveScores" in df.columns:
        for pid, cell in zip(df["patientId"], df["falsePositiveScores"], strict=True):
            out.setdefault(str(pid), []).extend(_parse_scores(cell))
    elif "fpScore" in df.columns:
        for pid, val in zip(df["patientId"], pd.to_numeric(df["fpScore"], errors="coerce"), strict=True):
            lst = out.setdefault(str(pid), [])
            if not np.isnan(val):
                lst.append(float(val))
    else:
        raise ValueError(f"{path}: expected a 'falsePositiveScores' (wide) or 'fpScore' (long) column")
    return out


def load_model_metrics(path: str | Path, metric: str | None = None) -> pd.Series:
    """Read a per-model metrics CSV (``model,<metric>``) and return a Series indexed by model."""
    df = pd.read_csv(path)
    if "model" not in df.columns:
        raise ValueError(f"{path}: missing 'model' column")
    if metric is None:
        numeric = [c for c in df.columns if c != "model" and pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric) != 1:
            raise ValueError(f"{path}: pass --metric; numeric columns found: {numeric}")
        metric = numeric[0]
    if metric not in df.columns:
        raise ValueError(f"{path}: metric column {metric!r} not found")
    if df["model"].duplicated().any():
        raise ValueError(f"{path}: duplicated model names")
    out: pd.Series = pd.Series(
        df[metric].astype(float).to_numpy(), index=df["model"].astype(str), name=metric
    )
    return out
