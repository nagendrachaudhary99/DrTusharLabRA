"""Reproducible statistics for iTRIALSPACE-style virtual-lesion trials."""

from itrialspace_eval.agreement import AgreementResult, bland_altman, synthetic_real_agreement
from itrialspace_eval.failure_modes import FailureMode, find_failure_modes
from itrialspace_eval.io import load_lesions, load_patients
from itrialspace_eval.stats import (
    CPM_FP_RATES,
    FrocResult,
    auc_delong,
    cluster_bootstrap_sensitivity,
    froc,
    stratified_sensitivity,
    wilson_ci,
)

__version__ = "0.1.0"

__all__ = [
    "CPM_FP_RATES",
    "AgreementResult",
    "FailureMode",
    "FrocResult",
    "__version__",
    "auc_delong",
    "bland_altman",
    "cluster_bootstrap_sensitivity",
    "find_failure_modes",
    "froc",
    "load_lesions",
    "load_patients",
    "stratified_sensitivity",
    "synthetic_real_agreement",
    "wilson_ci",
]
