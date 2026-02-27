"""E-value computation for sensitivity to unmeasured confounding."""

from typing import Dict


def evalue_ols(estimate: float, se: float, ci_lower: float, ci_upper: float) -> Dict:
    """E-value for OLS regression estimate (continuous outcome)."""
    raise NotImplementedError("Sensitivity analysis implementation")


def evalue_rr(rr: float, ci_lower: float, ci_upper: float) -> Dict:
    """E-value for risk ratio (binary outcome)."""
    raise NotImplementedError("Sensitivity analysis implementation")
