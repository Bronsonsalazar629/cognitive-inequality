"""
E-value computation for sensitivity to unmeasured confounding.

VanderWeele & Ding (2017): Sensitivity Analysis in Observational Research:
Introducing the E-Value. Annals of Internal Medicine, 167(4), 268-274.

The E-value is the minimum strength of association (on the risk ratio scale)
that an unmeasured confounder must have with BOTH the exposure AND the outcome
to fully explain away the observed effect, conditional on measured covariates.

For continuous outcomes (OLS):
    1. Standardize: d = |β| / SD_outcome          (Cohen's d)
    2. Approximate RR: RR = exp(0.91 × d)          (Chinn 2000)
    3. E-value: E = RR + sqrt(RR × (RR − 1))

For risk ratios (binary outcomes):
    E = RR + sqrt(RR × (RR − 1))

For the CI E-value, use the weaker (closer-to-null) CI bound.
If the CI already contains the null, E-value for the CI = 1.
"""

import math
from typing import Dict


def _evalue_from_rr(rr: float) -> float:
    """Core E-value formula given a risk ratio >= 1."""
    if rr <= 1.0:
        return 1.0
    return rr + math.sqrt(rr * (rr - 1.0))


def evalue_rr(rr: float, ci_lower: float, ci_upper: float) -> Dict:
    """
    E-value for a risk ratio (binary outcome).

    Args:
        rr:       Point estimate of risk ratio.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.

    Returns:
        Dict with keys: evalue_point, evalue_ci, rr.
    """
    # Ensure RR >= 1 (flip if protective)
    if rr < 1.0:
        rr_adj = 1.0 / rr
        ci_lower_adj = 1.0 / ci_upper
    else:
        rr_adj = rr
        ci_lower_adj = ci_lower

    evalue_point = _evalue_from_rr(rr_adj)

    # CI E-value: use the bound closer to the null (ci_lower for RR > 1)
    if ci_lower_adj <= 1.0:
        evalue_ci = 1.0
    else:
        evalue_ci = _evalue_from_rr(ci_lower_adj)

    return {
        'evalue_point': evalue_point,
        'evalue_ci': evalue_ci,
        'rr': rr_adj,
    }


def evalue_ols(estimate: float, se: float, sd_outcome: float = 1.0) -> Dict:
    """
    E-value for an OLS regression coefficient (continuous outcome).

    Converts the unstandardized coefficient to Cohen's d, approximates
    the risk ratio using Chinn (2000), then applies the E-value formula.

    Args:
        estimate:   OLS regression coefficient (unstandardized).
        se:         Standard error of the coefficient.
        sd_outcome: Standard deviation of the outcome variable.

    Returns:
        Dict with keys: evalue_point, evalue_ci, rr_approx.
    """
    # Standardize to Cohen's d
    d = abs(estimate) / sd_outcome

    # Approximate risk ratio (Chinn 2000: d → OR, then OR → RR approximation)
    rr = math.exp(0.91 * d)

    evalue_point = _evalue_from_rr(rr)

    # CI bound closest to null (lower bound for positive estimates)
    ci_bound = abs(estimate) - 1.96 * se
    if ci_bound <= 0:
        # CI crosses zero — null already within CI
        evalue_ci = 1.0
    else:
        d_ci = ci_bound / sd_outcome
        rr_ci = math.exp(0.91 * d_ci)
        evalue_ci = _evalue_from_rr(rr_ci)

    return {
        'evalue_point': evalue_point,
        'evalue_ci': evalue_ci,
        'rr_approx': rr,
    }
