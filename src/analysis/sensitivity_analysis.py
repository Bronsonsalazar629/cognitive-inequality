"""
Sensitivity Analysis for Unmeasured Confounding

E-value analysis for the SES→cognition causal pathway.
Quantifies the minimum strength of association an unmeasured confounder
would need with both SES and cognition to fully explain away observed effects.

Reference: VanderWeele & Ding (2017), Annals of Internal Medicine 167(4).
"""

import logging
from typing import Dict, Optional

from src.utils.evalue import evalue_ols

logger = logging.getLogger(__name__)

# Assumed SE for the total effect when not available from model output.
# Based on typical OLS SE for c-path in MIDUS MR2 (N≈421).
_DEFAULT_SE_ASSUMPTION = 0.12


def compute_evalue(estimate: float, se: float,
                   sd_outcome: float = 1.0) -> Dict:
    """
    Compute E-value for an OLS estimate with interpretation string.

    Args:
        estimate:   Regression coefficient.
        se:         Standard error of the coefficient.
        sd_outcome: SD of the outcome (for standardization).

    Returns:
        Dict with evalue_point, evalue_ci, rr_approx, interpretation.
    """
    result = evalue_ols(estimate=estimate, se=se, sd_outcome=sd_outcome)

    ep = result['evalue_point']
    ec = result['evalue_ci']
    rr = result['rr_approx']

    interp = (
        f"An unmeasured confounder associated with both SES and cognition "
        f"by a factor of {ep:.2f}-fold (RR≈{rr:.2f}) would be needed to "
        f"explain away the observed effect. The CI E-value is {ec:.2f}."
    )
    result['interpretation'] = interp
    return result


def run_ses_cognition_sensitivity(
    bk_results: Dict,
    sd_outcome: float = 0.821,
    se_assumption: float = _DEFAULT_SE_ASSUMPTION,
) -> Dict:
    """
    Compute E-values for all paths in the SES→cognition mediation model.

    Computes:
      - Total SES effect E-value (c-path, same across all mediators)
      - Direct SES effect E-value (c'-path) per mediator

    Args:
        bk_results:    Dict of mediator → MediationResult from baron_kenny_mediation.
        sd_outcome:    SD of cognitive_score (default 0.821 for MIDUS MR2 z-scores).
        se_assumption: SE used for E-value CI computation (applied to all paths).

    Returns:
        Dict with keys:
          total_effect:   E-value dict for the total SES→cognition effect.
          direct_effects: {mediator: E-value dict} for direct effects.
          se_assumption:  The SE used.
    """
    if not bk_results:
        return {'total_effect': {}, 'direct_effects': {}, 'se_assumption': se_assumption}

    # Total effect (c) is the same across all mediators — take from first
    first = next(iter(bk_results.values()))
    c = first.c

    logger.info("=" * 70)
    logger.info("SENSITIVITY ANALYSIS — E-VALUES")
    logger.info("=" * 70)
    logger.info(f"  Outcome SD: {sd_outcome:.3f}  |  SE assumption: {se_assumption:.3f}")
    logger.info(f"  Total SES effect (c): {c:.4f}")

    total_ev = compute_evalue(estimate=c, se=se_assumption, sd_outcome=sd_outcome)
    logger.info(f"  Total effect E-value: {total_ev['evalue_point']:.2f} "
                f"(CI E-value: {total_ev['evalue_ci']:.2f})")
    logger.info(f"  → {total_ev['interpretation']}")

    direct_effects = {}
    logger.info("\n  Direct effects (c') per mediator:")
    for mediator, bk in bk_results.items():
        ev = compute_evalue(estimate=bk.c_prime, se=se_assumption,
                            sd_outcome=sd_outcome)
        direct_effects[mediator] = ev
        logger.info(f"    [{mediator}]  c'={bk.c_prime:.4f}  "
                    f"E-value={ev['evalue_point']:.2f}  "
                    f"CI E-value={ev['evalue_ci']:.2f}")

    return {
        'total_effect': total_ev,
        'direct_effects': direct_effects,
        'se_assumption': se_assumption,
    }
