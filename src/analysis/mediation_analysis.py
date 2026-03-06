"""
Baron & Kenny Mediation Analysis with Bootstrap CIs

Implements multi-mediator mediation analysis for decomposing the
SES -> Cognitive Function pathway into direct and indirect effects
through available mediators per dataset.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import statsmodels.api as sm

logger = logging.getLogger(__name__)


@dataclass
class MediationResult:
    """Result of Baron-Kenny mediation for a single mediator."""
    a: float          # X→M path coefficient
    b: float          # M→Y path coefficient (controlling for X)
    c: float          # total effect (X→Y)
    c_prime: float    # direct effect (X→Y controlling for M)
    indirect: float   # a * b
    proportion_mediated: float


@dataclass
class BootstrapResult:
    """Bootstrap confidence interval for indirect effect."""
    point_estimate: float
    ci_lower: float
    ci_upper: float
    n_boot: int


def _run_ols(y, X, weights=None):
    """Run OLS or WLS regression, return fitted model."""
    X_with_const = sm.add_constant(X)
    if weights is not None:
        model = sm.WLS(y, X_with_const, weights=weights).fit()
    else:
        model = sm.OLS(y, X_with_const).fit()
    return model


def baron_kenny_mediation(df: pd.DataFrame,
                          x: str, m: str, y: str,
                          covariates: Optional[List[str]] = None,
                          weights: Optional[np.ndarray] = None) -> MediationResult:
    """
    Baron-Kenny mediation analysis for a single mediator.

    Step 1: Y ~ X → total effect c
    Step 2: M ~ X → path a
    Step 3: Y ~ X + M → direct effect c', path b
    """
    data = df[[x, m, y]].copy()
    if covariates:
        for cov in covariates:
            data[cov] = df[cov]
    data = data.dropna()

    w = weights[data.index] if weights is not None else None

    predictors_base = [x]
    if covariates:
        predictors_base.extend(covariates)

    # Step 1: Y ~ X (total effect)
    model_c = _run_ols(data[y], data[predictors_base], weights=w)
    c = model_c.params[x]

    # Step 2: M ~ X (path a)
    model_a = _run_ols(data[m], data[predictors_base], weights=w)
    a = model_a.params[x]

    # Step 3: Y ~ X + M (direct effect, path b)
    predictors_full = predictors_base + [m]
    model_b = _run_ols(data[y], data[predictors_full], weights=w)
    c_prime = model_b.params[x]
    b = model_b.params[m]

    indirect = a * b
    proportion_mediated = indirect / c if abs(c) > 1e-10 else 0.0

    return MediationResult(
        a=a, b=b, c=c, c_prime=c_prime,
        indirect=indirect,
        proportion_mediated=proportion_mediated,
    )


def _compute_indirect(data, x, m, y, covariates, weights):
    """Compute indirect effect (a*b) for one sample."""
    predictors_base = [x]
    if covariates:
        predictors_base.extend(covariates)

    model_a = _run_ols(data[m], data[predictors_base], weights=weights)
    a = model_a.params[x]

    predictors_full = predictors_base + [m]
    model_b = _run_ols(data[y], data[predictors_full], weights=weights)
    b = model_b.params[m]

    return a * b


def bootstrap_mediation(df: pd.DataFrame,
                        x: str, m: str, y: str,
                        covariates: Optional[List[str]] = None,
                        weights: Optional[np.ndarray] = None,
                        n_boot: int = 1000,
                        ci: float = 0.95,
                        seed: int = 42) -> BootstrapResult:
    """
    Bootstrap confidence interval for indirect effect.

    Resamples with replacement, computes indirect effect each iteration.
    Returns percentile CI.
    """
    cols = [x, m, y]
    if covariates:
        cols.extend(covariates)
    data = df[cols].dropna()

    rng = np.random.default_rng(seed)
    indirect_samples = []

    for _ in range(n_boot):
        idx = rng.choice(len(data), size=len(data), replace=True)
        boot_data = data.iloc[idx].reset_index(drop=True)
        boot_w = weights[idx] if weights is not None else None
        try:
            ab = _compute_indirect(boot_data, x, m, y, covariates, boot_w)
            indirect_samples.append(ab)
        except Exception:
            continue

    indirect_samples = np.array(indirect_samples)
    alpha = 1 - ci
    ci_lower = np.percentile(indirect_samples, 100 * alpha / 2)
    ci_upper = np.percentile(indirect_samples, 100 * (1 - alpha / 2))
    point_estimate = np.median(indirect_samples)

    return BootstrapResult(
        point_estimate=point_estimate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_boot=len(indirect_samples),
    )


def analyze_all_mediators(df: pd.DataFrame,
                          x: str, y: str,
                          mediators: List[str],
                          covariates: Optional[List[str]] = None,
                          weights: Optional[np.ndarray] = None,
                          n_boot: int = 1000,
                          seed: int = 42) -> Dict[str, BootstrapResult]:
    """
    Run bootstrap mediation for each mediator.

    Returns dict mapping mediator name to BootstrapResult.
    """
    results = {}
    for m in mediators:
        logger.info(f"Analyzing mediator: {m}")
        results[m] = bootstrap_mediation(
            df, x=x, m=m, y=y,
            covariates=covariates, weights=weights,
            n_boot=n_boot, seed=seed,
        )
    return results


def get_significant_mediators(results: Dict[str, BootstrapResult],
                              alpha: float = 0.05) -> List[str]:
    """Return mediators where bootstrap CI excludes zero."""
    significant = []
    for name, result in results.items():
        if result.ci_lower > 0 or result.ci_upper < 0:
            significant.append(name)
    return significant
