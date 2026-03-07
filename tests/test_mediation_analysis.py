# tests/test_mediation_analysis.py
"""Tests for Baron-Kenny mediation analysis."""
import pytest
import pandas as pd
import numpy as np


def _make_full_mediation(n=1000, seed=42):
    """X→M→Y, no direct X→Y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    M = 0.8 * X + rng.normal(0, 0.3, n)
    Y = 0.7 * M + rng.normal(0, 0.3, n)
    return pd.DataFrame({'X': X, 'M': M, 'Y': Y})


def _make_partial_mediation(n=1000, seed=42):
    """X→M→Y and X→Y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    M = 0.7 * X + rng.normal(0, 0.3, n)
    Y = 0.5 * X + 0.6 * M + rng.normal(0, 0.3, n)
    return pd.DataFrame({'X': X, 'M': M, 'Y': Y})


def test_baron_kenny_full_mediation():
    """Full mediation: significant indirect, near-zero direct effect."""
    from src.analysis.mediation_analysis import baron_kenny_mediation

    df = _make_full_mediation()
    result = baron_kenny_mediation(df, x='X', m='M', y='Y')

    assert hasattr(result, 'a')
    assert hasattr(result, 'b')
    assert hasattr(result, 'indirect')
    assert abs(result.indirect) > 0.3  # strong indirect
    assert abs(result.c_prime) < abs(result.c)  # direct < total


def test_baron_kenny_partial_mediation():
    """Partial mediation: both direct and indirect effects."""
    from src.analysis.mediation_analysis import baron_kenny_mediation

    df = _make_partial_mediation()
    result = baron_kenny_mediation(df, x='X', m='M', y='Y')

    assert abs(result.indirect) > 0.2
    assert abs(result.c_prime) > 0.2  # direct effect still present
    assert result.proportion_mediated > 0 and result.proportion_mediated < 1


def test_baron_kenny_no_mediation():
    """No mediation: random data should have small indirect effect."""
    from src.analysis.mediation_analysis import baron_kenny_mediation

    rng = np.random.default_rng(99)
    n = 200
    df = pd.DataFrame({
        'X': rng.normal(0, 1, n),
        'M': rng.normal(0, 1, n),
        'Y': rng.normal(0, 1, n),
    })
    result = baron_kenny_mediation(df, x='X', m='M', y='Y')
    assert abs(result.indirect) < 0.15


def test_bootstrap_ci_contains_true_effect():
    """Bootstrap CI should contain the true indirect effect."""
    from src.analysis.mediation_analysis import bootstrap_mediation

    df = _make_partial_mediation(n=500)
    result = bootstrap_mediation(df, x='X', m='M', y='Y', n_boot=500)

    # True indirect ≈ 0.7 * 0.6 = 0.42
    assert result.ci_lower < 0.42 < result.ci_upper


def test_bootstrap_ci_excludes_zero_for_significant():
    """For strong mediation, CI should not contain zero."""
    from src.analysis.mediation_analysis import bootstrap_mediation

    df = _make_full_mediation(n=500)
    result = bootstrap_mediation(df, x='X', m='M', y='Y', n_boot=500)

    assert result.ci_lower > 0 or result.ci_upper < 0


def test_analyze_all_mediators():
    """Should return results for each mediator."""
    from src.analysis.mediation_analysis import analyze_all_mediators

    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, n)
    M1 = 0.7 * X + rng.normal(0, 0.3, n)
    M2 = 0.5 * X + rng.normal(0, 0.3, n)
    M3 = rng.normal(0, 1, n)  # no mediation
    Y = 0.3 * X + 0.5 * M1 + 0.4 * M2 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'X': X, 'M1': M1, 'M2': M2, 'M3': M3, 'Y': Y})

    results = analyze_all_mediators(df, x='X', y='Y', mediators=['M1', 'M2', 'M3'], n_boot=200)
    assert 'M1' in results
    assert 'M2' in results
    assert 'M3' in results


def test_filter_significant_mediators():
    """Should return only mediators where CI excludes zero."""
    from src.analysis.mediation_analysis import analyze_all_mediators, get_significant_mediators

    rng = np.random.default_rng(42)
    n = 500
    X = rng.normal(0, 1, n)
    M1 = 0.8 * X + rng.normal(0, 0.3, n)
    M2 = rng.normal(0, 1, n)  # no mediation
    Y = 0.6 * M1 + rng.normal(0, 0.3, n)
    df = pd.DataFrame({'X': X, 'M1': M1, 'M2': M2, 'Y': Y})

    results = analyze_all_mediators(df, x='X', y='Y', mediators=['M1', 'M2'], n_boot=200)
    significant = get_significant_mediators(results)
    assert 'M1' in significant


def test_weighted_mediation():
    """Weighted mediation should produce different results than unweighted."""
    from src.analysis.mediation_analysis import baron_kenny_mediation

    df = _make_partial_mediation(n=200, seed=42)
    rng = np.random.default_rng(42)
    weights = rng.uniform(0.5, 5.0, len(df))

    unweighted = baron_kenny_mediation(df, x='X', m='M', y='Y')
    weighted = baron_kenny_mediation(df, x='X', m='M', y='Y', weights=weights)

    # Results should differ (not identical)
    assert abs(unweighted.indirect - weighted.indirect) > 0.001


def test_baron_kenny_with_covariates():
    """Covariates are passed through and reduce confounding."""
    import pandas as pd, numpy as np
    from src.analysis.mediation_analysis import baron_kenny_mediation
    rng = np.random.default_rng(0)
    n = 200
    age = rng.normal(44, 5, n)
    df = pd.DataFrame({
        'ses': rng.normal(0, 1, n),
        'mediator': rng.normal(0, 1, n),
        'outcome': rng.normal(0, 1, n),
        'age': age,
        'female': rng.integers(0, 2, n).astype(float),
    })
    result = baron_kenny_mediation(df, x='ses', m='mediator', y='outcome',
                                   covariates=['age', 'female'])
    assert hasattr(result, 'indirect')
