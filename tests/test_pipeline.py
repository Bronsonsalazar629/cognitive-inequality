# tests/test_pipeline.py
"""Tests for pipeline orchestrator."""
import pytest
import pandas as pd
import numpy as np


def _make_synthetic_pipeline_data(n=200, seed=42):
    """Create minimal synthetic data mimicking harmonized output."""
    rng = np.random.default_rng(seed)
    ses = rng.normal(0, 1, n)
    ins = rng.binomial(1, 0.5, n).astype(float)
    health = rng.normal(3, 1, n)
    cog = 0.5 * ses + 0.3 * ins + 0.2 * health + rng.normal(0, 0.3, n)
    return pd.DataFrame({
        'ses_index': ses,
        'cognitive_score': cog,
        'has_insurance': ins,
        'general_health': health,
        'age': rng.integers(33, 44, n).astype(float),
        'female': rng.binomial(1, 0.5, n).astype(float),
        'dataset': 'addhealth',
    })


def test_pipeline_runs_end_to_end():
    """Pipeline should run all stages on synthetic data without error."""
    from src.pipeline.run_pipeline import run_full_pipeline

    df = _make_synthetic_pipeline_data()
    config = {
        'alpha': 0.05,
        'n_bootstrap': 100,
        'use_llm': False,
        'mediators': ['has_insurance', 'general_health'],
    }

    results = run_full_pipeline(df, config)
    assert 'pc_discovery' in results
    assert 'mediation' in results
    assert 'prediction' in results
    assert 'counterfactuals' in results


def test_pipeline_skips_missing_stages():
    """If cognitive_score is all NaN, prediction and counterfactual stages should be skipped."""
    from src.pipeline.run_pipeline import run_full_pipeline

    rng = np.random.default_rng(42)
    n = 100
    df = pd.DataFrame({
        'ses_index': rng.normal(0, 1, n),
        'cognitive_score': [np.nan] * n,
        'has_insurance': rng.binomial(1, 0.5, n).astype(float),
        'dataset': 'nhanes',
    })
    config = {
        'alpha': 0.05,
        'n_bootstrap': 50,
        'use_llm': False,
        'mediators': ['has_insurance'],
    }

    results = run_full_pipeline(df, config)
    assert results['prediction'] is None
    assert results['counterfactuals'] is None
