import numpy as np
import pandas as pd
import pytest
from src.analysis.longitudinal_analysis import (
    merge_baseline_followup,
    run_longitudinal_regression,
)


def _make_panel(n=200, ses_effect=0.3):
    rng = np.random.default_rng(42)
    m2id = np.arange(n)
    ses = rng.uniform(0, 1, n)
    baseline_cog = ses * ses_effect + rng.normal(0, 0.8, n)
    followup_cog = baseline_cog + ses * ses_effect + rng.normal(0, 0.5, n)
    mr2 = pd.DataFrame({
        'M2ID': m2id, 'ses_index': ses,
        'cognitive_score': baseline_cog,
        'RB1PRAGE': rng.uniform(34, 55, n),
        'female': rng.integers(0, 2, n).astype(float),
    })
    m3 = pd.DataFrame({
        'M2ID': m2id, 'cognitive_score_m3': followup_cog,
        'ses_index_m3': ses + rng.normal(0, 0.05, n),
    })
    return mr2, m3


def test_merge_baseline_followup():
    mr2, m3 = _make_panel()
    panel = merge_baseline_followup(mr2, m3)
    assert 'cognitive_change' in panel.columns
    assert len(panel) == len(mr2)
    assert panel['cognitive_change'].notna().sum() > 150


def test_longitudinal_regression_ses_positive():
    mr2, m3 = _make_panel(ses_effect=0.5)
    panel = merge_baseline_followup(mr2, m3)
    result = run_longitudinal_regression(panel)
    assert result['ses_coef'] > 0
    assert 'ses_pvalue' in result
    assert 'n' in result
    assert result['n'] > 150
