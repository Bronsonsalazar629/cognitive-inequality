# tests/test_data_loader_piaac.py
"""Tests for PIAAC Cycle 2 data loader."""
import pytest
import pandas as pd
import numpy as np


def _make_piaac_data(n=10):
    """Create synthetic PIAAC-like data."""
    rng = np.random.default_rng(42)
    data = {
        'AGEG5LFS': [1, 2, 3, 4, 5, 1, 2, 3, 4, 5],
        'GENDER_R': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        'EARNMTHALLDCLC2': ['3', '7', '5', '.n', '9', '2', '8', '.v', '6', '10'],
        'PAREDC2': ['1', '2', '3', '.d', '2', '1', '3', '.r', '2', '1'],
        'CNTRYID_E': [840] * n,
    }
    # Add 10 plausible values for each domain
    for i in range(1, 11):
        data[f'PVLIT{i}'] = rng.normal(270, 40, n).tolist()
        data[f'PVNUM{i}'] = rng.normal(260, 45, n).tolist()
        data[f'PVAPS{i}'] = rng.normal(280, 35, n).tolist()
    return pd.DataFrame(data)


def test_filter_piaac_young_adults():
    """Should filter to AGEG5LFS 1-3 (ages 16-29)."""
    from src.data.data_loader_piaac import filter_young_adults

    df = _make_piaac_data()
    result = filter_young_adults(df, max_age_group=3)
    assert len(result) == 6  # groups 1,2,3 appear twice each
    assert result['AGEG5LFS'].max() <= 3


def test_piaac_cognitive_scores():
    """Should average plausible values and create z-scored composite."""
    from src.data.data_loader_piaac import create_cognitive_scores

    df = _make_piaac_data()
    result = create_cognitive_scores(df)
    assert 'literacy_score' in result.columns
    assert 'numeracy_score' in result.columns
    assert 'problem_solving_score' in result.columns
    assert 'cognitive_score' in result.columns
    # Z-scored composite should have mean ~ 0
    assert abs(result['cognitive_score'].mean()) < 0.1
    # All rows should have valid scores (no missing PVs in synthetic data)
    assert result['cognitive_score'].notna().all()


def test_piaac_cognitive_missing():
    """Should handle .n/.v as NaN in PV columns."""
    from src.data.data_loader_piaac import create_cognitive_scores

    df = _make_piaac_data()
    # Set some PV values to missing codes
    df.loc[0, 'PVLIT1'] = '.n'
    df.loc[0, 'PVLIT2'] = '.v'
    result = create_cognitive_scores(df)
    # Row 0 should still have a literacy score (8 out of 10 PVs valid)
    assert result['literacy_score'].notna().iloc[0]


def test_piaac_ses_index():
    """Should combine earnings + parental education into 0-1 SES index."""
    from src.data.data_loader_piaac import create_ses_index

    df = pd.DataFrame({
        'EARNMTHALLDCLC2': ['1', '5', '10'],
        'PAREDC2': ['1', '2', '3'],
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].min() >= 0.0
    assert result['ses_index'].max() <= 1.0
    # Lowest earnings + lowest education = lowest SES
    assert result['ses_index'].iloc[0] == result['ses_index'].min()


def test_piaac_ses_missing():
    """Should treat .n/.v/.d/.r as NaN in SES variables."""
    from src.data.data_loader_piaac import create_ses_index

    df = pd.DataFrame({
        'EARNMTHALLDCLC2': ['.n', '5', '.v'],
        'PAREDC2': ['1', '.d', '3'],
    })
    result = create_ses_index(df)
    assert pd.isna(result['ses_index'].iloc[0])
    assert pd.isna(result['ses_index'].iloc[1])
    assert pd.isna(result['ses_index'].iloc[2])


def test_load_piaac():
    """End-to-end: should return DataFrame with expected columns."""
    from src.data.data_loader_piaac import load_piaac

    result = load_piaac(data_path='docs/plans/prgusap2.csv')
    assert 'cognitive_score' in result.columns
    assert 'ses_index' in result.columns
    assert 'female' in result.columns
    assert 'age_group' in result.columns
    assert len(result) > 0
    # Should be filtered to young adults by default
    assert result['age_group'].max() <= 3
