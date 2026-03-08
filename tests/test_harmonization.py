# tests/test_harmonization.py
"""Tests for cross-dataset harmonization."""
import pytest
import pandas as pd
import numpy as np


def test_z_standardize():
    """Z-standardization should produce mean~0, sd~1."""
    from src.data.harmonization import z_standardize

    s = pd.Series([10, 20, 30, 40, 50])
    result = z_standardize(s)
    assert abs(result.mean()) < 0.01
    assert abs(result.std() - 1.0) < 0.01


def test_harmonize_datasets():
    """Harmonization should produce combined DataFrame with dataset labels."""
    from src.data.harmonization import harmonize_datasets

    nhanes = pd.DataFrame({
        'cognitive_score': [0.5, -0.3, 0.1],
        'ses_index': [0.2, 0.5, 0.8],
    })
    gss = pd.DataFrame({
        'cognitive_score': [60, 70, 80],
        'ses_index': [0.3, 0.6, 0.9],
    })

    result = harmonize_datasets({'nhanes': nhanes, 'gss': gss})
    assert 'dataset' in result.columns
    assert 'cognitive_z' in result.columns
    assert 'ses_z' in result.columns
    assert len(result) == 6


def test_harmonize_addhealth_columns():
    """Harmonization should include Add Health with expected columns."""
    from src.data.harmonization import harmonize_datasets

    addhealth = pd.DataFrame({
        'cognitive_score': [0.5, -0.3, 0.1],
        'ses_index': [0.2, 0.5, 0.8],
        'age': [35, 38, 40],
        'female': [0.0, 1.0, 0.0],
        'has_insurance': [1.0, 0.0, 1.0],
        'general_health': [4.0, 3.0, 5.0],
        'bmi': [25.0, 30.0, 22.0],
        'digits_backward': [4, 5, 3],
        'social_ladder': [5.0, 7.0, 3.0],
    })

    result = harmonize_datasets({'addhealth': addhealth})
    assert 'dataset' in result.columns
    assert result['dataset'].iloc[0] == 'addhealth'
    assert 'cognitive_z' in result.columns
    assert 'ses_z' in result.columns
    assert len(result) == 3


def test_harmonize_nhanes_no_cognitive():
    """NHANES with all-zero cognitive_score should have cognitive_z as NaN."""
    from src.data.harmonization import harmonize_datasets

    nhanes = pd.DataFrame({
        'cognitive_score': [0.0, 0.0, 0.0],
        'ses_index': [0.2, 0.5, 0.8],
        'has_insurance': [1.0, 0.0, 1.0],
    })
    addhealth = pd.DataFrame({
        'cognitive_score': [0.5, -0.3, 0.1],
        'ses_index': [0.2, 0.5, 0.8],
    })

    result = harmonize_datasets({'nhanes': nhanes, 'addhealth': addhealth})
    nhanes_rows = result[result['dataset'] == 'nhanes']
    # All-zero cognitive_score has zero variance -> cognitive_z should be NaN
    assert nhanes_rows['cognitive_z'].isna().all() or (nhanes_rows['cognitive_z'] == 0).all()
    # NHANES rows still present for SES analysis
    assert len(nhanes_rows) == 3


def test_get_available_mediators():
    """Should return dict of available mediator columns per dataset."""
    from src.data.harmonization import get_available_mediators

    addhealth = pd.DataFrame({
        'has_insurance': [1.0, 0.0, 1.0],
        'general_health': [4.0, 3.0, 5.0],
        'bmi': [25.0, np.nan, np.nan],  # <50% valid, should be excluded
        'cognitive_score': [0.5, -0.3, 0.1],
        'ses_index': [0.2, 0.5, 0.8],
        'dataset': ['addhealth'] * 3,
    })
    nhanes = pd.DataFrame({
        'has_insurance': [1.0, 0.0, 1.0],
        'depression_score': [2.0, 5.0, 8.0],
        'cognitive_score': [0.0, 0.0, 0.0],
        'ses_index': [0.2, 0.5, 0.8],
        'dataset': ['nhanes'] * 3,
    })

    combined = pd.concat([addhealth, nhanes], ignore_index=True)
    mediator_cols = ['has_insurance', 'general_health', 'bmi', 'depression_score']
    result = get_available_mediators(combined, mediator_cols)

    assert 'addhealth' in result
    assert 'has_insurance' in result['addhealth']
    assert 'general_health' in result['addhealth']
    assert 'bmi' not in result['addhealth']  # too many missing
    assert 'nhanes' in result
    assert 'depression_score' in result['nhanes']


def test_harmonize_with_piaac():
    """PIAAC cognitive_score should harmonize with z-standardization."""
    from src.data.harmonization import harmonize_datasets

    piaac = pd.DataFrame({
        'cognitive_score': [0.5, -0.2, 0.3, 0.1],
        'ses_index': [0.1, 0.4, 0.7, 0.9],
        'female': [0.0, 1.0, 0.0, 1.0],
    })
    result = harmonize_datasets({'piaac': piaac})
    assert result['dataset'].iloc[0] == 'piaac'
    assert 'cognitive_z' in result.columns
    assert abs(result['cognitive_z'].mean()) < 0.01


def test_harmonize_with_nsduh():
    """NSDUH k6_score should be available as mediator; no cognitive_score."""
    from src.data.harmonization import harmonize_datasets, get_available_mediators

    nsduh = pd.DataFrame({
        'k6_score': [8.0, 12.0, 18.0, 10.0],
        'ses_index': [0.2, 0.5, 0.3, 0.8],
        'functional_impairment': [2.0, 5.0, 10.0, 3.0],
        'has_insurance': [1.0, 0.0, 1.0, 1.0],
        'dataset': ['nsduh'] * 4,
    })
    result = harmonize_datasets({'nsduh': nsduh})
    assert result['dataset'].iloc[0] == 'nsduh'
    assert 'ses_z' in result.columns
    # No cognitive_score → no cognitive_z
    assert 'cognitive_z' not in result.columns or result['cognitive_z'].isna().all()

    mediators = get_available_mediators(
        nsduh, ['k6_score', 'functional_impairment', 'has_insurance']
    )
    assert 'k6_score' in mediators['nsduh']
    assert 'functional_impairment' in mediators['nsduh']
