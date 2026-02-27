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
