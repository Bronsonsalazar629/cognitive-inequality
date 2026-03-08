# tests/test_data_loader_brfss.py
"""Tests for BRFSS data loader."""
import pytest
import pandas as pd
import numpy as np


def test_create_brfss_ses_index():
    """SES index should use 70/30 income/education weighting."""
    from src.data.data_loader_brfss import create_ses_index

    df = pd.DataFrame({
        'INCOME2': [1, 4, 8],   # 1=<$10K, 8=>=75K
        'EDUCA': [1, 3, 6],     # 1=Never, 6=College grad
    })
 = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].iloc[0] < result['ses_index'].iloc[2]
    assert result['ses_index'].min() >= 0.0
    assert result['ses_index'].max() <= 1.0


def test_create_cognitive_impairment():
    """Binary cognitive impairment from DECIDE variable."""
    from src.data.data_loader_brfss import create_cognitive_impairment

    df = pd.DataFrame({
        'DECIDE': [1, 2, 7, 9],  # 1=Yes, 2=No, 7=Refused, 9=DK
    })
    result = create_cognitive_impairment(df)
    assert result['cognitive_impairment'].iloc[0] == 1.0
    assert result['cognitive_impairment'].iloc[1] == 0.0
    assert pd.isna(result['cognitive_impairment'].iloc[2])
    assert pd.isna(result['cognitive_impairment'].iloc[3])


def test_filter_brfss_age():
    """Should filter to ages 25-44."""
    from src.data.data_loader_brfss import filter_age_range

    df = pd.DataFrame({
        '_AGE_G': [1, 2, 3, 4, 5],  # age groups
        'val': range(5),
    })
    result = filter_age_range(df)
    assert all(result['_AGE_G'].isin([2, 3]))
