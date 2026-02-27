# tests/test_data_loader_nhanes.py
"""Tests for NHANES data loader."""
import pytest
import pandas as pd
import numpy as np


def test_create_cognitive_composite():
    """Cognitive composite should be mean of z-scored tests."""
    from src.data.data_loader_nhanes import create_cognitive_composite

    df = pd.DataFrame({
        'CFDDS': [50.0, 60.0, 70.0, 80.0],
        'CFDCST': [15.0, 20.0, 25.0, 30.0],
        'CFDCSR': [3.0, 5.0, 7.0, 9.0],
    })
    result = create_cognitive_composite(df)
    assert 'cognitive_score' in result.columns
    assert abs(result['cognitive_score'].mean()) < 0.01  # z-scored mean ~ 0
    assert abs(result['cognitive_score'].std() - 1.0) < 0.3  # SD ~ 1


def test_create_ses_index():
    """SES index should be 0-1 normalized weighted composite."""
    from src.data.data_loader_nhanes import create_ses_index

    df = pd.DataFrame({
        'INDFMPIR': [0.0, 2.5, 5.0],
        'DMDEDUC2': [1.0, 3.0, 5.0],
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].min() >= 0.0
    assert result['ses_index'].max() <= 1.0
    # Lowest income + lowest education = lowest SES
    assert result['ses_index'].iloc[0] == result['ses_index'].min()


def test_create_depression_score():
    """Depression score should sum PHQ-9 items."""
    from src.data.data_loader_nhanes import create_depression_score

    df = pd.DataFrame({
        'DPQ020': [0, 1, 2],
        'DPQ030': [0, 1, 2],
        'DPQ040': [0, 1, 2],
        'DPQ050': [0, 1, 2],
        'DPQ060': [0, 1, 2],
        'DPQ070': [0, 1, 2],
        'DPQ080': [0, 1, 2],
        'DPQ090': [0, 1, 2],
        'DPQ100': [0, 0, 0],
    })
    result = create_depression_score(df)
    assert 'depression_score' in result.columns
    assert result['depression_score'].iloc[0] == 0
    assert result['depression_score'].iloc[1] == 8
    assert result['depression_score'].iloc[2] == 16


def test_create_screen_time():
    """Screen time should sum computer and TV hours."""
    from src.data.data_loader_nhanes import create_screen_time

    df = pd.DataFrame({
        'PAQ710': [2.0, 4.0, float('nan')],
        'PAQ715': [3.0, 2.0, 1.0],
    })
    result = create_screen_time(df)
    assert 'screen_time_hours' in result.columns
    assert result['screen_time_hours'].iloc[0] == 5.0
    assert result['screen_time_hours'].iloc[1] == 6.0
    assert pd.isna(result['screen_time_hours'].iloc[2])


def test_filter_age_range():
    """Should filter to ages 25-45."""
    from src.data.data_loader_nhanes import filter_age_range

    df = pd.DataFrame({
        'RIDAGEYR': [20, 25, 35, 45, 50, 60],
        'val': [1, 2, 3, 4, 5, 6],
    })
    result = filter_age_range(df)
    assert len(result) == 3
    assert result['RIDAGEYR'].min() >= 25
    assert result['RIDAGEYR'].max() <= 45
