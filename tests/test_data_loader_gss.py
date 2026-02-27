# tests/test_data_loader_gss.py
"""Tests for GSS data loader."""
import pytest
import pandas as pd
import numpy as np


def test_create_gss_ses_index():
    """SES index should use 50/30/20 weighting."""
    from src.data.data_loader_gss import create_ses_index

    df = pd.DataFrame({
        'REALINC': [10000, 30000, 60000],
        'EDUC': [8, 12, 18],
        'PRESTG80': [20, 40, 70],
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].iloc[0] < result['ses_index'].iloc[2]


def test_create_cognitive_score():
    """Cognitive score should scale WORDSUM to 0-100."""
    from src.data.data_loader_gss import create_cognitive_score

    df = pd.DataFrame({'WORDSUM': [0, 5, 10]})
    result = create_cognitive_score(df)
    assert result['cognitive_score'].iloc[0] == 0.0
    assert result['cognitive_score'].iloc[1] == 50.0
    assert result['cognitive_score'].iloc[2] == 100.0


def test_create_screen_time():
    """Screen time should convert weekly to daily hours."""
    from src.data.data_loader_gss import create_screen_time

    df = pd.DataFrame({'WWWHRS': [7, 14, 0]})
    result = create_screen_time(df)
    assert abs(result['screen_hours_daily'].iloc[0] - 1.0) < 0.01
    assert abs(result['screen_hours_daily'].iloc[1] - 2.0) < 0.01
