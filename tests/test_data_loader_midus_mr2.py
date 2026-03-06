"""
Tests for MIDUS Refresher 2 data loader.

Tests use synthetic DataFrames that mirror the real MR2 variable
structure — no actual SAV files required.
"""

import pytest
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# create_cognitive_composite
# ---------------------------------------------------------------------------

def test_cognitive_composite_uses_btact():
    from src.data.data_loader_midus_mr2 import create_cognitive_composite

    df = pd.DataFrame({
        'RB3TCOMPZ': [0.5, -0.3, 1.2, np.nan],
        'RB3TEMZ':   [0.4, -0.1, 1.0, 0.8],
        'RB3TEFZ':   [0.6, -0.5, 1.4, 0.9],
    })
    result = create_cognitive_composite(df)
    assert 'cognitive_score' in result.columns
    # Row with NaN composite should still get a score from EM + EF
    assert result['cognitive_score'].notna().sum() == 4


def test_cognitive_composite_requires_at_least_two():
    from src.data.data_loader_midus_mr2 import create_cognitive_composite

    df = pd.DataFrame({
        'RB3TCOMPZ': [np.nan, np.nan],
        'RB3TEMZ':   [np.nan, 0.5],
        'RB3TEFZ':   [np.nan, np.nan],
    })
    result = create_cognitive_composite(df)
    # Row 0: all NaN → NaN. Row 1: only 1 non-NaN → NaN
    assert result['cognitive_score'].isna().all()


def test_cognitive_composite_prefers_btact_composite():
    from src.data.data_loader_midus_mr2 import create_cognitive_composite

    df = pd.DataFrame({
        'RB3TCOMPZ': [1.0],
        'RB3TEMZ':   [0.0],
        'RB3TEFZ':   [0.0],
    })
    result = create_cognitive_composite(df)
    # Should use mean of all three, not just TCOMPZ
    assert result['cognitive_score'].iloc[0] == pytest.approx((1.0 + 0.0 + 0.0) / 3, abs=0.01)


# ---------------------------------------------------------------------------
# create_ses_index
# ---------------------------------------------------------------------------

def test_ses_index_normalized_0_to_1():
    from src.data.data_loader_midus_mr2 import create_ses_index

    df = pd.DataFrame({
        'RB1PB16': [10000, 50000, 100000],
        'RB1PB1':  [3.0, 6.0, 9.0],   # education 1-12 scale
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].min() >= 0.0
    assert result['ses_index'].max() <= 1.0


def test_ses_index_monotonic():
    from src.data.data_loader_midus_mr2 import create_ses_index

    df = pd.DataFrame({
        'RB1PB16': [10000, 50000, 150000],
        'RB1PB1':  [1.0, 5.0, 12.0],
    })
    result = create_ses_index(df)
    ses = result['ses_index']
    assert ses.iloc[0] < ses.iloc[1] < ses.iloc[2]


def test_ses_index_nan_propagation():
    from src.data.data_loader_midus_mr2 import create_ses_index

    df = pd.DataFrame({
        'RB1PB16': [np.nan, 50000],
        'RB1PB1':  [5.0, 5.0],
    })
    result = create_ses_index(df)
    assert pd.isna(result['ses_index'].iloc[0])
    assert pd.notna(result['ses_index'].iloc[1])


# ---------------------------------------------------------------------------
# create_depression_score
# ---------------------------------------------------------------------------

def test_depression_score_binary_flag():
    from src.data.data_loader_midus_mr2 import create_depression_score

    df = pd.DataFrame({'RB1PA60': [1.0, 2.0, 7.0, np.nan]})
    result = create_depression_score(df)
    assert 'depression_score' in result.columns
    assert result['depression_score'].iloc[0] == 1.0   # yes
    assert result['depression_score'].iloc[1] == 0.0   # no
    assert pd.isna(result['depression_score'].iloc[2])  # refused
    assert pd.isna(result['depression_score'].iloc[3])  # missing


# ---------------------------------------------------------------------------
# create_screen_time
# ---------------------------------------------------------------------------

def test_screen_time_change_score():
    from src.data.data_loader_midus_mr2 import create_screen_time

    df = pd.DataFrame({
        'RB1PA108B': [1.0, 2.0, 3.0, 4.0, 5.0],
        # 1=much less, 2=less, 3=same, 4=more, 5=much more
    })
    result = create_screen_time(df)
    assert 'screen_time_change' in result.columns
    # Centered: 3=baseline (0), 5=+2, 1=-2
    assert result['screen_time_change'].iloc[2] == pytest.approx(0.0)
    assert result['screen_time_change'].iloc[4] > 0
    assert result['screen_time_change'].iloc[0] < 0


def test_screen_time_invalid_codes_nan():
    from src.data.data_loader_midus_mr2 import create_screen_time

    df = pd.DataFrame({'RB1PA108B': [7.0, 8.0, 9.0, 3.0]})
    result = create_screen_time(df)
    assert pd.isna(result['screen_time_change'].iloc[0])
    assert pd.isna(result['screen_time_change'].iloc[1])
    assert pd.isna(result['screen_time_change'].iloc[2])
    assert pd.notna(result['screen_time_change'].iloc[3])


# ---------------------------------------------------------------------------
# create_sleep_variable
# ---------------------------------------------------------------------------

def test_sleep_change_centered():
    from src.data.data_loader_midus_mr2 import create_sleep_variable

    df = pd.DataFrame({'RB1PA108J': [1.0, 3.0, 5.0]})
    result = create_sleep_variable(df)
    assert 'sleep_change' in result.columns
    assert result['sleep_change'].iloc[1] == pytest.approx(0.0)


def test_sleep_invalid_codes_nan():
    from src.data.data_loader_midus_mr2 import create_sleep_variable

    df = pd.DataFrame({'RB1PA108J': [7.0, 8.0, 9.0, 3.0]})
    result = create_sleep_variable(df)
    assert result['sleep_change'].isna().sum() == 3
    assert pd.notna(result['sleep_change'].iloc[3])


# ---------------------------------------------------------------------------
# create_healthcare_access
# ---------------------------------------------------------------------------

def test_healthcare_access_binary():
    from src.data.data_loader_midus_mr2 import create_healthcare_access

    df = pd.DataFrame({'RB1SC1': [1.0, 2.0, 8.0, 9.0, np.nan]})
    result = create_healthcare_access(df)
    assert result['has_insurance'].iloc[0] == 1.0
    assert result['has_insurance'].iloc[1] == 0.0
    assert pd.isna(result['has_insurance'].iloc[2])
    assert pd.isna(result['has_insurance'].iloc[3])
    assert pd.isna(result['has_insurance'].iloc[4])


# ---------------------------------------------------------------------------
# filter_age_range
# ---------------------------------------------------------------------------

def test_filter_age_range():
    from src.data.data_loader_midus_mr2 import filter_age_range

    df = pd.DataFrame({'RB1PRAGE': [30, 34, 45, 55, 56, 70]})
    result = filter_age_range(df, min_age=34, max_age=55)
    assert result['RB1PRAGE'].min() >= 34
    assert result['RB1PRAGE'].max() <= 55
    assert len(result) == 3


# ---------------------------------------------------------------------------
# create_confounders
# ---------------------------------------------------------------------------

def test_confounders_sex():
    from src.data.data_loader_midus_mr2 import create_confounders

    df = pd.DataFrame({'RB1PRSEX': [1.0, 2.0, 2.0, np.nan]})
    result = create_confounders(df)
    assert 'female' in result.columns
    assert result['female'].iloc[0] == 0.0
    assert result['female'].iloc[1] == 1.0
    assert pd.isna(result['female'].iloc[3])
