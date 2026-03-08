# tests/test_data_loader_nsduh.py
"""Tests for NSDUH 2024 data loader."""
import pytest
import pandas as pd
import numpy as np


def _make_nsduh_data(n=10):
    """Create synthetic NSDUH-like data."""
    return pd.DataFrame({
        'CATAGE': [1.0, 1.0, 2.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0],
        'AGE3': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 1.0, 4.0, 6.0, 8.0],
        'IRSEX': [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        'NEWRACE2': [1.0, 2.0, 7.0, 1.0, 3.0, 5.0, 2.0, 1.0, 4.0, 6.0],
        'INCOME': [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 1.0, 4.0, 2.0, 3.0],
        'ANYEDUC3': [1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0],
        'DSTNRV30': [1.0, 3.0, 2.0, 4.0, 1.0, 5.0, 2.0, 3.0, 97.0, 1.0],
        'DSTHOP30': [1.0, 2.0, 3.0, 5.0, 1.0, 4.0, 1.0, 2.0, 1.0, 3.0],
        'DSTRST30': [2.0, 3.0, 1.0, 3.0, 2.0, 4.0, 3.0, 1.0, 2.0, 2.0],
        'DSTCHR30': [1.0, 4.0, 2.0, 5.0, 85.0, 3.0, 1.0, 2.0, 3.0, 1.0],
        'DSTEFF30': [1.0, 2.0, 1.0, 4.0, 1.0, 5.0, 2.0, 3.0, 1.0, 2.0],
        'DSTNGD30': [1.0, 3.0, 2.0, 3.0, 1.0, 4.0, 1.0, 2.0, 2.0, 94.0],
        'WHODASTOTSC': [0.0, 5.0, 3.0, 12.0, np.nan, 8.0, 1.0, 6.0, np.nan, 4.0],
        'SPDPSTYR': [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'SMIPY': [0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'AMDELT': [2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0],
        'IRINSUR4': [1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        'HEALTH': [1.0, 3.0, 2.0, 4.0, 5.0, 94.0, 1.0, 2.0, 97.0, 3.0],
        'ANALWT2_C': [1000.0] * 10,
    })


def test_filter_nsduh_young():
    """Should filter to CATAGE 1-2 (ages 12-25)."""
    from src.data.data_loader_nsduh import filter_adolescents_young_adults

    df = _make_nsduh_data()
    result = filter_adolescents_young_adults(df)
    assert len(result) == 6  # CATAGE 1 (3 rows) + CATAGE 2 (3 rows)
    assert result['CATAGE'].max() <= 2.0


def test_k6_score():
    """K6 should sum 6 items (each 1-5), range 6-30."""
    from src.data.data_loader_nsduh import create_k6_score

    df = pd.DataFrame({
        'DSTNRV30': [1.0, 5.0],
        'DSTHOP30': [1.0, 5.0],
        'DSTRST30': [1.0, 5.0],
        'DSTCHR30': [1.0, 5.0],
        'DSTEFF30': [1.0, 5.0],
        'DSTNGD30': [1.0, 5.0],
    })
    result = create_k6_score(df)
    assert 'k6_score' in result.columns
    assert result['k6_score'].iloc[0] == 6.0   # all 1s
    assert result['k6_score'].iloc[1] == 30.0  # all 5s


def test_k6_missing():
    """Rows with missing K6 items (85/94/97/98/99) should get NaN."""
    from src.data.data_loader_nsduh import create_k6_score

    df = pd.DataFrame({
        'DSTNRV30': [1.0, 97.0],
        'DSTHOP30': [1.0, 2.0],
        'DSTRST30': [1.0, 3.0],
        'DSTCHR30': [85.0, 2.0],
        'DSTEFF30': [1.0, 1.0],
        'DSTNGD30': [1.0, 94.0],
    })
    result = create_k6_score(df)
    assert pd.isna(result['k6_score'].iloc[0])  # DSTCHR30=85
    assert pd.isna(result['k6_score'].iloc[1])  # DSTNRV30=97, DSTNGD30=94


def test_nsduh_ses_index():
    """Should combine INCOME (1-4) and ANYEDUC3 (1-2) into 0-1 SES."""
    from src.data.data_loader_nsduh import create_ses_index

    df = pd.DataFrame({
        'INCOME': [1.0, 2.5, 4.0],
        'ANYEDUC3': [1.0, 1.5, 2.0],
    })
    result = create_ses_index(df)
    assert 'ses_index' in result.columns
    assert result['ses_index'].min() >= 0.0
    assert result['ses_index'].max() <= 1.0
    assert result['ses_index'].iloc[0] == result['ses_index'].min()


def test_functional_impairment():
    """WHODAS total score should be preserved as functional_impairment."""
    from src.data.data_loader_nsduh import create_functional_impairment

    df = pd.DataFrame({
        'WHODASTOTSC': [0.0, 12.0, 24.0, np.nan],
    })
    result = create_functional_impairment(df)
    assert 'functional_impairment' in result.columns
    assert result['functional_impairment'].iloc[0] == 0.0
    assert result['functional_impairment'].iloc[2] == 24.0
    assert pd.isna(result['functional_impairment'].iloc[3])


def test_nsduh_confounders():
    """Should create female, race indicators, general_health (reversed)."""
    from src.data.data_loader_nsduh import create_confounders

    df = pd.DataFrame({
        'IRSEX': [1.0, 2.0],
        'NEWRACE2': [1.0, 2.0],
        'HEALTH': [1.0, 5.0],
    })
    result = create_confounders(df)
    assert result['female'].iloc[0] == 0.0
    assert result['female'].iloc[1] == 1.0
    assert result['race_white'].iloc[0] == 1.0
    assert result['race_black'].iloc[1] == 1.0
    # Health reversed: 1=excellent→5, 5=poor→1
    assert result['general_health'].iloc[0] == 5.0
    assert result['general_health'].iloc[1] == 1.0


def test_load_nsduh():
    """End-to-end: should load from RData and return processed DataFrame."""
    from src.data.data_loader_nsduh import load_nsduh

    result = load_nsduh(data_path='docs/plans/NSDUH_2024.RData')
    assert 'k6_score' in result.columns
    assert 'ses_index' in result.columns
    assert 'functional_impairment' in result.columns
    assert 'has_insurance' in result.columns
    assert 'female' in result.columns
    assert 'general_health' in result.columns
    assert len(result) > 20000  # should have ~25K adolescents+young adults
    # Should be filtered to ages 12-25
    assert result['age_category'].max() <= 2.0
