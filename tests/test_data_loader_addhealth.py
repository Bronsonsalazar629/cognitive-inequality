# tests/test_data_loader_addhealth.py
"""Tests for Add Health Wave V data loader."""

import pytest
import pandas as pd
import numpy as np


def test_create_cognitive_composite():
    """Cognitive composite should be mean of z-scored word recall scores."""
    from src.data.data_loader_addhealth import create_cognitive_composite

    df = pd.DataFrame(
        {
            "C5WD90_1": [5.0, 8.0, 10.0, 12.0],  # immediate recall (0-15)
            "C5WD60_1": [3.0, 5.0, 7.0, 10.0],  # delayed recall (0-12)
        }
    )
    result = create_cognitive_composite(df)
    assert "cognitive_score" in result.columns
    assert abs(result["cognitive_score"].mean()) < 0.01  # z-scored mean ~ 0
    assert abs(result["cognitive_score"].std() - 1.0) < 0.3  # SD ~ 1


def test_create_cognitive_composite_missing_codes():
    """Should treat 995/996/999 as NaN in cognitive scores."""
    from src.data.data_loader_addhealth import create_cognitive_composite

    df = pd.DataFrame(
        {
            "C5WD90_1": [5.0, 995.0, 8.0],
            "C5WD60_1": [3.0, 5.0, 999.0],
        }
    )
    result = create_cognitive_composite(df)
    assert pd.isna(result["cognitive_score"].iloc[1])
    assert pd.isna(result["cognitive_score"].iloc[2])
    assert not pd.isna(result["cognitive_score"].iloc[0])


# hello this is testing


def test_create_digits_backward_score():
    """Digits backward should count consecutive correct pairs."""
    from src.data.data_loader_addhealth import create_digits_backward_score

    # H5MH3A-9B: pairs of trials at increasing lengths
    # Each A/B pair: 0=fail, 1=pass. Score = highest length with at least 1 pass
    df = pd.DataFrame(
        {
            "H5MH3A": [1.0, 1.0, 0.0],  # 2-digit: pass
            "H5MH3B": [1.0, 0.0, 0.0],  # 2-digit trial B
            "H5MH4A": [1.0, 1.0, 0.0],  # 3-digit
            "H5MH4B": [1.0, 0.0, 0.0],  # 3-digit trial B
            "H5MH5A": [1.0, 0.0, 0.0],  # 4-digit
            "H5MH5B": [0.0, 0.0, 0.0],  # 4-digit trial B
            "H5MH6A": [0.0, 0.0, 0.0],  # 5-digit
            "H5MH6B": [0.0, 0.0, 0.0],
            "H5MH7A": [0.0, 0.0, 0.0],  # 6-digit
            "H5MH7B": [0.0, 0.0, 0.0],
            "H5MH8A": [0.0, 0.0, 0.0],  # 7-digit
            "H5MH8B": [0.0, 0.0, 0.0],
            "H5MH9A": [0.0, 0.0, 0.0],  # 8-digit
            "H5MH9B": [0.0, 0.0, 0.0],
        }
    )
    result = create_digits_backward_score(df)
    assert "digits_backward" in result.columns
    assert result["digits_backward"].iloc[0] == 4  # passed up to 4-digit
    assert result["digits_backward"].iloc[1] == 3  # passed up to 3-digit
    assert result["digits_backward"].iloc[2] == 0  # failed all


def test_create_ses_index():
    """SES index should combine household income and education."""
    from src.data.data_loader_addhealth import create_ses_index

    df = pd.DataFrame(
        {
            "H5EC2": [1.0, 7.0, 13.0],  # household income 1-13
            "H5EL7": [1.0, 5.0, 10.0],  # highest education level
        }
    )
    result = create_ses_index(df)
    assert "ses_index" in result.columns
    assert result["ses_index"].min() >= 0.0
    assert result["ses_index"].max() <= 1.0
    # Lowest income + lowest education = lowest SES
    assert result["ses_index"].iloc[0] == result["ses_index"].min()


def test_create_ses_index_missing_codes():
    """SES should treat 997/998 as NaN."""
    from src.data.data_loader_addhealth import create_ses_index

    df = pd.DataFrame(
        {
            "H5EC2": [997.0, 7.0, 998.0],
            "H5EL7": [1.0, 5.0, 10.0],
        }
    )
    result = create_ses_index(df)
    assert pd.isna(result["ses_index"].iloc[0])
    assert pd.isna(result["ses_index"].iloc[2])
    assert not pd.isna(result["ses_index"].iloc[1])


def test_create_healthcare_access():
    """Healthcare access from employer insurance variables."""
    from src.data.data_loader_addhealth import create_healthcare_access

    df = pd.DataFrame(
        {
            "H5LM13A": [1.0, 0.0, 97.0, 97.0],  # current employer provides health ins
            "H5LM23A": [0.0, 1.0, 0.0, 97.0],  # previous employer provided health ins
        }
    )
    result = create_healthcare_access(df)
    assert "has_insurance" in result.columns
    assert result["has_insurance"].iloc[0] == 1.0  # current yes
    assert result["has_insurance"].iloc[1] == 1.0  # previous yes
    assert result["has_insurance"].iloc[2] == 0.0  # current skip, previous no
    assert pd.isna(result["has_insurance"].iloc[3])  # both skip = unknown


def test_create_confounders():
    """Should create sex, race, BMI, and health confounders."""
    from src.data.data_loader_addhealth import create_confounders

    df = pd.DataFrame(
        {
            "H5OD2A": [1.0, 2.0, 1.0],  # 1=male, 2=female
            "H5OD4A": [1.0, 0.0, 0.0],  # white
            "H5OD4B": [0.0, 1.0, 0.0],  # black
            "H5OD4C": [0.0, 0.0, 1.0],  # hispanic
            "H5ID1": [1.0, 3.0, 5.0],  # general health 1-5
            "H5ID3": [150.0, 200.0, 250.0],  # weight lbs
            "H5ID2F": [5.0, 6.0, 5.0],  # height feet
            "H5ID2I": [8.0, 0.0, 4.0],  # height inches
        }
    )
    result = create_confounders(df)
    assert "female" in result.columns
    assert result["female"].iloc[0] == 0.0
    assert result["female"].iloc[1] == 1.0
    assert "bmi" in result.columns
    assert result["bmi"].iloc[0] > 0


def test_compute_age():
    """Should compute age from birth year and survey year."""
    from src.data.data_loader_addhealth import compute_age

    df = pd.DataFrame(
        {
            "H5OD1Y": [1980, 1978, 1982],
            "IYEAR5": [2017, 2017, 2018],
        }
    )
    result = compute_age(df)
    assert "age" in result.columns
    assert result["age"].iloc[0] == 37
    assert result["age"].iloc[1] == 39
    assert result["age"].iloc[2] == 36


def test_clean_missing_codes():
    """Should replace Add Health missing codes with NaN."""
    from src.data.data_loader_addhealth import clean_missing_codes

    df = pd.DataFrame(
        {
            "A": [1.0, 95.0, 97.0, 5.0],
            "B": [10.0, 995.0, 997.0, 999.0],
        }
    )
    result = clean_missing_codes(df, small_cols=["A"], large_cols=["B"])
    assert pd.isna(result["A"].iloc[1])
    assert pd.isna(result["A"].iloc[2])
    assert not pd.isna(result["A"].iloc[0])
    assert pd.isna(result["B"].iloc[1])
    assert pd.isna(result["B"].iloc[2])
    assert pd.isna(result["B"].iloc[3])
