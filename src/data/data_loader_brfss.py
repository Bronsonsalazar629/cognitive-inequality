"""
BRFSS 2022 Data Loader

Loads Behavioral Risk Factor Surveillance System data for validation
of the SES-cognition gradient. Uses self-reported cognitive difficulty
as outcome (binary), with stress, sleep, and insurance as mediators.

Target population: US adults ages 25-44.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

BRFSS_URL = 'https://www.cdc.gov/brfss/annual_data/2022/files/LLCP2022XPT.zip'

BRFSS_COLUMNS = [
    'DECIDE', 'INCOME2', 'EDUCA', 'EMPLOY1',
    'MENTHLTH', 'SLEPTIM1', 'HLTHPLN1', 'GENHLTH',
    '_AGE_G', 'SEX1', '_RACE', 'SMOKDAY2', '_BMI5',
    '_LLCPWT', '_STSTR',
]


def filter_age_range(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to ages 25-44 using _AGE_G (2=25-34, 3=35-44)."""
    return df[df['_AGE_G'].isin([2, 3])].copy()


def create_cognitive_impairment(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary cognitive impairment from DECIDE variable."""
    df = df.copy()
    df['cognitive_impairment'] = np.nan
    df.loc[df['DECIDE'] == 1, 'cognitive_impairment'] = 1.0
    df.loc[df['DECIDE'] == 2, 'cognitive_impairment'] = 0.0
    return df


def create_ses_index(df: pd.DataFrame, income_weight: float = 0.7, education_weight: float = 0.3) -> pd.DataFrame:
    """Create SES index from income and education (70/30 weighting)."""
    df = df.copy()

    if 'INCOME2' in df.columns:
        valid_income = df['INCOME2'].isin(range(1, 9))
        df['income_norm'] = np.nan
        df.loc[valid_income, 'income_norm'] = (df.loc[valid_income, 'INCOME2'] - 1) / 7
    else:
        df['income_norm'] = np.nan

    if 'EDUCA' in df.columns:
        valid_educ = df['EDUCA'].isin(range(1, 7))
        df['education_norm'] = np.nan
        df.loc[valid_educ, 'education_norm'] = (df.loc[valid_educ, 'EDUCA'] - 1) / 5
    else:
        df['education_norm'] = np.nan

    df['ses_index'] = income_weight * df['income_norm'] + education_weight * df['education_norm']
    return df


def create_mediators(df: pd.DataFrame) -> pd.DataFrame:
    """Create mediator variables from BRFSS fields."""
    df = df.copy()

    # Mental health days (stress proxy)
    if 'MENTHLTH' in df.columns:
        df['mental_health_days'] = df['MENTHLTH'].copy()
        df.loc[df['MENTHLTH'] == 88, 'mental_health_days'] = 0  # 88 = None
        df.loc[df['MENTHLTH'].isin([77, 99]), 'mental_health_days'] = np.nan

    # Sleep hours
    if 'SLEPTIM1' in df.columns:
        df['sleep_hours'] = df['SLEPTIM1'].copy()
        df.loc[df['SLEPTIM1'].isin([77, 99]), 'sleep_hours'] = np.nan

    # Insurance
    if 'HLTHPLN1' in df.columns:
        df['has_insurance'] = np.nan
        df.loc[df['HLTHPLN1'] == 1, 'has_insurance'] = 1.0
        df.loc[df['HLTHPLN1'] == 2, 'has_insurance'] = 0.0

    return df


def load_brfss(raw_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess BRFSS 2022 dataset.

    Args:
        raw_path: Path to already-downloaded XPT file (skip download if provided)
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame filtered to ages 25-44
    """
    if raw_path and Path(raw_path).exists():
        logger.info(f"Loading BRFSS from {raw_path}")
        df = pd.read_sas(raw_path)
    else:
        logger.info(f"Downloading BRFSS from CDC (this is ~500MB)...")
        logger.info("Download URL: " + BRFSS_URL)
        logger.info("Please download manually and pass raw_path parameter.")
        raise FileNotFoundError(
            f"BRFSS data not found. Download from {BRFSS_URL}, "
            f"extract the XPT file, and pass its path as raw_path."
        )

    # Keep only needed columns
    available = [c for c in BRFSS_COLUMNS if c in df.columns]
    df = df[available]

    df = filter_age_range(df)
    logger.info(f"After age filter (25-44): {len(df)} rows")

    df = create_cognitive_impairment(df)
    df = create_ses_index(df)
    df = create_mediators(df)

    # SES quartiles
    df['ses_quartile'] = pd.qcut(
        df['ses_index'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High']
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed BRFSS data to {output_path}")

    return df
