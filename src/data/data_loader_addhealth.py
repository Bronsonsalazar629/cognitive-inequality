"""
Add Health Wave V Data Loader

Loads and preprocesses the National Longitudinal Study of Adolescent to Adult
Health (Add Health) Wave V public-use data for cognitive inequality analysis.

Data source: ICPSR Study 21600, DS32 (main survey) and DS42 (weights).
Target population: US adults ages 33-43 (born 1976-1983), surveyed 2016-2018.
Output: Processed DataFrame with cognitive composites, SES index, and survey weights.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Missing value codes used in Add Health
SMALL_MISSING = {95, 96, 97, 98, 99}       # for single/double digit fields
LARGE_MISSING = {995, 996, 997, 998, 999}   # for triple digit fields
VERY_LARGE_MISSING = {9995, 9997, 99997, 99999, 999997}  # for continuous fields


def clean_missing_codes(df: pd.DataFrame,
                        small_cols: list = None,
                        large_cols: list = None) -> pd.DataFrame:
    """Replace Add Health missing value codes with NaN."""
    df = df.copy()
    if small_cols:
        for col in small_cols:
            if col in df.columns:
                df.loc[df[col].isin(SMALL_MISSING), col] = np.nan
    if large_cols:
        for col in large_cols:
            if col in df.columns:
                df.loc[df[col].isin(LARGE_MISSING), col] = np.nan
    return df


def compute_age(df: pd.DataFrame) -> pd.DataFrame:
    """Compute age from birth year and survey year."""
    df = df.copy()
    if 'H5OD1Y' in df.columns and 'IYEAR5' in df.columns:
        df['age'] = df['IYEAR5'] - df['H5OD1Y']
    else:
        df['age'] = np.nan
    return df


def create_cognitive_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create z-scored cognitive composite from word recall tests.

    Uses C5WD90_1 (immediate recall, 90sec) and C5WD60_1 (delayed recall, 60sec).
    Missing codes (995, 996, 999) are treated as NaN.
    Requires both tests present for a valid composite.
    """
    df = df.copy()
    recall_cols = ['C5WD90_1', 'C5WD60_1']

    # Clean missing codes
    for col in recall_cols:
        if col in df.columns:
            df.loc[df[col].isin(LARGE_MISSING), col] = np.nan

    # Z-score each test
    z_cols = []
    for col in recall_cols:
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            z_col = f'{col}_z'
            if std > 0:
                df[z_col] = (df[col] - mean) / std
            else:
                df[z_col] = 0.0
            z_cols.append(z_col)
        else:
            z_col = f'{col}_z'
            df[z_col] = np.nan
            z_cols.append(z_col)

    # Composite = mean of z-scores, require both present
    valid_count = df[z_cols].notna().sum(axis=1)
    df['cognitive_score'] = df[z_cols].mean(axis=1)
    df.loc[valid_count < 2, 'cognitive_score'] = np.nan

    return df


def create_digits_backward_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create digits backward span score from H5MH3A-H5MH9B.

    The test presents number strings of increasing length (2-8 digits).
    Each length has two trials (A and B). Score = highest length where
    at least one trial was passed (value=1).
    """
    df = df.copy()

    # Pairs: (trial_A, trial_B) for digit lengths 2 through 8
    pairs = [
        ('H5MH3A', 'H5MH3B', 2),  # 2-digit
        ('H5MH4A', 'H5MH4B', 3),  # 3-digit
        ('H5MH5A', 'H5MH5B', 4),  # 4-digit
        ('H5MH6A', 'H5MH6B', 5),  # 5-digit
        ('H5MH7A', 'H5MH7B', 6),  # 6-digit
        ('H5MH8A', 'H5MH8B', 7),  # 7-digit
        ('H5MH9A', 'H5MH9B', 8),  # 8-digit
    ]

    df['digits_backward'] = 0
    for col_a, col_b, length in pairs:
        if col_a in df.columns and col_b in df.columns:
            passed = (df[col_a] == 1) | (df[col_b] == 1)
            df.loc[passed, 'digits_backward'] = length

    return df


def create_ses_index(df: pd.DataFrame,
                     income_weight: float = 0.6,
                     education_weight: float = 0.4) -> pd.DataFrame:
    """
    Create SES index from household income (H5EC2) and education (H5EL7).

    H5EC2: Total household income, 1-13 scale (1=<$5K to 13=$200K+).
           Missing codes: 997=legitimate skip, 998=don't know.
    H5EL7: Highest education completed (ordinal).

    Normalized to 0-1 scale with configurable weights.
    """
    df = df.copy()

    # Clean missing codes for income
    if 'H5EC2' in df.columns:
        df.loc[df['H5EC2'].isin({997, 998}), 'H5EC2'] = np.nan
        inc_min, inc_max = df['H5EC2'].min(), df['H5EC2'].max()
        if pd.notna(inc_min) and pd.notna(inc_max) and inc_max > inc_min:
            df['income_norm'] = (df['H5EC2'] - inc_min) / (inc_max - inc_min)
        else:
            df['income_norm'] = np.where(df['H5EC2'].isna(), np.nan, 0.5)
    else:
        df['income_norm'] = np.nan

    # Normalize education
    if 'H5EL7' in df.columns:
        edu_min, edu_max = df['H5EL7'].min(), df['H5EL7'].max()
        if pd.notna(edu_min) and pd.notna(edu_max) and edu_max > edu_min:
            df['education_norm'] = (df['H5EL7'] - edu_min) / (edu_max - edu_min)
        else:
            df['education_norm'] = np.where(df['H5EL7'].isna(), np.nan, 0.5)
    else:
        df['education_norm'] = np.nan

    # If either component is NaN, the index is NaN
    df['ses_index'] = np.where(
        df['income_norm'].isna() | df['education_norm'].isna(),
        np.nan,
        income_weight * df['income_norm'] + education_weight * df['education_norm']
    )

    return df


def create_healthcare_access(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create healthcare access indicator from employer insurance.

    H5LM13A: Current employer provides health insurance (0=no, 1=yes, 97=skip)
    H5LM23A: Previous employer provided health insurance (0=no, 1=yes, 97=skip)
    """
    df = df.copy()

    for col in ['H5LM13A', 'H5LM23A']:
        if col in df.columns:
            df.loc[df[col].isin(SMALL_MISSING), col] = np.nan

    if 'H5LM13A' in df.columns and 'H5LM23A' in df.columns:
        # Has insurance if either current or previous employer provided it
        df['has_insurance'] = np.where(
            (df['H5LM13A'] == 1) | (df['H5LM23A'] == 1),
            1.0,
            np.where(
                df['H5LM13A'].isna() & df['H5LM23A'].isna(),
                np.nan,
                0.0
            )
        )
    elif 'H5LM13A' in df.columns:
        df['has_insurance'] = df['H5LM13A']
    else:
        df['has_insurance'] = np.nan

    return df


def create_confounders(df: pd.DataFrame) -> pd.DataFrame:
    """Create confounder variables: sex, race, BMI, general health."""
    df = df.copy()

    # Sex: H5OD2A (1=male, 2=female)
    if 'H5OD2A' in df.columns:
        df['female'] = (df['H5OD2A'] == 2).astype(float)

    # Race indicators (mark-all-that-apply)
    race_map = {
        'H5OD4A': 'race_white',
        'H5OD4B': 'race_black',
        'H5OD4C': 'race_hispanic',
    }
    for src, dst in race_map.items():
        if src in df.columns:
            df[dst] = df[src].astype(float)

    # General health (1=excellent to 5=poor, reverse so higher=better)
    if 'H5ID1' in df.columns:
        df['general_health'] = 6 - df['H5ID1']

    # BMI from height/weight
    if all(c in df.columns for c in ['H5ID3', 'H5ID2F', 'H5ID2I']):
        # Clean missing codes
        for col in ['H5ID3', 'H5ID2F', 'H5ID2I']:
            df.loc[df[col].isin(SMALL_MISSING | {998}), col] = np.nan

        height_inches = df['H5ID2F'] * 12 + df['H5ID2I']
        weight_lbs = df['H5ID3']
        # BMI = (weight_lbs / height_inches^2) * 703
        df['bmi'] = (weight_lbs / (height_inches ** 2)) * 703

    return df


def load_addhealth(data_path: str = None,
                   weights_path: str = None,
                   output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess Add Health Wave V dataset for cognitive analysis.

    Args:
        data_path: Path to DS32 .rda file
        weights_path: Path to DS42 .rda file (survey weights)
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame with all variables
    """
    import pyreadr

    # Default paths
    if data_path is None:
        data_path = 'ICPSR_21600/DS0032/21600-0032-Data.rda'
    if weights_path is None:
        weights_path = 'DS0042/21600-0042-Data.rda'

    # Load main data
    logger.info(f"Loading Add Health Wave V from {data_path}")
    result = pyreadr.read_r(data_path)
    df = list(result.values())[0]
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Load and merge weights
    weights_file = Path(weights_path)
    if weights_file.exists():
        logger.info(f"Loading survey weights from {weights_path}")
        wt_result = pyreadr.read_r(weights_path)
        wt_df = list(wt_result.values())[0]
        df = df.merge(wt_df, on='AID', how='left')
        logger.info(f"Merged weights: {wt_df.columns.tolist()}")
    else:
        logger.warning(f"Weights file not found: {weights_path}")

    # Compute age
    df = compute_age(df)
    logger.info(f"Age range: {df['age'].min()}-{df['age'].max()}")

    # Create composite variables
    df = create_cognitive_composite(df)
    df = create_digits_backward_score(df)
    df = create_ses_index(df)
    df = create_healthcare_access(df)
    df = create_confounders(df)

    # Create SES quartiles
    valid_ses = df['ses_index'].dropna()
    if len(valid_ses) > 0:
        df['ses_quartile'] = pd.qcut(
            df['ses_index'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High']
        )

    # Subjective social status ladder (H5EC9, 1-10)
    if 'H5EC9' in df.columns:
        df['social_ladder'] = df['H5EC9'].replace(
            {95: np.nan, 97: np.nan, 98: np.nan, 99: np.nan}
        )

    # Select final columns
    final_cols = [
        'AID', 'age', 'IYEAR5', 'H5OD2A', 'female',
        'race_white', 'race_black', 'race_hispanic',
        'H5EC1', 'H5EC2', 'H5EL7', 'ses_index', 'ses_quartile',
        'social_ladder',
        'C5WD90_1', 'C5WD60_1', 'cognitive_score', 'digits_backward',
        'has_insurance', 'general_health', 'bmi',
        'H5ID6C',  # high blood pressure
        'GSW5', 'CLUSTER2',  # survey weights
    ]
    available_final = [c for c in final_cols if c in df.columns]
    df = df[available_final]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed Add Health data to {output_path}")

    return df
