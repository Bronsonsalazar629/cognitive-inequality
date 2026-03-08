"""
PIAAC Cycle 2 (2023) Data Loader

Loads and preprocesses the Programme for International Assessment of Adult
Competencies public-use data for cognitive inequality analysis.

Data source: OECD/NCES PIAAC Cycle 2 public-use files (semicolon-delimited CSV).
Target population: Adults ages 16-65, filtered to 16-29 for young adult analysis.
Output: Processed DataFrame with cognitive composites, SES index, and demographics.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# PIAAC missing value codes (string representations in CSV)
PIAAC_MISSING = {'.n', '.v', '.', '.d', '.r'}


def _clean_piaac_missing(series: pd.Series) -> pd.Series:
    """Replace PIAAC missing codes with NaN, convert to float."""
    result = series.copy()
    result = result.replace(PIAAC_MISSING, np.nan)
    return pd.to_numeric(result, errors='coerce')


def filter_young_adults(df: pd.DataFrame, max_age_group: int = 3) -> pd.DataFrame:
    """
    Filter to young adults by AGEG5LFS.

    Age groups: 1=16-19, 2=20-24, 3=25-29, 4=30-34, ..., 10=60-65.
    Default max_age_group=3 gives ages 16-29.
    """
    return df[df['AGEG5LFS'] <= max_age_group].copy()


def create_cognitive_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cognitive composite from plausible values.

    Averages 10 plausible values each for literacy (PVLIT), numeracy (PVNUM),
    and adaptive problem-solving (PVAPS). Z-scores each domain, then
    takes the mean as the composite cognitive_score.
    """
    df = df.copy()

    for domain, prefix in [('literacy_score', 'PVLIT'),
                           ('numeracy_score', 'PVNUM'),
                           ('problem_solving_score', 'PVAPS')]:
        pv_cols = [f'{prefix}{i}' for i in range(1, 11)]
        available = [c for c in pv_cols if c in df.columns]
        if available:
            for col in available:
                df[col] = _clean_piaac_missing(df[col])
            df[domain] = df[available].mean(axis=1)
        else:
            df[domain] = np.nan

    # Z-score each domain
    z_cols = []
    for domain in ['literacy_score', 'numeracy_score', 'problem_solving_score']:
        z_col = f'{domain}_z'
        mean = df[domain].mean()
        std = df[domain].std()
        if std > 0:
            df[z_col] = (df[domain] - mean) / std
        else:
            df[z_col] = 0.0
        z_cols.append(z_col)

    # Composite = mean of z-scores
    df['cognitive_score'] = df[z_cols].mean(axis=1)

    return df


def create_ses_index(df: pd.DataFrame,
                     earnings_weight: float = 0.6,
                     education_weight: float = 0.4) -> pd.DataFrame:
    """
    Create SES index from earnings decile and parental education.

    EARNMTHALLDCLC2: Monthly earnings decile (1-10)
    PAREDC2: Parental education (1=below secondary, 2=secondary, 3=tertiary)
    """
    df = df.copy()

    # Clean and normalize earnings
    if 'EARNMTHALLDCLC2' in df.columns:
        df['earnings'] = _clean_piaac_missing(df['EARNMTHALLDCLC2'])
        e_min, e_max = df['earnings'].min(), df['earnings'].max()
        if pd.notna(e_min) and pd.notna(e_max) and e_max > e_min:
            df['earnings_norm'] = (df['earnings'] - e_min) / (e_max - e_min)
        else:
            df['earnings_norm'] = np.nan
    else:
        df['earnings_norm'] = np.nan

    # Clean and normalize parental education
    if 'PAREDC2' in df.columns:
        df['pared'] = _clean_piaac_missing(df['PAREDC2'])
        p_min, p_max = df['pared'].min(), df['pared'].max()
        if pd.notna(p_min) and pd.notna(p_max) and p_max > p_min:
            df['education_norm'] = (df['pared'] - p_min) / (p_max - p_min)
        else:
            df['education_norm'] = np.nan
    else:
        df['education_norm'] = np.nan

    # Weighted composite
    df['ses_index'] = np.where(
        df['earnings_norm'].isna() | df['education_norm'].isna(),
        np.nan,
        earnings_weight * df['earnings_norm'] + education_weight * df['education_norm'],
    )

    return df


def load_piaac(data_path: str = None,
               max_age_group: int = 3,
               output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess PIAAC Cycle 2 dataset.

    Args:
        data_path: Path to semicolon-delimited CSV
        max_age_group: Maximum AGEG5LFS group to include (default 3 = ages 16-29)
        output_path: Optional path to save processed CSV

    Returns:
        Processed DataFrame
    """
    if data_path is None:
        data_path = 'docs/plans/prgusap2.csv'

    logger.info(f"Loading PIAAC from {data_path}")
    df = pd.read_csv(data_path, sep=';', low_memory=False)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Filter to young adults
    df = filter_young_adults(df, max_age_group=max_age_group)
    logger.info(f"After age filter: {len(df)} rows")

    # Create composites
    df = create_cognitive_scores(df)
    df = create_ses_index(df)

    # Demographics
    df['female'] = (df['GENDER_R'] == 2).astype(float)
    df['age_group'] = df['AGEG5LFS']

    # Select final columns
    final_cols = [
        'age_group', 'female', 'GENDER_R', 'CNTRYID_E',
        'literacy_score', 'numeracy_score', 'problem_solving_score',
        'cognitive_score', 'ses_index',
        'earnings_norm', 'education_norm',
    ]
    available = [c for c in final_cols if c in df.columns]
    df = df[available]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed PIAAC to {output_path}")

    return df
