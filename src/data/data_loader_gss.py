"""
GSS 2010-2022 Data Loader

Loads General Social Survey cumulative data for longitudinal trend analysis
of the SES-cognition gradient. Uses vocabulary test (WORDSUM) as cognitive
measure and internet hours (WWWHRS) as screen time proxy.

Target population: US adults ages 25-45, years 2010-2022.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

GSS_COLUMNS = [
    'YEAR', 'AGE', 'SEX', 'RACE', 'HEALTH',
    'WORDSUM', 'REALINC', 'EDUC', 'DEGREE', 'PRESTG80',
    'WWWHRS', 'EMAILHRS',
]


def filter_gss(df: pd.DataFrame, min_year: int = 2010, max_year: int = 2022,
               min_age: int = 25, max_age: int = 45) -> pd.DataFrame:
    """Filter to target years and age range."""
    mask = (
        (df['YEAR'] >= min_year) & (df['YEAR'] <= max_year) &
        (df['AGE'] >= min_age) & (df['AGE'] <= max_age)
    )
    return df[mask].copy()


def create_cognitive_score(df: pd.DataFrame) -> pd.DataFrame:
    """Scale WORDSUM (0-10) to 0-100."""
    df = df.copy()
    if 'WORDSUM' in df.columns:
        df['cognitive_score'] = (df['WORDSUM'] / 10) * 100
        # Treat out-of-range as missing
        df.loc[~df['WORDSUM'].between(0, 10), 'cognitive_score'] = np.nan
    else:
        df['cognitive_score'] = np.nan
    return df


def create_ses_index(df: pd.DataFrame, income_w: float = 0.5,
                     educ_w: float = 0.3, prestige_w: float = 0.2) -> pd.DataFrame:
    """Create 3-component SES index (income 50%, education 30%, prestige 20%)."""
    df = df.copy()

    for col, weight_name in [('REALINC', 'income_norm'), ('EDUC', 'educ_norm'), ('PRESTG80', 'prestige_norm')]:
        if col in df.columns:
            col_min, col_max = df[col].min(), df[col].max()
            if col_max > col_min:
                df[weight_name] = (df[col] - col_min) / (col_max - col_min)
            else:
                df[weight_name] = 0.5
        else:
            df[weight_name] = np.nan

    df['ses_index'] = (income_w * df['income_norm'] +
                       educ_w * df['educ_norm'] +
                       prestige_w * df['prestige_norm'])
    return df


def create_screen_time(df: pd.DataFrame) -> pd.DataFrame:
    """Convert weekly internet hours to daily."""
    df = df.copy()
    if 'WWWHRS' in df.columns:
        df['screen_hours_daily'] = df['WWWHRS'] / 7
        # Cap at reasonable maximum
        df.loc[df['screen_hours_daily'] > 18, 'screen_hours_daily'] = np.nan
    else:
        df['screen_hours_daily'] = np.nan
    return df


def load_gss(raw_path: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess GSS cumulative dataset.

    Args:
        raw_path: Path to GSS .dta Stata file
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame filtered to 2010-2022, ages 25-45
    """
    if not raw_path or not Path(raw_path).exists():
        raise FileNotFoundError(
            "GSS data not found. Download the cumulative Stata file from "
            "https://gss.norc.org/get-the-data/stata and pass its path."
        )

    import pyreadstat
    logger.info(f"Loading GSS from {raw_path}")
    df, meta = pyreadstat.read_dta(raw_path, usecols=GSS_COLUMNS)

    df = filter_gss(df)
    logger.info(f"After filtering (2010-2022, ages 25-45): {len(df)} rows")

    df = create_cognitive_score(df)
    df = create_ses_index(df)
    df = create_screen_time(df)

    # SES quartiles
    df['ses_quartile'] = pd.qcut(
        df['ses_index'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High']
    )

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed GSS data to {output_path}")

    return df
