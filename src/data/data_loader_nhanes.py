"""
NHANES 2013-2014 Data Loader

Loads and preprocesses National Health and Nutrition Examination Survey data
for cognitive inequality analysis. Fetches demographic, cognitive, depression,
screen time, sleep, healthcare access, and confounder variables.

Target population: US adults ages 25-45.
Output: Processed DataFrame with composite scores and survey weights.
"""

import logging
import io
import tempfile
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# NHANES 2013-2014 table URLs (XPT format) - direct data download links
_BASE = 'https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2013/DataFiles'
NHANES_TABLES = {
    'DEMO_H': f'{_BASE}/DEMO_H.XPT',
    'CFQ_H': f'{_BASE}/CFQ_H.XPT',
    'DPQ_H': f'{_BASE}/DPQ_H.XPT',
    'PAQ_H': f'{_BASE}/PAQ_H.XPT',
    'SLQ_H': f'{_BASE}/SLQ_H.XPT',
    'HIQ_H': f'{_BASE}/HIQ_H.XPT',
    'SMQ_H': f'{_BASE}/SMQ_H.XPT',
    'ALQ_H': f'{_BASE}/ALQ_H.XPT',
    'BMX_H': f'{_BASE}/BMX_H.XPT',
    'BPQ_H': f'{_BASE}/BPQ_H.XPT',
    'DIQ_H': f'{_BASE}/DIQ_H.XPT',
}

# Columns to keep from each table
TABLE_COLUMNS = {
    'DEMO_H': ['SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'INDFMPIR',
               'DMDEDUC2', 'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA'],
    'CFQ_H': ['SEQN', 'CFDDS', 'CFDCST1', 'CFDCSR'],
    'DPQ_H': ['SEQN', 'DPQ020', 'DPQ030', 'DPQ040', 'DPQ050', 'DPQ060',
              'DPQ070', 'DPQ080', 'DPQ090', 'DPQ100'],
    'PAQ_H': ['SEQN', 'PAQ710', 'PAQ715'],
    'SLQ_H': ['SEQN', 'SLD010H'],
    'HIQ_H': ['SEQN', 'HIQ011'],
    'SMQ_H': ['SEQN', 'SMQ020'],
    'ALQ_H': ['SEQN', 'ALQ130'],
    'BMX_H': ['SEQN', 'BMXBMI'],
    'BPQ_H': ['SEQN', 'BPQ020'],
    'DIQ_H': ['SEQN', 'DIQ010'],
}


def fetch_nhanes_table(table_name: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Fetch a single NHANES table from CDC website or local cache.

    Args:
        table_name: NHANES table identifier (e.g., 'DEMO_H')
        cache_dir: Directory to cache downloaded files

    Returns:
        DataFrame with requested columns
    """
    if cache_dir:
        cache_path = cache_dir / f'{table_name}.csv'
        if cache_path.exists():
            logger.info(f"Loading {table_name} from cache")
            return pd.read_csv(cache_path)

    url = NHANES_TABLES[table_name]
    logger.info(f"Downloading {table_name} from {url}")
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix='.xpt', delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name
    df = pd.read_sas(tmp_path, format='xport')
    Path(tmp_path).unlink()

    # Keep only needed columns (some may be missing)
    wanted = TABLE_COLUMNS.get(table_name, [])
    available = [c for c in wanted if c in df.columns]
    df = df[available]

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_dir / f'{table_name}.csv', index=False)

    return df


def filter_age_range(df: pd.DataFrame, min_age: int = 25, max_age: int = 45) -> pd.DataFrame:
    """Filter to target age range."""
    return df[(df['RIDAGEYR'] >= min_age) & (df['RIDAGEYR'] <= max_age)].copy()


def create_cognitive_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create z-scored cognitive composite from CFDDS, CFDCST, CFDCSR.

    Requires at least 2 of 3 tests present for a valid composite.
    """
    df = df.copy()
    tests = ['CFDDS', 'CFDCST1', 'CFDCSR']

    for test in tests:
        if test in df.columns:
            mean = df[test].mean()
            std = df[test].std()
            if std > 0:
                df[f'{test}_z'] = (df[test] - mean) / std
            else:
                df[f'{test}_z'] = 0.0
        else:
            df[f'{test}_z'] = np.nan

    z_cols = [f'{t}_z' for t in tests]
    valid_count = df[z_cols].notna().sum(axis=1)
    df['cognitive_score'] = df[z_cols].mean(axis=1)
    df.loc[valid_count < 2, 'cognitive_score'] = np.nan

    return df


def create_ses_index(df: pd.DataFrame, income_weight: float = 0.6, education_weight: float = 0.4) -> pd.DataFrame:
    """
    Create SES index from poverty income ratio and education level.

    Normalized to 0-1 scale with configurable weights.
    """
    df = df.copy()

    if 'INDFMPIR' in df.columns:
        pir_min, pir_max = df['INDFMPIR'].min(), df['INDFMPIR'].max()
        if pir_max > pir_min:
            df['income_norm'] = (df['INDFMPIR'] - pir_min) / (pir_max - pir_min)
        else:
            df['income_norm'] = 0.5
    else:
        df['income_norm'] = np.nan

    if 'DMDEDUC2' in df.columns:
        # DMDEDUC2: 1=<9th grade, 2=9-11, 3=HS/GED, 4=Some college, 5=College+
        df['education_norm'] = (df['DMDEDUC2'] - 1) / 4
    else:
        df['education_norm'] = np.nan

    df['ses_index'] = income_weight * df['income_norm'] + education_weight * df['education_norm']

    return df


def create_depression_score(df: pd.DataFrame) -> pd.DataFrame:
    """Create PHQ-9 depression score from DPQ items."""
    df = df.copy()
    phq9_cols = [f'DPQ0{i}0' for i in range(2, 10)] + ['DPQ100']
    available = [c for c in phq9_cols if c in df.columns]
    df['depression_score'] = df[available].sum(axis=1)
    # Set to NaN if more than 2 items missing
    missing_count = df[available].isna().sum(axis=1)
    df.loc[missing_count > 2, 'depression_score'] = np.nan
    return df


def create_screen_time(df: pd.DataFrame) -> pd.DataFrame:
    """Create screen time composite from computer + TV hours."""
    df = df.copy()
    if 'PAQ710' in df.columns and 'PAQ715' in df.columns:
        df['screen_time_hours'] = df['PAQ710'] + df['PAQ715']
    elif 'PAQ710' in df.columns:
        df['screen_time_hours'] = df['PAQ710']
    elif 'PAQ715' in df.columns:
        df['screen_time_hours'] = df['PAQ715']
    else:
        df['screen_time_hours'] = np.nan
    return df


def create_healthcare_access(df: pd.DataFrame) -> pd.DataFrame:
    """Create healthcare access indicator from insurance status."""
    df = df.copy()
    if 'HIQ011' in df.columns:
        # HIQ011: 1=Yes, 2=No, 7=Refused, 9=Don't know
        df['has_insurance'] = (df['HIQ011'] == 1).astype(float)
        df.loc[df['HIQ011'].isin([7, 9]), 'has_insurance'] = np.nan
    else:
        df['has_insurance'] = np.nan
    return df


def create_sleep_variable(df: pd.DataFrame) -> pd.DataFrame:
    """Extract sleep hours and create deprivation indicator."""
    df = df.copy()
    if 'SLD010H' in df.columns:
        df['sleep_hours'] = df['SLD010H']
        df['sleep_deprived'] = ((df['SLD010H'] < 6) | (df['SLD010H'] > 9)).astype(float)
    else:
        df['sleep_hours'] = np.nan
        df['sleep_deprived'] = np.nan
    return df


def create_confounders(df: pd.DataFrame) -> pd.DataFrame:
    """Encode confounder variables."""
    df = df.copy()
    if 'RIAGENDR' in df.columns:
        df['female'] = (df['RIAGENDR'] == 2).astype(float)
    if 'SMQ020' in df.columns:
        df['smoker'] = (df['SMQ020'] == 1).astype(float)
        df.loc[df['SMQ020'].isin([7, 9]), 'smoker'] = np.nan
    if 'BPQ020' in df.columns:
        df['hypertension'] = (df['BPQ020'] == 1).astype(float)
        df.loc[df['BPQ020'].isin([7, 9]), 'hypertension'] = np.nan
    if 'DIQ010' in df.columns:
        df['diabetes'] = (df['DIQ010'] == 1).astype(float)
        df.loc[df['DIQ010'].isin([7, 9]), 'diabetes'] = np.nan
    return df


def load_nhanes(cache_dir: Optional[str] = None, output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess complete NHANES 2013-2014 dataset for cognitive analysis.

    Args:
        cache_dir: Directory for caching raw downloads
        output_path: Path to save processed CSV

    Returns:
        Processed DataFrame with all variables, filtered to ages 25-45
    """
    cache = Path(cache_dir) if cache_dir else None

    # Fetch all tables
    tables = {}
    for name in NHANES_TABLES:
        tables[name] = fetch_nhanes_table(name, cache)

    # Merge all on SEQN
    df = tables['DEMO_H']
    for name, table in tables.items():
        if name != 'DEMO_H':
            df = df.merge(table, on='SEQN', how='left')

    logger.info(f"Merged dataset: {len(df)} rows, {len(df.columns)} columns")

    # Filter age range
    df = filter_age_range(df)
    logger.info(f"After age filter (25-45): {len(df)} rows")

    # Create composite variables
    df = create_cognitive_composite(df)
    df = create_ses_index(df)
    df = create_depression_score(df)
    df = create_screen_time(df)
    df = create_healthcare_access(df)
    df = create_sleep_variable(df)
    df = create_confounders(df)

    # Create SES quartiles
    df['ses_quartile'] = pd.qcut(
        df['ses_index'], q=4, labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High']
    )

    # Select final columns
    final_cols = [
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH1', 'female',
        'INDFMPIR', 'DMDEDUC2', 'ses_index', 'ses_quartile',
        'CFDDS', 'CFDCST1', 'CFDCSR', 'cognitive_score',
        'depression_score', 'screen_time_hours', 'sleep_hours', 'sleep_deprived',
        'has_insurance', 'smoker', 'ALQ130', 'BMXBMI', 'hypertension', 'diabetes',
        'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA',
    ]
    available_final = [c for c in final_cols if c in df.columns]
    df = df[available_final]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed NHANES data to {output_path}")

    return df
