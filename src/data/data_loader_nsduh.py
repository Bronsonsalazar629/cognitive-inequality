"""
NSDUH 2024 Data Loader

Loads and preprocesses the National Survey on Drug Use and Health
public-use data for cognitive inequality analysis.

Data source: SAMHSA NSDUH 2024 public-use file (.RData format).
Target population: Adolescents and young adults ages 12-25 (CATAGE 1-2).
Output: Processed DataFrame with K6 distress score, SES index,
        functional impairment, and demographics.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# NSDUH missing/skip codes
NSDUH_MISSING = {85, 94, 97, 98, 99}


def filter_adolescents_young_adults(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to CATAGE 1 (12-17) and 2 (18-25)."""
    return df[df['CATAGE'] <= 2.0].copy()


def create_k6_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sum six K6 Kessler distress items into k6_score (range 6-30).

    Items: DSTNRV30, DSTHOP30, DSTRST30, DSTCHR30, DSTEFF30, DSTNGD30
    Each scored 1-5. Missing codes (85/94/97/98/99) → NaN for that row.
    """
    df = df.copy()
    k6_items = ['DSTNRV30', 'DSTHOP30', 'DSTRST30', 'DSTCHR30', 'DSTEFF30', 'DSTNGD30']

    for col in k6_items:
        df[col] = df[col].where(~df[col].isin(NSDUH_MISSING), np.nan)

    df['k6_score'] = df[k6_items].sum(axis=1, min_count=6)
    return df


def create_ses_index(df: pd.DataFrame,
                     income_weight: float = 0.6,
                     education_weight: float = 0.4) -> pd.DataFrame:
    """
    Create SES index from INCOME (1-4) and ANYEDUC3 (1-2), normalized 0-1.
    """
    df = df.copy()

    i_min, i_max = 1.0, 4.0
    e_min, e_max = 1.0, 2.0

    df['income_norm'] = (df['INCOME'] - i_min) / (i_max - i_min)
    df['education_norm'] = (df['ANYEDUC3'] - e_min) / (e_max - e_min)

    df['ses_index'] = income_weight * df['income_norm'] + education_weight * df['education_norm']
    return df


def create_functional_impairment(df: pd.DataFrame) -> pd.DataFrame:
    """Map WHODASTOTSC to functional_impairment (0-24, higher=worse)."""
    df = df.copy()
    df['functional_impairment'] = df['WHODASTOTSC']
    return df


def create_confounders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create demographic confounders:
    - female: IRSEX 2=female→1, 1=male→0
    - race_white, race_black, race_hispanic, race_asian, race_other: from NEWRACE2
    - general_health: HEALTH reversed (1=excellent→5, 5=poor→1), missing codes→NaN
    """
    df = df.copy()

    df['female'] = (df['IRSEX'] == 2.0).astype(float)

    df['race_white'] = (df['NEWRACE2'] == 1.0).astype(float)
    df['race_black'] = (df['NEWRACE2'] == 2.0).astype(float)
    df['race_hispanic'] = (df['NEWRACE2'] == 7.0).astype(float)
    df['race_asian'] = (df['NEWRACE2'] == 5.0).astype(float)
    df['race_other'] = (~df['NEWRACE2'].isin([1.0, 2.0, 5.0, 7.0])).astype(float)

    health = df['HEALTH'].copy()
    health = health.where(~health.isin(NSDUH_MISSING), np.nan)
    df['general_health'] = 6.0 - health

    return df


def load_nsduh(data_path: str = None,
               output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess NSDUH 2024 dataset from RData file.

    Args:
        data_path: Path to NSDUH_2024.RData
        output_path: Optional path to save processed CSV

    Returns:
        Processed DataFrame filtered to ages 12-25
    """
    if data_path is None:
        data_path = 'docs/plans/NSDUH_2024.RData'

    logger.info(f"Loading NSDUH from {data_path}")
    import pyreadr
    rdata = pyreadr.read_r(data_path)
    df = rdata['NSDUH_2024']
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    # Filter to adolescents + young adults
    df = filter_adolescents_young_adults(df)
    logger.info(f"After age filter: {len(df)} rows")

    # Create scores
    df = create_k6_score(df)
    df = create_ses_index(df)
    df = create_functional_impairment(df)
    df = create_confounders(df)

    # Binary indicators
    df['has_insurance'] = (df['IRINSUR4'] == 1.0).astype(float)
    df['spd'] = df['SPDPSTYR'].astype(float)
    df['smi'] = df['SMIPY'].astype(float)
    df['mde'] = (df['AMDELT'] == 1.0).astype(float)
    df['age_category'] = df['CATAGE']

    # Select final columns
    final_cols = [
        'age_category', 'female', 'k6_score', 'ses_index',
        'functional_impairment', 'has_insurance', 'general_health',
        'spd', 'smi', 'mde',
        'race_white', 'race_black', 'race_hispanic', 'race_asian', 'race_other',
        'ANALWT2_C',
    ]
    available = [c for c in final_cols if c in df.columns]
    df = df[available]

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved processed NSDUH to {output_path}")

    return df
