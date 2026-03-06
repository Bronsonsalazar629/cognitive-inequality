"""
MIDUS Refresher 2 Data Loader

Loads and preprocesses the Midlife in the United States Refresher 2 survey
(MR2) for cognitive inequality analysis. Uses Project 1 (Survey) and
Project 3 (BTACT + phone MoCA) files.

Survey year: 2022 (MR2 Wave)
Target population: US adults ages 34-55
Cognitive battery: BTACT (Brief Test of Adult Cognition by Telephone) +
                   phone-administered MoCA (max score 22, not 30)

File naming convention:
    Survey:    MR2_P1_SURVEY_N<n>_<date>.sav
    Cognitive: MR2_P3_BTACT+MOCA_N<n>_<date>.sav

Variables use RB1P* prefix (survey) and RB3T* prefix (cognitive).
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# MR2 variable mappings
SURVEY_COLS = [
    'MRID',
    'RB1PRAGE',     # age
    'RB1PRSEX',     # sex (1=male, 2=female)
    'RB1PB1',       # education (1-12 ordinal)
    'RB1PB16',      # pre-tax income last year
    'RB1SRINC',     # respondent total income
    'RB1PA60',      # depression: felt sad/depressed 2+ weeks (1=yes, 2=no)
    'RB1PA108B',    # screen time vs pre-pandemic (1-5 scale)
    'RB1PA108J',    # sleep vs pre-pandemic (1-5 scale)
    'RB1SC1',       # health insurance (1=yes, 2=no)
    'RB1PCV1',      # overall COVID difficulty (1-10)
    # Survey weights
    'RB1PWGHT6',    # full post-stratification weight
    'RB1SWGHT1',    # phone + SAQ weight
]

COG_COLS = [
    'MRID',
    'RB1PRAGE',     # age (also in cog file)
    'RB3TCOMPZ',    # BTACT overall composite (z-score)
    'RB3TEMZ',      # episodic memory factor (z-score)
    'RB3TEFZ',      # executive function factor (z-score)
    'RB3TEMZ_Long', # longitudinal episodic memory (z-score vs MR1)
    'RB3TEFZ_Long', # longitudinal executive function
    'RB3TCOMPZ_Long',# longitudinal composite
    'RB3MOCATOT',   # phone MoCA total (0-22)
    'RB3MOCATOTADJ',# MoCA education-adjusted
    'RB3TBKTOT',    # backward counting total
    'RB3TNSTOT',    # number series total
    'RB3TCTFLU',    # category fluency (unique words)
    'RB3TWLF',      # word list proportion forgotten
]

# Missing/refused codes in MR2 survey responses
SURVEY_MISSING = {7, 8, 97, 98, 99}


def filter_age_range(df: pd.DataFrame,
                     min_age: int = 34,
                     max_age: int = 55) -> pd.DataFrame:
    """Filter to target age range."""
    return df[(df['RB1PRAGE'] >= min_age) & (df['RB1PRAGE'] <= max_age)].copy()


def create_cognitive_composite(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cognitive composite from BTACT components.

    Uses mean of RB3TCOMPZ, RB3TEMZ, RB3TEFZ (all already z-scored
    by MIDUS). Requires at least 2 of 3 components present.
    """
    df = df.copy()
    components = ['RB3TCOMPZ', 'RB3TEMZ', 'RB3TEFZ']

    available = [c for c in components if c in df.columns]
    if not available:
        df['cognitive_score'] = np.nan
        return df

    valid_count = df[available].notna().sum(axis=1)
    df['cognitive_score'] = df[available].mean(axis=1)
    df.loc[valid_count < 2, 'cognitive_score'] = np.nan

    return df


def create_ses_index(df: pd.DataFrame,
                     income_weight: float = 0.6,
                     education_weight: float = 0.4) -> pd.DataFrame:
    """
    Create SES index from income and education (60/40 weighting).

    RB1PB16: Pre-tax income (continuous $)
    RB1PB1:  Highest education (1-12 ordinal scale)

    Both normalized 0-1 before weighting. NaN in either → NaN index.
    """
    df = df.copy()

    # Income normalization
    if 'RB1PB16' in df.columns:
        inc = df['RB1PB16'].copy()
        inc_min = inc.min()   # pandas min() skips NaN
        inc_max = inc.max()
        if pd.notna(inc_min) and pd.notna(inc_max) and inc_max > inc_min:
            df['income_norm'] = (inc - inc_min) / (inc_max - inc_min)
        elif pd.notna(inc_min):
            # Only one unique value — assign 0.5 to valid rows
            df['income_norm'] = np.where(inc.notna(), 0.5, np.nan)
        else:
            df['income_norm'] = np.nan
    else:
        df['income_norm'] = np.nan

    # Education normalization (1-12 scale)
    if 'RB1PB1' in df.columns:
        edu = df['RB1PB1'].copy()
        edu_min = edu.min()
        edu_max = edu.max()
        if pd.notna(edu_min) and pd.notna(edu_max) and edu_max > edu_min:
            df['education_norm'] = (edu - edu_min) / (edu_max - edu_min)
        elif pd.notna(edu_min):
            df['education_norm'] = np.where(edu.notna(), 0.5, np.nan)
        else:
            df['education_norm'] = np.nan
    else:
        df['education_norm'] = np.nan

    # Composite — NaN if either component missing
    df['ses_index'] = np.where(
        df['income_norm'].isna() | df['education_norm'].isna(),
        np.nan,
        income_weight * df['income_norm'] + education_weight * df['education_norm']
    )

    return df


def create_depression_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary depression indicator from RB1PA60.

    RB1PA60: Felt sad/depressed for 2+ weeks in last 12 months
        1 = Yes
        2 = No
        7/8/9 = Refused / DK / missing
    """
    df = df.copy()
    df['depression_score'] = np.nan

    if 'RB1PA60' not in df.columns:
        return df

    df.loc[df['RB1PA60'] == 1.0, 'depression_score'] = 1.0
    df.loc[df['RB1PA60'] == 2.0, 'depression_score'] = 0.0
    # 7, 8, 9 remain NaN

    return df


def create_screen_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create screen time change score from RB1PA108B.

    RB1PA108B: Compared to pre-pandemic, time spent doing screen time activities
        1 = Much less
        2 = Somewhat less
        3 = About the same
        4 = Somewhat more
        5 = Much more
        7/8/9 = Refused / DK / missing

    Centered so that 3 (same as pre-pandemic) = 0.
    """
    df = df.copy()
    df['screen_time_change'] = np.nan

    if 'RB1PA108B' not in df.columns:
        return df

    valid = df['RB1PA108B'].isin([1.0, 2.0, 3.0, 4.0, 5.0])
    df.loc[valid, 'screen_time_change'] = df.loc[valid, 'RB1PA108B'] - 3.0

    return df


def create_sleep_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create sleep change score from RB1PA108J.

    RB1PA108J: Compared to pre-pandemic, time spent sleeping
        1 = Much less
        2 = Somewhat less
        3 = About the same
        4 = Somewhat more
        5 = Much more
        7/8/9 = Refused / DK / missing

    Centered so that 3 = 0.
    """
    df = df.copy()
    df['sleep_change'] = np.nan

    if 'RB1PA108J' not in df.columns:
        return df

    valid = df['RB1PA108J'].isin([1.0, 2.0, 3.0, 4.0, 5.0])
    df.loc[valid, 'sleep_change'] = df.loc[valid, 'RB1PA108J'] - 3.0

    return df


def create_healthcare_access(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary insurance indicator from RB1SC1.

    RB1SC1: Covered by healthcare insurance currently
        1 = Yes
        2 = No
        8/9 = DK / missing
    """
    df = df.copy()
    df['has_insurance'] = np.nan

    if 'RB1SC1' not in df.columns:
        return df

    df.loc[df['RB1SC1'] == 1.0, 'has_insurance'] = 1.0
    df.loc[df['RB1SC1'] == 2.0, 'has_insurance'] = 0.0

    return df


def create_confounders(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode confounder variables.

    RB1PRSEX: 1=male, 2=female → female (0/1)
    RB1PCV1:  Overall COVID difficulty (1-10) → covid_hardship
    """
    df = df.copy()

    if 'RB1PRSEX' in df.columns:
        df['female'] = np.nan
        df.loc[df['RB1PRSEX'] == 1.0, 'female'] = 0.0
        df.loc[df['RB1PRSEX'] == 2.0, 'female'] = 1.0

    if 'RB1PCV1' in df.columns:
        valid = df['RB1PCV1'].between(1, 10)
        df['covid_hardship'] = np.where(valid, df['RB1PCV1'], np.nan)

    return df


def load_midus_mr2(survey_path: Optional[str] = None,
                   cognitive_path: Optional[str] = None,
                   min_age: int = 34,
                   max_age: int = 55,
                   output_path: Optional[str] = None) -> pd.DataFrame:
    """
    Load and preprocess MIDUS Refresher 2 dataset.

    Merges Project 1 (Survey) and Project 3 (BTACT+MoCA) on MRID,
    then applies all variable construction and age filtering.

    Args:
        survey_path:    Path to MR2_P1_SURVEY_*.sav
        cognitive_path: Path to MR2_P3_BTACT+MOCA_*.sav
        min_age:        Lower age bound (default 34)
        max_age:        Upper age bound (default 55)
        output_path:    Path to save processed CSV

    Returns:
        Processed DataFrame ready for analysis
    """
    import pyreadstat

    # Default paths
    if survey_path is None:
        candidates = list(Path('.').glob('MR2_P1_SURVEY_*.sav'))
        if not candidates:
            raise FileNotFoundError("No MR2_P1_SURVEY_*.sav file found in current directory.")
        survey_path = str(candidates[0])

    if cognitive_path is None:
        candidates = list(Path('.').glob('MR2_P3_BTACT*.sav'))
        if not candidates:
            raise FileNotFoundError("No MR2_P3_BTACT*.sav file found in current directory.")
        cognitive_path = str(candidates[0])

    logger.info(f"Loading MR2 survey from {survey_path}")
    survey, _ = pyreadstat.read_sav(survey_path)
    logger.info(f"Survey: {len(survey)} rows, {len(survey.columns)} variables")

    logger.info(f"Loading MR2 cognitive from {cognitive_path}")
    cog, _ = pyreadstat.read_sav(cognitive_path)
    logger.info(f"Cognitive: {len(cog)} rows, {len(cog.columns)} variables")

    # Keep only needed columns (ignore missing ones gracefully)
    survey_keep = [c for c in SURVEY_COLS if c in survey.columns]
    cog_keep = [c for c in COG_COLS if c in cog.columns]

    survey = survey[survey_keep]
    cog = cog[cog_keep]

    # Merge on MRID (inner join — must have both survey and cognitive)
    df = cog.merge(survey, on='MRID', how='inner', suffixes=('', '_surv'))
    logger.info(f"After merge: {len(df)} rows")

    # Age filter
    df = filter_age_range(df, min_age=min_age, max_age=max_age)
    logger.info(f"After age filter ({min_age}-{max_age}): {len(df)} rows")

    # Build composite variables
    df = create_cognitive_composite(df)
    df = create_ses_index(df)
    df = create_depression_score(df)
    df = create_screen_time(df)
    df = create_sleep_variable(df)
    df = create_healthcare_access(df)
    df = create_confounders(df)

    # SES quartiles
    valid_ses = df['ses_index'].dropna()
    if len(valid_ses) >= 4:
        df['ses_quartile'] = pd.qcut(
            df['ses_index'], q=4,
            labels=['Q1_Low', 'Q2', 'Q3', 'Q4_High'],
            duplicates='drop'
        )

    # Final column selection
    final_cols = [
        'MRID', 'RB1PRAGE', 'female', 'covid_hardship',
        'RB1PB1', 'RB1PB16', 'ses_index', 'ses_quartile',
        'RB3TCOMPZ', 'RB3TEMZ', 'RB3TEFZ',
        'RB3TEMZ_Long', 'RB3TEFZ_Long', 'RB3TCOMPZ_Long',
        'RB3MOCATOT', 'RB3MOCATOTADJ',
        'RB3TBKTOT', 'RB3TNSTOT', 'RB3TCTFLU',
        'cognitive_score',
        'depression_score', 'screen_time_change', 'sleep_change',
        'has_insurance',
        'RB1PWGHT6', 'RB1SWGHT1',
    ]
    available_final = [c for c in final_cols if c in df.columns]
    df = df[available_final]

    # Report
    n = len(df)
    cog_n = df['cognitive_score'].notna().sum()
    logger.info(f"Final dataset: N={n}, cognitive_score valid={cog_n} ({cog_n/n*100:.1f}%)")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved to {output_path}")

    return df
