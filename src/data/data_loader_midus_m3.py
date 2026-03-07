"""MIDUS 3 data loader — survey + BTACT cognitive battery."""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
import pyreadstat

logger = logging.getLogger(__name__)

SURVEY_SAV = 'M3_P1_SURVEY_N3294_20251029.sav'
BTACT_SAV  = 'M3_P3_BTACT_N3291_20210922.sav'

# NOTE: sex variable is C1PRSEX in the actual file (not C1SEX as documented)
SURVEY_COLS = ['M2ID', 'C1PRAGE', 'C1SRINC', 'C1STINC', 'C1PRSEX']
BTACT_COLS  = ['M2ID', 'C3TCOMP']


def _find_sav(filename: str) -> Path:
    """Search project root and data/raw for SAV file."""
    for base in [Path('.'), Path('data/raw')]:
        p = base / filename
        if p.exists():
            return p
    raise FileNotFoundError(f"{filename} not found in . or data/raw/")


def load_midus_m3(survey_path: str = None, btact_path: str = None) -> pd.DataFrame:
    """
    Load MIDUS 3 survey + BTACT, return merged DataFrame.

    Columns returned:
        M2ID, C1PRAGE, female, ses_index_m3, cognitive_score_m3
    """
    survey_path = Path(survey_path) if survey_path else _find_sav(SURVEY_SAV)
    btact_path  = Path(btact_path)  if btact_path  else _find_sav(BTACT_SAV)

    logger.info(f"Loading M3 survey from {survey_path}")
    survey, _ = pyreadstat.read_sav(str(survey_path), usecols=SURVEY_COLS)
    survey = pd.DataFrame(survey)

    logger.info(f"Loading M3 BTACT from {btact_path}")
    btact, _ = pyreadstat.read_sav(str(btact_path), usecols=BTACT_COLS)
    btact = pd.DataFrame(btact)

    df = survey.merge(btact, on='M2ID', how='inner')
    logger.info(f"Merged M3 N={len(df)}")

    # Sex: 1=male 2=female in MIDUS coding (variable is C1PRSEX in actual file)
    df['female'] = (df['C1PRSEX'] == 2).astype(float)

    # SES index: log income (use total household income C1STINC, fallback to C1SRINC)
    income = df['C1STINC'].where(df['C1STINC'] > 0, df['C1SRINC'])
    income = income.where(income > 0)
    log_income = np.log1p(income)
    min_i, max_i = log_income.min(), log_income.max()
    df['ses_index_m3'] = (log_income - min_i) / (max_i - min_i) if max_i > min_i else 0.5

    # Cognitive score: z-score C3TCOMP
    cog = df['C3TCOMP'].where(df['C3TCOMP'] > 0)
    df['cognitive_score_m3'] = (cog - cog.mean()) / cog.std()

    final_cols = ['M2ID', 'C1PRAGE', 'female', 'ses_index_m3', 'cognitive_score_m3']
    return df[final_cols].copy()
