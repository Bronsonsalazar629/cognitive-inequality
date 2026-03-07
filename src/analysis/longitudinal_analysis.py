"""Longitudinal analysis: MR2 baseline → MIDUS 3 follow-up cognitive change."""

import logging
from typing import Dict
import numpy as np
import pandas as pd
import statsmodels.api as sm

logger = logging.getLogger(__name__)


def merge_baseline_followup(mr2_df: pd.DataFrame, m3_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge MR2 baseline with M3 follow-up on M2ID.

    Computes cognitive_change = cognitive_score_m3 - cognitive_score (both z-scored).
    Returns merged panel with baseline SES, covariates, and change score.
    """
    # MR2 uses 'MRID' for the cross-wave participant ID; M3 uses 'M2ID'.
    # They are the same identifier — align before merging.
    mr2_key = 'M2ID' if 'M2ID' in mr2_df.columns else 'MRID'
    mr2_merge = mr2_df.rename(columns={mr2_key: 'M2ID'}) if mr2_key == 'MRID' else mr2_df
    panel = mr2_merge.merge(m3_df[['M2ID', 'cognitive_score_m3']], on='M2ID', how='inner')
    panel['cognitive_change'] = panel['cognitive_score_m3'] - panel['cognitive_score']
    logger.info(f"Panel N={len(panel)}, cognitive_change mean={panel['cognitive_change'].mean():.3f}")
    return panel


def run_longitudinal_regression(panel: pd.DataFrame,
                                 x: str = 'ses_index',
                                 y: str = 'cognitive_change',
                                 covariates: list = None) -> Dict:
    """
    OLS regression: cognitive_change ~ ses_index + covariates.

    Returns dict with ses_coef, ses_pvalue, ses_ci_lower, ses_ci_upper, r2, n.
    """
    if covariates is None:
        covariates = ['RB1PRAGE', 'female']

    cols = [x, y] + [c for c in covariates if c in panel.columns]
    data = panel[cols].dropna()

    X = sm.add_constant(data[[x] + [c for c in covariates if c in data.columns]])
    model = sm.OLS(data[y], X).fit()

    return {
        'ses_coef':     model.params[x],
        'ses_pvalue':   model.pvalues[x],
        'ses_ci_lower': model.conf_int().loc[x, 0],
        'ses_ci_upper': model.conf_int().loc[x, 1],
        'r2':           model.rsquared,
        'n':            int(len(data)),
        'model':        model,
    }
