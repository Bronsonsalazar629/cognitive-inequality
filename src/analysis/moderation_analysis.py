"""Moderation analysis: does the SES→cognition gap vary by sex or age cohort?"""

import logging
from typing import Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.analysis.mediation_analysis import bootstrap_mediation

logger = logging.getLogger(__name__)


def run_interaction_model(df: pd.DataFrame,
                           moderator: str,
                           x: str = 'ses_index',
                           y: str = 'cognitive_score') -> Dict:
    """
    Fit OLS with interaction term: y ~ x * moderator + covariates.

    moderator='female'      → uses female column directly
    moderator='age_cohort'  → creates binary 0/1 split at median age
    """
    data = df.copy()

    if moderator == 'age_cohort':
        median_age = data['RB1PRAGE'].median()
        data['age_cohort'] = (data['RB1PRAGE'] > median_age).astype(float)
        mod_col = 'age_cohort'
        covariates = ['female']
    else:
        mod_col = moderator
        covariates = ['RB1PRAGE']

    cols = [x, y, mod_col] + covariates
    data = data[cols].dropna()

    data = data.copy()
    data['interaction'] = data[x] * data[mod_col]
    predictors = [x, mod_col, 'interaction'] + covariates
    X = sm.add_constant(data[predictors])
    model = sm.OLS(data[y], X).fit()

    return {
        'interaction_coef':   model.params['interaction'],
        'interaction_pvalue': model.pvalues['interaction'],
        'interaction_ci':     tuple(model.conf_int().loc['interaction']),
        'n':                  int(len(data)),
        'model':              model,
    }


def run_subgroup_mediation(df: pd.DataFrame,
                            moderator: str,
                            mediators: List[str],
                            x: str = 'ses_index',
                            y: str = 'cognitive_score',
                            n_boot: int = 500) -> Dict:
    """
    Run bootstrap mediation separately per subgroup.

    moderator='female'     → groups: {0: male, 1: female}
    moderator='age_cohort' → groups: {'younger': age<=median, 'older': age>median}

    Returns dict of {group_label: {mediator: BootstrapResult}}
    """
    data = df.copy()

    if moderator == 'age_cohort':
        median_age = data['RB1PRAGE'].median()
        data['_group'] = np.where(data['RB1PRAGE'] <= median_age, 'younger', 'older')
        group_labels = ['younger', 'older']
    else:
        data['_group'] = data[moderator]
        group_labels = sorted(data['_group'].dropna().unique())

    results = {}
    for label in group_labels:
        subset = data[data['_group'] == label].copy()
        logger.info(f"  Subgroup {moderator}={label}, N={len(subset)}")
        group_results = {}
        for m in mediators:
            if m not in subset.columns:
                continue
            try:
                boot = bootstrap_mediation(subset, x=x, m=m, y=y, n_boot=n_boot)
                group_results[m] = boot
            except Exception as e:
                logger.warning(f"  bootstrap failed for {m} in group {label}: {e}")
        results[label] = group_results

    return results
