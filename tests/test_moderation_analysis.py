import numpy as np
import pandas as pd
import pytest
from src.analysis.moderation_analysis import (
    run_interaction_model,
    run_subgroup_mediation,
)


def _make_df(n=400):
    rng = np.random.default_rng(0)
    ses = rng.uniform(0, 1, n)
    age = rng.uniform(34, 55, n)
    female = rng.integers(0, 2, n).astype(float)
    mediator = 0.3 * ses + rng.normal(0, 0.5, n)
    cog = 0.5 * ses + 0.2 * mediator - 0.1 * female + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        'ses_index': ses, 'RB1PRAGE': age, 'female': female,
        'mediator': mediator, 'cognitive_score': cog,
    })


def test_interaction_model_sex():
    df = _make_df()
    result = run_interaction_model(df, moderator='female')
    assert 'interaction_coef' in result
    assert 'interaction_pvalue' in result
    assert 'n' in result


def test_interaction_model_age_cohort():
    df = _make_df()
    result = run_interaction_model(df, moderator='age_cohort')
    assert 'interaction_coef' in result


def test_subgroup_mediation_sex():
    df = _make_df()
    result = run_subgroup_mediation(df, moderator='female',
                                    mediators=['mediator'])
    assert 0 in result and 1 in result
    assert 'mediator' in result[0]


def test_subgroup_mediation_age_cohort():
    df = _make_df()
    result = run_subgroup_mediation(df, moderator='age_cohort',
                                    mediators=['mediator'])
    assert 'younger' in result and 'older' in result
