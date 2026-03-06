"""
Tests for E-value sensitivity analysis.

E-values quantify the minimum strength of unmeasured confounding
required to explain away an observed effect (VanderWeele & Ding 2017).
"""

import pytest
import numpy as np
import pandas as pd

from src.utils.evalue import evalue_ols, evalue_rr
from src.analysis.sensitivity_analysis import (
    compute_evalue,
    run_ses_cognition_sensitivity,
)


# ---------------------------------------------------------------------------
# evalue_rr — risk ratio E-values
# ---------------------------------------------------------------------------

class TestEvalueRR:
    def test_null_effect_returns_one(self):
        result = evalue_rr(rr=1.0, ci_lower=0.8, ci_upper=1.2)
        assert result['evalue_point'] == pytest.approx(1.0)

    def test_rr_two_formula(self):
        # E = RR + sqrt(RR*(RR-1)) = 2 + sqrt(2) ≈ 3.414
        result = evalue_rr(rr=2.0, ci_lower=1.5, ci_upper=2.5)
        assert result['evalue_point'] == pytest.approx(3.414, abs=0.01)

    def test_larger_rr_larger_evalue(self):
        e2 = evalue_rr(rr=2.0, ci_lower=1.0, ci_upper=3.0)['evalue_point']
        e3 = evalue_rr(rr=3.0, ci_lower=1.0, ci_upper=4.0)['evalue_point']
        assert e3 > e2

    def test_ci_lower_exceeds_one_evalue_gt_one(self):
        # CI lower = 1.5 → E-value for CI > 1
        result = evalue_rr(rr=2.0, ci_lower=1.5, ci_upper=2.5)
        assert result['evalue_ci'] > 1.0

    def test_ci_crosses_null_evalue_ci_is_one(self):
        # CI includes 1.0 → already contains null → E-value for CI = 1
        result = evalue_rr(rr=2.0, ci_lower=0.9, ci_upper=3.0)
        assert result['evalue_ci'] == pytest.approx(1.0)

    def test_result_contains_required_keys(self):
        result = evalue_rr(rr=2.0, ci_lower=1.2, ci_upper=3.0)
        assert 'evalue_point' in result
        assert 'evalue_ci' in result
        assert 'rr' in result


# ---------------------------------------------------------------------------
# evalue_ols — OLS regression E-values (continuous outcome)
# ---------------------------------------------------------------------------

class TestEvalueOLS:
    def test_null_effect_returns_one(self):
        result = evalue_ols(estimate=0.0, se=0.1, sd_outcome=1.0)
        assert result['evalue_point'] == pytest.approx(1.0)

    def test_large_ses_effect(self):
        # c=1.42, SD=0.821 → d=1.730, RR≈4.83, E≈9.1
        result = evalue_ols(estimate=1.42, se=0.15, sd_outcome=0.821)
        assert result['evalue_point'] > 8.0
        assert result['rr_approx'] == pytest.approx(4.83, abs=0.1)

    def test_larger_effect_larger_evalue(self):
        e_small = evalue_ols(estimate=0.5, se=0.1, sd_outcome=1.0)['evalue_point']
        e_large = evalue_ols(estimate=1.5, se=0.1, sd_outcome=1.0)['evalue_point']
        assert e_large > e_small

    def test_ci_crosses_zero_evalue_ci_is_one(self):
        # estimate=0.05, se=0.1 → CI lower = -0.146 → crosses zero
        result = evalue_ols(estimate=0.05, se=0.1, sd_outcome=1.0)
        assert result['evalue_ci'] == pytest.approx(1.0)

    def test_ci_excludes_zero_evalue_ci_gt_one(self):
        # estimate=0.5, se=0.1 → CI lower=0.304 → excludes zero
        result = evalue_ols(estimate=0.5, se=0.1, sd_outcome=1.0)
        assert result['evalue_ci'] > 1.0
        assert result['evalue_ci'] < result['evalue_point']

    def test_negative_estimate_treated_symmetrically(self):
        pos = evalue_ols(estimate=1.0, se=0.1, sd_outcome=1.0)['evalue_point']
        neg = evalue_ols(estimate=-1.0, se=0.1, sd_outcome=1.0)['evalue_point']
        assert pos == pytest.approx(neg, rel=1e-6)

    def test_result_contains_required_keys(self):
        result = evalue_ols(estimate=1.0, se=0.1, sd_outcome=1.0)
        for key in ('evalue_point', 'evalue_ci', 'rr_approx'):
            assert key in result


# ---------------------------------------------------------------------------
# compute_evalue — convenience wrapper
# ---------------------------------------------------------------------------

class TestComputeEvalue:
    def test_returns_dict_with_interpretation(self):
        result = compute_evalue(estimate=1.42, se=0.15, sd_outcome=0.821)
        assert 'evalue_point' in result
        assert 'interpretation' in result

    def test_interpretation_is_string(self):
        result = compute_evalue(estimate=1.0, se=0.2, sd_outcome=1.0)
        assert isinstance(result['interpretation'], str)
        assert len(result['interpretation']) > 10


# ---------------------------------------------------------------------------
# run_ses_cognition_sensitivity — full sensitivity report
# ---------------------------------------------------------------------------

class TestRunSESCognitionSensitivity:

    @pytest.fixture
    def bk_results(self):
        from src.analysis.mediation_analysis import MediationResult
        return {
            'depression_score': MediationResult(
                a=0.12, b=-0.20, c=1.42, c_prime=1.39,
                indirect=-0.024, proportion_mediated=0.017,
            ),
            'sleep_change': MediationResult(
                a=0.08, b=0.71, c=1.42, c_prime=1.36,
                indirect=0.057, proportion_mediated=0.040,
            ),
        }

    def test_returns_total_effect_evalue(self, bk_results):
        result = run_ses_cognition_sensitivity(bk_results, sd_outcome=0.821)
        assert 'total_effect' in result
        assert result['total_effect']['evalue_point'] > 1.0

    def test_returns_evalue_per_mediator(self, bk_results):
        result = run_ses_cognition_sensitivity(bk_results, sd_outcome=0.821)
        assert 'direct_effects' in result
        for mediator in bk_results:
            assert mediator in result['direct_effects']

    def test_total_effect_evalue_high_for_strong_ses_effect(self, bk_results):
        # c=1.42 is large → E-value should be > 8
        result = run_ses_cognition_sensitivity(bk_results, sd_outcome=0.821)
        assert result['total_effect']['evalue_point'] > 8.0

    def test_result_includes_se_estimates(self, bk_results):
        result = run_ses_cognition_sensitivity(bk_results, sd_outcome=0.821)
        assert 'se_assumption' in result
