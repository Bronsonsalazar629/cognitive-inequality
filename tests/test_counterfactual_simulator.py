# tests/test_counterfactual_simulator.py
"""Tests for counterfactual intervention simulator."""
import pytest
import pandas as pd
import numpy as np


def _make_trained_model():
    """Create and train a prediction model on synthetic data."""
    from src.analysis.prediction_model import CognitivePredictionModel

    rng = np.random.default_rng(42)
    n = 300
    ses = rng.normal(0, 1, n)
    ins = rng.binomial(1, 0.5, n).astype(float)
    health = rng.normal(3, 1, n)
    cog = 0.5 * ses + 0.4 * ins + 0.3 * health + rng.normal(0, 0.3, n)

    X = pd.DataFrame({'ses_index': ses, 'has_insurance': ins, 'general_health': health})
    y = pd.Series(cog, name='cognitive_score')

    model = CognitivePredictionModel()
    model.train(X, y)
    return model, X


def test_simulate_ses_increase():
    """Shifting ses_index up by 1 SD should change predicted cognitive_score."""
    from src.simulation.counterfactual_simulator import InterventionSimulator

    model, X = _make_trained_model()
    sim = InterventionSimulator(model)

    result = sim.simulate_intervention(X, variable='ses_index', shift=1.0)
    assert result.effect_size != 0
    assert result.counterfactual_mean > result.baseline_mean  # positive relationship


def test_simulate_mediator_intervention():
    """Setting a mediator to a target value should change prediction."""
    from src.simulation.counterfactual_simulator import InterventionSimulator

    model, X = _make_trained_model()
    sim = InterventionSimulator(model)

    result = sim.simulate_intervention(X, variable='has_insurance', target_value=1.0)
    assert result.effect_size != 0
    assert result.affected_n > 0


def test_select_interventions_from_mediation():
    """Should only generate interventions for significant mediators."""
    from src.simulation.counterfactual_simulator import generate_interventions
    from src.analysis.mediation_analysis import BootstrapResult

    model, X = _make_trained_model()

    mediation_results = {
        'has_insurance': BootstrapResult(point_estimate=0.3, ci_lower=0.1, ci_upper=0.5, n_boot=500),
        'general_health': BootstrapResult(point_estimate=0.01, ci_lower=-0.1, ci_upper=0.12, n_boot=500),
    }

    results = generate_interventions(mediation_results, model, X)
    # Only has_insurance is significant (CI excludes zero)
    variables_simulated = [r.variable for r in results]
    assert 'has_insurance' in variables_simulated
    assert 'general_health' not in variables_simulated


def test_no_interventions_when_none_significant():
    """Should return empty list when no mediators are significant."""
    from src.simulation.counterfactual_simulator import generate_interventions
    from src.analysis.mediation_analysis import BootstrapResult

    model, X = _make_trained_model()

    mediation_results = {
        'has_insurance': BootstrapResult(point_estimate=0.01, ci_lower=-0.1, ci_upper=0.12, n_boot=500),
    }

    results = generate_interventions(mediation_results, model, X)
    assert len(results) == 0


def test_population_impact():
    """Should scale effect to population level using survey weights."""
    from src.simulation.counterfactual_simulator import InterventionSimulator

    model, X = _make_trained_model()
    sim = InterventionSimulator(model)

    result = sim.simulate_intervention(X, variable='ses_index', shift=1.0)
    weights = np.ones(len(X)) * 1000.0  # each person represents 1000
    impact = sim.estimate_population_impact(result, weights)

    assert impact.weighted_effect != 0
    assert impact.weighted_n > len(X)  # scaled up by weights
