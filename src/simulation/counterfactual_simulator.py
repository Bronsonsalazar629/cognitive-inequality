"""
Counterfactual Intervention Simulator

Simulates policy interventions by modifying mediator values and predicting
cognitive outcomes using the trained XGBoost model. Only simulates
interventions for mediators with statistically significant mediation effects.
"""

import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class InterventionResult:
    """Result of a single counterfactual intervention."""
    variable: str
    baseline_mean: float
    counterfactual_mean: float
    effect_size: float
    affected_n: int


@dataclass
class PopulationImpact:
    """Population-level impact estimate."""
    weighted_effect: float
    weighted_n: float


class InterventionSimulator:
    """Simulate counterfactual outcomes under interventions."""

    def __init__(self, model):
        self.model = model

    def simulate_intervention(self, X: pd.DataFrame,
                              variable: str,
                              shift: Optional[float] = None,
                              target_value: Optional[float] = None) -> InterventionResult:
        """
        Simulate an intervention on a single variable.

        Args:
            X: Feature DataFrame
            variable: Column to modify
            shift: Add this amount to the variable (e.g., +1 SD)
            target_value: Set variable to this value for all rows
        """
        baseline_preds = self.model.predict(X)
        baseline_mean = baseline_preds.mean()

        X_cf = X.copy()
        if target_value is not None:
            affected_n = int((X_cf[variable] != target_value).sum())
            X_cf[variable] = target_value
        elif shift is not None:
            affected_n = len(X_cf)
            X_cf[variable] = X_cf[variable] + shift
        else:
            raise ValueError("Must specify either shift or target_value")

        cf_preds = self.model.predict(X_cf)
        cf_mean = cf_preds.mean()

        return InterventionResult(
            variable=variable,
            baseline_mean=float(baseline_mean),
            counterfactual_mean=float(cf_mean),
            effect_size=float(cf_mean - baseline_mean),
            affected_n=affected_n,
        )

    def estimate_population_impact(self, result: InterventionResult,
                                   weights: np.ndarray) -> PopulationImpact:
        """Scale intervention effect to population level using survey weights."""
        weighted_n = float(weights.sum())
        weighted_effect = result.effect_size  # effect per person, scaled by weight count

        return PopulationImpact(
            weighted_effect=weighted_effect,
            weighted_n=weighted_n,
        )


def generate_interventions(mediation_results: Dict,
                           model,
                           X: pd.DataFrame) -> List[InterventionResult]:
    """
    Generate interventions for significant mediators only.

    A mediator is significant if its bootstrap CI excludes zero.
    For each significant mediator, simulates shifting it by 1 SD.
    """
    from src.analysis.mediation_analysis import get_significant_mediators

    significant = get_significant_mediators(mediation_results)
    if not significant:
        return []

    sim = InterventionSimulator(model)
    results = []

    for mediator in significant:
        if mediator not in X.columns:
            continue
        sd = X[mediator].std()
        if sd <= 0:
            continue
        # Shift by 1 SD in the direction of the mediation effect
        direction = 1.0 if mediation_results[mediator].point_estimate > 0 else -1.0
        result = sim.simulate_intervention(X, variable=mediator, shift=direction * sd)
        results.append(result)

    # Sort by effect size descending
    results.sort(key=lambda r: abs(r.effect_size), reverse=True)
    return results
