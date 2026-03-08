"""
Full Analysis Pipeline Orchestrator

Runs all stages: PC discovery → mediation → prediction → counterfactuals.
Each stage is wrapped in try/except so failures don't kill the whole pipeline.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def run_full_pipeline(df: pd.DataFrame, config: Dict) -> Dict:
    """
    Run the full cognitive inequality analysis pipeline.

    Args:
        df: Harmonized DataFrame with ses_index, cognitive_score, mediators
        config: Dict with alpha, n_bootstrap, use_llm, mediators

    Returns:
        Dict with results from each stage (None if stage skipped/failed)
    """
    alpha = config.get('alpha', 0.05)
    n_bootstrap = config.get('n_bootstrap', 1000)
    use_llm = config.get('use_llm', False)
    mediators = config.get('mediators', [])

    results = {
        'pc_discovery': None,
        'mediation': None,
        'prediction': None,
        'counterfactuals': None,
    }

    has_cognition = (
        'cognitive_score' in df.columns
        and df['cognitive_score'].notna().sum() > 10
    )

    # Stage 1: PC Discovery
    try:
        from src.analysis.pc_algorithm import discover_ses_cognition_paths
        dataset_name = df['dataset'].iloc[0] if 'dataset' in df.columns else 'unknown'
        pc_result = discover_ses_cognition_paths(
            df, dataset_name=dataset_name,
            mediators=mediators, alpha=alpha, use_llm=use_llm,
        )
        results['pc_discovery'] = pc_result
        logger.info("PC discovery complete")
    except Exception as e:
        logger.error(f"PC discovery failed: {e}")

    if not has_cognition:
        logger.warning("No valid cognitive_score data — skipping prediction and counterfactual stages")
        return results

    # Stage 2: Mediation Analysis
    try:
        from src.analysis.mediation_analysis import analyze_all_mediators
        available_mediators = [m for m in mediators if m in df.columns]
        med_results = analyze_all_mediators(
            df, x='ses_index', y='cognitive_score',
            mediators=available_mediators, n_boot=n_bootstrap,
        )
        results['mediation'] = med_results
        logger.info("Mediation analysis complete")
    except Exception as e:
        logger.error(f"Mediation analysis failed: {e}")

    # Stage 3: Prediction Model
    try:
        from src.analysis.prediction_model import CognitivePredictionModel
        feature_cols = ['ses_index'] + [m for m in mediators if m in df.columns]
        model_df = df[feature_cols + ['cognitive_score']].dropna()
        X = model_df[feature_cols]
        y = model_df['cognitive_score']

        model = CognitivePredictionModel()
        model.train(X, y)
        model.compute_shap_values(X)
        cv = model.cross_validate(X, y, cv=5)

        results['prediction'] = {
            'model': model,
            'cv_results': cv,
            'feature_importance': model.feature_importance(),
        }
        logger.info("Prediction model complete")
    except Exception as e:
        logger.error(f"Prediction model failed: {e}")

    # Stage 4: Counterfactual Simulations
    try:
        if results['mediation'] is not None and results['prediction'] is not None:
            from src.simulation.counterfactual_simulator import generate_interventions
            model = results['prediction']['model']
            interventions = generate_interventions(results['mediation'], model, X)
            results['counterfactuals'] = interventions
            logger.info(f"Counterfactual simulations complete: {len(interventions)} interventions")
        else:
            logger.warning("Skipping counterfactuals: missing mediation or prediction results")
    except Exception as e:
        logger.error(f"Counterfactual simulation failed: {e}")

    return results
