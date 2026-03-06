"""
PC Algorithm for Causal Discovery (SES → Cognition)

Wraps causal-learn's PC algorithm for discovering causal structure
between socioeconomic status, mediators, and cognitive outcomes.
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

logger = logging.getLogger(__name__)


def refine_graph(adj_matrix: pd.DataFrame, variable_names: list) -> pd.DataFrame:
    """
    LLM-based edge validation via causal_graph_refiner.

    Placeholder that calls the existing CausalGraphRefiner.
    Can be monkeypatched in tests.
    """
    from src.llm.causal_graph_refiner import CausalGraphRefiner
    refiner = CausalGraphRefiner()
    # For now, return the adjacency matrix unchanged
    # Full integration would pass edges through refiner.refine_causal_graph()
    logger.info("LLM graph refinement called (pass-through)")
    return adj_matrix


def run_pc_discovery(df: pd.DataFrame,
                     variables: list,
                     alpha: float = 0.05) -> pd.DataFrame:
    """
    Run PC algorithm on selected variables.

    Args:
        df: DataFrame with data
        variables: Column names to include in discovery
        alpha: Significance level for conditional independence tests

    Returns:
        Labeled adjacency matrix as pd.DataFrame
    """
    from causallearn.search.ConstraintBased.PC import pc

    # Select columns and drop NaN rows
    data = df[variables].dropna()
    logger.info(f"PC discovery: {len(data)} complete rows, {len(variables)} variables")

    if len(data) < 10:
        logger.warning("Too few complete rows for PC algorithm")
        return pd.DataFrame(0, index=variables, columns=variables)

    # Run PC algorithm
    data_array = data.values.astype(float)
    cg = pc(data_array, alpha=alpha, indep_test='fisherz')

    # Extract adjacency matrix
    adj = cg.G.graph
    adj_df = pd.DataFrame(adj, index=variables, columns=variables)

    return adj_df


def discover_ses_cognition_paths(df: pd.DataFrame,
                                  dataset_name: str,
                                  mediators: Optional[list] = None,
                                  alpha: float = 0.05,
                                  use_llm: bool = False) -> dict:
    """
    Discover causal paths from SES to cognition.

    Args:
        df: DataFrame with ses_index, cognitive_score, and optional mediators
        dataset_name: Name of the dataset (for logging)
        mediators: List of mediator column names to include
        alpha: Significance level for PC algorithm
        use_llm: If True, validate edges with LLM (causal_graph_refiner)

    Returns:
        Dict with 'adjacency' (pd.DataFrame), 'dataset' (str),
        'variables' (list), 'llm_refined' (bool)
    """
    variables = ['ses_index', 'cognitive_score']
    if mediators:
        # Only include mediators that exist in the data
        variables.extend([m for m in mediators if m in df.columns])

    logger.info(f"Discovering SES→cognition paths for {dataset_name}: {variables}")

    adj = run_pc_discovery(df, variables=variables, alpha=alpha)

    result = {
        'adjacency': adj,
        'dataset': dataset_name,
        'variables': variables,
        'llm_refined': False,
    }

    if use_llm:
        adj = refine_graph(adj, variables)
        result['adjacency'] = adj 
        result['llm_refined'] = True

    return result
