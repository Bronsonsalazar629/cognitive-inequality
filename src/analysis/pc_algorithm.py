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


def refine_graph(adj_matrix: pd.DataFrame, variable_names: list,
                 data: pd.DataFrame = None, llm_client=None) -> dict:
    """
    LLM-based edge validation via CausalGraphRefiner + Ollama.

    Extracts edges from adjacency matrix, validates them with the local
    LLM using domain knowledge about SES→cognition pathways, and returns
    a validation report. Falls back silently if Ollama is unreachable.
    """
    from src.llm.causal_graph_refiner import CausalGraphRefiner, CausalEdge

    discovered_edges = []
    for i, src in enumerate(variable_names):
        for j, tgt in enumerate(variable_names):
            if i != j and adj_matrix.iloc[i, j] != 0:
                discovered_edges.append((src, tgt))

    if not discovered_edges:
        logger.info("No edges to refine.")
        return {'adjacency': adj_matrix, 'llm_report': None}

    if llm_client is None:
        from src.llm.ollama_client import OllamaClient
        llm_client = OllamaClient()

    if not llm_client.is_available():
        logger.warning("Ollama not available — skipping LLM graph refinement.")
        return {'adjacency': adj_matrix, 'llm_report': None}

    expert_edges = [
        CausalEdge('ses_index', 'cognitive_score', 0.95, 'expert',
                   'SES consistently predicts cognitive function (Farah 2018)',
                   'Farah 2018; Marmot 2005'),
        CausalEdge('ses_index', 'purpose_in_life', 0.85, 'expert',
                   'Higher SES enables access to meaningful work and civic engagement',
                   'Ryff 1989; Kim et al. 2014'),
        CausalEdge('ses_index', 'sense_of_control', 0.88, 'expert',
                   'Economic security supports perceived mastery over life outcomes',
                   'Pearlin & Schooler 1978'),
    ]

    refiner = CausalGraphRefiner(llm_client, domain='cognitive_inequality')
    dummy_df = data if data is not None else pd.DataFrame()

    report = refiner.refine_causal_graph(
        expert_edges=expert_edges,
        discovered_edges=discovered_edges,
        data=dummy_df,
        protected_attr='ses_index',
        outcome='cognitive_score',
    )
    logger.info(f"LLM graph refinement: {report['summary']}")
    return {'adjacency': adj_matrix, 'llm_report': report}


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
        refined = refine_graph(adj, variables, data=df)
        result['adjacency'] = refined['adjacency']
        result['llm_report'] = refined.get('llm_report')
        result['llm_refined'] = refined.get('llm_report') is not None

    return result
