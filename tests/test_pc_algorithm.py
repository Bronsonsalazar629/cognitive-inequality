# tests/test_pc_algorithm.py
"""Tests for PC algorithm causal discovery."""

import pytest
import pandas as pd
import numpy as np


def _make_synthetic_dag(n=500, seed=42):
    """Create synthetic data with known DAG: X→M→Y, X→Y."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, n)
    M = 0.7 * X + rng.normal(0, 0.5, n)
    Y = 0.5 * X + 0.6 * M + rng.normal(0, 0.5, n)
    return pd.DataFrame({"X": X, "M": M, "Y": Y})


def test_pc_discovers_known_structure():
    """PC should discover edges in synthetic DAG with known structure."""
    from src.analysis.pc_algorithm import run_pc_discovery

    df = _make_synthetic_dag(n=1000)
    adj = run_pc_discovery(df, variables=["X", "M", "Y"], alpha=0.05)

    assert isinstance(adj, pd.DataFrame)
    # X and M should be connected
    assert adj.loc["X", "M"] != 0 or adj.loc["M", "X"] != 0
    # M and Y should be connected
    assert adj.loc["M", "Y"] != 0 or adj.loc["Y", "M"] != 0


def test_pc_returns_adjacency_matrix():
    """Output should be a labeled adjacency matrix (pd.DataFrame)."""
    from src.analysis.pc_algorithm import run_pc_discovery

    df = _make_synthetic_d
    adj = run_pc_discovery(df, variables=["X", "M", "Y"])

    assert isinstance(adj, pd.DataFrame)
    assert list(adj.index) == ["X", "M", "Y"]
    assert list(adj.columns) == ["X", "M", "Y"]


def test_pc_handles_missing_data():
    """PC should drop NaN rows and still run."""
    from src.analysis.pc_algorithm import run_pc_discovery

    df = _make_synthetic_dag(n=500)
    # Introduce NaN in 10% of rows
    df.loc[df.index[:50], "X"] = np.nan
    adj = run_pc_discovery(df, variables=["X", "M", "Y"])

    assert isinstance(adj, pd.DataFrame)
    assert adj.shape == (3, 3)


def test_run_ses_cognition_discovery():
    """Should return graph with ses_index and cognitive_score nodes."""
    from src.analysis.pc_algorithm import discover_ses_cognition_paths

    rng = np.random.default_rng(42)
    n = 500
    ses = rng.normal(0, 1, n)
    mediator = 0.6 * ses + rng.normal(0, 0.5, n)
    cog = 0.4 * ses + 0.5 * mediator + rng.normal(0, 0.5, n)
    df = pd.DataFrame(
        {
            "ses_index": ses,
            "has_insurance": mediator,
            "cognitive_score": cog,
        }
    )

    result = discover_ses_cognition_paths(
        df, dataset_name="test", mediators=["has_insurance"]
    )
    assert "adjacency" in result
    assert "ses_index" in result["adjacency"].index
    assert "cognitive_score" in result["adjacency"].columns


def test_pc_with_mediators():
    """Including mediators should allow discovery of mediation paths."""
    from src.analysis.pc_algorithm import discover_ses_cognition_paths

    rng = np.random.default_rng(42)
    n = 1000
    ses = rng.normal(0, 1, n)
    ins = 0.7 * ses + rng.normal(0, 0.3, n)
    cog = 0.3 * ses + 0.6 * ins + rng.normal(0, 0.3, n)
    df = pd.DataFrame(
        {
            "ses_index": ses,
            "has_insurance": ins,
            "cognitive_score": cog,
        }
    )

    result = discover_ses_cognition_paths(
        df, dataset_name="test", mediators=["has_insurance"]
    )
    adj = result["adjacency"]
    # SES and insurance should be connected
    assert (
        adj.loc["ses_index", "has_insurance"] != 0
        or adj.loc["has_insurance", "ses_index"] != 0
    )


def test_pc_without_llm():
    """Without LLM flag, should return raw PC result."""
    from src.analysis.pc_algorithm import discover_ses_cognition_paths

    df = _make_synthetic_dag(n=200)
    df = df.rename(columns={"X": "ses_index", "Y": "cognitive_score", "M": "mediator"})

    result = discover_ses_cognition_paths(
        df, dataset_name="test", mediators=["mediator"], use_llm=False
    )
    assert "adjacency" in result
    assert "llm_refined" not in result or result["llm_refined"] is False


def test_pc_with_llm_flag(monkeypatch):
    """With use_llm=True, should call causal_graph_refiner."""
    from src.analysis.pc_algorithm import discover_ses_cognition_paths

    calls = []

    def mock_refine(adj_matrix, variable_names):
        calls.append((adj_matrix, variable_names))
        return adj_matrix  # return unchanged

    monkeypatch.setattr("src.analysis.pc_algorithm.refine_graph", mock_refine)

    df = _make_synthetic_dag(n=200)
    df = df.rename(columns={"X": "ses_index", "Y": "cognitive_score", "M": "mediator"})

    result = discover_ses_cognition_paths(
        df, dataset_name="test", mediators=["mediator"], use_llm=True
    )
    assert len(calls) == 1
    assert result["llm_refined"] is True
