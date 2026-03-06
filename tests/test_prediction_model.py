# tests/test_prediction_model.py
"""Tests for XGBoost + SHAP prediction model."""
import pytest
import pandas as pd
import numpy as np


def _make_prediction_data(n=300, seed=42):
    """Create synthetic data for prediction tests."""
    rng = np.random.default_rng(seed)
    ses = rng.normal(0, 1, n)
    insurance = rng.binomial(1, 0.6, n).astype(float)
    health = rng.normal(3, 1, n)
    cog = 0.5 * ses + 0.3 * insurance + 0.2 * health + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        'ses_index': ses,
        'has_insurance': insurance,
        'general_health': health,
        'cognitive_score': cog,
    })


def test_model_trains_on_data():
    """Model should train without errors."""
    from src.analysis.prediction_model import CognitivePredictionModel

    df = _make_prediction_data()
    model = CognitivePredictionModel()
    X = df[['ses_index', 'has_insurance', 'general_health']]
    y = df['cognitive_score']
    model.train(X, y)

    assert model.model is not None


def test_model_predicts_continuous():
    """Predictions should be continuous floats."""
    from src.analysis.prediction_model import CognitivePredictionModel

    df = _make_prediction_data()
    model = CognitivePredictionModel()
    X = df[['ses_index', 'has_insurance', 'general_health']]
    y = df['cognitive_score']
    model.train(X, y)

    preds = model.predict(X)
    assert len(preds) == len(X)
    assert preds.dtype == np.float32 or preds.dtype == np.float64


def test_model_cross_validate():
    """Cross-validation should return R² and RMSE for each fold."""
    from src.analysis.prediction_model import CognitivePredictionModel

    df = _make_prediction_data(n=200)
    model = CognitivePredictionModel()
    X = df[['ses_index', 'has_insurance', 'general_health']]
    y = df['cognitive_score']

    cv_results = model.cross_validate(X, y, cv=5)
    assert 'r2_scores' in cv_results
    assert 'rmse_scores' in cv_results
    assert len(cv_results['r2_scores']) == 5
    assert len(cv_results['rmse_scores']) == 5


def test_shap_values_computed():
    """SHAP values should have same shape as X."""
    from src.analysis.prediction_model import CognitivePredictionModel

    df = _make_prediction_data()
    model = CognitivePredictionModel()
    X = df[['ses_index', 'has_insurance', 'general_health']]
    y = df['cognitive_score']
    model.train(X, y)

    shap_vals = model.compute_shap_values(X)
    assert shap_vals.shape == X.shape


def test_shap_feature_importance():
    """Feature importance should return sorted dict of feature→mean|SHAP|."""
    from src.analysis.prediction_model import CognitivePredictionModel

    df = _make_prediction_data()
    model = CognitivePredictionModel()
    X = df[['ses_index', 'has_insurance', 'general_health']]
    y = df['cognitive_score']
    model.train(X, y)
    model.compute_shap_values(X)

    importance = model.feature_importance()
    assert isinstance(importance, dict)
    assert 'ses_index' in importance
    # ses_index has strongest effect (0.5), should rank high
    keys = list(importance.keys())
    assert keys[0] == 'ses_index'


def test_partial_dependence():
    """Should return (feature_values, predicted_values) arrays."""
    from src.analysis.prediction_model import CognitivePredictionModel

    df = _make_prediction_data()
    model = CognitivePredictionModel()
    X = df[['ses_index', 'has_insurance', 'general_health']]
    y = df['cognitive_score']
    model.train(X, y)

    feat_vals, pred_vals = model.partial_dependence(X, 'ses_index', grid_points=20)
    assert len(feat_vals) == 20
    assert len(pred_vals) == 20
    # Predictions should increase with SES (positive relationship)
    assert pred_vals[-1] > pred_vals[0]
