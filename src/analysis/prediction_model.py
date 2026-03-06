"""
XGBoost + SHAP Prediction Model

Trains a non-linear model to predict cognitive scores and uses SHAP
to decompose predictions into feature contributions. Validates
Baron-Kenny mediation results with data-driven feature importance.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

import xgboost as xgb
import shap
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error

logger = logging.getLogger(__name__)


class CognitivePredictionModel:
    """XGBoost model for cognitive score prediction with SHAP explanations."""

    def __init__(self, max_depth=6, n_estimators=100, learning_rate=0.1):
        self.model = None
        self.shap_values_ = None
        self.feature_names_ = None
        self.params = {
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
        }

    def train(self, X: pd.DataFrame, y: pd.Series,
              weights: Optional[np.ndarray] = None):
        """Train XGBoost regressor."""
        self.feature_names_ = list(X.columns)
        self.model = xgb.XGBRegressor(
            max_depth=self.params['max_depth'],
            n_estimators=self.params['n_estimators'],
            learning_rate=self.params['learning_rate'],
            random_state=42,
        )
        self.model.fit(X, y, sample_weight=weights)
        logger.info(f"Trained XGBoost on {len(X)} samples, {len(self.feature_names_)} features")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict cognitive scores."""
        return self.model.predict(X)

    def cross_validate(self, X: pd.DataFrame, y: pd.Series,
                       cv: int = 5, weights: Optional[np.ndarray] = None) -> Dict:
        """K-fold cross-validation returning R² and RMSE per fold."""
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        r2_scores = []
        rmse_scores = []

        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            w_train = weights[train_idx] if weights is not None else None

            model = xgb.XGBRegressor(
                max_depth=self.params['max_depth'],
                n_estimators=self.params['n_estimators'],
                learning_rate=self.params['learning_rate'],
                random_state=42,
            )
            model.fit(X_train, y_train, sample_weight=w_train)
            preds = model.predict(X_test)

            r2_scores.append(r2_score(y_test, preds))
            rmse_scores.append(np.sqrt(mean_squared_error(y_test, preds)))

        return {'r2_scores': r2_scores, 'rmse_scores': rmse_scores}

    def compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values using TreeExplainer."""
        explainer = shap.TreeExplainer(self.model)
        self.shap_values_ = explainer.shap_values(X)
        return self.shap_values_

    def feature_importance(self) -> Dict[str, float]:
        """Return mean |SHAP| per feature, sorted descending."""
        if self.shap_values_ is None:
            raise ValueError("Call compute_shap_values first")
        mean_abs = np.abs(self.shap_values_).mean(axis=0)
        importance = dict(zip(self.feature_names_, mean_abs))
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    def partial_dependence(self, X: pd.DataFrame, feature: str,
                           grid_points: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute partial dependence for a feature.

        Varies the feature across its range while holding others at observed values.
        """
        feat_vals = np.linspace(X[feature].min(), X[feature].max(), grid_points)
        pred_means = []

        for val in feat_vals:
            X_mod = X.copy()
            X_mod[feature] = val
            preds = self.model.predict(X_mod)
            pred_means.append(preds.mean())

        return feat_vals, np.array(pred_means)
