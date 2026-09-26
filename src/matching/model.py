"""
Stage 4: LightGBM Matching Model.

Trains a gradient-boosted decision tree binary classifier on similarity features
to predict pairwise match probability P(match | pair_features).
"""

import os
from typing import Dict, List, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd


class MatcherModel:
    """LightGBM Classifier for pairwise entity resolution."""

    def __init__(self, booster: Optional[lgb.Booster] = None, feature_names: Optional[List[str]] = None):
        self.booster = booster
        self.feature_names = feature_names or []

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict match probabilities (values between 0.0 and 1.0)."""
        if self.booster is None:
            raise ValueError("Model is not trained or loaded yet.")
        if self.feature_names:
            X = X[self.feature_names]
        return self.booster.predict(X)

    def save(self, model_path: str) -> None:
        """Save booster model to text file."""
        os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
        self.booster.save_model(model_path)

    @classmethod
    def load(cls, model_path: str) -> "MatcherModel":
        """Load booster model from file."""
        booster = lgb.Booster(model_file=model_path)
        feature_names = booster.feature_name()
        return cls(booster=booster, feature_names=feature_names)

    def get_feature_importances(self) -> pd.DataFrame:
        """Return table of feature importances sorted descending."""
        if self.booster is None:
            return pd.DataFrame()
        imp = self.booster.feature_importance(importance_type="gain")
        names = self.booster.feature_name()
        df = pd.DataFrame({"feature": names, "importance_gain": imp})
        return df.sort_values("importance_gain", ascending=False).reset_index(drop=True)


def train_matcher(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[np.ndarray] = None,
    params: Optional[Dict] = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 30,
) -> MatcherModel:
    """Train a LightGBM model on candidate pair features."""
    feature_names = list(X_train.columns)

    default_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_samples": 20,
        "verbosity": -1,
        "n_jobs": -1,
        "random_state": 42,
    }

    if params:
        default_params.update(params)

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)

    valid_sets = [dtrain]
    valid_names = ["train"]

    callbacks = []
    if X_val is not None and y_val is not None:
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_names)
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False))

    callbacks.append(lgb.log_evaluation(period=50))

    booster = lgb.train(
        default_params,
        dtrain,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    return MatcherModel(booster=booster, feature_names=feature_names)
