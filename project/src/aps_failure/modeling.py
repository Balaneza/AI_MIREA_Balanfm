from __future__ import annotations

from collections import OrderedDict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_model_candidates(random_state: int = 42, n_jobs: int = -1) -> OrderedDict[str, object]:
    """Models are intentionally classical and reproducible for a tabular APS task."""
    return OrderedDict(
        {
            "LogisticRegression": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=900,
                            class_weight="balanced",
                            solver="lbfgs",
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "RandomForest": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=180,
                            max_depth=16,
                            min_samples_leaf=2,
                            class_weight="balanced_subsample",
                            n_jobs=n_jobs,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                learning_rate=0.06,
                max_iter=220,
                max_leaf_nodes=31,
                l2_regularization=0.05,
                early_stopping=True,
                class_weight="balanced",
                random_state=random_state,
            ),
            "MLP": Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                    ("scaler", StandardScaler()),
                    (
                        "model",
                        MLPClassifier(
                            hidden_layer_sizes=(64, 32),
                            activation="relu",
                            alpha=1e-4,
                            learning_rate_init=5e-4,
                            max_iter=45,
                            early_stopping=True,
                            validation_fraction=0.15,
                            n_iter_no_change=8,
                            random_state=random_state,
                        ),
                    ),
                ]
            ),
        }
    )


def predict_positive_probability(model: object, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x)
        return 1.0 / (1.0 + np.exp(-scores))
    raise TypeError(f"Model {type(model).__name__} does not expose probability scores")
