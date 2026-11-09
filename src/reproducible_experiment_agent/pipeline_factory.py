"""The finite pipeline allowlist: imputers, encoders, scalers and three model families.

Every transform lives inside one scikit-learn Pipeline, so ``fit`` touches the
training rows only and dev or holdout rows are only ever transformed.
"""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, RobustScaler, StandardScaler

from .schemas import ExperimentSpec, TaskType


def _model(spec: ExperimentSpec, task_type: TaskType, seed: int) -> Any:
    p = dict(spec.params)
    if spec.model == "linear":
        if task_type == "classification":
            return LogisticRegression(C=float(p.get("C", 1.0)), max_iter=2000)
        return Ridge(alpha=float(p.get("alpha", 1.0)))
    if spec.model == "random_forest":
        kw = {
            "n_estimators": int(p.get("n_estimators", 100)),
            "max_depth": None if p.get("max_depth") is None else int(p["max_depth"]),
            "min_samples_leaf": int(p.get("min_samples_leaf", 1)),
            "random_state": seed,
            "n_jobs": 1,
        }
        if task_type == "classification":
            return RandomForestClassifier(**kw)
        return RandomForestRegressor(**kw)
    kw = {
        "learning_rate": float(p.get("learning_rate", 0.1)),
        "max_iter": int(p.get("max_iter", 100)),
        "max_leaf_nodes": int(p.get("max_leaf_nodes", 31)),
        "l2_regularization": float(p.get("l2_regularization", 0.0)),
        "min_samples_leaf": int(p.get("min_samples_leaf", 20)),
        "random_state": seed,
    }
    if task_type == "classification":
        return HistGradientBoostingClassifier(**kw)
    return HistGradientBoostingRegressor(**kw)


def build_pipeline(
    spec: ExperimentSpec,
    task_type: TaskType,
    numeric: list[str],
    categorical: list[str],
    seed: int,
) -> Pipeline:
    """Assemble preprocessing and model for one trial.

    Args:
        spec: Validated trial spec.
        task_type: Classification or regression.
        numeric: Numeric feature columns to use.
        categorical: Categorical feature columns to use.
        seed: Random state for stochastic models.

    Returns:
        An unfitted Pipeline. Unseen categories at transform time are encoded
        as all-zero (one-hot) or -1 (ordinal), never an error.
    """
    imputer = {"median": "median", "mean": "mean", "zero": "constant"}[spec.numeric_imputer]
    num_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy=imputer, fill_value=0.0)),
    ]
    if spec.scaler != "none":
        num_steps.append(
            ("scale", StandardScaler() if spec.scaler == "standard" else RobustScaler())
        )
    if spec.categorical_encoder == "onehot":
        enc: Any = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")), ("encode", enc)]
    transformers = []
    if numeric:
        transformers.append(("num", Pipeline(num_steps), numeric))
    if categorical:
        transformers.append(("cat", Pipeline(cat_steps), categorical))
    pre = ColumnTransformer(transformers, remainder="drop")
    return Pipeline([("pre", pre), ("model", _model(spec, task_type, seed))])
