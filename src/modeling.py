"""Model training utilities following the required development sequence."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

try:
    from .config import RANDOM_STATE
except ImportError:  # pragma: no cover - supports direct notebook imports
    from config import RANDOM_STATE


def build_logistic_regression() -> LogisticRegression:
    """Create the stage-2 Logistic Regression classifier."""

    return LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=RANDOM_STATE,
    )


def build_random_forest() -> RandomForestClassifier:
    """Create the stage-3 Random Forest classifier."""

    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_xgboost(scale_pos_weight: float | None = None) -> XGBClassifier:
    """Create the stage-4 XGBoost classifier.

    Args:
        scale_pos_weight: Optional imbalance weight calculated from the
            training set only.
    """

    params = {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "eval_metric": "logloss",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight
    return XGBClassifier(**params)
