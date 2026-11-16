"""Model C: unweighted XGBoost."""

from xgboost import XGBClassifier

from .common import fit_ensemble, predict


def fit(frame, y):
    """Return three benchmark XGBoost estimators fitted to the training rows."""
    return fit_ensemble(frame, y, "C", lambda seed: XGBClassifier(
        n_estimators=150, max_depth=2, learning_rate=.03, min_child_weight=3,
        reg_lambda=5, subsample=.8, colsample_bytree=.8, scale_pos_weight=1,
        random_state=seed, n_jobs=2, eval_metric="logloss"))


def score(frame, fitted):
    """Return Model C scores and warnings for prepared input rows."""
    return predict(frame, fitted, "C")
