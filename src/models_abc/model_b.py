"""Model B: unweighted Random Forest."""

from sklearn.ensemble import RandomForestClassifier

from .common import fit_ensemble, predict


def fit(frame, y):
    """Return three benchmark Random Forest estimators fitted to the training rows."""
    return fit_ensemble(frame, y, "B", lambda seed: RandomForestClassifier(
        n_estimators=150, max_depth=5, min_samples_leaf=5, class_weight={0: 1, 1: 1},
        random_state=seed, n_jobs=2))


def score(frame, fitted):
    """Return Model B scores and warnings for prepared input rows."""
    return predict(frame, fitted, "B")
