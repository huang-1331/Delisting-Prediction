"""Model A: unweighted logistic regression."""

from sklearn.linear_model import LogisticRegression

from .common import fit_ensemble, predict


def fit(frame, y):
    """Return the three benchmark logistic models fitted to the supplied training rows."""
    return fit_ensemble(frame, y, "A", lambda seed: LogisticRegression(
        C=.1, class_weight={0: 1, 1: 1}, max_iter=2000, random_state=seed))


def score(frame, fitted):
    """Return Model A scores and warnings for prepared input rows."""
    return predict(frame, fitted, "A")
