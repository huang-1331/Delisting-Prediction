"""Shared feature contract, fitted-model loader, and scoring for Models A–C."""

from hashlib import sha256
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CONTRACT = json.loads((ROOT / "feature_contract.json").read_text(encoding="utf-8"))
SEEDS = (42, 43, 44)


def load_fitted(model):
    """Load a bundled benchmark ensemble after checking its recorded checksum."""
    expected = CONTRACT["model_sha256"][model]
    artifact = ROOT / "weights" / f"model_{model.lower()}.joblib"
    if sha256(artifact.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Model {model} artifact checksum mismatch")
    return joblib.load(artifact)


def features(frame):
    """Return the 56 numeric inputs in their original training order."""
    columns = CONTRACT["feature_columns_ABC"]
    return frame[columns].astype(float).replace([np.inf, -np.inf], np.nan)


def fit_ensemble(frame, y, model, estimator):
    """Fit the specified ensemble using only the provided training rows."""
    if model not in CONTRACT["thresholds"]:
        raise ValueError(f"Unknown model: {model}")
    x = features(frame)
    y = np.asarray(y)
    if len(y) != len(x) or set(y) != {0, 1}:
        raise ValueError("Supply aligned binary labels containing both classes")
    return [make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
        StandardScaler(), estimator(seed),
    ).fit(x, y) for seed in SEEDS]


def predict(frame, fitted, model):
    """Return the mean positive-class scores and warnings at the locked cutoff."""
    x = features(frame)
    score = np.mean([estimator.predict_proba(x)[:, 1] for estimator in fitted], axis=0)
    return score, score >= CONTRACT["thresholds"][model]
