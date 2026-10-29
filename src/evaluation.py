"""Model evaluation utilities for imbalanced delisting prediction."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class ClassificationMetrics:
    """Core metrics required by the research protocol."""

    recall: float
    precision: float
    f1: float
    roc_auc: float | None
    pr_auc: float | None
    confusion_matrix: np.ndarray


def evaluate_binary_classifier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> ClassificationMetrics:
    """Evaluate a binary classifier without making accuracy central.

    Args:
        y_true: Ground-truth labels where delisting within 12 months is 1.
        y_pred: Predicted binary labels.
        y_score: Optional positive-class score or probability.

    Returns:
        Required classification metrics. AUC values are None when scores are
        unavailable or when only one class is present.
    """

    roc_auc = None
    pr_auc = None
    if y_score is not None and len(np.unique(y_true)) > 1:
        roc_auc = float(roc_auc_score(y_true, y_score))
        pr_auc = float(average_precision_score(y_true, y_score))

    return ClassificationMetrics(
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=confusion_matrix(y_true, y_pred),
    )




def evaluate_algorithm_scores(y_true, scores, threshold):
    """Return JSON-compatible required metrics for caller-supplied binary labels.

    Inputs are actual labels, finite scores in [0,1], and a frozen threshold.
    Output includes a fixed 2x2 [normal, event] confusion matrix. No threshold
    selection occurs here. Scores need not be calibrated event probabilities.
    """
    y = np.asarray(y_true)
    p = np.asarray(scores, dtype=float)
    if not len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Evaluation requires nonempty binary labels')
    if len(y) != len(p) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid evaluation scores')
    decision = p >= threshold
    result = evaluate_binary_classifier(y, decision, p)
    cm = confusion_matrix(y, decision, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return dict(n=len(y), recall=result.recall, precision=result.precision,
                f1=result.f1, roc_auc=result.roc_auc, pr_auc=result.pr_auc,
                confusion_matrix=cm.tolist(), tn=int(tn), fp=int(fp),
                fn=int(fn), tp=int(tp), threshold=float(threshold),
                false_positive_rate=float(fp/(tn+fp)) if tn+fp else None,
                accuracy=float(np.mean(decision == y)))
