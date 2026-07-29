"""Standard aggregate metrics and bootstrap intervals for membership inference."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def tpr_at_fpr(labels: np.ndarray, scores: np.ndarray, low_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    allowed = tpr[fpr <= low_fpr]
    return float(np.max(allowed)) if len(allowed) else 0.0


def attack_advantage(labels: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    return float(np.max(tpr - fpr))


def membership_metrics(labels: np.ndarray, scores: np.ndarray, predictions: np.ndarray, low_fpr: float) -> dict[str, float | int | bool]:
    y_true = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=float)
    y_pred = np.asarray(predictions, dtype=np.int64)
    if len(y_true) != len(score_values) or len(y_true) != len(y_pred) or len(np.unique(y_true)) != 2:
        raise ValueError("Membership metrics require aligned binary labels, scores, and predictions")
    negatives = int(np.sum(y_true == 0))
    return {
        "roc_auc": float(roc_auc_score(y_true, score_values)),
        "tpr_at_fpr": tpr_at_fpr(y_true, score_values, low_fpr),
        "attack_advantage": attack_advantage(y_true, score_values),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "target_member_count": int(np.sum(y_true == 1)),
        "target_nonmember_count": negatives,
        "low_fpr_resolution_limited": negatives < int(round(1.0 / low_fpr)),
    }


def bootstrap_confidence_intervals(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    low_fpr: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, float]:
    """Stratified target-test bootstrap percentile intervals, reproducible by seed."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be greater than zero")
    y_true = np.asarray(labels, dtype=np.int64)
    score_values = np.asarray(scores, dtype=float)
    y_pred = np.asarray(predictions, dtype=np.int64)
    positive_ids, negative_ids = np.flatnonzero(y_true == 1), np.flatnonzero(y_true == 0)
    if not len(positive_ids) or not len(negative_ids):
        raise ValueError("Bootstrap requires both membership classes")
    rng = np.random.default_rng(seed)
    values = {"roc_auc": [], "tpr_at_fpr": [], "attack_advantage": []}
    for _ in range(bootstrap_samples):
        sample_ids = np.concatenate((rng.choice(positive_ids, len(positive_ids), replace=True), rng.choice(negative_ids, len(negative_ids), replace=True)))
        sample = membership_metrics(y_true[sample_ids], score_values[sample_ids], y_pred[sample_ids], low_fpr)
        for key in values:
            values[key].append(float(sample[key]))
    result: dict[str, float] = {}
    for key, samples in values.items():
        result[f"{key}_ci_low"] = float(np.percentile(samples, 2.5))
        result[f"{key}_ci_high"] = float(np.percentile(samples, 97.5))
    return result
