"""Aggregate metrics for protected-vector sensitive attribute inference."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve


def tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """Maximum true-positive rate reachable without exceeding ``target_fpr``."""
    labels = np.asarray(y_true, dtype=np.int64)
    values = np.asarray(scores, dtype=float)
    if labels.ndim != 1 or values.ndim != 1 or len(labels) != len(values):
        raise ValueError("y_true and scores must be equally sized 1D arrays")
    if len(np.unique(labels)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(labels, values)
    allowed = tpr[fpr <= float(target_fpr)]
    return float(np.max(allowed)) if len(allowed) else 0.0


def aggregate_oof_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    predicted_labels: np.ndarray,
    fold_metrics: Sequence[dict[str, float]],
    target_fpr: float = 0.01,
) -> dict[str, float | int | bool]:
    """Summarize out-of-fold predictions and per-fold variation."""
    labels = np.asarray(y_true, dtype=np.int64)
    score_values = np.asarray(scores, dtype=float)
    predictions = np.asarray(predicted_labels, dtype=np.int64)
    negatives = int(np.sum(labels == 0))
    result: dict[str, float | int | bool] = {
        "positive_count": int(np.sum(labels == 1)),
        "negative_count": negatives,
        "roc_auc": float(roc_auc_score(labels, score_values)),
        "tpr_at_fpr_1pct": tpr_at_fpr(labels, score_values, target_fpr),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "macro_precision": float(precision_score(labels, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, predictions, average="macro", zero_division=0)),
        "low_fpr_resolution_limited": negatives < int(round(1.0 / target_fpr)),
    }
    for metric_name in ("roc_auc", "tpr_at_fpr_1pct", "macro_f1", "macro_precision", "macro_recall"):
        values = np.asarray([fold[metric_name] for fold in fold_metrics], dtype=float)
        result[f"{metric_name}_fold_mean"] = float(np.nanmean(values))
        result[f"{metric_name}_fold_std"] = float(np.nanstd(values, ddof=0))
    return result
