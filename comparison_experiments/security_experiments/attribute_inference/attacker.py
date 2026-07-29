"""Auxiliary-data attribute attacker with strict out-of-fold evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from comparison_experiments.security_experiments.attribute_inference.metrics import tpr_at_fpr


def run_stratified_attribute_attack(
    protected_vectors: np.ndarray,
    labels: np.ndarray,
    cv_folds: int,
    random_seed: int,
) -> dict[str, object]:
    """Train only on each fold's auxiliary subset and predict its held-out target subset."""
    vectors = np.asarray(protected_vectors, dtype=np.float32)
    targets = np.asarray(labels, dtype=np.int64)
    if vectors.ndim != 2 or len(vectors) != len(targets):
        raise ValueError("protected_vectors must be a 2D matrix aligned with labels")
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least two")
    class_counts = np.bincount(targets, minlength=2)
    if np.any(class_counts < cv_folds):
        raise ValueError("Each class needs at least cv_folds samples for stratified evaluation")

    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    oof_scores = np.full(len(targets), np.nan, dtype=float)
    oof_predictions = np.full(len(targets), -1, dtype=np.int64)
    fold_metrics: list[dict[str, float]] = []
    split_sizes: list[tuple[int, int]] = []
    split_indices: list[tuple[np.ndarray, np.ndarray]] = []
    for train_ids, test_ids in splitter.split(vectors, targets):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=random_seed),
        )
        classifier.fit(vectors[train_ids], targets[train_ids])
        scores = classifier.predict_proba(vectors[test_ids])[:, 1]
        predictions = classifier.predict(vectors[test_ids])
        oof_scores[test_ids] = scores
        oof_predictions[test_ids] = predictions
        test_labels = targets[test_ids]
        fold_metrics.append(
            {
                "roc_auc": float(roc_auc_score(test_labels, scores)),
                "tpr_at_fpr_1pct": tpr_at_fpr(test_labels, scores),
                "macro_f1": float(f1_score(test_labels, predictions, average="macro", zero_division=0)),
                "macro_precision": float(precision_score(test_labels, predictions, average="macro", zero_division=0)),
                "macro_recall": float(recall_score(test_labels, predictions, average="macro", zero_division=0)),
            }
        )
        split_sizes.append((int(len(train_ids)), int(len(test_ids))))
        split_indices.append((train_ids.copy(), test_ids.copy()))
    if np.isnan(oof_scores).any() or np.any(oof_predictions < 0):
        raise RuntimeError("Out-of-fold evaluation did not score every target vector")
    return {
        "scores": oof_scores,
        "predictions": oof_predictions,
        "fold_metrics": fold_metrics,
        "split_sizes": split_sizes,
        "split_indices": split_indices,
    }
