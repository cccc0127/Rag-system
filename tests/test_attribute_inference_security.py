import numpy as np
import pytest

pytest.importorskip("sklearn")

from comparison_experiments.security_experiments.attribute_inference.attacker import (
    run_stratified_attribute_attack,
)
from comparison_experiments.security_experiments.attribute_inference.labels import (
    build_sensitive_attribute_labels,
)
from comparison_experiments.security_experiments.attribute_inference.metrics import (
    aggregate_oof_metrics,
    tpr_at_fpr,
)


def test_labels_return_binary_attributes_without_returning_entity_values():
    records = [
        {"content": "Reach us at alpha@example.com"},
        {"content": "Documentation is at https://example.org/help"},
        {"content": "Plain public content only."},
    ]
    labels = build_sensitive_attribute_labels(records, ("email", "url"))
    assert labels["has_email"].tolist() == [1, 0, 0]
    assert labels["has_url"].tolist() == [0, 1, 0]
    assert labels["has_any_sensitive"].tolist() == [1, 1, 0]


def test_attribute_attacker_uses_disjoint_train_and_test_folds_and_learns_signal():
    rng = np.random.default_rng(7)
    labels = np.array([0] * 20 + [1] * 20, dtype=np.int64)
    vectors = np.column_stack([labels * 4.0 + rng.normal(0, 0.15, len(labels)), rng.normal(0, 1, len(labels))]).astype(np.float32)
    result = run_stratified_attribute_attack(vectors, labels, cv_folds=5, random_seed=2029)
    assert all(set(train).isdisjoint(test) for train, test in result["split_indices"])
    summary = aggregate_oof_metrics(labels, result["scores"], result["predictions"], result["fold_metrics"])
    assert float(summary["roc_auc"]) > 0.95


def test_tpr_at_low_fpr_and_resolution_flag_are_reported():
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    scores = np.array([0.01, 0.05, 0.10, 0.80, 0.90, 0.99])
    assert tpr_at_fpr(labels, scores, 0.01) == 1.0
    folds = [{"roc_auc": 1.0, "tpr_at_fpr_1pct": 1.0, "macro_f1": 1.0, "macro_precision": 1.0, "macro_recall": 1.0}]
    summary = aggregate_oof_metrics(labels, scores, labels, folds)
    assert summary["low_fpr_resolution_limited"] is True
