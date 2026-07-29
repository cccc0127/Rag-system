import numpy as np
import pytest

pytest.importorskip("sklearn")

from comparison_experiments.security_experiments.membership_inference.attacker import (
    shadow_logistic_regression_attack,
)
from comparison_experiments.security_experiments.membership_inference.dataset_split import (
    GROUP_NAMES,
    build_membership_data_split,
    split_source_documents,
)
from comparison_experiments.security_experiments.membership_inference.metrics import (
    attack_advantage,
    bootstrap_confidence_intervals,
    membership_metrics,
    tpr_at_fpr,
)


def _documents(count: int) -> list[dict[str, str]]:
    return [{"filename": f"doc_{index}.txt", "content": "independent source document sentence. " * 20} for index in range(count)]


def test_source_document_split_is_disjoint_and_balanced_chunk_groups():
    source_groups = split_source_documents(_documents(12), seed=2030)
    names = [{record["filename"] for record in source_groups[group]} for group in GROUP_NAMES]
    assert all(not (left & right) for index, left in enumerate(names) for right in names[index + 1:])
    data = build_membership_data_split(_documents(12), 2, 2, 80, 0, False, 2030)
    assert all(len(data.groups[group]) == 2 for group in GROUP_NAMES)
    source_sets = list(data.source_document_sets().values())
    assert all(not (left & right) for index, left in enumerate(source_sets) for right in source_sets[index + 1:])


def test_source_document_split_reports_insufficient_groups():
    with pytest.raises(ValueError, match="four source documents"):
        split_source_documents(_documents(3), seed=2030)


def test_shadow_classifier_does_not_train_on_target_vectors_and_learns_separable_signal():
    rng = np.random.default_rng(4)
    shadow_member = rng.normal(2.0, 0.2, size=(30, 4)).astype(np.float32)
    shadow_nonmember = rng.normal(-2.0, 0.2, size=(30, 4)).astype(np.float32)
    target_member = rng.normal(2.0, 0.2, size=(20, 4)).astype(np.float32)
    target_nonmember = rng.normal(-2.0, 0.2, size=(20, 4)).astype(np.float32)
    result = shadow_logistic_regression_attack(shadow_member, shadow_nonmember, np.vstack((target_member, target_nonmember)), 2031)
    labels = np.array([1] * 20 + [0] * 20)
    assert result.train_size == 60
    assert membership_metrics(labels, result.scores, result.predictions, 0.01)["roc_auc"] > 0.95


def test_random_uninformative_vectors_have_near_chance_shadow_auc():
    rng = np.random.default_rng(18)
    shadow_member = rng.normal(0, 1, size=(100, 5)).astype(np.float32)
    shadow_nonmember = rng.normal(0, 1, size=(100, 5)).astype(np.float32)
    target = rng.normal(0, 1, size=(200, 5)).astype(np.float32)
    labels = np.array([1] * 100 + [0] * 100)
    result = shadow_logistic_regression_attack(shadow_member, shadow_nonmember, target, 2031)
    auc = float(membership_metrics(labels, result.scores, result.predictions, 0.01)["roc_auc"])
    assert 0.3 < auc < 0.7


def test_advantage_low_fpr_and_bootstrap_intervals_are_correct_and_reproducible():
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.01, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.99])
    predictions = (scores >= 0.5).astype(np.int64)
    assert attack_advantage(labels, scores) == 1.0
    assert tpr_at_fpr(labels, scores, 0.01) == 1.0
    first = bootstrap_confidence_intervals(labels, scores, predictions, 0.01, 40, 2031)
    second = bootstrap_confidence_intervals(labels, scores, predictions, 0.01, 40, 2031)
    assert first == second
    assert first["roc_auc_ci_low"] == 1.0


def test_result_metrics_mark_low_fpr_resolution_and_have_no_record_fields():
    labels = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    result = membership_metrics(labels, scores, (scores >= 0.5).astype(int), 0.01)
    assert result["low_fpr_resolution_limited"] is True
    forbidden = {"content", "filename", "path", "chunk_id", "prediction_scores"}
    assert not forbidden & set(result)
