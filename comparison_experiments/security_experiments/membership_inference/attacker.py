"""Shadow-data membership attackers using only protected vector values."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dimension_reduction import l2_normalize


@dataclass(frozen=True)
class MembershipAttackResult:
    scores: np.ndarray
    predictions: np.ndarray
    train_size: int
    threshold: float | None = None


def shadow_logistic_regression_attack(
    shadow_member_vectors: np.ndarray,
    shadow_nonmember_vectors: np.ndarray,
    target_vectors: np.ndarray,
    seed: int,
) -> MembershipAttackResult:
    """Fit only on shadow protected vectors; target vectors remain unseen until scoring."""
    members, nonmembers, target = _validate_vectors(shadow_member_vectors, shadow_nonmember_vectors, target_vectors)
    train_vectors = np.vstack((members, nonmembers))
    train_labels = np.concatenate((np.ones(len(members), dtype=np.int64), np.zeros(len(nonmembers), dtype=np.int64)))
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000, random_state=seed),
    )
    classifier.fit(train_vectors, train_labels)
    scores = classifier.predict_proba(target)[:, 1]
    return MembershipAttackResult(scores=scores, predictions=(scores >= 0.5).astype(np.int64), train_size=len(train_labels))


def shadow_density_knn_attack(
    shadow_member_vectors: np.ndarray,
    shadow_nonmember_vectors: np.ndarray,
    target_vectors: np.ndarray,
    density_knn_k: int,
) -> MembershipAttackResult:
    """Exact normalized-vector density attack calibrated entirely on shadow data.

    Membership score is the negative mean distance to the shadow-member cloud.
    The threshold is selected only from leave-one-out shadow-member scores and
    shadow-nonmember scores, never from target membership labels.
    """
    members, nonmembers, target = _validate_vectors(shadow_member_vectors, shadow_nonmember_vectors, target_vectors)
    if density_knn_k <= 0 or len(members) <= density_knn_k:
        raise ValueError("density_knn_k must be positive and smaller than shadow_member count")
    normalized_members = l2_normalize(members)
    normalized_nonmembers = l2_normalize(nonmembers)
    normalized_target = l2_normalize(target)
    member_scores = -_mean_neighbor_distance(normalized_members, normalized_members, density_knn_k, exclude_self=True)
    nonmember_scores = -_mean_neighbor_distance(normalized_nonmembers, normalized_members, density_knn_k, exclude_self=False)
    threshold = _shadow_balanced_threshold(
        np.concatenate((np.ones(len(member_scores), dtype=np.int64), np.zeros(len(nonmember_scores), dtype=np.int64))),
        np.concatenate((member_scores, nonmember_scores)),
    )
    target_scores = -_mean_neighbor_distance(normalized_target, normalized_members, density_knn_k, exclude_self=False)
    return MembershipAttackResult(scores=target_scores, predictions=(target_scores >= threshold).astype(np.int64), train_size=len(members) + len(nonmembers), threshold=threshold)


def _mean_neighbor_distance(query_vectors: np.ndarray, reference_vectors: np.ndarray, k: int, exclude_self: bool) -> np.ndarray:
    distances = np.maximum(
        np.sum(query_vectors**2, axis=1, keepdims=True) + np.sum(reference_vectors**2, axis=1)[None, :] - 2.0 * (query_vectors @ reference_vectors.T),
        0.0,
    )
    if exclude_self:
        if query_vectors.shape != reference_vectors.shape or not np.allclose(query_vectors, reference_vectors):
            raise ValueError("Self exclusion is only valid for the same shadow-member matrix")
        np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    return np.mean(np.sqrt(nearest), axis=1)


def _shadow_balanced_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    candidates = np.unique(scores)
    best_threshold, best_score = float(candidates[0]), -np.inf
    for threshold in candidates:
        value = balanced_accuracy_score(labels, scores >= threshold)
        if value > best_score:
            best_threshold, best_score = float(threshold), float(value)
    return best_threshold


def _validate_vectors(*matrices: np.ndarray) -> tuple[np.ndarray, ...]:
    converted = tuple(np.asarray(matrix, dtype=np.float32) for matrix in matrices)
    if any(matrix.ndim != 2 or not len(matrix) for matrix in converted):
        raise ValueError("All attack vector matrices must be non-empty and two-dimensional")
    dimensions = {matrix.shape[1] for matrix in converted}
    if len(dimensions) != 1:
        raise ValueError("All attack vector matrices must have the same vector dimension")
    return converted
