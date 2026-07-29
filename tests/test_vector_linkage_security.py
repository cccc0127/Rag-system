import numpy as np
import pytest

from comparison_experiments.security_experiments.vector_linkage.attacker import exact_linkage_topk
from comparison_experiments.security_experiments.vector_linkage.attacker import build_public_candidate_vectors
from comparison_experiments.security_experiments.vector_linkage.metrics import compute_vector_linkage_metrics
from comparison_experiments.schemes.private_rag_random_projection import PrivateRAGRandomProjectionScheme


def test_exact_cosine_linkage_recovers_matching_candidate_identity():
    candidates = np.eye(6, dtype=np.float32)
    protected = candidates.copy()
    ranked = exact_linkage_topk(protected, candidates, "cosine", top_k=5)
    metrics = compute_vector_linkage_metrics(ranked, {1, 4})
    assert metrics["linkage_top1_recovery_rate"] == 1.0
    assert metrics["linkage_recall_at_5"] == 1.0
    assert metrics["sensitive_linkage_top1_recovery_rate"] == 1.0


def test_exact_l2_linkage_recovers_matching_candidate_identity():
    candidates = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 3.0], [5.0, 5.0], [7.0, 7.0]], dtype=np.float32)
    ranked = exact_linkage_topk(candidates, candidates, "l2", top_k=5)
    metrics = compute_vector_linkage_metrics(ranked, {0, 3})
    assert metrics["linkage_top1_recovery_rate"] == 1.0
    assert metrics["sensitive_linkage_recall_at_5"] == 1.0


def test_linkage_rejects_dimension_mismatch():
    with pytest.raises(ValueError, match="dimensions must match"):
        exact_linkage_topk(np.ones((2, 3), dtype=np.float32), np.ones((2, 4), dtype=np.float32), "cosine")


def test_public_private_rag_rp_projection_is_linkable_when_seed_is_known():
    documents = np.arange(48, dtype=np.float32).reshape(6, 8) + 1.0
    scheme = PrivateRAGRandomProjectionScheme(projection_dim=6, projection_sigma=0.1, random_seed=42)
    protected = scheme.run(documents, documents[:1], [])
    candidates, metric = build_public_candidate_vectors(scheme, documents)
    ranked = exact_linkage_topk(protected.document_vectors, candidates, metric, top_k=5)
    assert np.array_equal(ranked[:, 0], np.arange(6, dtype=np.int64))
