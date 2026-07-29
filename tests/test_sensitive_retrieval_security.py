import numpy as np

from comparison_experiments.security_experiments.sensitive_retrieval.metrics import (
    compute_sensitive_retrieval_metrics,
)
from comparison_experiments.security_experiments.sensitive_retrieval.sensitive_data import (
    AttackQuery,
    build_attack_queries,
    find_sensitive_chunks,
    redact_sensitive_entities,
)


def test_sensitive_entities_are_detected_and_redacted_from_attack_queries():
    records = [
        {"content": "Contact alice@example.com at 555-123-4567 and visit https://example.org/project details."},
        {"content": "General public documentation without private identifiers."},
    ]
    targets = find_sensitive_chunks(records, ("email", "url", "phone"))
    assert len(targets) == 1

    redacted = redact_sensitive_entities(records[0]["content"], ("email", "url", "phone"))
    assert "alice@example.com" not in redacted
    assert "555-123-4567" not in redacted
    assert "https://example.org/project" not in redacted

    attacks = build_attack_queries(records, targets, ("email", "url", "phone"), seed=7)
    assert len(attacks) == 1
    assert "alice@example.com" not in attacks[0].query
    assert "555-123-4567" not in attacks[0].query
    assert "https://example.org/project" not in attacks[0].query


def test_targeted_sensitive_retrieval_metrics():
    attacks = [
        AttackQuery(target_chunk_id=2, entity_types=("email",), query="unused"),
        AttackQuery(target_chunk_id=4, entity_types=("url",), query="unused"),
    ]
    topk = np.array([[2, 1, 0, 3, 4], [1, 4, 2, 0, 3]], dtype=np.int64)
    metrics = compute_sensitive_retrieval_metrics(topk, attacks, {2, 4}, top_k=5)
    assert metrics["sensitive_target_recall_at_1"] == 0.5
    assert metrics["sensitive_target_recall_at_5"] == 1.0
    assert metrics["sensitive_top1_exposure_rate"] == 0.5
    assert metrics["mean_sensitive_chunks_at_5"] == 2.0
