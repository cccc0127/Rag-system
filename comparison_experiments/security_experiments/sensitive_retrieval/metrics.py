"""Aggregate metrics for targeted sensitive-retrieval exposure."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from comparison_experiments.security_experiments.sensitive_retrieval.sensitive_data import AttackQuery


def compute_sensitive_retrieval_metrics(
    topk_indices: np.ndarray,
    attack_queries: Sequence[AttackQuery],
    sensitive_chunk_ids: set[int],
    top_k: int,
) -> Dict[str, float | int]:
    if len(topk_indices) != len(attack_queries):
        raise ValueError("topk_indices and attack_queries must have the same number of rows")
    if not attack_queries:
        raise ValueError("At least one attack query is required")

    target_at_1: list[float] = []
    target_at_k: list[float] = []
    sensitive_counts: list[float] = []
    retrieved_ranks: list[int] = []
    if top_k < 5:
        raise ValueError("Targeted exposure metrics require --top-k to be at least 5")
    for ranked_ids, attack in zip(topk_indices, attack_queries):
        ranked = [int(item) for item in ranked_ids[:5]]
        target = int(attack.target_chunk_id)
        target_at_1.append(float(bool(ranked and ranked[0] == target)))
        target_at_k.append(float(target in ranked))
        sensitive_counts.append(float(sum(item in sensitive_chunk_ids for item in ranked)))
        if target in ranked:
            retrieved_ranks.append(ranked.index(target) + 1)

    return {
        "sensitive_target_recall_at_1": float(np.mean(target_at_1)),
        "sensitive_target_recall_at_5": float(np.mean(target_at_k)),
        "sensitive_top1_exposure_rate": float(np.mean(target_at_1)),
        "mean_sensitive_chunks_at_5": float(np.mean(sensitive_counts)),
        "mean_target_rank_when_retrieved": float(np.mean(retrieved_ranks)) if retrieved_ranks else float("nan"),
        "num_attack_queries": int(len(attack_queries)),
        "num_sensitive_targets": int(len({attack.target_chunk_id for attack in attack_queries})),
    }
