"""Aggregate metrics for known-candidate vector linkage."""

from __future__ import annotations

from typing import Dict

import numpy as np


def compute_vector_linkage_metrics(
    ranked_candidate_ids: np.ndarray,
    sensitive_chunk_ids: set[int],
) -> Dict[str, float | int]:
    ranked = np.asarray(ranked_candidate_ids, dtype=np.int64)
    if ranked.ndim != 2 or ranked.shape[1] < 5:
        raise ValueError("ranked_candidate_ids must be a 2D Top-5-or-greater array")
    true_ids = np.arange(ranked.shape[0], dtype=np.int64)
    all_metrics = _subset_metrics(ranked, true_ids)
    sensitive_ids = np.array(sorted(item for item in sensitive_chunk_ids if 0 <= item < len(true_ids)))
    sensitive_metrics = _subset_metrics(ranked[sensitive_ids], sensitive_ids) if len(sensitive_ids) else _nan_metrics()
    return {
        "candidate_pool_size": int(ranked.shape[0]),
        "linkage_top1_recovery_rate": all_metrics["top1"],
        "linkage_recall_at_5": all_metrics["recall_at_5"],
        "linkage_mrr_at_5": all_metrics["mrr_at_5"],
        "num_sensitive_chunks": int(len(sensitive_ids)),
        "sensitive_linkage_top1_recovery_rate": sensitive_metrics["top1"],
        "sensitive_linkage_recall_at_5": sensitive_metrics["recall_at_5"],
        "sensitive_linkage_mrr_at_5": sensitive_metrics["mrr_at_5"],
    }


def _subset_metrics(ranked: np.ndarray, true_ids: np.ndarray) -> dict[str, float]:
    top1 = ranked[:, 0] == true_ids
    ranks: list[float] = []
    for true_id, candidate_ids in zip(true_ids, ranked[:, :5]):
        matching = np.flatnonzero(candidate_ids == true_id)
        ranks.append(float(matching[0] + 1) if len(matching) else float("inf"))
    recovered = np.array([np.isfinite(rank) for rank in ranks], dtype=bool)
    reciprocal = [1.0 / rank if np.isfinite(rank) else 0.0 for rank in ranks]
    return {
        "top1": float(np.mean(top1)),
        "recall_at_5": float(np.mean(recovered)),
        "mrr_at_5": float(np.mean(reciprocal)),
    }


def _nan_metrics() -> dict[str, float]:
    return {"top1": float("nan"), "recall_at_5": float("nan"), "mrr_at_5": float("nan")}
