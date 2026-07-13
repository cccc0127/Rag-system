"""Metrics for Our DP-RAG semantic decomposition experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class RetrievalStage:
    name: str
    vectors: np.ndarray
    query_vectors: np.ndarray
    topk_indices: np.ndarray
    query_times: np.ndarray
    vector_dim: int


def exact_retrieval(
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Run exact cosine retrieval with per-query timing."""
    document_vectors = l2_normalize(document_vectors)
    query_vectors = l2_normalize(query_vectors)

    topk_rows: List[np.ndarray] = []
    query_times: List[float] = []
    for query_vector in query_vectors:
        start = time.perf_counter()
        scores = document_vectors @ query_vector
        topk = np.argsort(scores)[::-1][:top_k]
        query_times.append(time.perf_counter() - start)
        topk_rows.append(topk.astype(np.int64))

    return np.vstack(topk_rows), np.asarray(query_times, dtype=np.float64)


def evaluate_comparison(
    comparison_name: str,
    reference_stage: RetrievalStage,
    candidate_stage: RetrievalStage,
    mean_noise_signal_ratio: float = float("nan"),
    mean_direction_cosine: float = float("nan"),
) -> Dict[str, float | int | str]:
    """Evaluate candidate retrieval against a reference stage Top-K."""
    reference_topk = reference_stage.topk_indices
    candidate_topk = candidate_stage.topk_indices

    return {
        "comparison_name": comparison_name,
        "reference_stage": reference_stage.name,
        "candidate_stage": candidate_stage.name,
        "recall_at_1": mean_overlap(reference_topk, candidate_topk, 1),
        "recall_at_3": mean_overlap(reference_topk, candidate_topk, 3),
        "recall_at_5": mean_overlap(reference_topk, candidate_topk, 5),
        "recall_at_10": mean_overlap(reference_topk, candidate_topk, 10),
        "mrr_at_5": mean_mrr_at_5(reference_topk, candidate_topk),
        "mean_query_time": float(np.mean(candidate_stage.query_times)),
        "reference_dim": int(reference_stage.vector_dim),
        "candidate_dim": int(candidate_stage.vector_dim),
        "mean_noise_signal_ratio": float(mean_noise_signal_ratio),
        "mean_direction_cosine": float(mean_direction_cosine),
    }


def mean_overlap(reference_topk: np.ndarray, candidate_topk: np.ndarray, k: int) -> float:
    values: List[float] = []
    for reference, candidate in zip(reference_topk, candidate_topk):
        reference_set = set(int(idx) for idx in reference[:k])
        candidate_set = set(int(idx) for idx in candidate[:k])
        values.append(len(reference_set & candidate_set) / max(1, min(k, len(reference_set))))
    return float(np.mean(values)) if values else float("nan")


def mean_mrr_at_5(reference_topk: np.ndarray, candidate_topk: np.ndarray) -> float:
    values: List[float] = []
    for reference, candidate in zip(reference_topk, candidate_topk):
        if len(reference) == 0:
            values.append(0.0)
            continue
        target_id = int(reference[0])
        reciprocal_rank = 0.0
        for rank, idx in enumerate(candidate[:5], start=1):
            if int(idx) == target_id:
                reciprocal_rank = 1.0 / rank
                break
        values.append(reciprocal_rank)
    return float(np.mean(values)) if values else float("nan")


def mean_direction_cosine(reference_vectors: np.ndarray, candidate_vectors: np.ndarray) -> float:
    reference_vectors = l2_normalize(reference_vectors)
    candidate_vectors = l2_normalize(candidate_vectors)
    if reference_vectors.shape != candidate_vectors.shape:
        return float("nan")
    values = np.sum(reference_vectors * candidate_vectors, axis=1)
    return float(np.mean(values))


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    norms = np.linalg.norm(array, ord=2, axis=1, keepdims=True)
    return (array / np.maximum(norms, eps)).astype(np.float32)


def rows_to_table(rows: Iterable[Dict[str, float | int | str]]) -> str:
    rows = list(rows)
    if not rows:
        return "(no rows)"
    columns = [
        "comparison_name",
        "recall_at_5",
        "mrr_at_5",
        "mean_query_time",
        "mean_noise_signal_ratio",
        "mean_direction_cosine",
    ]
    widths = {column: len(column) for column in columns}
    rendered_rows: List[Dict[str, str]] = []
    for row in rows:
        rendered = {}
        for column in columns:
            value = row.get(column, "")
            if isinstance(value, float):
                rendered[column] = "nan" if np.isnan(value) else f"{value:.6f}"
            else:
                rendered[column] = str(value)
            widths[column] = max(widths[column], len(rendered[column]))
        rendered_rows.append(rendered)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    rule = "-+-".join("-" * widths[column] for column in columns)
    body = [
        " | ".join(row[column].ljust(widths[column]) for column in columns)
        for row in rendered_rows
    ]
    return "\n".join([header, rule, *body])
