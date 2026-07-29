"""Shared metrics for comparison experiments."""

from __future__ import annotations

from typing import Dict

import numpy as np

from comparison_experiments.shared.retrievers import RetrievalResult, exact_topk
from comparison_experiments.shared.types import SchemeOutput


def compute_scheme_metrics(
    scheme_output: SchemeOutput,
    retrieval: RetrievalResult,
    top_k: int,
    ef_search: int | None = None,
) -> Dict[str, float | int | str]:
    reference_depth = max(10, top_k, 5)
    reference_document_vectors = (
        scheme_output.reference_document_vectors
        if scheme_output.reference_document_vectors is not None
        else scheme_output.document_vectors
    )
    reference_query_vectors = (
        scheme_output.reference_query_vectors
        if scheme_output.reference_query_vectors is not None
        else scheme_output.query_vectors
    )
    reference_topk = exact_topk(
        reference_document_vectors,
        reference_query_vectors,
        reference_depth,
    )
    hnsw_topk = retrieval.topk_indices

    metrics: Dict[str, float | int | str] = {
        "scheme": scheme_output.name,
        "backend_type": scheme_output.backend_type,
        "ef_search": int(ef_search) if ef_search is not None else "",
        "vector_dim": int(scheme_output.vector_dim),
        "mean_query_time": float(np.mean(retrieval.query_times)),
        "index_build_time": float(retrieval.index_build_time),
        "mean_noise_signal_ratio": _metadata_float(scheme_output, "mean_noise_signal_ratio"),
        "mean_sigma": _metadata_float(scheme_output, "mean_sigma"),
        "mean_epsilon": _metadata_float(scheme_output, "mean_epsilon"),
        "hnsw_recall_at_5": _mean_overlap(reference_topk, hnsw_topk, 5),
        "hnsw_mrr_at_5": _mean_mrr_at_5(reference_topk, hnsw_topk),
        "sap_noise_signal_ratio": _metadata_float(scheme_output, "sap_noise_signal_ratio"),
        "beta": _metadata_float(scheme_output, "beta"),
        "ratio_k": _metadata_float(scheme_output, "ratio_k"),
        "he_absolute_error_mean": _retrieval_or_metadata_float(
            retrieval,
            scheme_output,
            "he_absolute_error_mean",
        ),
        "he_relative_error_mean": _retrieval_or_metadata_float(
            retrieval,
            scheme_output,
            "he_relative_error_mean",
        ),
        "he_scan_time": _retrieval_or_metadata_float(retrieval, scheme_output, "he_scan_time"),
        "he_refine_time": _retrieval_or_metadata_float(retrieval, scheme_output, "he_refine_time"),
        "ciphertext_size_kb": _retrieval_or_metadata_float(
            retrieval,
            scheme_output,
            "ciphertext_size_kb",
        ),
        "plain_size_kb": _retrieval_or_metadata_float(retrieval, scheme_output, "plain_size_kb"),
        "cipher_expansion_ratio": _retrieval_or_metadata_float(
            retrieval,
            scheme_output,
            "cipher_expansion_ratio",
        ),
    }
    if retrieval.metadata:
        metrics.update(
            {
                f"retrieval_{key}": value
                for key, value in retrieval.metadata.items()
                if isinstance(value, (int, float, str))
            }
        )
    metrics.update(
        {
            f"metadata_{key}": value
            for key, value in scheme_output.metadata.items()
            if isinstance(value, (int, float, str))
        }
    )
    return metrics


def _metadata_float(scheme_output: SchemeOutput, key: str) -> float:
    value = scheme_output.metadata.get(key, float("nan"))
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _retrieval_or_metadata_float(
    retrieval: RetrievalResult,
    scheme_output: SchemeOutput,
    key: str,
) -> float:
    if retrieval.metadata and key in retrieval.metadata:
        try:
            return float(retrieval.metadata[key])
        except (TypeError, ValueError):
            return float("nan")
    return _metadata_float(scheme_output, key)


def _mean_overlap(reference_topk: np.ndarray, candidate_topk: np.ndarray, k: int) -> float:
    values = []
    for reference, candidate in zip(reference_topk, candidate_topk):
        reference_set = set(int(idx) for idx in reference[:k])
        candidate_set = set(int(idx) for idx in candidate[:k])
        values.append(len(reference_set & candidate_set) / max(1, min(k, len(reference_set))))
    return float(np.mean(values)) if values else float("nan")


def _mean_mrr_at_5(reference_topk: np.ndarray, candidate_topk: np.ndarray) -> float:
    values = []
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
