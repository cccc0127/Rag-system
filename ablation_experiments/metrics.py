"""Metric computation for DP-RAG ablation experiments."""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np

from dimension_reduction import l2_normalize
from evaluator import cosine_scores, safe_pearson, top_k_indices
from ablation_experiments.schemes import SchemeOutput


def evaluate_scheme_metrics(
    scheme_output: SchemeOutput,
    reduced_raw_embeddings: np.ndarray,
    query_reduced: np.ndarray,
    utility_scale: float,
) -> Dict[str, float | str]:
    signal_norms = _l2_norms(scheme_output.signal_vectors)
    noise_norms = _l2_norms(scheme_output.noise_vectors)
    nsr = noise_norms / np.maximum(signal_norms, 1e-12)
    direction_cosines = _paired_cosine(reduced_raw_embeddings, scheme_output.vectors)

    overlap_values: Dict[int, list[float]] = {1: [], 3: [], 5: [], 10: []}
    mrr5_values: list[float] = []
    pearson_values: list[float] = []
    mean_drift_values: list[float] = []
    max_drift_values: list[float] = []
    raw_times: list[float] = []
    noised_times: list[float] = []

    retrieval_depth = 10
    for query_vec in query_reduced:
        raw_start = time.perf_counter()
        raw_scores = cosine_scores(query_vec, reduced_raw_embeddings)
        raw_top = top_k_indices(raw_scores, retrieval_depth)
        raw_times.append(time.perf_counter() - raw_start)

        noised_start = time.perf_counter()
        noised_scores = cosine_scores(query_vec, scheme_output.vectors)
        noised_top = top_k_indices(noised_scores, retrieval_depth)
        noised_times.append(time.perf_counter() - noised_start)

        for k in overlap_values:
            overlap_values[k].append(_overlap_at_k(raw_top, noised_top, k))
        mrr5_values.append(_reciprocal_rank_at_k(int(raw_top[0]), noised_top, 5))
        pearson_values.append(safe_pearson(raw_scores, noised_scores))
        abs_drift = np.abs(noised_scores - raw_scores)
        mean_drift_values.append(float(np.mean(abs_drift)))
        max_drift_values.append(float(np.max(abs_drift)))

    return {
        "scheme": scheme_output.name,
        "utility_scale": float(utility_scale),
        "mean_nsr": float(np.mean(nsr)),
        "mean_overlap1": float(np.mean(overlap_values[1])),
        "mean_overlap3": float(np.mean(overlap_values[3])),
        "mean_overlap5": float(np.mean(overlap_values[5])),
        "mean_overlap10": float(np.mean(overlap_values[10])),
        "mean_mrr5": float(np.mean(mrr5_values)),
        "mean_pearson": float(np.nanmean(pearson_values)),
        "mean_drift": float(np.mean(mean_drift_values)),
        "max_drift": float(np.max(max_drift_values)),
        "mean_direction_cosine": float(np.mean(direction_cosines)),
        "dp_noise_time": float(scheme_output.dp_noise_time),
        "mean_raw_retrieval_time": float(np.mean(raw_times)),
        "mean_noised_retrieval_time": float(np.mean(noised_times)),
        "mean_sigma": _nanmean(scheme_output.sigmas),
        "max_sigma": _nanmax(scheme_output.sigmas),
        "mean_sigma_per_dim": _nanmean(scheme_output.sigma_per_dim),
        "mean_epsilon": _nanmean(scheme_output.epsilons),
        "min_epsilon": _nanmin(scheme_output.epsilons),
    }


def metric_rows(results: Sequence[Dict[str, float | str]]) -> list[list[object]]:
    return [
        [
            item["scheme"],
            f"{float(item['utility_scale']):g}",
            f"{float(item['mean_nsr']):.6f}",
            f"{float(item['mean_overlap5']):.6f}",
            f"{float(item['mean_mrr5']):.6f}",
            f"{float(item['mean_pearson']):.6f}",
            f"{float(item['mean_drift']):.6f}",
            f"{float(item['mean_direction_cosine']):.6f}",
        ]
        for item in results
    ]


def _paired_cosine(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.sum(l2_normalize(left) * l2_normalize(right), axis=1)


def _l2_norms(vectors: np.ndarray) -> np.ndarray:
    return np.linalg.norm(vectors, ord=2, axis=1)


def _overlap_at_k(raw_top: np.ndarray, noised_top: np.ndarray, k: int) -> float:
    raw_set = set(int(idx) for idx in raw_top[:k])
    noised_set = set(int(idx) for idx in noised_top[:k])
    return len(raw_set & noised_set) / max(1, min(k, len(raw_set)))


def _reciprocal_rank_at_k(target_id: int, ranked_ids: np.ndarray, k: int) -> float:
    for rank, idx in enumerate(ranked_ids[:k], start=1):
        if int(idx) == int(target_id):
            return 1.0 / rank
    return 0.0


def _nanmean(values: np.ndarray) -> float:
    return float(np.nanmean(values)) if not np.all(np.isnan(values)) else float("nan")


def _nanmax(values: np.ndarray) -> float:
    return float(np.nanmax(values)) if not np.all(np.isnan(values)) else float("nan")


def _nanmin(values: np.ndarray) -> float:
    return float(np.nanmin(values)) if not np.all(np.isnan(values)) else float("nan")
