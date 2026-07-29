"""Shared retrieval helpers for comparison experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import numpy as np

from hnsw_experiments.hnsw_index import HNSWRetriever
from comparison_experiments.shared.ckks_utils import (
    ckks_squared_l2,
    create_ckks_context,
)


@dataclass
class RetrievalResult:
    topk_indices: np.ndarray
    query_times: np.ndarray
    index_build_time: float
    filter_topk_indices: np.ndarray | None = None
    metadata: Dict[str, float | int | str] | None = None


def run_hnsw_retrieval(
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int,
    ef_search: int,
    M: int,
    ef_construction: int,
    space: str = "cosine",
    random_seed: int = 42,
) -> RetrievalResult:
    document_vectors = np.asarray(document_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    if document_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("document_vectors and query_vectors must be 2D arrays")
    if document_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(
            f"Vector dim mismatch: documents={document_vectors.shape[1]}, "
            f"queries={query_vectors.shape[1]}"
        )

    build_start = time.perf_counter()
    retriever = HNSWRetriever(
        dim=document_vectors.shape[1],
        space=space,
        M=M,
        ef_construction=ef_construction,
        ef_search=ef_search,
        random_seed=random_seed,
    ).build(document_vectors)
    index_build_time = time.perf_counter() - build_start

    topk_rows: list[np.ndarray] = []
    query_times: list[float] = []
    for query_vector in query_vectors:
        search_start = time.perf_counter()
        topk, _ = retriever.search(query_vector, top_k=top_k)
        query_times.append(time.perf_counter() - search_start)
        topk_rows.append(topk)

    return RetrievalResult(
        topk_indices=np.vstack(topk_rows).astype(np.int64),
        query_times=np.array(query_times, dtype=np.float64),
        index_build_time=float(index_build_time),
    )

# 过滤-精简两阶段检索
def run_hnsw_filter_refine_retrieval(
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    reference_document_vectors: np.ndarray,
    reference_query_vectors: np.ndarray,
    top_k: int,
    k_prime: int,
    ef_search: int,
    M: int,
    ef_construction: int,
    space: str = "l2",
    random_seed: int = 42,
) -> RetrievalResult:
    filter_result = run_hnsw_retrieval(
        document_vectors=document_vectors,
        query_vectors=query_vectors,
        top_k=max(top_k, k_prime),
        ef_search=ef_search,
        M=M,
        ef_construction=ef_construction,
        space=space,
        random_seed=random_seed,
    )

    reference_document_vectors = np.asarray(reference_document_vectors, dtype=np.float32)
    reference_query_vectors = np.asarray(reference_query_vectors, dtype=np.float32)
    refined_rows: list[np.ndarray] = []
    refine_times: list[float] = []
    for query_id, candidates in enumerate(filter_result.topk_indices):
        refine_start = time.perf_counter()
        query = reference_query_vectors[query_id]
        candidate_vectors = reference_document_vectors[candidates]
        distances = np.sum((candidate_vectors - query) ** 2, axis=1)
        order = np.argsort(distances)[:top_k]
        refined_rows.append(candidates[order])
        refine_times.append(time.perf_counter() - refine_start)

    return RetrievalResult(
        topk_indices=np.vstack(refined_rows).astype(np.int64),
        query_times=filter_result.query_times + np.array(refine_times, dtype=np.float64),
        index_build_time=filter_result.index_build_time,
        filter_topk_indices=filter_result.topk_indices,
    )


def run_ckks_full_scan_retrieval(
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int,
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: list[int],
    global_scale: float,
) -> RetrievalResult:
    document_vectors = np.asarray(document_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    _validate_vectors(document_vectors, query_vectors)

    context = create_ckks_context(
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        global_scale=global_scale,
    )
    topk_rows: list[np.ndarray] = []
    query_times: list[float] = []
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    cipher_bytes: list[int] = []
    plain_bytes: list[int] = []

    for query_vector in query_vectors:
        search_start = time.perf_counter()
        distances: list[float] = []
        for document_vector in document_vectors:
            result = ckks_squared_l2(context, query_vector, document_vector)
            distances.append(result.score)
            absolute_errors.append(result.absolute_error)
            relative_errors.append(result.relative_error)
            cipher_bytes.append(result.left_cipher_bytes + result.right_cipher_bytes)
            plain_bytes.append(result.left_plain_bytes + result.right_plain_bytes)
        order = np.argsort(np.array(distances, dtype=np.float64))[:top_k]
        topk_rows.append(order.astype(np.int64))
        query_times.append(time.perf_counter() - search_start)

    return RetrievalResult(
        topk_indices=np.vstack(topk_rows).astype(np.int64),
        query_times=np.array(query_times, dtype=np.float64),
        index_build_time=0.0,
        metadata=_ckks_metadata(
            absolute_errors=absolute_errors,
            relative_errors=relative_errors,
            cipher_bytes=cipher_bytes,
            plain_bytes=plain_bytes,
            time_key="he_scan_time",
            elapsed_times=query_times,
        ),
    )


def run_hnsw_ckks_refine_retrieval(
    document_vectors: np.ndarray,
    query_vectors: np.ndarray,
    top_k: int,
    k_prime: int,
    ef_search: int,
    M: int,
    ef_construction: int,
    space: str,
    random_seed: int,
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: list[int],
    global_scale: float,
) -> RetrievalResult:
    document_vectors = np.asarray(document_vectors, dtype=np.float32)
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    _validate_vectors(document_vectors, query_vectors)

    filter_result = run_hnsw_retrieval(
        document_vectors=document_vectors,
        query_vectors=query_vectors,
        top_k=max(top_k, k_prime),
        ef_search=ef_search,
        M=M,
        ef_construction=ef_construction,
        space=space,
        random_seed=random_seed,
    )
    context = create_ckks_context(
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        global_scale=global_scale,
    )
    refined_rows: list[np.ndarray] = []
    refine_times: list[float] = []
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    cipher_bytes: list[int] = []
    plain_bytes: list[int] = []

    for query_id, candidates in enumerate(filter_result.topk_indices):
        refine_start = time.perf_counter()
        query_vector = query_vectors[query_id]
        distances: list[float] = []
        for candidate_id in candidates:
            result = ckks_squared_l2(context, query_vector, document_vectors[int(candidate_id)])
            distances.append(result.score)
            absolute_errors.append(result.absolute_error)
            relative_errors.append(result.relative_error)
            cipher_bytes.append(result.left_cipher_bytes + result.right_cipher_bytes)
            plain_bytes.append(result.left_plain_bytes + result.right_plain_bytes)
        order = np.argsort(np.array(distances, dtype=np.float64))[:top_k]
        refined_rows.append(candidates[order])
        refine_times.append(time.perf_counter() - refine_start)

    return RetrievalResult(
        topk_indices=np.vstack(refined_rows).astype(np.int64),
        query_times=filter_result.query_times + np.array(refine_times, dtype=np.float64),
        index_build_time=filter_result.index_build_time,
        filter_topk_indices=filter_result.topk_indices,
        metadata=_ckks_metadata(
            absolute_errors=absolute_errors,
            relative_errors=relative_errors,
            cipher_bytes=cipher_bytes,
            plain_bytes=plain_bytes,
            time_key="he_refine_time",
            elapsed_times=refine_times,
        ),
    )


def exact_topk(document_vectors: np.ndarray, query_vectors: np.ndarray, top_k: int) -> np.ndarray:
    document_vectors = _l2_normalize(document_vectors)
    query_vectors = _l2_normalize(query_vectors)
    scores = query_vectors @ document_vectors.T
    return np.argsort(scores, axis=1)[:, ::-1][:, :top_k].astype(np.int64)


def _validate_vectors(document_vectors: np.ndarray, query_vectors: np.ndarray) -> None:
    if document_vectors.ndim != 2 or query_vectors.ndim != 2:
        raise ValueError("document_vectors and query_vectors must be 2D arrays")
    if document_vectors.shape[1] != query_vectors.shape[1]:
        raise ValueError(
            f"Vector dim mismatch: documents={document_vectors.shape[1]}, "
            f"queries={query_vectors.shape[1]}"
        )


def _ckks_metadata(
    absolute_errors: list[float],
    relative_errors: list[float],
    cipher_bytes: list[int],
    plain_bytes: list[int],
    time_key: str,
    elapsed_times: list[float],
) -> Dict[str, float | int | str]:
    mean_cipher_bytes = float(np.mean(cipher_bytes)) if cipher_bytes else float("nan")
    mean_plain_bytes = float(np.mean(plain_bytes)) if plain_bytes else float("nan")
    expansion_ratio = mean_cipher_bytes / mean_plain_bytes if mean_plain_bytes > 0 else float("nan")
    return {
        "he_absolute_error_mean": float(np.mean(absolute_errors)) if absolute_errors else float("nan"),
        "he_relative_error_mean": float(np.mean(relative_errors)) if relative_errors else float("nan"),
        time_key: float(np.mean(elapsed_times)) if elapsed_times else float("nan"),
        "ciphertext_size_kb": mean_cipher_bytes / 1024.0,
        "plain_size_kb": mean_plain_bytes / 1024.0,
        "cipher_expansion_ratio": float(expansion_ratio),
    }


def _l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(array, ord=2, axis=1, keepdims=True)
    return array / np.maximum(norms, eps)
