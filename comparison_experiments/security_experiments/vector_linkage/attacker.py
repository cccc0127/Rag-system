"""Exact known-candidate linkage attacker for protected vector indices."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from comparison_experiments.schemes.dcpe_dce import DCPEDCEScheme
from comparison_experiments.schemes.our_dp_rag import OurDPRAGScheme
from comparison_experiments.schemes.private_rag_random_projection import PrivateRAGRandomProjectionScheme
from comparison_experiments.schemes.raw_hnsw import RawHNSWScheme
from comparison_experiments.shared.types import SchemeOutput
from dimension_reduction import l2_normalize
from representation_layer import RepresentationConfig, build_representation


def build_public_candidate_vectors(scheme: object, raw_embeddings: np.ndarray) -> tuple[np.ndarray, str]:
    """Reconstruct each scheme's public deterministic pre-noise representation.

    The attacker knows algorithms and public parameters but not the per-vector
    random noise realization used by Our DP-RAG or DCPE+DCE.
    """
    if isinstance(scheme, RawHNSWScheme):
        return l2_normalize(raw_embeddings), "cosine"
    if isinstance(scheme, OurDPRAGScheme):
        representation = build_representation(
            raw_embeddings=raw_embeddings,
            config=RepresentationConfig(
                mode=scheme.representation_mode,
                jl_target_dim=scheme.jl_target_dim,
                jl_epsilon=scheme.jl_epsilon,
                jl_random_seed=scheme.jl_seed,
            ),
        )
        return representation.document_vectors, "cosine"
    if isinstance(scheme, PrivateRAGRandomProjectionScheme):
        normalized = l2_normalize(raw_embeddings)
        rng = np.random.default_rng(scheme.random_seed)
        matrix = rng.normal(
            loc=0.0,
            scale=scheme.projection_sigma,
            size=(normalized.shape[1], scheme.projection_dim),
        ).astype(np.float32)
        return (normalized @ matrix).astype(np.float32), "l2"
    if isinstance(scheme, DCPEDCEScheme):
        normalized = l2_normalize(raw_embeddings)
        scale = float(np.sqrt(normalized.shape[1]) if scheme.s is None else scheme.s)
        return (scale * normalized).astype(np.float32), "l2"
    raise TypeError(f"Unsupported linkage scheme type: {type(scheme).__name__}")


def exact_linkage_topk(
    protected_vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    distance_metric: str,
    top_k: int = 5,
) -> np.ndarray:
    """Return exact Top-K candidate identities for every protected vector."""
    protected = _as_matrix(protected_vectors, "protected_vectors")
    candidates = _as_matrix(candidate_vectors, "candidate_vectors")
    if protected.shape[1] != candidates.shape[1]:
        raise ValueError(
            "Protected and candidate vector dimensions must match: "
            f"{protected.shape[1]} != {candidates.shape[1]}"
        )
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    k = min(int(top_k), candidates.shape[0])
    normalized_metric = str(distance_metric).lower()
    if normalized_metric == "cosine":
        protected = l2_normalize(protected)
        candidates = l2_normalize(candidates)
        scores = protected @ candidates.T
        return np.argsort(scores, axis=1)[:, ::-1][:, :k].astype(np.int64)
    if normalized_metric == "l2":
        distances = (
            np.sum(protected**2, axis=1, keepdims=True)
            + np.sum(candidates**2, axis=1)[None, :]
            - 2.0 * (protected @ candidates.T)
        )
        return np.argsort(distances, axis=1)[:, :k].astype(np.int64)
    raise ValueError("distance_metric must be 'cosine' or 'l2'")


def _as_matrix(vectors: np.ndarray, name: str) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D array")
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"{name} must not be empty")
    return matrix
