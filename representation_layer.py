"""Shared representation layer for DP-RAG retrieval vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from config import config as global_config
from dimension_reduction import JLProjector, l2_normalize


VALID_REPRESENTATION_MODES = {"jl", "no_jl"}


@dataclass(frozen=True)
class RepresentationConfig:
    mode: str = "jl"
    jl_target_dim: int = global_config.JL_TARGET_DIM
    jl_epsilon: float = global_config.JL_EPSILON
    jl_random_seed: int = global_config.JL_RANDOM_SEED


@dataclass(frozen=True)
class RepresentationState:
    mode: str
    vector_dim: int
    projector: JLProjector | None
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RepresentationOutput:
    document_vectors: np.ndarray
    query_vectors: np.ndarray | None
    state: RepresentationState
    metadata: dict[str, Any]


def build_representation(
    raw_embeddings: np.ndarray,
    query_embeddings: np.ndarray | None = None,
    config: RepresentationConfig | None = None,
) -> RepresentationOutput:
    rep_config = config or RepresentationConfig()
    mode = _normalize_mode(rep_config.mode)
    raw_embeddings = _as_2d_float_array(raw_embeddings)

    if mode == "no_jl":
        document_vectors = l2_normalize(raw_embeddings)
        query_vectors = (
            l2_normalize(query_embeddings)
            if query_embeddings is not None
            else None
        )
        projector = None
        vector_dim = int(document_vectors.shape[1])
    else:
        projector = JLProjector(
            target_dim=rep_config.jl_target_dim,
            eps=rep_config.jl_epsilon,
            random_state=rep_config.jl_random_seed,
        )
        document_vectors = projector.fit_transform(raw_embeddings)
        query_vectors = (
            projector.transform(query_embeddings)
            if query_embeddings is not None
            else None
        )
        vector_dim = int(document_vectors.shape[1])

    metadata: dict[str, Any] = {
        "representation_mode": mode,
        "vector_dim": vector_dim,
        "jl_target_dim": int(rep_config.jl_target_dim) if mode == "jl" else "",
        "jl_epsilon": float(rep_config.jl_epsilon) if mode == "jl" else "",
        "jl_random_seed": int(rep_config.jl_random_seed) if mode == "jl" else "",
        "uses_jl": mode == "jl",
    }
    state = RepresentationState(
        mode=mode,
        vector_dim=vector_dim,
        projector=projector,
        metadata=metadata,
    )
    return RepresentationOutput(
        document_vectors=document_vectors.astype(np.float32),
        query_vectors=query_vectors.astype(np.float32) if query_vectors is not None else None,
        state=state,
        metadata=metadata,
    )


def transform_query_representation(
    query_embeddings: np.ndarray,
    state: RepresentationState,
) -> np.ndarray:
    query_embeddings = _as_2d_float_array(query_embeddings)
    if state.mode == "no_jl":
        return l2_normalize(query_embeddings).astype(np.float32)
    if state.mode == "jl":
        if state.projector is None:
            raise ValueError("JL representation state requires a fitted projector")
        return state.projector.transform(query_embeddings).astype(np.float32)
    raise ValueError(f"Unsupported representation mode: {state.mode}")


def parse_representation_config_from_args(args: Any) -> RepresentationConfig:
    return RepresentationConfig(
        mode=getattr(args, "representation_mode", getattr(global_config, "REPRESENTATION_MODE", "jl")),
        jl_target_dim=int(getattr(args, "jl_target_dim", global_config.JL_TARGET_DIM)),
        jl_epsilon=float(getattr(args, "jl_epsilon", global_config.JL_EPSILON)),
        jl_random_seed=int(getattr(args, "jl_seed", global_config.JL_RANDOM_SEED)),
    )


def _normalize_mode(mode: str) -> str:
    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized not in VALID_REPRESENTATION_MODES:
        raise ValueError(
            f"Unsupported representation mode: {mode}. "
            f"Expected one of: {sorted(VALID_REPRESENTATION_MODES)}"
        )
    return normalized


def _as_2d_float_array(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"vectors must be 1D or 2D, got shape {array.shape}")
    return array
