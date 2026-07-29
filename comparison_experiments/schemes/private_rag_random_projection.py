"""Private RAG with Random Projection baseline adapter.

This reproduces the retrieval representation in Yao and Li (ICLR 2025
Building Trust Workshop): normalize document/query embeddings and project both
with one shared Gaussian matrix before nearest-neighbor retrieval.  It is an
empirical random-projection privacy baseline, not a calibrated DP mechanism.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from comparison_experiments.shared.types import SchemeOutput
from dimension_reduction import l2_normalize


class PrivateRAGRandomProjectionScheme:
    """Shared Gaussian random projection followed by HNSW L2 retrieval."""

    name = "Private RAG-RP"
    backend_type = "hnsw"

    def __init__(
        self,
        projection_dim: int = 64,
        projection_sigma: float = 0.1,
        clip_lower: float = 1.0,
        clip_upper: float = 2.0,
        random_seed: int = 42,
    ):
        if projection_dim <= 0:
            raise ValueError("projection_dim must be greater than 0")
        if projection_sigma <= 0.0:
            raise ValueError("projection_sigma must be greater than 0")
        if clip_lower <= 0.0 or clip_upper < clip_lower:
            raise ValueError("clip bounds must satisfy 0 < clip_lower <= clip_upper")
        self.projection_dim = int(projection_dim)
        self.projection_sigma = float(projection_sigma)
        self.clip_lower = float(clip_lower)
        self.clip_upper = float(clip_upper)
        self.random_seed = int(random_seed)

    def run(
        self,
        raw_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        chunk_records: Sequence[Dict[str, object]],
    ) -> SchemeOutput:
        del chunk_records
        normalized_docs = l2_normalize(raw_embeddings)
        normalized_queries = l2_normalize(query_embeddings)
        input_dim = int(normalized_docs.shape[1])
        if normalized_queries.shape[1] != input_dim:
            raise ValueError(
                "Document/query dimension mismatch: "
                f"{input_dim} != {normalized_queries.shape[1]}"
            )

        # Input vectors have unit L2 norm, so the paper's gamma=1, Delta=2
        # normalization/clipping constraint is satisfied without perturbation.
        rng = np.random.default_rng(self.random_seed)
        projection_matrix = rng.normal(
            loc=0.0,
            scale=self.projection_sigma,
            size=(input_dim, self.projection_dim),
        ).astype(np.float32)
        document_vectors = (normalized_docs @ projection_matrix).astype(np.float32)
        query_vectors = (normalized_queries @ projection_matrix).astype(np.float32)

        return SchemeOutput(
            name=self.name,
            backend_type=self.backend_type,
            document_vectors=document_vectors,
            query_vectors=query_vectors,
            vector_dim=self.projection_dim,
            reference_document_vectors=normalized_docs.astype(np.float32),
            reference_query_vectors=normalized_queries.astype(np.float32),
            metadata={
                "projection_dim": self.projection_dim,
                "projection_sigma": self.projection_sigma,
                "clip_lower": self.clip_lower,
                "clip_upper": self.clip_upper,
                "random_seed": self.random_seed,
                "vector_dim": self.projection_dim,
                "uses_dp": False,
                "uses_jl": True,
                "uses_encryption": False,
                "distance_metric": "l2",
                "hnsw_space": "l2",
                "privacy_claim": "empirical_random_projection_only_no_formal_dp_epsilon",
                "normalization_note": "input vectors are L2-normalized and satisfy gamma=1, Delta=2",
                # Projection scale is not DP Gaussian-mechanism noise.
                "mean_noise_signal_ratio": float("nan"),
                "mean_sigma": float("nan"),
                "mean_epsilon": float("nan"),
            },
        )
