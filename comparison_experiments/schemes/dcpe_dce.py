"""DCPE+DCE baseline adapter.

This is a retrieval-behavior reproduction of the ICDE 2025 PP-ANNS scheme:
SAP/DCPE dense vectors are used for HNSW filtering, while DCE refine is
emulated by exact Euclidean distance over normalized raw embeddings.
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from dimension_reduction import l2_normalize
from comparison_experiments.shared.types import SchemeOutput


class DCPEDCEScheme:
    name = "DCPE+DCE"
    backend_type = "hnsw_filter_refine"

    def __init__(
        self,
        beta: float = 0.5,  #工程默认值，没有经过严谨调参
        ratio_k: int = 4,   #可修改的点，影响方案性能的参数
        random_seed: int = 42,
        s: float | None = None,
    ):
        if beta <= 0.0:
            raise ValueError("beta must be greater than 0")
        if ratio_k <= 0:
            raise ValueError("ratio_k must be greater than 0")
        self.beta = float(beta)
        self.ratio_k = int(ratio_k)
        self.random_seed = int(random_seed)
        self.s = None if s is None else float(s)

    def run(
        self,
        raw_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        chunk_records: Sequence[Dict[str, object]],
    ) -> SchemeOutput:
        del chunk_records
        normalized_docs = l2_normalize(raw_embeddings)
        normalized_queries = l2_normalize(query_embeddings)
        dim = int(normalized_docs.shape[1])
        scale = float(np.sqrt(dim) if self.s is None else self.s)

        rng = np.random.default_rng(self.random_seed)
        doc_noise = _sample_l2_ball(
            rng,
            shape=normalized_docs.shape,
            radius=scale * self.beta / 4.0,
        )
        query_noise = _sample_l2_ball(
            rng,
            shape=normalized_queries.shape,
            radius=scale * self.beta / 4.0,
        )

        document_vectors = (scale * normalized_docs + doc_noise).astype(np.float32)
        query_vectors = (scale * normalized_queries + query_noise).astype(np.float32)

        signal_norms = np.linalg.norm(scale * normalized_docs, ord=2, axis=1)
        noise_norms = np.linalg.norm(doc_noise, ord=2, axis=1)
        sap_nsr = noise_norms / np.maximum(signal_norms, 1e-12)

        return SchemeOutput(
            name=self.name,
            backend_type=self.backend_type,
            document_vectors=document_vectors,
            query_vectors=query_vectors,
            vector_dim=dim,
            reference_document_vectors=normalized_docs.astype(np.float32),
            reference_query_vectors=normalized_queries.astype(np.float32),
            metadata={
                "beta": self.beta,
                "ratio_k": self.ratio_k,
                "s": scale,
                "vector_dim": dim,
                "uses_dp": False,
                "uses_jl": False,
                "uses_encryption": True,
                "refine": "exact_distance_equivalent_to_DCE",
                "distance_metric": "l2",
                "hnsw_space": "l2",
                "sap_noise_signal_ratio": float(np.mean(sap_nsr)),
                "mean_noise_signal_ratio": float(np.mean(sap_nsr)),
                "mean_sigma": float("nan"),
                "mean_epsilon": float("nan"),
            },
        )


def _sample_l2_ball(
    rng: np.random.Generator,
    shape: tuple[int, int],
    radius: float,
) -> np.ndarray:
    directions = rng.normal(size=shape).astype(np.float32)
    norms = np.linalg.norm(directions, ord=2, axis=1, keepdims=True)
    directions = directions / np.maximum(norms, 1e-12)
    radial = rng.random((shape[0], 1), dtype=np.float32) ** (1.0 / shape[1])
    return (directions * radial * float(radius)).astype(np.float32)
