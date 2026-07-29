"""Unprotected raw-HNSW baseline used by retrieval-security experiments."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from comparison_experiments.shared.types import SchemeOutput
from dimension_reduction import l2_normalize


class RawHNSWScheme:
    """Raw normalized embeddings with no privacy transformation."""

    name = "Vanilla Raw HNSW"
    backend_type = "hnsw"

    def run(
        self,
        raw_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        chunk_records: Sequence[Dict[str, object]],
    ) -> SchemeOutput:
        del chunk_records
        document_vectors = l2_normalize(raw_embeddings)
        query_vectors = l2_normalize(query_embeddings)
        return SchemeOutput(
            name=self.name,
            backend_type=self.backend_type,
            document_vectors=document_vectors,
            query_vectors=query_vectors,
            vector_dim=int(document_vectors.shape[1]),
            reference_document_vectors=document_vectors,
            reference_query_vectors=query_vectors,
            metadata={
                "uses_dp": False,
                "uses_jl": False,
                "uses_encryption": False,
                "distance_metric": "cosine",
                "hnsw_space": "cosine",
                "security_boundary": "no_protection_raw_retrieval_baseline",
                "mean_noise_signal_ratio": float("nan"),
                "mean_sigma": float("nan"),
                "mean_epsilon": float("nan"),
            },
        )
