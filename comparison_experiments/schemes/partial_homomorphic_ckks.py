"""Partial homomorphic CKKS baselines for comparison experiments."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from dimension_reduction import l2_normalize
from comparison_experiments.shared.ckks_utils import (
    DEFAULT_CKKS_COEFF_MOD_BIT_SIZES,
    DEFAULT_CKKS_GLOBAL_SCALE,
    DEFAULT_CKKS_POLY_MODULUS_DEGREE,
)
from comparison_experiments.shared.types import SchemeOutput


class PartialHomomorphicCKKSScheme:
    """Adapter that exposes CKKS distance evaluation through SchemeOutput."""

    def __init__(
        self,
        mode: str,
        poly_modulus_degree: int = DEFAULT_CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes: Sequence[int] = DEFAULT_CKKS_COEFF_MOD_BIT_SIZES,
        global_scale: float = DEFAULT_CKKS_GLOBAL_SCALE,
        ratio_k: int = 4,
    ):
        normalized_mode = mode.strip().lower().replace("-", "_")
        if normalized_mode not in {"fullscan", "refine"}:
            raise ValueError("mode must be 'fullscan' or 'refine'")
        if ratio_k <= 0:
            raise ValueError("ratio_k must be greater than 0")
        self.mode = normalized_mode
        self.poly_modulus_degree = int(poly_modulus_degree)
        self.coeff_mod_bit_sizes = [int(value) for value in coeff_mod_bit_sizes]
        self.global_scale = float(global_scale)
        self.ratio_k = int(ratio_k)
        self.name = (
            "PartialHE-CKKS-FullScan"
            if self.mode == "fullscan"
            else "HNSW+PartialHE-CKKS-Refine"
        )
        self.backend_type = (
            "ckks_full_scan"
            if self.mode == "fullscan"
            else "hnsw_ckks_refine"
        )

    def run(
        self,
        raw_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        chunk_records: Sequence[Dict[str, object]],
    ) -> SchemeOutput:
        del chunk_records
        document_vectors = l2_normalize(raw_embeddings).astype(np.float32)
        query_vectors = l2_normalize(query_embeddings).astype(np.float32)
        return SchemeOutput(
            name=self.name,
            backend_type=self.backend_type,
            document_vectors=document_vectors,
            query_vectors=query_vectors,
            vector_dim=int(document_vectors.shape[1]),
            reference_document_vectors=document_vectors,
            reference_query_vectors=query_vectors,
            metadata={
                "poly_modulus_degree": self.poly_modulus_degree,
                "coeff_mod_bit_sizes": ",".join(str(value) for value in self.coeff_mod_bit_sizes),
                "global_scale": self.global_scale,
                "ratio_k": self.ratio_k,
                "vector_dim": int(document_vectors.shape[1]),
                "uses_dp": False,
                "uses_jl": False,
                "uses_encryption": True,
                "encryption_scheme": "CKKS",
                "mean_noise_signal_ratio": float("nan"),
                "mean_sigma": float("nan"),
                "mean_epsilon": float("nan"),
                "distance_metric": "squared_l2",
                "hnsw_space": "l2",
                "security_boundary": (
                    "full_scan_ckks_distance"
                    if self.mode == "fullscan"
                    else "plaintext_hnsw_candidates_ckks_refine"
                ),
            },
        )
