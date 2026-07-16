"""Our DP-RAG scheme adapter for external comparison experiments."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from config import config
from dimension_reduction import l2_normalize
from gaussian_noise import AnalyticGaussianCalibrator, NoiseApplication
from comparison_experiments.shared.types import SchemeOutput
from representation_layer import RepresentationConfig, build_representation


class OurDPRAGScheme:
    """JL -> clipping -> dynamic analytic Gaussian DP -> final normalization."""

    name = "Our DP-RAG"
    backend_type = "hnsw"

    def __init__(
        self,
        representation_mode: str = config.REPRESENTATION_MODE,
        jl_target_dim: int = config.JL_TARGET_DIM,
        jl_epsilon: float = config.JL_EPSILON,
        jl_seed: int = config.JL_RANDOM_SEED,
        dp_delta: float = 1e-5,
        utility_scale: float = 0.01,
        noise_seed: int = 42,
        name: str | None = None,
    ):
        self.representation_mode = str(representation_mode)
        self.jl_target_dim = int(jl_target_dim)
        self.jl_epsilon = float(jl_epsilon)
        self.jl_seed = int(jl_seed)
        self.dp_delta = float(dp_delta)
        self.utility_scale = float(utility_scale)
        self.noise_seed = int(noise_seed)
        self.name = name or self._default_name()

    def _default_name(self) -> str:
        mode = self.representation_mode.strip().lower().replace("-", "_")
        if mode == "no_jl":
            return "Our DP-RAG-NoJL"
        if mode == "jl":
            return f"Our DP-RAG-JL{self.jl_target_dim}"
        return "Our DP-RAG"

    def run(
        self,
        raw_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        chunk_records: Sequence[Dict[str, object]],
    ) -> SchemeOutput:
        representation = build_representation(
            raw_embeddings=raw_embeddings,
            query_embeddings=query_embeddings,
            config=RepresentationConfig(
                mode=self.representation_mode,
                jl_target_dim=self.jl_target_dim,
                jl_epsilon=self.jl_epsilon,
                jl_random_seed=self.jl_seed,
            ),
        )
        reduced_embeddings = representation.document_vectors
        reduced_queries = representation.query_vectors
        if reduced_queries is None:
            raise RuntimeError("Representation layer did not return query vectors.")

        calibrator = AnalyticGaussianCalibrator(
            delta=self.dp_delta,
            utility_scale=self.utility_scale,
            random_state=self.noise_seed,
        )
        applications: list[NoiseApplication] = [
            calibrator.apply_noise_with_diagnostics(
                vector,
                raw_score=float(record["raw_sensitivity_score"]),
            )
            for vector, record in zip(reduced_embeddings, chunk_records)
        ]

        noised_vectors = np.vstack([item.noised_vector for item in applications]).astype(np.float32)
        clipped_vectors = np.vstack([item.clipped_vector for item in applications]).astype(np.float32)
        noise_vectors = np.vstack([item.noise_vector for item in applications]).astype(np.float32)
        document_vectors = l2_normalize(noised_vectors)
        query_vectors = l2_normalize(reduced_queries)

        signal_norms = np.linalg.norm(clipped_vectors, ord=2, axis=1)
        noise_norms = np.linalg.norm(noise_vectors, ord=2, axis=1)
        nsr = noise_norms / np.maximum(signal_norms, 1e-12)
        sigmas = np.array([item.calibration.sigma for item in applications], dtype=np.float32)
        epsilons = np.array([item.calibration.epsilon for item in applications], dtype=np.float32)

        metadata: Dict[str, float | int | str] = {
            "utility_scale": self.utility_scale,
            "dp_delta": self.dp_delta,
            "representation_mode": representation.metadata["representation_mode"],
            "representation_label": self.name,
            "jl_target_dim": self.jl_target_dim,
            "jl_effective_target_dim": representation.metadata.get("jl_target_dim", ""),
            "mean_sigma": float(np.mean(sigmas)),
            "mean_epsilon": float(np.mean(epsilons)),
            "mean_noise_signal_ratio": float(np.mean(nsr)),
            "vector_dim": int(document_vectors.shape[1]),
            "uses_dp": True,
            "uses_jl": bool(representation.metadata["uses_jl"]),
            "uses_encryption": False,
            "distance_metric": "cosine",
            "hnsw_space": "cosine",
        }

        return SchemeOutput(
            name=self.name,
            backend_type=self.backend_type,
            document_vectors=document_vectors.astype(np.float32),
            query_vectors=query_vectors.astype(np.float32),
            vector_dim=int(document_vectors.shape[1]),
            metadata=metadata,
            reference_document_vectors=l2_normalize(raw_embeddings),
            reference_query_vectors=l2_normalize(query_embeddings),
        )
