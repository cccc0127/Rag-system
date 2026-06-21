"""Our DP-RAG scheme adapter for external comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from dimension_reduction import JLProjector, l2_normalize
from gaussian_noise import AnalyticGaussianCalibrator, NoiseApplication


@dataclass
class SchemeOutput:
    name: str
    backend_type: str
    document_vectors: np.ndarray
    query_vectors: np.ndarray
    vector_dim: int
    metadata: Dict[str, float | int | str]


class OurDPRAGScheme:
    """JL -> clipping -> dynamic analytic Gaussian DP -> final normalization."""

    name = "Our DP-RAG"
    backend_type = "hnsw"

    def __init__(
        self,
        jl_target_dim: int = 256,
        jl_epsilon: float = 0.3,
        jl_seed: int = 42,
        dp_delta: float = 1e-5,
        utility_scale: float = 0.01,
        noise_seed: int = 42,
    ):
        self.jl_target_dim = int(jl_target_dim)
        self.jl_epsilon = float(jl_epsilon)
        self.jl_seed = int(jl_seed)
        self.dp_delta = float(dp_delta)
        self.utility_scale = float(utility_scale)
        self.noise_seed = int(noise_seed)

    def run(
        self,
        raw_embeddings: np.ndarray,
        query_embeddings: np.ndarray,
        chunk_records: Sequence[Dict[str, object]],
    ) -> SchemeOutput:
        projector = JLProjector(
            target_dim=self.jl_target_dim,
            eps=self.jl_epsilon,
            random_state=self.jl_seed,
        )
        reduced_embeddings = projector.fit_transform(raw_embeddings)
        reduced_queries = projector.transform(query_embeddings)

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
            "jl_target_dim": self.jl_target_dim,
            "mean_sigma": float(np.mean(sigmas)),
            "mean_epsilon": float(np.mean(epsilons)),
            "mean_noise_signal_ratio": float(np.mean(nsr)),
            "vector_dim": int(document_vectors.shape[1]),
        }

        return SchemeOutput(
            name=self.name,
            backend_type=self.backend_type,
            document_vectors=document_vectors.astype(np.float32),
            query_vectors=query_vectors.astype(np.float32),
            vector_dim=int(document_vectors.shape[1]),
            metadata=metadata,
        )
