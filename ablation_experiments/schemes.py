"""Ablation schemes for DP-RAG privacy/utility experiments."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Dict, Sequence

import numpy as np

from dimension_reduction import JLProjector, l2_normalize
from gaussian_noise import AnalyticGaussianCalibrator, NoiseApplication


@dataclass(frozen=True)
class SchemeOutput:
    name: str
    vectors: np.ndarray
    signal_vectors: np.ndarray
    noise_vectors: np.ndarray
    sigmas: np.ndarray
    sigma_per_dim: np.ndarray
    epsilons: np.ndarray
    dp_noise_time: float


def run_full_current(
    raw_embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    chunk_records: Sequence[Dict[str, object]],
    projector: JLProjector,
    calibrator: AnalyticGaussianCalibrator,
    utility_scale: float,
) -> SchemeOutput:
    del raw_embeddings, projector, utility_scale
    start = time.perf_counter()
    applications = [
        calibrator.apply_noise_with_diagnostics(
            vector,
            raw_score=float(record["raw_sensitivity_score"]),
        )
        for vector, record in zip(reduced_embeddings, chunk_records)
    ]
    return _build_output("Full Current", applications, start)


def run_no_dp(
    raw_embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    chunk_records: Sequence[Dict[str, object]],
    projector: JLProjector,
    calibrator: AnalyticGaussianCalibrator,
    utility_scale: float,
) -> SchemeOutput:
    del raw_embeddings, chunk_records, projector, calibrator, utility_scale
    start = time.perf_counter()
    zeros = np.zeros_like(reduced_embeddings, dtype=np.float32)
    nan_values = np.full((reduced_embeddings.shape[0],), np.nan, dtype=np.float32)
    return SchemeOutput(
        name="No DP Baseline",
        vectors=l2_normalize(reduced_embeddings),
        signal_vectors=l2_normalize(reduced_embeddings),
        noise_vectors=zeros,
        sigmas=nan_values,
        sigma_per_dim=nan_values,
        epsilons=nan_values,
        dp_noise_time=time.perf_counter() - start,
    )


def run_old_pipeline_dp_before_jl(
    raw_embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    chunk_records: Sequence[Dict[str, object]],
    projector: JLProjector,
    calibrator: AnalyticGaussianCalibrator,
    utility_scale: float,
) -> SchemeOutput:
    del reduced_embeddings, utility_scale
    start = time.perf_counter()
    applications = [
        calibrator.apply_noise_with_diagnostics(
            vector,
            raw_score=float(record["raw_sensitivity_score"]),
        )
        for vector, record in zip(raw_embeddings, chunk_records)
    ]
    noised_raw = np.vstack([item.noised_vector for item in applications]).astype(np.float32)
    vectors = projector.transform(noised_raw)
    return _build_output("Old Pipeline DP Before JL", applications, start, vectors=vectors)


def run_no_dimension_aware_scaling(
    raw_embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    chunk_records: Sequence[Dict[str, object]],
    projector: JLProjector,
    calibrator: AnalyticGaussianCalibrator,
    utility_scale: float,
) -> SchemeOutput:
    del raw_embeddings, projector
    start = time.perf_counter()
    applications = [
        _apply_noise_without_dim_scaling(
            calibrator,
            vector,
            raw_score=float(record["raw_sensitivity_score"]),
            utility_scale=utility_scale,
        )
        for vector, record in zip(reduced_embeddings, chunk_records)
    ]
    return _build_output("No Dimension-Aware Scaling", applications, start)


def run_fixed_dp_calibration(
    raw_embeddings: np.ndarray,
    reduced_embeddings: np.ndarray,
    chunk_records: Sequence[Dict[str, object]],
    projector: JLProjector,
    calibrator: AnalyticGaussianCalibrator,
    utility_scale: float,
) -> SchemeOutput:
    del raw_embeddings, chunk_records, projector
    start = time.perf_counter()
    fixed_epsilon = 4.0
    fixed_local_sensitivity = 0.35
    u_star, solved_by_brentq = calibrator.solve_noise_multiplier(fixed_epsilon)
    sigma = float(u_star * fixed_local_sensitivity)
    applications = [
        _apply_fixed_noise(
            calibrator,
            vector,
            epsilon=fixed_epsilon,
            local_sensitivity=fixed_local_sensitivity,
            sigma=sigma,
            solved_by_brentq=solved_by_brentq,
            utility_scale=utility_scale,
        )
        for vector in reduced_embeddings
    ]
    return _build_output("Fixed DP Calibration", applications, start)


SCHEMES: list[tuple[str, Callable[..., SchemeOutput]]] = [
    ("Full Current", run_full_current),
    ("No DP Baseline", run_no_dp),
    ("Old Pipeline DP Before JL", run_old_pipeline_dp_before_jl),
    ("No Dimension-Aware Scaling", run_no_dimension_aware_scaling),
    ("Fixed DP Calibration", run_fixed_dp_calibration),
]


def _build_output(
    name: str,
    applications: Sequence[NoiseApplication],
    start_time: float,
    vectors: np.ndarray | None = None,
) -> SchemeOutput:
    noised = np.vstack([item.noised_vector for item in applications]).astype(np.float32)
    final_vectors = l2_normalize(noised) if vectors is None else l2_normalize(vectors)
    return SchemeOutput(
        name=name,
        vectors=final_vectors,
        signal_vectors=np.vstack([item.clipped_vector for item in applications]).astype(np.float32),
        noise_vectors=np.vstack([item.noise_vector for item in applications]).astype(np.float32),
        sigmas=np.array([item.calibration.sigma for item in applications], dtype=np.float32),
        sigma_per_dim=np.array([item.sigma_per_dim for item in applications], dtype=np.float32),
        epsilons=np.array([item.calibration.epsilon for item in applications], dtype=np.float32),
        dp_noise_time=time.perf_counter() - start_time,
    )


def _clip_vector(vector: np.ndarray, l2_clip_norm: float) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32)
    norm_value = float(np.linalg.norm(vector, ord=2))
    return (vector / max(1.0, norm_value / l2_clip_norm)).astype(np.float32)


def _noise_application(
    noised_vector: np.ndarray,
    clipped_vector: np.ndarray,
    noise_vector: np.ndarray,
    raw_score: float,
    normalized_score: float,
    epsilon: float,
    local_sensitivity: float,
    sigma: float,
    solved_by_brentq: bool,
    sigma_per_dim: float,
) -> NoiseApplication:
    from gaussian_noise import NoiseApplication, NoiseCalibration

    calibration = NoiseCalibration(
        raw_score=float(raw_score),
        normalized_score=float(normalized_score),
        epsilon=float(epsilon),
        local_sensitivity=float(local_sensitivity),
        sigma=float(sigma),
        solved_by_brentq=bool(solved_by_brentq),
    )
    return NoiseApplication(
        noised_vector=noised_vector.astype(np.float32),
        clipped_vector=clipped_vector.astype(np.float32),
        noise_vector=noise_vector.astype(np.float32),
        calibration=calibration,
        sigma_per_dim=float(sigma_per_dim),
    )


def _apply_noise_without_dim_scaling(
    calibrator: AnalyticGaussianCalibrator,
    vector: np.ndarray,
    raw_score: float,
    utility_scale: float,
) -> NoiseApplication:
    calibration = calibrator.calibrate(raw_score)
    clipped = _clip_vector(vector, calibrator.l2_clip_norm)
    sigma_per_dim = float(calibration.sigma * utility_scale)
    noise = calibrator.rng.normal(0.0, sigma_per_dim, size=clipped.shape).astype(np.float32)
    noised = clipped + noise
    return _noise_application(
        noised,
        clipped,
        noise,
        raw_score=calibration.raw_score,
        normalized_score=calibration.normalized_score,
        epsilon=calibration.epsilon,
        local_sensitivity=calibration.local_sensitivity,
        sigma=calibration.sigma,
        solved_by_brentq=calibration.solved_by_brentq,
        sigma_per_dim=sigma_per_dim,
    )


def _apply_fixed_noise(
    calibrator: AnalyticGaussianCalibrator,
    vector: np.ndarray,
    epsilon: float,
    local_sensitivity: float,
    sigma: float,
    solved_by_brentq: bool,
    utility_scale: float,
) -> NoiseApplication:
    clipped = _clip_vector(vector, calibrator.l2_clip_norm)
    dim = int(clipped.size)
    sigma_per_dim = float((sigma * utility_scale) / np.sqrt(dim))
    noise = calibrator.rng.normal(0.0, sigma_per_dim, size=clipped.shape).astype(np.float32)
    noised = clipped + noise
    return _noise_application(
        noised,
        clipped,
        noise,
        raw_score=np.nan,
        normalized_score=np.nan,
        epsilon=epsilon,
        local_sensitivity=local_sensitivity,
        sigma=sigma,
        solved_by_brentq=solved_by_brentq,
        sigma_per_dim=sigma_per_dim,
    )
