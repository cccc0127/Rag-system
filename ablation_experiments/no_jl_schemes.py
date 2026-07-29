"""No-JL-only internal ablations for the deployed DP-RAG representation."""

from __future__ import annotations

import time
from typing import Dict, Sequence

import numpy as np

from dimension_reduction import l2_normalize
from gaussian_noise import AnalyticGaussianCalibrator, NoiseApplication
from ablation_experiments.schemes import (
    SchemeOutput,
    _apply_fixed_noise,
    _apply_noise_without_dim_scaling,
    _build_output,
)


def run_full_current_no_jl(
    raw_embeddings: np.ndarray,
    chunk_records: Sequence[Dict[str, object]],
    calibrator: AnalyticGaussianCalibrator,
) -> SchemeOutput:
    start = time.perf_counter()
    applications = [
        calibrator.apply_noise_with_diagnostics(vector, raw_score=float(record["raw_sensitivity_score"]))
        for vector, record in zip(raw_embeddings, chunk_records)
    ]
    return _rename(_build_output("Full Current", applications, start), "Full Current NoJL")


def run_no_dp_baseline_no_jl(raw_embeddings: np.ndarray) -> SchemeOutput:
    start = time.perf_counter()
    zeros = np.zeros_like(raw_embeddings, dtype=np.float32)
    nan_values = np.full((raw_embeddings.shape[0],), np.nan, dtype=np.float32)
    return SchemeOutput(
        name="No DP Baseline NoJL",
        vectors=l2_normalize(raw_embeddings),
        signal_vectors=l2_normalize(raw_embeddings),
        noise_vectors=zeros,
        sigmas=nan_values,
        sigma_per_dim=nan_values,
        epsilons=nan_values,
        dp_noise_time=time.perf_counter() - start,
    )


def run_no_dimension_aware_scaling_no_jl(
    raw_embeddings: np.ndarray,
    chunk_records: Sequence[Dict[str, object]],
    calibrator: AnalyticGaussianCalibrator,
    utility_scale: float,
) -> SchemeOutput:
    start = time.perf_counter()
    applications = [
        _apply_noise_without_dim_scaling(
            calibrator,
            vector,
            raw_score=float(record["raw_sensitivity_score"]),
            utility_scale=utility_scale,
        )
        for vector, record in zip(raw_embeddings, chunk_records)
    ]
    return _rename(
        _build_output("No Dimension-Aware Scaling", applications, start),
        "No Dimension-Aware Scaling NoJL",
    )


def run_fixed_dp_calibration_no_jl(
    raw_embeddings: np.ndarray,
    calibrator: AnalyticGaussianCalibrator,
    utility_scale: float,
    matched_epsilon: float,
    matched_local_sensitivity: float,
) -> SchemeOutput:
    """Use constant calibration matched to the dynamic calibration's mean values."""
    start = time.perf_counter()
    multiplier, solved_by_brentq = calibrator.solve_noise_multiplier(matched_epsilon)
    sigma = float(multiplier * matched_local_sensitivity)
    applications = [
        _apply_fixed_noise(
            calibrator,
            vector,
            epsilon=matched_epsilon,
            local_sensitivity=matched_local_sensitivity,
            sigma=sigma,
            solved_by_brentq=solved_by_brentq,
            utility_scale=utility_scale,
        )
        for vector in raw_embeddings
    ]
    return _rename(_build_output("Fixed DP Calibration", applications, start), "Fixed DP Calibration NoJL")


def matched_dynamic_calibration(
    chunk_records: Sequence[Dict[str, object]],
    calibrator: AnalyticGaussianCalibrator,
) -> tuple[float, float]:
    calibrations = [calibrator.calibrate(float(record["raw_sensitivity_score"])) for record in chunk_records]
    return (
        float(np.mean([item.epsilon for item in calibrations])),
        float(np.mean([item.local_sensitivity for item in calibrations])),
    )


def _rename(output: SchemeOutput, name: str) -> SchemeOutput:
    return SchemeOutput(name=name, **{field: getattr(output, field) for field in output.__dataclass_fields__ if field != "name"})
