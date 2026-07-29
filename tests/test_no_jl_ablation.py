from pathlib import Path
import sys

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from gaussian_noise import AnalyticGaussianCalibrator
from ablation_experiments.no_jl_schemes import (
    matched_dynamic_calibration,
    run_fixed_dp_calibration_no_jl,
    run_full_current_no_jl,
    run_no_dimension_aware_scaling_no_jl,
    run_no_dp_baseline_no_jl,
)
from ablation_experiments.plotting import plot_no_jl_main


def test_no_jl_ablation_schemes_keep_raw_dimension():
    vectors = np.random.default_rng(7).normal(size=(6, 1024)).astype(np.float32)
    records = [{"raw_sensitivity_score": 1.0 + index} for index in range(6)]
    calibrator = AnalyticGaussianCalibrator(utility_scale=0.01, random_state=9)
    epsilon, sensitivity = matched_dynamic_calibration(records, calibrator)
    outputs = [
        run_full_current_no_jl(vectors, records, calibrator),
        run_no_dp_baseline_no_jl(vectors),
        run_no_dimension_aware_scaling_no_jl(
            vectors, records, AnalyticGaussianCalibrator(utility_scale=0.01, random_state=9), 0.01,
        ),
        run_fixed_dp_calibration_no_jl(
            vectors, AnalyticGaussianCalibrator(utility_scale=0.01, random_state=9), 0.01, epsilon, sensitivity,
        ),
    ]
    assert [item.name for item in outputs] == [
        "Full Current NoJL",
        "No DP Baseline NoJL",
        "No Dimension-Aware Scaling NoJL",
        "Fixed DP Calibration NoJL",
    ]
    assert all(item.vectors.shape == (6, 1024) for item in outputs)


def test_no_jl_plotter_writes_only_the_four_requested_figures(tmp_path):
    schemes = [
        "Full Current NoJL",
        "No DP Baseline NoJL",
        "No Dimension-Aware Scaling NoJL",
        "Fixed DP Calibration NoJL",
    ]
    rows = [
        {
            "scheme": scheme,
            "utility_scale": scale,
            "mean_mrr5": 0.8,
            "mean_overlap5": 0.8,
            "mean_nsr": 0.01,
            "mean_direction_cosine": 0.99,
        }
        for scheme in schemes
        for scale in (0.01, 0.1)
    ]
    paths = plot_no_jl_main(rows, tmp_path)
    assert {path.name for path in paths} == {
        "no_jl_ablation_mrr_at_5_curve.png",
        "no_jl_ablation_overlap_at_5_curve.png",
        "no_jl_ablation_noise_signal_ratio_curve.png",
        "no_jl_ablation_direction_cosine_curve.png",
    }
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)
