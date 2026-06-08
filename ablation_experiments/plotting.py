"""Plotting utilities for DP-RAG ablation experiments."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


PLOT_SPECS = [
    ("mean_nsr", "ablation_noise_signal_ratio_curve.png", "Mean Noise/Signal Ratio", "Mean NSR", True, None),
    ("mean_overlap1", "ablation_overlap_at_1_curve.png", "Mean Overlap@1", "Overlap@1", False, (0.0, 1.1)),
    ("mean_overlap3", "ablation_overlap_at_3_curve.png", "Mean Overlap@3", "Overlap@3", False, (0.0, 1.1)),
    ("mean_overlap5", "ablation_overlap_at_5_curve.png", "Mean Overlap@5", "Overlap@5", False, (0.0, 1.1)),
    ("mean_overlap10", "ablation_overlap_at_10_curve.png", "Mean Overlap@10", "Overlap@10", False, (0.0, 1.1)),
    ("mean_pearson", "ablation_pearson_correlation_curve.png", "Mean Pearson Correlation", "Pearson", False, (0.0, 1.1)),
    ("mean_drift", "ablation_mean_absolute_drift_curve.png", "Mean Absolute Drift", "Mean Absolute Drift", False, None),
    ("max_drift", "ablation_max_absolute_drift_curve.png", "Max Absolute Drift", "Max Absolute Drift", False, None),
    ("mean_mrr5", "ablation_mrr_at_5_curve.png", "Mean MRR@5", "MRR@5", False, (0.0, 1.1)),
    ("mean_direction_cosine", "ablation_direction_cosine_curve.png", "Mean Direction Cosine", "Direction Cosine", False, (0.0, 1.1)),
    ("dp_noise_time", "ablation_dp_noise_time_curve.png", "DP Noise Time", "Seconds", False, None),
]


def plot_all(results: Sequence[Dict[str, float | str]], result_dir: Path) -> list[Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for metric_key, filename, title, ylabel, y_log, y_limit in PLOT_SPECS:
        path = result_dir / filename
        plot_metric(results, metric_key, path, title, ylabel, y_log=y_log, y_limit=y_limit)
        saved_paths.append(path)

    retrieval_path = result_dir / "ablation_retrieval_time_curve.png"
    plot_retrieval_time(results, retrieval_path)
    saved_paths.append(retrieval_path)
    return saved_paths


def plot_metric(
    results: Sequence[Dict[str, float | str]],
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
    y_log: bool = False,
    y_limit: tuple[float, float] | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=160)
    for scheme in _scheme_names(results):
        scheme_results = _by_scheme(results, scheme)
        x = np.array([float(item["utility_scale"]) for item in scheme_results], dtype=np.float64)
        y = np.array([float(item[metric_key]) for item in scheme_results], dtype=np.float64)
        if y_log:
            y = np.maximum(y, 1e-12)
        ax.plot(x, y, marker="o", linewidth=2.0, label=scheme)

    ax.set_title(title, pad=12)
    ax.set_xlabel("Utility Scale")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_retrieval_time(results: Sequence[Dict[str, float | str]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6), dpi=160)
    for scheme in _scheme_names(results):
        scheme_results = _by_scheme(results, scheme)
        x = np.array([float(item["utility_scale"]) for item in scheme_results], dtype=np.float64)
        raw_y = np.array([float(item["mean_raw_retrieval_time"]) for item in scheme_results], dtype=np.float64)
        noised_y = np.array([float(item["mean_noised_retrieval_time"]) for item in scheme_results], dtype=np.float64)
        ax.plot(x, raw_y, marker="o", linewidth=1.8, linestyle="--", label=f"{scheme} Raw")
        ax.plot(x, noised_y, marker="s", linewidth=1.8, label=f"{scheme} Noised")

    ax.set_title("Mean Retrieval Time per Query", pad=12)
    ax.set_xlabel("Utility Scale")
    ax.set_ylabel("Seconds / Query")
    ax.set_xscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _scheme_names(results: Sequence[Dict[str, float | str]]) -> list[str]:
    return list(dict.fromkeys(str(item["scheme"]) for item in results))


def _by_scheme(results: Sequence[Dict[str, float | str]], scheme: str) -> list[Dict[str, float | str]]:
    return sorted(
        [item for item in results if str(item["scheme"]) == scheme],
        key=lambda item: float(item["utility_scale"]),
    )
