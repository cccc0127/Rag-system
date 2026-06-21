"""Plotting helpers for HNSW retrieval experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_hnsw_curves(results: Sequence[Dict[str, float]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [
        _plot_recall_curve(results, output_dir / "hnsw_recall_curve.png"),
        _plot_latency_curve(results, output_dir / "hnsw_latency_curve.png"),
        _plot_speedup_curve(results, output_dir / "hnsw_speedup_curve.png"),
    ]
    return saved_paths


def _x_values(results: Sequence[Dict[str, float]]) -> np.ndarray:
    return np.array([float(item["ef_search"]) for item in results], dtype=np.float64)


def _plot_recall_curve(results: Sequence[Dict[str, float]], output_path: Path) -> Path:
    x = _x_values(results)
    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    for key, label in [
        ("recall_at_1", "Recall@1"),
        ("recall_at_3", "Recall@3"),
        ("recall_at_5", "Recall@5"),
        ("recall_at_10", "Recall@10"),
    ]:
        y = np.array([float(item[key]) for item in results], dtype=np.float64)
        ax.plot(x, y, marker="o", linewidth=2.0, label=label)
    ax.set_title("HNSW Recall against Exact Retrieval")
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Recall")
    ax.set_ylim(0.0, 1.1)
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_latency_curve(results: Sequence[Dict[str, float]], output_path: Path) -> Path:
    x = _x_values(results)
    exact = np.array([float(item["mean_exact_query_time"]) for item in results], dtype=np.float64)
    hnsw = np.array([float(item["mean_hnsw_query_time"]) for item in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    ax.plot(x, exact, marker="o", linewidth=2.0, label="Exact query time")
    ax.plot(x, hnsw, marker="o", linewidth=2.0, label="HNSW query time")
    ax.set_title("Exact vs HNSW Query Latency")
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Mean query time (seconds)")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_speedup_curve(results: Sequence[Dict[str, float]], output_path: Path) -> Path:
    x = _x_values(results)
    speedup = np.array([float(item["speedup_ratio"]) for item in results], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    ax.plot(x, speedup, marker="o", linewidth=2.0, label="Exact / HNSW speedup")
    ax.axhline(1.0, color="#6b7280", linestyle="--", linewidth=1.2, label="Parity")
    ax.set_title("HNSW Query Speedup")
    ax.set_xlabel("ef_search")
    ax.set_ylabel("Speedup ratio")
    ax.set_xscale("log", base=2)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
