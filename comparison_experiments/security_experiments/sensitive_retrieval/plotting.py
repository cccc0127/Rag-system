"""Plots scoped to targeted sensitive-retrieval exposure results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt


def plot_sensitive_retrieval_results(
    metrics: Sequence[Dict[str, float | int | str]], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _bar(metrics, "sensitive_target_recall_at_1", "Target Sensitive Recall@1", "Rate", output_dir / "sensitive_target_recall_at_1.png"),
        _bar(metrics, "sensitive_target_recall_at_5", "Target Sensitive Recall@5", "Rate", output_dir / "sensitive_target_recall_at_5.png"),
        _bar(metrics, "sensitive_top1_exposure_rate", "Sensitive Top-1 Exposure Rate", "Rate", output_dir / "sensitive_top1_exposure_rate.png"),
        _bar(metrics, "mean_sensitive_chunks_at_5", "Mean Sensitive Chunks in Top-5", "Mean count", output_dir / "mean_sensitive_chunks_at_5.png", None),
        _tradeoff(metrics, output_dir / "security_utility_tradeoff.png"),
    ]


def _bar(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    y_limit: tuple[float, float] | None = (0.0, 1.05),
) -> Path:
    labels = [str(row["scheme"]) for row in metrics]
    values = [float(row[value_key]) for row in metrics]
    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=160)
    bars = ax.bar(labels, values, color="#dc2626", alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel("Scheme")
    ax.set_ylabel(ylabel)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.bar_label(bars, fmt="%.4g", padding=3)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _tradeoff(metrics: Sequence[Dict[str, float | int | str]], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=160)
    for row in metrics:
        x_value = float(row["normal_hnsw_recall_at_5"])
        y_value = float(row["sensitive_target_recall_at_5"])
        ax.scatter(x_value, y_value, s=70, color="#2563eb")
        ax.annotate(str(row["scheme"]), (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Security–Utility Trade-off")
    ax.set_xlabel("Normal-query Recall@5 (higher is better)")
    ax.set_ylabel("Sensitive Target Recall@5 (lower is better)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
