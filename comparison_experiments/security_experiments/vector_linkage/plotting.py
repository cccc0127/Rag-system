"""Plots scoped to known-candidate vector linkage results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt


def plot_vector_linkage_results(
    metrics: Sequence[Dict[str, float | int | str]], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _bar(metrics, "linkage_top1_recovery_rate", "Vector Linkage Top-1 Recovery Rate", output_dir / "linkage_top1_recovery_rate.png"),
        _bar(metrics, "linkage_recall_at_5", "Vector Linkage Recall@5", output_dir / "linkage_recall_at_5.png"),
        _bar(metrics, "linkage_mrr_at_5", "Vector Linkage MRR@5", output_dir / "linkage_mrr_at_5.png"),
        _bar(metrics, "sensitive_linkage_top1_recovery_rate", "Sensitive Vector Linkage Top-1 Recovery", output_dir / "sensitive_linkage_top1_recovery_rate.png"),
        _tradeoff(metrics, output_dir / "linkage_utility_tradeoff.png"),
    ]


def _bar(
    metrics: Sequence[Dict[str, float | int | str]], value_key: str, title: str, output_path: Path
) -> Path:
    labels = [str(row["scheme"]) for row in metrics]
    values = [float(row[value_key]) for row in metrics]
    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=160)
    bars = ax.bar(labels, values, color="#7c3aed", alpha=0.82)
    ax.set_title(title)
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Recovery rate (lower is better)")
    ax.set_ylim(0.0, 1.05)
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
        y_value = float(row["sensitive_linkage_top1_recovery_rate"])
        ax.scatter(x_value, y_value, s=70, color="#7c3aed")
        ax.annotate(str(row["scheme"]), (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    ax.set_title("Linkage Security–Utility Trade-off")
    ax.set_xlabel("Normal-query Recall@5 (higher is better)")
    ax.set_ylabel("Sensitive Top-1 Linkage Recovery (lower is better)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
