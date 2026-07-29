"""Plotting helpers scoped to the Private RAG-RP k-sensitivity experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt


def plot_private_rag_rp_sensitivity(
    metrics: Sequence[Dict[str, float | int | str]],
    output_dir: Path,
) -> list[Path]:
    """Save k-ordered performance and resource curves for Private RAG-RP."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_metrics = sorted(metrics, key=lambda row: int(row["k"]))
    return [
        _plot_curve(
            ordered_metrics,
            value_key="hnsw_recall_at_5",
            title="Private RAG-RP: Recall@5 vs Projection Dimension",
            ylabel="Recall@5",
            output_path=output_dir / "private_rag_rp_recall_at_5_vs_k.png",
            y_limit=(0.0, 1.05),
        ),
        _plot_curve(
            ordered_metrics,
            value_key="hnsw_mrr_at_5",
            title="Private RAG-RP: MRR@5 vs Projection Dimension",
            ylabel="MRR@5",
            output_path=output_dir / "private_rag_rp_mrr_at_5_vs_k.png",
            y_limit=(0.0, 1.05),
        ),
        _plot_curve(
            ordered_metrics,
            value_key="mean_query_time",
            title="Private RAG-RP: Mean Query Time vs Projection Dimension",
            ylabel="Mean query time (seconds)",
            output_path=output_dir / "private_rag_rp_query_time_vs_k.png",
            prefer_log_y=True,
        ),
        _plot_curve(
            ordered_metrics,
            value_key="index_build_time",
            title="Private RAG-RP: Index Build Time vs Projection Dimension",
            ylabel="Index build time (seconds)",
            output_path=output_dir / "private_rag_rp_index_build_time_vs_k.png",
            prefer_log_y=True,
        ),
        _plot_curve(
            ordered_metrics,
            value_key="vector_dim",
            title="Private RAG-RP: Vector Dimension vs Projection Dimension",
            ylabel="Vector dimension",
            output_path=output_dir / "private_rag_rp_vector_dim_vs_k.png",
        ),
    ]


def _plot_curve(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    y_limit: tuple[float, float] | None = None,
    prefer_log_y: bool = False,
) -> Path:
    x_values = [int(row["k"]) for row in metrics]
    y_values = [float(row[value_key]) for row in metrics]
    use_log_y = prefer_log_y and all(value > 0.0 and value == value for value in y_values)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    ax.plot(x_values, y_values, marker="o", linewidth=2.0, color="#2563eb")
    ax.set_title(title)
    ax.set_xlabel("Projection dimension k")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_values)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    if use_log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
