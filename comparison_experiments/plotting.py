"""Comparison plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt


def plot_comparison_figures(
    metrics: Sequence[Dict[str, float | int | str]],
    output_dir: Path,
    default_ef_search: int,
) -> list[Path]:
    default_dir = output_dir / "default_config"
    ef_dir = output_dir / "ef_search"
    default_dir.mkdir(parents=True, exist_ok=True)
    ef_dir.mkdir(parents=True, exist_ok=True)

    default_metrics = _select_default_metrics(metrics, default_ef_search)
    saved_paths = [
        _plot_bar(
            default_metrics,
            value_key="mean_query_time",
            title="Mean Query Time by Scheme",
            ylabel="Mean query time (seconds)",
            output_path=default_dir / "comparison_query_time.png",
        ),
        _plot_bar(
            default_metrics,
            value_key="vector_dim",
            title="Vector Dimension by Scheme",
            ylabel="Vector dimension",
            output_path=default_dir / "comparison_vector_dim.png",
        ),
    ]
    saved_paths.extend(
        [
            _plot_ef_search_curve(
                metrics,
                value_key="hnsw_recall_at_1",
                title="HNSW Recall@1 across ef_search",
                ylabel="Recall@1",
                output_path=ef_dir / "comparison_ef_search_recall_at_1.png",
                y_limit=(0.0, 1.1),
            ),
            _plot_ef_search_curve(
                metrics,
                value_key="hnsw_recall_at_3",
                title="HNSW Recall@3 across ef_search",
                ylabel="Recall@3",
                output_path=ef_dir / "comparison_ef_search_recall_at_3.png",
                y_limit=(0.0, 1.1),
            ),
            _plot_ef_search_curve(
                metrics,
                value_key="hnsw_recall_at_5",
                title="HNSW Recall@5 across ef_search",
                ylabel="Recall@5",
                output_path=ef_dir / "comparison_ef_search_recall_at_5.png",
                y_limit=(0.0, 1.1),
            ),
            _plot_ef_search_curve(
                metrics,
                value_key="hnsw_recall_at_10",
                title="HNSW Recall@10 across ef_search",
                ylabel="Recall@10",
                output_path=ef_dir / "comparison_ef_search_recall_at_10.png",
                y_limit=(0.0, 1.1),
            ),
            _plot_ef_search_curve(
                metrics,
                value_key="hnsw_mrr_at_5",
                title="HNSW MRR@5 across ef_search",
                ylabel="MRR@5",
                output_path=ef_dir / "comparison_ef_search_mrr_at_5.png",
                y_limit=(0.0, 1.1),
            ),
            _plot_ef_search_curve(
                metrics,
                value_key="mean_query_time",
                title="Mean Query Time across ef_search",
                ylabel="Mean query time (seconds)",
                output_path=ef_dir / "comparison_ef_search_query_time.png",
            ),
            _plot_ef_search_curve(
                metrics,
                value_key="index_build_time",
                title="Index Build Time across ef_search",
                ylabel="Index build time (seconds)",
                output_path=ef_dir / "comparison_ef_search_index_build_time.png",
            ),
        ]
    )
    return saved_paths


def _plot_bar(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> Path:
    labels = [str(item["scheme"]) for item in metrics]
    values = [float(item[value_key]) for item in metrics]

    fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=160)
    bars = ax.bar(labels, values, color="#2563eb", alpha=0.82, label=ylabel)
    ax.set_title(title)
    ax.set_xlabel("Scheme")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    ax.bar_label(bars, fmt="%.6g", padding=3)
    fig.autofmt_xdate(rotation=15, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _plot_ef_search_curve(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    y_limit: tuple[float, float] | None = None,
) -> Path:
    grouped: dict[str, list[Dict[str, float | int | str]]] = {}
    for item in metrics:
        grouped.setdefault(str(item["scheme"]), []).append(item)

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    for scheme, rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda row: float(row["ef_search"]))
        x_values = [float(row["ef_search"]) for row in sorted_rows]
        y_values = [float(row[value_key]) for row in sorted_rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=scheme)

    ax.set_title(title)
    ax.set_xlabel("ef_search")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _select_default_metrics(
    metrics: Sequence[Dict[str, float | int | str]],
    default_ef_search: int,
) -> list[Dict[str, float | int | str]]:
    by_scheme: dict[str, list[Dict[str, float | int | str]]] = {}
    for item in metrics:
        by_scheme.setdefault(str(item["scheme"]), []).append(item)

    selected: list[Dict[str, float | int | str]] = []
    for rows in by_scheme.values():
        exact_matches = [row for row in rows if int(row["ef_search"]) == int(default_ef_search)]
        if exact_matches:
            selected.append(exact_matches[0])
        else:
            selected.append(min(rows, key=lambda row: abs(int(row["ef_search"]) - int(default_ef_search))))
    return selected
