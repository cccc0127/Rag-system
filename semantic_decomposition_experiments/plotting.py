"""Plotting helpers for semantic decomposition experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np


FIGURE_DIR = Path("Result_picture") / "semantic_decomposition"
JOINT_TRADEOFF_DIR = FIGURE_DIR / "joint_tradeoff"


def plot_semantic_decomposition(
    rows: Iterable[Dict[str, float | int | str]],
    output_dir: Path = FIGURE_DIR,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

    paths = [
        _plot_metric_bar(
            rows,
            metric="recall_at_5",
            ylabel="Recall@5",
            title="Semantic Decomposition: Recall@5",
            filename=output_dir / "semantic_decomposition_recall_at_5.png",
            ylim=(0.0, 1.1),
        ),
        _plot_metric_bar(
            rows,
            metric="mrr_at_5",
            ylabel="MRR@5",
            title="Semantic Decomposition: MRR@5",
            filename=output_dir / "semantic_decomposition_mrr_at_5.png",
            ylim=(0.0, 1.1),
        ),
        _plot_loss_breakdown(rows, output_dir / "semantic_decomposition_loss_breakdown.png"),
        _plot_direction_cosine(rows, output_dir / "semantic_decomposition_direction_cosine.png"),
    ]
    return paths


def _plot_metric_bar(
    rows: List[Dict[str, float | int | str]],
    metric: str,
    ylabel: str,
    title: str,
    filename: Path,
    ylim: tuple[float, float] | None = None,
) -> Path:
    labels = [str(row["comparison_name"]) for row in rows]
    values = [_as_float(row.get(metric, float("nan"))) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(labels, values, color="#7EA6E0", edgecolor="#2F5F98")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Comparison")
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def _plot_loss_breakdown(rows: List[Dict[str, float | int | str]], filename: Path) -> Path:
    by_name = {str(row["comparison_name"]): row for row in rows}
    losses = {
        "JL loss": 1.0 - _as_float(by_name["JL-Exact vs Raw-Exact"]["recall_at_5"]),
        "DP loss": 1.0 - _as_float(by_name["JL-DP-Exact vs JL-Exact"]["recall_at_5"]),
        "HNSW loss": 1.0 - _as_float(by_name["JL-DP-HNSW vs JL-DP-Exact"]["recall_at_5"]),
        "Total deployed loss": 1.0
        - _as_float(by_name["JL-DP-HNSW vs Raw-Exact"]["recall_at_5"]),
    }

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(losses.keys(), losses.values(), color="#F2B36D", edgecolor="#B36317")
    ax.set_title("Semantic Decomposition: Recall@5 Loss Breakdown")
    ax.set_ylabel("Loss = 1 - Recall@5")
    ax.set_xlabel("Loss Component")
    ax.set_ylim(0.0, max(1.0, max(losses.values()) * 1.15))
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def _plot_direction_cosine(rows: List[Dict[str, float | int | str]], filename: Path) -> Path:
    selected = [
        row
        for row in rows
        if str(row["comparison_name"]) == "JL-DP-Exact vs JL-Exact"
    ]
    labels = [str(row["comparison_name"]) for row in selected]
    values = [_as_float(row.get("mean_direction_cosine", float("nan"))) for row in selected]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(labels, values, color="#83C9A5", edgecolor="#248F5B")
    ax.set_title("DP Noise Direction Preservation")
    ax.set_ylabel("Mean Direction Cosine")
    ax.set_xlabel("Comparison")
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def _as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def plot_joint_tradeoff(
    rows: Iterable[Dict[str, float | int | str]],
    output_dir: Path = JOINT_TRADEOFF_DIR,
    max_dp_loss_for_recommendation: float = 0.02,
) -> List[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(rows)

    paths = [
        _plot_heatmap(
            rows,
            metric="final_hnsw_recall_at_5",
            title="Final HNSW Recall@5",
            colorbar_label="Recall@5",
            filename=output_dir / "final_hnsw_recall_heatmap.png",
            vmin=0.0,
            vmax=1.0,
        ),
        _plot_heatmap(
            rows,
            metric="dp_loss_at_5",
            title="DP Loss@5",
            colorbar_label="1 - DP Recall@5",
            filename=output_dir / "dp_loss_heatmap.png",
            vmin=0.0,
        ),
        _plot_heatmap(
            rows,
            metric="mean_noise_signal_ratio",
            title="Noise/Signal Ratio",
            colorbar_label="Mean NSR",
            filename=output_dir / "noise_signal_ratio_heatmap.png",
        ),
        _plot_heatmap(
            rows,
            metric="mean_direction_cosine",
            title="DP Direction Cosine",
            colorbar_label="Mean Direction Cosine",
            filename=output_dir / "direction_cosine_heatmap.png",
            vmin=0.0,
            vmax=1.0,
        ),
        _plot_heatmap(
            rows,
            metric="mean_hnsw_query_time",
            title="HNSW Query Time",
            colorbar_label="Seconds / Query",
            filename=output_dir / "hnsw_query_time_heatmap.png",
        ),
        _plot_heatmap(
            rows,
            metric="estimated_storage_mb",
            title="Estimated Vector Storage",
            colorbar_label="MB",
            filename=output_dir / "storage_cost_heatmap.png",
        ),
        _plot_pareto_scatter(
            rows,
            x_metric="mean_noise_signal_ratio",
            y_metric="final_hnsw_recall_at_5",
            xlabel="Mean Noise/Signal Ratio",
            ylabel="Final HNSW Recall@5",
            title="Pareto: Privacy Perturbation vs Utility",
            filename=output_dir / "pareto_scatter_recall_vs_nsr.png",
        ),
        _plot_pareto_scatter(
            rows,
            x_metric="mean_hnsw_query_time",
            y_metric="final_hnsw_recall_at_5",
            xlabel="Mean HNSW Query Time (s)",
            ylabel="Final HNSW Recall@5",
            title="Pareto: Retrieval Latency vs Utility",
            filename=output_dir / "pareto_scatter_recall_vs_latency.png",
        ),
        _plot_best_by_representation(
            rows,
            max_dp_loss=max_dp_loss_for_recommendation,
            filename=output_dir / "representation_comparison_best_by_dim.png",
        ),
    ]
    return paths


def _plot_heatmap(
    rows: List[Dict[str, float | int | str]],
    metric: str,
    title: str,
    colorbar_label: str,
    filename: Path,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    x_labels, scales, grid = _make_grid(rows, metric)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    image = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("Representation")
    ax.set_ylabel("Utility Scale")
    ax.set_xticks(np.arange(len(x_labels)))
    ax.set_xticklabels(x_labels)
    ax.set_yticks(np.arange(len(scales)))
    ax.set_yticklabels([_format_scale(scale) for scale in scales])
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)

    for y_idx in range(len(scales)):
        for x_idx in range(len(x_labels)):
            value = grid[y_idx, x_idx]
            if np.isfinite(value):
                ax.text(x_idx, y_idx, f"{value:.3g}", ha="center", va="center", fontsize=8, color="white")

    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def _plot_pareto_scatter(
    rows: List[Dict[str, float | int | str]],
    x_metric: str,
    y_metric: str,
    xlabel: str,
    ylabel: str,
    title: str,
    filename: Path,
) -> Path:
    dims = np.array([_as_float(row["vector_dim"]) for row in rows], dtype=np.float64)
    scales = np.array([_as_float(row["utility_scale"]) for row in rows], dtype=np.float64)
    x_values = np.array([_as_float(row[x_metric]) for row in rows], dtype=np.float64)
    y_values = np.array([_as_float(row[y_metric]) for row in rows], dtype=np.float64)
    sizes = 55.0 + 320.0 * (scales / max(float(np.nanmax(scales)), 1e-12))
    is_no_jl = np.array([_is_truthy(row.get("is_no_jl", False)) for row in rows], dtype=bool)

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    jl_mask = ~is_no_jl
    scatter = ax.scatter(
        x_values[jl_mask],
        y_values[jl_mask],
        c=dims[jl_mask],
        s=sizes[jl_mask],
        cmap="plasma",
        alpha=0.85,
        marker="o",
        edgecolor="#1F2937",
        linewidth=0.6,
        label="JL",
    )
    ax.scatter(
        x_values[is_no_jl],
        y_values[is_no_jl],
        c=dims[is_no_jl],
        s=sizes[is_no_jl] * 1.25,
        cmap="plasma",
        alpha=0.95,
        marker="*",
        edgecolor="#111827",
        linewidth=1.1,
        label="No-JL",
        vmin=float(np.nanmin(dims)) if len(dims) else None,
        vmax=float(np.nanmax(dims)) if len(dims) else None,
    )
    for row, x, y, dim, scale in zip(rows, x_values, y_values, dims, scales):
        if np.isfinite(x) and np.isfinite(y):
            if _is_truthy(row.get("is_no_jl", False)):
                label = f"No-JL/{int(dim)}d/{_format_scale(scale)}"
            else:
                label = f"{int(dim)}d/{_format_scale(scale)}"
            ax.annotate(label, (x, y), fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.1)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("Vector Dim")
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def _make_grid(
    rows: List[Dict[str, float | int | str]],
    metric: str,
) -> tuple[List[str], List[float], np.ndarray]:
    x_keys = sorted({_x_key(row) for row in rows}, key=_x_sort_key)
    x_labels = [_x_label_from_key(key) for key in x_keys]
    scales = sorted({float(_as_float(row["utility_scale"])) for row in rows})
    x_to_idx = {key: idx for idx, key in enumerate(x_keys)}
    scale_to_idx = {scale: idx for idx, scale in enumerate(scales)}
    grid = np.full((len(scales), len(x_keys)), np.nan, dtype=np.float64)

    for row in rows:
        x_key = _x_key(row)
        scale = float(_as_float(row["utility_scale"]))
        grid[scale_to_idx[scale], x_to_idx[x_key]] = _as_float(row.get(metric, float("nan")))
    return x_labels, scales, grid


def _format_scale(value: float) -> str:
    return f"{value:g}"


def _plot_best_by_representation(
    rows: List[Dict[str, float | int | str]],
    max_dp_loss: float,
    filename: Path,
) -> Path:
    x_keys = sorted({_x_key(row) for row in rows}, key=_x_sort_key)
    labels = [_x_label_from_key(key) for key in x_keys]
    best_rows = [_select_best_row([row for row in rows if _x_key(row) == key], max_dp_loss) for key in x_keys]
    values = [_as_float(row["final_hnsw_recall_at_5"]) for row, _ in best_rows]
    annotations = [
        f"scale={_format_scale(_as_float(row['utility_scale']))}" + ("\nfallback" if fallback else "")
        for row, fallback in best_rows
    ]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bars = ax.bar(labels, values, color="#8AB6D6", edgecolor="#22577A")
    ax.set_title(f"Best Final Recall@5 by Representation (DP loss <= {max_dp_loss:g})")
    ax.set_xlabel("Representation")
    ax.set_ylabel("Best Final HNSW Recall@5")
    ax.set_ylim(0.0, 1.1)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.tick_params(axis="x", rotation=20)
    for bar, annotation in zip(bars, annotations):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.025,
            annotation,
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(filename, dpi=160)
    plt.close(fig)
    return filename


def _select_best_row(
    rows: List[Dict[str, float | int | str]],
    max_dp_loss: float,
) -> tuple[Dict[str, float | int | str], bool]:
    eligible = [row for row in rows if _as_float(row.get("dp_loss_at_5", float("nan"))) <= max_dp_loss]
    if eligible:
        return max(eligible, key=lambda row: _as_float(row["final_hnsw_recall_at_5"])), False
    return min(rows, key=lambda row: _as_float(row.get("dp_loss_at_5", float("inf")))), True


def _x_key(row: Dict[str, float | int | str]) -> tuple[str, int]:
    dim = int(_as_float(row.get("vector_dim", row.get("jl_target_dim", 0))))
    if _is_truthy(row.get("is_no_jl", False)):
        return ("No-JL", dim)
    return ("JL", int(_as_float(row.get("jl_target_dim", dim))))


def _x_sort_key(key: tuple[str, int]) -> tuple[int, int]:
    representation, dim = key
    return (1 if representation == "No-JL" else 0, dim)


def _x_label_from_key(key: tuple[str, int]) -> str:
    representation, dim = key
    if representation == "No-JL":
        return f"No-JL {dim}"
    return str(dim)


def _is_truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)
