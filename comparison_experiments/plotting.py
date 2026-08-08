"""Comparison plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt


# Keep scheme identity stable across all ef_search figures.  Marker shapes and
# line styles remain distinguishable when values overlap exactly.
EF_SEARCH_STYLES = {
    "Our DP-RAG-NoJL": {"color": "#147D82", "marker": "o", "linestyle": "-", "zorder": 6},
    "Our DP-RAG-JL768": {"color": "#3465C5", "marker": "s", "linestyle": "--", "zorder": 5},
    "Our DP-RAG-JL256": {"color": "#8B5AA3", "marker": "^", "linestyle": "-.", "zorder": 4},
    "Private RAG-RP": {"color": "#C85A43", "marker": "*", "linestyle": ":", "zorder": 7},
    "DCPE+DCE": {"color": "#7A8E33", "marker": "D", "linestyle": "--", "zorder": 3},
}

FALLBACK_COLORS = ["#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD"]
FALLBACK_MARKERS = ["P", "X", "v", "<", ">"]


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
            value_key="hnsw_recall_at_5",
            title="HNSW Recall@5 by Scheme",
            ylabel="Recall@5",
            output_path=default_dir / "comparison_recall_at_5.png",
            y_limit=(0.0, 1.1),
        ),
        _plot_bar(
            default_metrics,
            value_key="hnsw_mrr_at_5",
            title="HNSW MRR@5 by Scheme",
            ylabel="MRR@5",
            output_path=default_dir / "comparison_mrr_at_5.png",
            y_limit=(0.0, 1.1),
        ),
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
    ef_metrics = _non_ckks_rows(metrics)
    saved_paths.extend(
        [
            _plot_ef_search_curve(
                ef_metrics,
                value_key="hnsw_recall_at_5",
                title="HNSW Recall@5 across ef_search",
                ylabel="Recall@5",
                output_path=ef_dir / "comparison_ef_search_recall_at_5.png",
                y_limit=(0.0, 1.1),
            ),
            _plot_ef_search_curve(
                ef_metrics,
                value_key="hnsw_mrr_at_5",
                title="HNSW MRR@5 across ef_search",
                ylabel="MRR@5",
                output_path=ef_dir / "comparison_ef_search_mrr_at_5.png",
                y_limit=(0.0, 1.1),
            ),
            _plot_ef_search_curve(
                ef_metrics,
                value_key="mean_query_time",
                title="Mean Query Time across ef_search",
                ylabel="Mean query time (seconds)",
                output_path=ef_dir / "comparison_ef_search_query_time.png",
            ),
        ]
    )
    return saved_paths


def plot_ckks_figures(
    metrics: Sequence[Dict[str, float | int | str]],
    output_dir: Path,
    default_ef_search: int,
) -> list[Path]:
    ckks_dir = output_dir / "ckks"
    default_metrics = _select_default_metrics(_ckks_rows(metrics), default_ef_search)
    if not default_metrics:
        return []

    rows_with_he_time = _with_ckks_he_time(default_metrics)
    ckks_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    saved_paths.extend(
        _maybe_plot_ckks_bar(
            default_metrics,
            value_key="he_relative_error_mean",
            title="CKKS Mean HE Relative Error",
            ylabel="Mean HE relative error",
            output_path=ckks_dir / "ckks_he_relative_error.png",
            log_y=True,
        )
    )
    saved_paths.extend(
        _maybe_plot_ckks_bar(
            default_metrics,
            value_key="he_absolute_error_mean",
            title="CKKS Mean HE Absolute Error",
            ylabel="Mean HE absolute error",
            output_path=ckks_dir / "ckks_he_absolute_error.png",
            log_y=True,
        )
    )
    saved_paths.extend(
        _maybe_plot_ckks_bar(
            default_metrics,
            value_key="cipher_expansion_ratio",
            title="CKKS Cipher/Plain Expansion Ratio",
            ylabel="Cipher/plain expansion ratio",
            output_path=ckks_dir / "ckks_cipher_expansion_ratio.png",
        )
    )
    saved_paths.extend(
        _maybe_plot_ckks_bar(
            default_metrics,
            value_key="ciphertext_size_kb",
            title="CKKS Mean Ciphertext Size",
            ylabel="Mean ciphertext size (KB)",
            output_path=ckks_dir / "ckks_ciphertext_size.png",
        )
    )
    saved_paths.extend(
        _maybe_plot_ckks_bar(
            rows_with_he_time,
            value_key="ckks_he_time",
            title="CKKS Mean HE Computation Time",
            ylabel="Mean HE computation time (seconds)",
            output_path=ckks_dir / "ckks_he_time.png",
            log_y=True,
        )
    )
    return saved_paths


def plot_database_scale_figures(
    metrics: Sequence[Dict[str, float | int | str]],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = [
        _plot_database_scale_curve(
            metrics,
            value_key="hnsw_recall_at_5",
            title="Database Scale: Recall@5",
            ylabel="Recall@5",
            output_path=output_dir / "database_scale_recall_at_5.png",
            y_limit=(0.0, 1.1),
        ),
        _plot_database_scale_curve(
            metrics,
            value_key="hnsw_mrr_at_5",
            title="Database Scale: MRR@5",
            ylabel="MRR@5",
            output_path=output_dir / "database_scale_mrr_at_5.png",
            y_limit=(0.0, 1.1),
        ),
        _plot_database_scale_curve(
            metrics,
            value_key="mean_query_time",
            title="Database Scale: Mean Query Time",
            ylabel="Mean query time (seconds)",
            output_path=output_dir / "database_scale_query_time.png",
            log_y=True,
        ),
        _plot_database_scale_curve(
            metrics,
            value_key="index_build_time",
            title="Database Scale: Index Build Time",
            ylabel="Index build time (seconds)",
            output_path=output_dir / "database_scale_index_build_time.png",
            log_y=True,
        ),
        _plot_database_scale_vector_dim(
            metrics,
            output_dir / "database_scale_vector_dim.png",
        ),
    ]
    return saved_paths


def plot_ckks_database_scale_figures(
    metrics: Sequence[Dict[str, float | int | str]],
    output_dir: Path,
) -> list[Path]:
    ckks_metrics = _ckks_rows(metrics)
    if not ckks_metrics:
        return []
    ckks_dir = output_dir / "ckks"
    ckks_dir.mkdir(parents=True, exist_ok=True)
    rows_with_he_time = _with_ckks_he_time(ckks_metrics)

    saved_paths: list[Path] = []
    saved_paths.extend(
        _maybe_plot_ckks_scale_curve(
            ckks_metrics,
            value_key="mean_query_time",
            title="CKKS Database Scale: Mean Query Time",
            ylabel="Mean query time (seconds)",
            output_path=ckks_dir / "ckks_scale_query_time.png",
            log_y=True,
        )
    )
    saved_paths.extend(
        _maybe_plot_ckks_scale_curve(
            rows_with_he_time,
            value_key="ckks_he_time",
            title="CKKS Database Scale: HE Computation Time",
            ylabel="Mean HE computation time (seconds)",
            output_path=ckks_dir / "ckks_scale_he_time.png",
            log_y=True,
        )
    )
    saved_paths.extend(
        _maybe_plot_ckks_scale_curve(
            ckks_metrics,
            value_key="cipher_expansion_ratio",
            title="CKKS Database Scale: Cipher/Plain Expansion Ratio",
            ylabel="Cipher/plain expansion ratio",
            output_path=ckks_dir / "ckks_scale_cipher_expansion_ratio.png",
        )
    )
    saved_paths.extend(
        _maybe_plot_ckks_scale_curve(
            ckks_metrics,
            value_key="he_relative_error_mean",
            title="CKKS Database Scale: Mean HE Relative Error",
            ylabel="Mean HE relative error",
            output_path=ckks_dir / "ckks_scale_relative_error.png",
            log_y=True,
        )
    )
    return saved_paths


def _plot_bar(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    y_limit: tuple[float, float] | None = None,
    log_y: bool = False,
) -> Path:
    labels = [str(item["scheme"]) for item in metrics]
    values = [float(item[value_key]) for item in metrics]

    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=160)
    bars = ax.bar(labels, values, color="#2563eb", alpha=0.82, label=ylabel)
    ax.set_title(title)
    ax.set_xlabel("Scheme")
    ax.set_ylabel(ylabel)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    ax.bar_label(bars, fmt="%.6g", padding=3)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def cleanup_old_ef_search_figures(output_dir: Path) -> None:
    ef_dir = output_dir / "ef_search"
    old_filenames = [
        "comparison_ef_search_recall_at_1.png",
        "comparison_ef_search_recall_at_3.png",
        "comparison_ef_search_recall_at_10.png",
        "comparison_ef_search_index_build_time.png",
    ]
    for filename in old_filenames:
        path = ef_dir / filename
        if path.is_file():
            path.unlink()


def _plot_database_scale_curve(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    y_limit: tuple[float, float] | None = None,
    log_y: bool = False,
) -> Path:
    grouped: dict[str, list[Dict[str, float | int | str]]] = {}
    for item in metrics:
        grouped.setdefault(str(item["scheme"]), []).append(item)

    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=160)
    for scheme, rows in grouped.items():
        sorted_rows = sorted(rows, key=lambda row: int(row["sample_chunks"]))
        x_values = [int(row["sample_chunks"]) for row in sorted_rows]
        y_values = [float(row[value_key]) for row in sorted_rows]
        ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=scheme)

    ax.set_title(title)
    ax.set_xlabel("sample_chunks")
    ax.set_ylabel(ylabel)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _maybe_plot_ckks_bar(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    log_y: bool = False,
) -> list[Path]:
    rows = _valid_metric_rows(metrics, value_key, positive=log_y)
    if not rows:
        return []
    return [
        _plot_bar(
            rows,
            value_key=value_key,
            title=title,
            ylabel=ylabel,
            output_path=output_path,
            log_y=log_y,
        )
    ]


def _maybe_plot_ckks_scale_curve(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    title: str,
    ylabel: str,
    output_path: Path,
    log_y: bool = False,
) -> list[Path]:
    rows = _valid_metric_rows(metrics, value_key, positive=log_y)
    if not rows:
        return []
    return [
        _plot_database_scale_curve(
            rows,
            value_key=value_key,
            title=title,
            ylabel=ylabel,
            output_path=output_path,
            log_y=log_y,
        )
    ]


def _ckks_rows(metrics: Sequence[Dict[str, float | int | str]]) -> list[Dict[str, float | int | str]]:
    return [row for row in metrics if "CKKS" in str(row.get("scheme", ""))]


def _non_ckks_rows(metrics: Sequence[Dict[str, float | int | str]]) -> list[Dict[str, float | int | str]]:
    return [row for row in metrics if "CKKS" not in str(row.get("scheme", ""))]


def _with_ckks_he_time(
    metrics: Sequence[Dict[str, float | int | str]],
) -> list[Dict[str, float | int | str]]:
    rows: list[Dict[str, float | int | str]] = []
    for row in metrics:
        he_time = _first_valid_float(row, ("he_scan_time", "he_refine_time"))
        if he_time is None:
            continue
        new_row = dict(row)
        new_row["ckks_he_time"] = he_time
        rows.append(new_row)
    return rows


def _valid_metric_rows(
    metrics: Sequence[Dict[str, float | int | str]],
    value_key: str,
    positive: bool = False,
) -> list[Dict[str, float | int | str]]:
    rows: list[Dict[str, float | int | str]] = []
    for row in metrics:
        value = _float_or_none(row.get(value_key))
        if value is None:
            continue
        if positive and value <= 0.0:
            continue
        rows.append(row)
    return rows


def _first_valid_float(
    row: Dict[str, float | int | str],
    keys: Sequence[str],
) -> float | None:
    for key in keys:
        value = _float_or_none(row.get(key))
        if value is not None:
            return value
    return None


def _float_or_none(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def _plot_database_scale_vector_dim(
    metrics: Sequence[Dict[str, float | int | str]],
    output_path: Path,
) -> Path:
    if not metrics:
        return output_path
    max_chunks = max(int(row["sample_chunks"]) for row in metrics)
    rows = [row for row in metrics if int(row["sample_chunks"]) == max_chunks]
    labels = [str(row["scheme"]) for row in rows]
    values = [float(row["vector_dim"]) for row in rows]

    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=160)
    bars = ax.bar(labels, values, color="#2563eb", alpha=0.82, label="Vector dimension")
    ax.set_title(f"Vector Dimension by Scheme ({max_chunks} chunks)")
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Vector dimension")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    ax.bar_label(bars, fmt="%.6g", padding=3)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_validation_tuning_figures(
    metrics: Sequence[Dict[str, float | int | str]],
    output_dir: Path,
    nsr_min_threshold: float,
    sap_nsr_min_threshold: float,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    our_rows = [row for row in metrics if str(row["scheme"]) == "Our DP-RAG"]
    dcpe_rows = [row for row in metrics if str(row["scheme"]) == "DCPE+DCE"]
    saved_paths: list[Path] = []
    if our_rows:
        saved_paths.append(
            _plot_tuning_curve(
                our_rows,
                x_key="utility_scale",
                y_key="hnsw_recall_at_5",
                title="Our DP-RAG Recall@5 vs utility_scale",
                xlabel="utility_scale",
                ylabel="Recall@5",
                output_path=output_dir / "tuning_our_dprag_recall_vs_utility_scale.png",
                y_limit=(0.0, 1.1),
            )
        )
        saved_paths.append(
            _plot_tuning_curve(
                our_rows,
                x_key="utility_scale",
                y_key="mean_noise_signal_ratio",
                title="Our DP-RAG NSR vs utility_scale",
                xlabel="utility_scale",
                ylabel="Mean Noise/Signal Ratio",
                output_path=output_dir / "tuning_our_dprag_nsr_vs_utility_scale.png",
                threshold=nsr_min_threshold,
                threshold_label="NSR threshold",
            )
        )
    if dcpe_rows:
        saved_paths.append(
            _plot_tuning_curve(
                dcpe_rows,
                x_key="beta",
                y_key="hnsw_recall_at_5",
                title="DCPE+DCE Recall@5 vs beta",
                xlabel="beta",
                ylabel="Recall@5",
                output_path=output_dir / "tuning_dcpe_dce_recall_vs_beta.png",
                y_limit=(0.0, 1.1),
            )
        )
        saved_paths.append(
            _plot_tuning_curve(
                dcpe_rows,
                x_key="beta",
                y_key="sap_noise_signal_ratio",
                title="DCPE+DCE SAP NSR vs beta",
                xlabel="beta",
                ylabel="SAP Noise/Signal Ratio",
                output_path=output_dir / "tuning_dcpe_dce_sap_nsr_vs_beta.png",
                threshold=sap_nsr_min_threshold,
                threshold_label="SAP NSR threshold",
            )
        )
    return saved_paths


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

    fig, ax = plt.subplots(figsize=(10.2, 5.4), dpi=160)
    for scheme_index, (scheme, rows) in enumerate(grouped.items()):
        sorted_rows = sorted(rows, key=lambda row: float(row["ef_search"]))
        x_values = [float(row["ef_search"]) for row in sorted_rows]
        y_values = [float(row[value_key]) for row in sorted_rows]
        style = EF_SEARCH_STYLES.get(
            scheme,
            {
                "color": FALLBACK_COLORS[scheme_index % len(FALLBACK_COLORS)],
                "marker": FALLBACK_MARKERS[scheme_index % len(FALLBACK_MARKERS)],
                "linestyle": "-",
                "zorder": 2 + scheme_index,
            },
        )
        ax.plot(
            x_values,
            y_values,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            markersize=8,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.4,
            zorder=style["zorder"],
            label=scheme,
        )

    ax.set_title(title)
    ax.set_xlabel("ef_search")
    ax.set_ylabel(ylabel)
    ax.set_xscale("log", base=2)
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(frameon=True, fontsize=8)
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


def _plot_tuning_curve(
    rows: Sequence[Dict[str, float | int | str]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    y_limit: tuple[float, float] | None = None,
    threshold: float | None = None,
    threshold_label: str | None = None,
) -> Path:
    sorted_rows = sorted(rows, key=lambda row: float(row[x_key]))
    x_values = [float(row[x_key]) for row in sorted_rows]
    y_values = [float(row[y_key]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(8.8, 5.2), dpi=160)
    ax.plot(x_values, y_values, marker="o", linewidth=2.0, label=ylabel)
    if threshold is not None:
        ax.axhline(
            float(threshold),
            color="#f97316",
            linestyle="--",
            linewidth=1.5,
            label=threshold_label or "threshold",
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log")
    if y_limit is not None:
        ax.set_ylim(*y_limit)
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
