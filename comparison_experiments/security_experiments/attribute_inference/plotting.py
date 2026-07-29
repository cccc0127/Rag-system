"""Plots for protected-vector sensitive attribute-inference results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt


PRIMARY_ATTRIBUTE = "has_any_sensitive"


def plot_attribute_inference_results(
    metrics: Sequence[Dict[str, float | int | str | bool]], output_dir: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary = [row for row in metrics if row["target_attribute"] == PRIMARY_ATTRIBUTE]
    figures: list[Path] = []
    if primary:
        figures.extend(
            [
                _bar(primary, "roc_auc", "Sensitive Attribute Inference ROC-AUC", output_dir / "any_sensitive_roc_auc.png", 0.5),
                _bar(primary, "tpr_at_fpr_1pct", "Sensitive Attribute Inference TPR at FPR=1%", output_dir / "any_sensitive_tpr_at_fpr_1pct.png", None),
                _bar(primary, "macro_f1", "Sensitive Attribute Inference Macro F1", output_dir / "any_sensitive_macro_f1.png", None),
                _tradeoff(primary, output_dir / "attribute_inference_utility_tradeoff.png"),
            ]
        )
    type_rows = [row for row in metrics if row["target_attribute"] != PRIMARY_ATTRIBUTE]
    if type_rows:
        figures.append(_grouped_auc(type_rows, output_dir / "attribute_type_roc_auc.png"))
    return figures


def _bar(
    rows: Sequence[Dict[str, float | int | str | bool]], key: str, title: str, output_path: Path, reference: float | None) -> Path:
    labels = [str(row["scheme"]) for row in rows]
    values = [float(row[key]) for row in rows]
    fig, ax = plt.subplots(figsize=(10.4, 5.4), dpi=160)
    bars = ax.bar(labels, values, color="#2563eb", alpha=0.82)
    if reference is not None:
        ax.axhline(reference, color="#dc2626", linestyle="--", linewidth=1.4, label="Random classifier (0.5)")
        ax.legend(loc="best")
    ax.set_title(title)
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Attack performance (lower is better)")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.bar_label(bars, fmt="%.4g", padding=3)
    fig.autofmt_xdate(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _grouped_auc(rows: Sequence[Dict[str, float | int | str | bool]], output_path: Path) -> Path:
    attributes = list(dict.fromkeys(str(row["target_attribute"]) for row in rows))
    schemes = list(dict.fromkeys(str(row["scheme"]) for row in rows))
    width = 0.8 / max(1, len(schemes))
    fig, ax = plt.subplots(figsize=(11.2, 5.6), dpi=160)
    for index, scheme in enumerate(schemes):
        lookup = {(str(row["scheme"]), str(row["target_attribute"])): float(row["roc_auc"]) for row in rows}
        values = [lookup.get((scheme, attribute), float("nan")) for attribute in attributes]
        positions = [item + (index - (len(schemes) - 1) / 2) * width for item in range(len(attributes))]
        ax.bar(positions, values, width=width, label=scheme)
    ax.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.3, label="Random classifier (0.5)")
    ax.set_title("Attribute-Type Inference ROC-AUC")
    ax.set_ylabel("ROC-AUC (lower is better)")
    ax.set_xticks(range(len(attributes)), [item.removeprefix("has_") for item in attributes])
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _tradeoff(rows: Sequence[Dict[str, float | int | str | bool]], output_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=160)
    for row in rows:
        x_value = float(row["normal_hnsw_recall_at_5"])
        y_value = float(row["roc_auc"])
        ax.scatter(x_value, y_value, s=70, color="#2563eb")
        ax.annotate(str(row["scheme"]), (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.3, label="Random classifier (0.5)")
    ax.set_title("Attribute-Inference Security–Utility Trade-off")
    ax.set_xlabel("Normal-query Recall@5 (higher is better)")
    ax.set_ylabel("Any-sensitive ROC-AUC (lower is better)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
