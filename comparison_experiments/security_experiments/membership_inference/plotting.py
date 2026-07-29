"""Aggregate-only visualizations for membership inference results."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt


def plot_membership_results(rows: Sequence[Dict[str, float | int | str | bool]], output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        _grouped(rows, "roc_auc", "Membership Inference ROC-AUC", output_dir / "membership_roc_auc.png", 0.5),
        _grouped(rows, "tpr_at_fpr", "Membership Inference TPR at Configured Low FPR", output_dir / "membership_tpr_at_fpr_1pct.png", None),
        _grouped(rows, "attack_advantage", "Membership Inference Attack Advantage", output_dir / "membership_attack_advantage.png", None),
        _tradeoff(rows, output_dir / "membership_utility_tradeoff.png"),
        _grouped(rows, "roc_auc", "Membership Attack Comparison (ROC-AUC)", output_dir / "membership_attack_comparison.png", 0.5),
    ]


def _grouped(rows: Sequence[Dict[str, float | int | str | bool]], key: str, title: str, path: Path, reference: float | None) -> Path:
    schemes = list(dict.fromkeys(str(row["scheme"]) for row in rows))
    attacks = list(dict.fromkeys(str(row["attack_name"]) for row in rows))
    width = 0.8 / max(1, len(attacks))
    fig, ax = plt.subplots(figsize=(11, 5.6), dpi=160)
    lookup = {(str(row["scheme"]), str(row["attack_name"])): float(row[key]) for row in rows}
    for attack_index, attack in enumerate(attacks):
        positions = [index + (attack_index - (len(attacks) - 1) / 2) * width for index in range(len(schemes))]
        values = [lookup.get((scheme, attack), float("nan")) for scheme in schemes]
        ax.bar(positions, values, width=width, label=attack)
    if reference is not None:
        ax.axhline(reference, color="#dc2626", linestyle="--", linewidth=1.3, label="Random classifier (0.5)")
    ax.set_title(title)
    ax.set_xlabel("Scheme")
    ax.set_ylabel("Attack performance (lower is better)")
    ax.set_xticks(range(len(schemes)), schemes, rotation=22, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _tradeoff(rows: Sequence[Dict[str, float | int | str | bool]], path: Path) -> Path:
    primary = [row for row in rows if row["attack_name"] == "shadow_logistic_regression"]
    fig, ax = plt.subplots(figsize=(7.8, 5.4), dpi=160)
    for row in primary:
        x_value, y_value = float(row["normal_hnsw_recall_at_5"]), float(row["roc_auc"])
        ax.scatter(x_value, y_value, s=70, color="#0891b2")
        ax.annotate(str(row["scheme"]), (x_value, y_value), xytext=(5, 5), textcoords="offset points")
    ax.axhline(0.5, color="#dc2626", linestyle="--", linewidth=1.3, label="Random classifier (0.5)")
    ax.set_title("Membership-Inference Security–Utility Trade-off")
    ax.set_xlabel("Normal-query Recall@5 (higher is better)")
    ax.set_ylabel("Shadow logistic ROC-AUC (lower is better)")
    ax.set_xlim(0.0, 1.05)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path
