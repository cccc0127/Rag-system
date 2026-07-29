"""Run auxiliary-data sensitive attribute inference against protected document vectors."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comparison_experiments.comparison_config import (  # noqa: E402
    DEFAULT_EF_SEARCH,
    DEFAULT_SAMPLE_CHUNKS,
    DEFAULT_TEST_QUERIES,
    DEFAULT_TOP_K,
    TEST_QUERY_SEED,
)
from comparison_experiments.comparison_runner import run_scheme_retrieval  # noqa: E402
from comparison_experiments.security_experiments.attribute_inference.attacker import (  # noqa: E402
    run_stratified_attribute_attack,
)
from comparison_experiments.security_experiments.attribute_inference.labels import (  # noqa: E402
    aggregate_label_counts,
    build_sensitive_attribute_labels,
)
from comparison_experiments.security_experiments.attribute_inference.metrics import (  # noqa: E402
    aggregate_oof_metrics,
)
from comparison_experiments.security_experiments.attribute_inference.plotting import (  # noqa: E402
    plot_attribute_inference_results,
)
from comparison_experiments.security_experiments.sensitive_retrieval.runner import (  # noqa: E402
    build_schemes,
)
from comparison_experiments.security_experiments.sensitive_retrieval.sensitive_data import (  # noqa: E402
    parse_sensitive_types,
)
from comparison_experiments.shared.context import add_context_args, prepare_comparison_context  # noqa: E402
from comparison_experiments.shared.metrics import compute_scheme_metrics  # noqa: E402
from comparison_experiments.shared.report import print_table  # noqa: E402


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
RESULTS_CSV = RESULTS_DIR / "security_attribute_inference_results.csv"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison" / "security" / "attribute_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run protected-vector sensitive attribute-inference attacks.")
    add_context_args(parser)
    parser.set_defaults(sample_chunks=DEFAULT_SAMPLE_CHUNKS, num_queries=DEFAULT_TEST_QUERIES, query_seed=TEST_QUERY_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument("--sensitive-types", default="email,url,phone")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--min-attribute-positives", type=int, default=10)
    parser.add_argument("--attack-seed", type=int, default=2029)
    parser.add_argument("--our-variant", choices=["no_jl", "jl256", "jl768"], default="jl256")
    parser.add_argument("--utility-scale", type=float, default=0.01)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--noise-seed", type=int, default=42)
    parser.add_argument("--jl-epsilon", type=float, default=0.3)
    parser.add_argument("--jl-seed", type=int, default=42)
    parser.add_argument("--disable-private-rag-rp", action="store_true")
    parser.add_argument("--private-rag-rp-dim", type=int, default=64)
    parser.add_argument("--private-rag-rp-sigma", type=float, default=0.1)
    parser.add_argument("--private-rag-rp-seed", type=int, default=42)
    parser.add_argument("--disable-dcpe-dce", action="store_true")
    parser.add_argument("--dcpe-beta", type=float, default=0.5)
    parser.add_argument("--dcpe-ratio-k", type=int, default=4)
    parser.add_argument("--dcpe-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    _validate_args(args)
    sensitive_types = parse_sensitive_types(args.sensitive_types)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    context = prepare_comparison_context(args)
    labels_by_attribute = build_sensitive_attribute_labels(context.chunk_records, sensitive_types)
    eligible = _eligible_attributes(labels_by_attribute, args.cv_folds, args.min_attribute_positives)
    if not eligible:
        raise RuntimeError("No attribute has enough positive and negative chunks for the requested stratified attack.")

    rows: list[Dict[str, float | int | str | bool]] = []
    for scheme in build_schemes(args):
        output = scheme.run(context.raw_embeddings, context.query_embeddings, context.chunk_records)
        retrieval = run_scheme_retrieval(output, args.top_k, args.ef_search, args.M, args.ef_construction, args.hnsw_space, args.hnsw_seed)
        normal_metrics = compute_scheme_metrics(output, retrieval, top_k=args.top_k, ef_search=args.ef_search)
        for attribute in eligible:
            labels = labels_by_attribute[attribute]
            attack = run_stratified_attribute_attack(output.document_vectors, labels, args.cv_folds, args.attack_seed)
            attack_metrics = aggregate_oof_metrics(
                labels, attack["scores"], attack["predictions"], attack["fold_metrics"]
            )
            row: Dict[str, float | int | str | bool] = {
                "scheme": output.name,
                "backend_type": output.backend_type,
                "target_attribute": attribute,
                "vector_dim": int(output.vector_dim),
                "sample_chunks": int(len(context.chunk_records)),
                "cv_folds": int(args.cv_folds),
                "attack_seed": int(args.attack_seed),
                "sensitive_types": ",".join(sensitive_types),
                "normal_hnsw_recall_at_5": float(normal_metrics["hnsw_recall_at_5"]),
                "normal_hnsw_mrr_at_5": float(normal_metrics["hnsw_mrr_at_5"]),
                "normal_mean_query_time": float(normal_metrics["mean_query_time"]),
                "normal_index_build_time": float(normal_metrics["index_build_time"]),
            }
            row.update(attack_metrics)
            rows.append(row)

    save_metrics_csv(rows, RESULTS_CSV)
    figures = plot_attribute_inference_results(rows, PICTURE_DIR)
    print_summary(rows, aggregate_label_counts(labels_by_attribute), eligible)
    print("\nSaved sensitive attribute-inference results:")
    print(f"- {RESULTS_CSV}")
    print("\nSaved figures:")
    for path in figures:
        print(f"- {path}")


def _eligible_attributes(labels_by_attribute: dict[str, object], cv_folds: int, min_positives: int) -> list[str]:
    eligible: list[str] = []
    for attribute, label_values in labels_by_attribute.items():
        labels = np.asarray(label_values, dtype=int)
        positives = int(labels.sum())
        negatives = int(len(labels) - positives)
        if positives >= min_positives and negatives >= cv_folds and positives >= cv_folds:
            eligible.append(attribute)
    return eligible


def _validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5 because normal retrieval reports Top-5 metrics")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least two")
    if args.min_attribute_positives < args.cv_folds:
        raise ValueError("--min-attribute-positives must be at least --cv-folds")


def save_metrics_csv(metrics: Sequence[Dict[str, float | int | str | bool]], output_path: Path) -> None:
    fieldnames = sorted({key for row in metrics for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def print_summary(rows: Sequence[Dict[str, float | int | str | bool]], counts: dict[str, int], eligible: Sequence[str]) -> None:
    print("\nSensitive attribute counts (values and record-level labels are never printed):")
    print_table(["Attribute", "Positive chunks"], [[name, value] for name, value in counts.items()])
    print(f"\nEvaluated attributes: {', '.join(eligible)}")
    primary = [row for row in rows if row["target_attribute"] == "has_any_sensitive"]
    print("\nProtected-Vector Attribute Inference (any sensitive attribute)")
    print_table(
        ["Scheme", "ROC-AUC", "TPR@1% FPR", "Macro F1", "Normal R@5", "Low-FPR limit"],
        [[row["scheme"], f"{float(row['roc_auc']):.4f}", f"{float(row['tpr_at_fpr_1pct']):.4f}", f"{float(row['macro_f1']):.4f}", f"{float(row['normal_hnsw_recall_at_5']):.4f}", str(bool(row['low_fpr_resolution_limited']))] for row in primary],
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
