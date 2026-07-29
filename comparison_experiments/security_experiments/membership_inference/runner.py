"""Run source-document-disjoint protected-vector membership inference attacks."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Dict, Sequence

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comparison_experiments.comparison_config import DEFAULT_EF_SEARCH, DEFAULT_TEST_QUERIES, DEFAULT_TOP_K, TEST_QUERY_SEED  # noqa: E402
from comparison_experiments.comparison_runner import run_scheme_retrieval  # noqa: E402
from comparison_experiments.security_experiments.membership_inference.attacker import (  # noqa: E402
    shadow_density_knn_attack,
    shadow_logistic_regression_attack,
)
from comparison_experiments.security_experiments.membership_inference.dataset_split import (  # noqa: E402
    MembershipDataSplit,
    build_membership_data_split,
)
from comparison_experiments.security_experiments.membership_inference.metrics import (  # noqa: E402
    bootstrap_confidence_intervals,
    membership_metrics,
)
from comparison_experiments.security_experiments.membership_inference.plotting import plot_membership_results  # noqa: E402
from comparison_experiments.security_experiments.sensitive_retrieval.runner import build_schemes  # noqa: E402
from comparison_experiments.shared.context import add_context_args  # noqa: E402
from comparison_experiments.shared.metrics import compute_scheme_metrics  # noqa: E402
from comparison_experiments.shared.report import print_table  # noqa: E402
from evaluator import generate_random_queries, iter_documents_recursive, load_embedding_model  # noqa: E402


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
RESULTS_CSV = RESULTS_DIR / "security_membership_inference_results.csv"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison" / "security" / "membership_inference"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run protected-document-vector membership inference attacks.")
    add_context_args(parser)
    parser.set_defaults(num_queries=DEFAULT_TEST_QUERIES, query_seed=TEST_QUERY_SEED)
    parser.add_argument("--membership-samples-per-class", type=int, default=100)
    parser.add_argument("--shadow-samples-per-class", type=int, default=100)
    parser.add_argument("--split-seed", type=int, default=2030)
    parser.add_argument("--attack-seed", type=int, default=2031)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--low-fpr", type=float, default=0.01)
    parser.add_argument("--density-knn-k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
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
    data_split = build_membership_data_split(
        iter_documents_recursive(args.knowledge_base), args.membership_samples_per_class, args.shadow_samples_per_class,
        args.chunk_size, args.overlap, args.enable_nlp_privacy, args.split_seed,
    )
    model = load_embedding_model(str(args.embedding_model))
    raw_vectors = _encode_split(model, data_split)
    target_queries = generate_random_queries(data_split.groups["target_member"], args.num_queries, args.query_seed)
    query_vectors = np.asarray(model.encode(target_queries, batch_size=max(1, len(target_queries)), show_progress_bar=False), dtype=np.float32)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[Dict[str, float | int | str | bool]] = []
    for scheme in build_schemes(args):
        target_member, target_nonmember, target_output = _protected_pair(
            scheme, data_split.groups["target_member"], raw_vectors["target_member"],
            data_split.groups["target_nonmember"], raw_vectors["target_nonmember"], query_vectors, args.attack_seed,
        )
        shadow_member, shadow_nonmember, _ = _protected_pair(
            scheme, data_split.groups["shadow_member"], raw_vectors["shadow_member"],
            data_split.groups["shadow_nonmember"], raw_vectors["shadow_nonmember"], query_vectors, args.attack_seed + 1,
        )
        retrieval = run_scheme_retrieval(target_output, args.top_k, args.ef_search, args.M, args.ef_construction, args.hnsw_space, args.hnsw_seed)
        utility = compute_scheme_metrics(target_output, retrieval, top_k=args.top_k, ef_search=args.ef_search)
        target_vectors = np.vstack((target_member, target_nonmember))
        target_labels = np.concatenate((np.ones(len(target_member), dtype=np.int64), np.zeros(len(target_nonmember), dtype=np.int64)))
        attacks = {
            "shadow_logistic_regression": shadow_logistic_regression_attack(shadow_member, shadow_nonmember, target_vectors, args.attack_seed),
            "shadow_density_knn": shadow_density_knn_attack(shadow_member, shadow_nonmember, target_vectors, args.density_knn_k),
        }
        for attack_name, attack in attacks.items():
            metrics = membership_metrics(target_labels, attack.scores, attack.predictions, args.low_fpr)
            metrics.update(bootstrap_confidence_intervals(target_labels, attack.scores, attack.predictions, args.low_fpr, args.bootstrap_samples, args.attack_seed))
            row: Dict[str, float | int | str | bool] = {
                "scheme": target_output.name,
                "backend_type": target_output.backend_type,
                "attack_name": attack_name,
                "vector_dim": int(target_output.vector_dim),
                "shadow_member_count": int(len(shadow_member)),
                "shadow_nonmember_count": int(len(shadow_nonmember)),
                "attack_train_size": int(attack.train_size),
                "split_seed": int(args.split_seed),
                "attack_seed": int(args.attack_seed),
                "bootstrap_samples": int(args.bootstrap_samples),
                "low_fpr": float(args.low_fpr),
                "normal_hnsw_recall_at_5": float(utility["hnsw_recall_at_5"]),
                "normal_hnsw_mrr_at_5": float(utility["hnsw_mrr_at_5"]),
                "normal_mean_query_time": float(utility["mean_query_time"]),
                "normal_index_build_time": float(utility["index_build_time"]),
            }
            row.update(metrics)
            rows.append(row)
    _save_rows(rows, RESULTS_CSV)
    figures = plot_membership_results(rows, PICTURE_DIR)
    _print_summary(rows)
    print("\nSaved membership-inference results:")
    print(f"- {RESULTS_CSV}")
    print("\nSaved figures:")
    for path in figures:
        print(f"- {path}")


def _encode_split(model: object, data_split: MembershipDataSplit) -> dict[str, np.ndarray]:
    records = [record for group in data_split.groups.values() for record in group]
    lengths = [len(group) for group in data_split.groups.values()]
    embeddings = np.asarray(model.encode([str(record["content"]) for record in records], batch_size=16, show_progress_bar=True), dtype=np.float32)
    result: dict[str, np.ndarray] = {}
    start = 0
    for name, length in zip(data_split.groups, lengths):
        result[name] = embeddings[start:start + length]
        start += length
    return result


def _protected_pair(scheme: object, member_records: Sequence[Dict[str, object]], member_raw: np.ndarray, nonmember_records: Sequence[Dict[str, object]], nonmember_raw: np.ndarray, query_vectors: np.ndarray, shuffle_seed: int):
    """Transform a shuffled member/nonmember mixture to avoid label-position artifacts."""
    labels = np.concatenate((np.ones(len(member_raw), dtype=bool), np.zeros(len(nonmember_raw), dtype=bool)))
    raw = np.vstack((member_raw, nonmember_raw))
    records = list(member_records) + list(nonmember_records)
    order = np.random.default_rng(shuffle_seed).permutation(len(raw))
    output = scheme.run(raw[order], query_vectors, [records[int(index)] for index in order])
    mixed_labels = labels[order]
    member_vectors = output.document_vectors[mixed_labels]
    nonmember_vectors = output.document_vectors[~mixed_labels]
    member_output = replace(
        output,
        document_vectors=member_vectors,
        reference_document_vectors=(output.reference_document_vectors[mixed_labels] if output.reference_document_vectors is not None else None),
    )
    return member_vectors, nonmember_vectors, member_output


def _validate_args(args: argparse.Namespace) -> None:
    if min(args.membership_samples_per_class, args.shadow_samples_per_class, args.bootstrap_samples, args.density_knn_k) <= 0:
        raise ValueError("Sample counts, bootstrap samples, and density_knn_k must be greater than zero")
    if args.shadow_samples_per_class <= args.density_knn_k:
        raise ValueError("--shadow-samples-per-class must exceed --density-knn-k")
    if not 0.0 < args.low_fpr < 1.0:
        raise ValueError("--low-fpr must be between zero and one")
    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5 because normal retrieval reports Top-5 metrics")


def _save_rows(rows: Sequence[Dict[str, float | int | str | bool]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: Sequence[Dict[str, float | int | str | bool]]) -> None:
    print("\nProtected-Vector Membership Inference (member is positive; lower is better)")
    print_table(
        ["Scheme", "Attack", "ROC-AUC", "TPR@low FPR", "Advantage", "Normal R@5", "Low-FPR limit"],
        [[row["scheme"], row["attack_name"], f"{float(row['roc_auc']):.4f}", f"{float(row['tpr_at_fpr']):.4f}", f"{float(row['attack_advantage']):.4f}", f"{float(row['normal_hnsw_recall_at_5']):.4f}", str(bool(row['low_fpr_resolution_limited']))] for row in rows],
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
