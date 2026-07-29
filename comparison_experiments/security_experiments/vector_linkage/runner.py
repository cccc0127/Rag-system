"""Run known-candidate linkage attacks against protected vector indices."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

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
from comparison_experiments.security_experiments.sensitive_retrieval.runner import (  # noqa: E402
    build_schemes,
)
from comparison_experiments.security_experiments.sensitive_retrieval.sensitive_data import (  # noqa: E402
    find_sensitive_chunks,
    parse_sensitive_types,
    safe_sensitive_summary,
)
from comparison_experiments.security_experiments.vector_linkage.attacker import (  # noqa: E402
    build_public_candidate_vectors,
    exact_linkage_topk,
)
from comparison_experiments.security_experiments.vector_linkage.metrics import (  # noqa: E402
    compute_vector_linkage_metrics,
)
from comparison_experiments.security_experiments.vector_linkage.plotting import (  # noqa: E402
    plot_vector_linkage_results,
)
from comparison_experiments.shared.context import add_context_args, prepare_comparison_context  # noqa: E402
from comparison_experiments.shared.metrics import compute_scheme_metrics  # noqa: E402
from comparison_experiments.shared.report import print_table  # noqa: E402


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
RESULTS_CSV = RESULTS_DIR / "security_vector_linkage_results.csv"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison" / "security" / "vector_linkage"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run protected-vector known-candidate linkage attacks.")
    add_context_args(parser)
    parser.set_defaults(
        sample_chunks=DEFAULT_SAMPLE_CHUNKS,
        num_queries=DEFAULT_TEST_QUERIES,
        query_seed=TEST_QUERY_SEED,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument("--sensitive-types", default="email,url,phone")
    parser.add_argument("--min-sensitive-chunks", type=int, default=5)
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
    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5 because this experiment reports Top-5 metrics")
    if args.min_sensitive_chunks <= 0:
        raise ValueError("--min-sensitive-chunks must be greater than zero")
    sensitive_types = parse_sensitive_types(args.sensitive_types)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    context = prepare_comparison_context(args)
    sensitive_chunks = find_sensitive_chunks(context.chunk_records, sensitive_types)
    if len(sensitive_chunks) < args.min_sensitive_chunks:
        raise RuntimeError(
            f"Only {len(sensitive_chunks)} sensitive chunks were found; "
            f"--min-sensitive-chunks requires {args.min_sensitive_chunks}."
        )
    sensitive_ids = {item.chunk_id for item in sensitive_chunks}

    all_metrics: list[Dict[str, float | int | str]] = []
    for scheme in build_schemes(args):
        scheme_output = scheme.run(context.raw_embeddings, context.query_embeddings, context.chunk_records)
        candidate_vectors, distance_metric = build_public_candidate_vectors(scheme, context.raw_embeddings)
        ranked_ids = exact_linkage_topk(
            protected_vectors=scheme_output.document_vectors,
            candidate_vectors=candidate_vectors,
            distance_metric=distance_metric,
            top_k=5,
        )
        linkage_metrics = compute_vector_linkage_metrics(ranked_ids, sensitive_ids)
        normal_retrieval = run_scheme_retrieval(
            scheme_output, args.top_k, args.ef_search, args.M, args.ef_construction,
            args.hnsw_space, args.hnsw_seed,
        )
        normal_metrics = compute_scheme_metrics(
            scheme_output, normal_retrieval, top_k=args.top_k, ef_search=args.ef_search
        )
        row: Dict[str, float | int | str] = {
            "scheme": scheme_output.name,
            "backend_type": scheme_output.backend_type,
            "vector_dim": int(scheme_output.vector_dim),
            "sample_chunks": int(len(context.chunk_records)),
            "sensitive_types": ",".join(sensitive_types),
            "normal_hnsw_recall_at_5": float(normal_metrics["hnsw_recall_at_5"]),
            "normal_hnsw_mrr_at_5": float(normal_metrics["hnsw_mrr_at_5"]),
            "normal_mean_query_time": float(normal_metrics["mean_query_time"]),
            "normal_index_build_time": float(normal_metrics["index_build_time"]),
        }
        row.update(linkage_metrics)
        all_metrics.append(row)

    save_metrics_csv(all_metrics, RESULTS_CSV)
    figures = plot_vector_linkage_results(all_metrics, PICTURE_DIR)
    print_summary(all_metrics, safe_sensitive_summary(sensitive_chunks))
    print("\nSaved vector linkage results:")
    print(f"- {RESULTS_CSV}")
    print("\nSaved figures:")
    for path in figures:
        print(f"- {path}")


def save_metrics_csv(metrics: Sequence[Dict[str, float | int | str]], output_path: Path) -> None:
    fieldnames = sorted({key for row in metrics for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def print_summary(metrics: Sequence[Dict[str, float | int | str]], sensitive_summary: Dict[str, int]) -> None:
    print("\nSensitive chunk summary (entity values are never printed):")
    print_table(["Field", "Count"], [[key, value] for key, value in sensitive_summary.items()])
    print("\nKnown-Candidate Vector Linkage")
    print_table(
        ["Scheme", "Link Top-1", "Link R@5", "Link MRR@5", "Sensitive Top-1", "Normal R@5"],
        [
            [
                row["scheme"],
                f"{float(row['linkage_top1_recovery_rate']):.4f}",
                f"{float(row['linkage_recall_at_5']):.4f}",
                f"{float(row['linkage_mrr_at_5']):.4f}",
                f"{float(row['sensitive_linkage_top1_recovery_rate']):.4f}",
                f"{float(row['normal_hnsw_recall_at_5']):.4f}",
            ]
            for row in metrics
        ],
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
