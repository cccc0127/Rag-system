"""Targeted sensitive-chunk retrieval exposure experiment.

The attacker receives Top-K chunk identifiers from a retrieval interface and
uses de-identified semantic context to target a chunk containing a sensitive
entity. This module deliberately does not invoke an LLM or print sensitive
content, queries, or matched entities.
"""

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
from comparison_experiments.schemes.dcpe_dce import DCPEDCEScheme  # noqa: E402
from comparison_experiments.schemes.our_dp_rag import OurDPRAGScheme  # noqa: E402
from comparison_experiments.schemes.private_rag_random_projection import (  # noqa: E402
    PrivateRAGRandomProjectionScheme,
)
from comparison_experiments.schemes.raw_hnsw import RawHNSWScheme  # noqa: E402
from comparison_experiments.security_experiments.sensitive_retrieval.metrics import (  # noqa: E402
    compute_sensitive_retrieval_metrics,
)
from comparison_experiments.security_experiments.sensitive_retrieval.plotting import (  # noqa: E402
    plot_sensitive_retrieval_results,
)
from comparison_experiments.security_experiments.sensitive_retrieval.sensitive_data import (  # noqa: E402
    AttackQuery,
    build_attack_queries,
    find_sensitive_chunks,
    parse_sensitive_types,
    safe_sensitive_summary,
)
from comparison_experiments.shared.context import (  # noqa: E402
    add_context_args,
    prepare_comparison_context,
)
from comparison_experiments.shared.metrics import compute_scheme_metrics  # noqa: E402
from comparison_experiments.shared.report import print_table  # noqa: E402
from comparison_experiments.shared.types import SchemeOutput  # noqa: E402


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
RESULTS_CSV = RESULTS_DIR / "security_sensitive_retrieval_results.csv"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison" / "security" / "sensitive_retrieval"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run targeted sensitive retrieval exposure experiments.")
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
    parser.add_argument("--max-sensitive-targets", type=int, default=50)
    parser.add_argument("--attack-query-seed", type=int, default=2028)
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
    validate_args(args)
    sensitive_types = parse_sensitive_types(args.sensitive_types)
    attack_state: dict[str, object] = {}

    def build_context_attacks(chunk_records: list[Dict[str, object]]) -> list[str]:
        all_sensitive_chunks = find_sensitive_chunks(chunk_records, sensitive_types)
        if len(all_sensitive_chunks) < args.min_sensitive_chunks:
            raise RuntimeError(
                f"Only {len(all_sensitive_chunks)} sensitive chunks were found; "
                f"--min-sensitive-chunks requires {args.min_sensitive_chunks}."
            )
        sensitive_chunks = all_sensitive_chunks
        if args.max_sensitive_targets > 0:
            sensitive_chunks = sensitive_chunks[: args.max_sensitive_targets]
        attacks = build_attack_queries(
            chunk_records, sensitive_chunks, sensitive_types, seed=args.attack_query_seed
        )
        if not attacks:
            raise RuntimeError("No de-identified attack queries could be constructed.")
        attack_state["all_sensitive_chunks"] = all_sensitive_chunks
        attack_state["attacks"] = attacks
        return [attack.query for attack in attacks]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    context = prepare_comparison_context(args, additional_query_builder=build_context_attacks)
    attack_queries = attack_state["attacks"]
    all_sensitive_chunks = attack_state["all_sensitive_chunks"]
    if not isinstance(attack_queries, list) or not isinstance(all_sensitive_chunks, list):
        raise RuntimeError("Attack query construction did not complete.")
    if context.additional_query_embeddings is None:
        raise RuntimeError("Attack query embeddings are unavailable.")

    sensitive_ids = {item.chunk_id for item in all_sensitive_chunks}
    schemes = build_schemes(args)
    all_metrics: list[Dict[str, float | int | str]] = []
    for scheme in schemes:
        normal_output = scheme.run(context.raw_embeddings, context.query_embeddings, context.chunk_records)
        normal_retrieval = run_scheme_retrieval(
            normal_output, args.top_k, args.ef_search, args.M, args.ef_construction,
            args.hnsw_space, args.hnsw_seed,
        )
        normal_metrics = compute_scheme_metrics(
            normal_output, normal_retrieval, top_k=args.top_k, ef_search=args.ef_search
        )

        attack_output = scheme.run(
            context.raw_embeddings, context.additional_query_embeddings, context.chunk_records
        )
        attack_retrieval = run_scheme_retrieval(
            attack_output, args.top_k, args.ef_search, args.M, args.ef_construction,
            args.hnsw_space, args.hnsw_seed,
        )
        security_metrics = compute_sensitive_retrieval_metrics(
            attack_retrieval.topk_indices, attack_queries, sensitive_ids, args.top_k
        )
        row: Dict[str, float | int | str] = {
            "scheme": attack_output.name,
            "backend_type": attack_output.backend_type,
            "vector_dim": attack_output.vector_dim,
            "top_k": int(args.top_k),
            "ef_search": int(args.ef_search),
            "sample_chunks": int(len(context.chunk_records)),
            "normal_hnsw_recall_at_5": float(normal_metrics["hnsw_recall_at_5"]),
            "normal_hnsw_mrr_at_5": float(normal_metrics["hnsw_mrr_at_5"]),
            "normal_mean_query_time": float(normal_metrics["mean_query_time"]),
            "normal_index_build_time": float(normal_metrics["index_build_time"]),
        }
        row.update(security_metrics)
        all_metrics.append(row)

    save_metrics_csv(all_metrics, RESULTS_CSV)
    figures = plot_sensitive_retrieval_results(all_metrics, PICTURE_DIR)
    print_summary(all_metrics, safe_sensitive_summary(all_sensitive_chunks))
    print("\nSaved targeted sensitive-retrieval results:")
    print(f"- {RESULTS_CSV}")
    print("\nSaved figures:")
    for path in figures:
        print(f"- {path}")


def build_schemes(args: argparse.Namespace) -> list[object]:
    variants = {"no_jl": ("no_jl", 256), "jl256": ("jl", 256), "jl768": ("jl", 768)}
    representation_mode, jl_target_dim = variants[args.our_variant]
    schemes: list[object] = [
        RawHNSWScheme(),
        OurDPRAGScheme(
            name=f"Our DP-RAG-{args.our_variant.upper().replace('_', '')}",
            representation_mode=representation_mode,
            jl_target_dim=jl_target_dim,
            jl_epsilon=args.jl_epsilon,
            jl_seed=args.jl_seed,
            dp_delta=args.dp_delta,
            utility_scale=args.utility_scale,
            noise_seed=args.noise_seed,
        ),
    ]
    if not args.disable_private_rag_rp:
        schemes.append(
            PrivateRAGRandomProjectionScheme(
                projection_dim=args.private_rag_rp_dim,
                projection_sigma=args.private_rag_rp_sigma,
                random_seed=args.private_rag_rp_seed,
            )
        )
    if not args.disable_dcpe_dce:
        schemes.append(
            DCPEDCEScheme(
                beta=args.dcpe_beta, ratio_k=args.dcpe_ratio_k, random_seed=args.dcpe_seed
            )
        )
    return schemes


def validate_args(args: argparse.Namespace) -> None:
    if args.top_k < 5:
        raise ValueError("--top-k must be at least 5 because this experiment reports Top-5 metrics")
    if args.min_sensitive_chunks <= 0 or args.max_sensitive_targets < 0:
        raise ValueError("Sensitive-target limits must be non-negative, with min greater than zero")


def save_metrics_csv(metrics: Sequence[Dict[str, float | int | str]], output_path: Path) -> None:
    fieldnames = sorted({key for row in metrics for key in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def print_summary(metrics: Sequence[Dict[str, float | int | str]], sensitive_summary: Dict[str, int]) -> None:
    print("\nSensitive chunk summary (entity values are never printed):")
    print_table(["Field", "Count"], [[key, value] for key, value in sensitive_summary.items()])
    print("\nTargeted Sensitive Retrieval Exposure")
    print_table(
        ["Scheme", "Target R@1", "Target R@5", "Top-1 Exposure", "Sensitive / Top-5", "Normal R@5"],
        [
            [
                row["scheme"],
                f"{float(row['sensitive_target_recall_at_1']):.4f}",
                f"{float(row['sensitive_target_recall_at_5']):.4f}",
                f"{float(row['sensitive_top1_exposure_rate']):.4f}",
                f"{float(row['mean_sensitive_chunks_at_5']):.4f}",
                f"{float(row['normal_hnsw_recall_at_5']):.4f}",
            ]
            for row in metrics
        ],
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
