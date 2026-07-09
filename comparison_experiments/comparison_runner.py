"""Main entry point for external comparison experiments."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config
from comparison_experiments.comparison_config import (
    DEFAULT_EF_SEARCH,
    DEFAULT_EF_SEARCH_LIST,
    DEFAULT_SAMPLE_CHUNKS,
    DEFAULT_TEST_QUERIES,
    DEFAULT_TOP_K,
    TEST_QUERY_SEED,
)
from comparison_experiments.plotting import cleanup_old_ef_search_figures, plot_comparison_figures
from comparison_experiments.shared.context import add_context_args, prepare_comparison_context
from comparison_experiments.shared.metrics import compute_scheme_metrics
from comparison_experiments.shared.report import (
    print_ef_search_summary,
    print_context_summary,
    print_scheme_report,
    print_table,
)
from comparison_experiments.shared.retrievers import (
    RetrievalResult,
    run_hnsw_filter_refine_retrieval,
    run_hnsw_retrieval,
)
from comparison_experiments.shared.types import SchemeOutput
from comparison_experiments.schemes.dcpe_dce import DCPEDCEScheme
from comparison_experiments.schemes.our_dp_rag import OurDPRAGScheme


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run external comparison experiments for DP-RAG schemes."
    )
    add_context_args(parser)
    parser.set_defaults(
        sample_chunks=DEFAULT_SAMPLE_CHUNKS,
        num_queries=DEFAULT_TEST_QUERIES,
        query_seed=TEST_QUERY_SEED,
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--utility-scale", type=float, default=getattr(config, "DP_UTILITY_SCALE", 0.01))
    parser.add_argument("--dp-delta", type=float, default=getattr(config, "DP_DELTA", 1e-5))
    parser.add_argument("--noise-seed", type=int, default=getattr(config, "DP_RANDOM_SEED", 42))
    parser.add_argument("--jl-target-dim", type=int, default=getattr(config, "JL_TARGET_DIM", 256))
    parser.add_argument("--jl-epsilon", type=float, default=getattr(config, "JL_EPSILON", 0.3))
    parser.add_argument("--jl-seed", type=int, default=getattr(config, "JL_RANDOM_SEED", 42))
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--ef-search-list", default=DEFAULT_EF_SEARCH_LIST)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument("--dcpe-beta", type=float, default=0.5)
    parser.add_argument("--dcpe-ratio-k", type=int, default=4)
    parser.add_argument("--dcpe-seed", type=int, default=42)
    parser.add_argument("--recommended-params", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--visual-text-chars", type=int, default=400)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_ef_search_figures(PICTURE_DIR)
    apply_recommended_params(args)

    context = prepare_comparison_context(args)
    if args.verbose:
        print_context_summary(context)

    schemes = [
        OurDPRAGScheme(
            jl_target_dim=args.jl_target_dim,
            jl_epsilon=args.jl_epsilon,
            jl_seed=args.jl_seed,
            dp_delta=args.dp_delta,
            utility_scale=args.utility_scale,
            noise_seed=args.noise_seed,
        ),
        DCPEDCEScheme(
            beta=args.dcpe_beta,
            ratio_k=args.dcpe_ratio_k,
            random_seed=args.dcpe_seed,
        ),
    ]

    all_metrics: list[Dict[str, float | int | str]] = []
    ef_search_list = parse_int_list(args.ef_search_list)
    if int(args.ef_search) not in ef_search_list:
        ef_search_list.append(int(args.ef_search))
        ef_search_list = sorted(set(ef_search_list))

    for scheme in schemes:
        scheme_output = scheme.run(
            raw_embeddings=context.raw_embeddings,
            query_embeddings=context.query_embeddings,
            chunk_records=context.chunk_records,
        )
        default_retrieval = None
        default_metrics = None
        for ef_search in ef_search_list:
            retrieval = run_scheme_retrieval(
                scheme_output=scheme_output,
                top_k=args.top_k,
                ef_search=ef_search,
                M=args.M,
                ef_construction=args.ef_construction,
                default_space=args.hnsw_space,
                random_seed=args.hnsw_seed,
            )
            metrics = compute_scheme_metrics(
                scheme_output,
                retrieval,
                top_k=args.top_k,
                ef_search=ef_search,
            )
            all_metrics.append(metrics)
            if ef_search == int(args.ef_search):
                default_retrieval = retrieval
                default_metrics = metrics

        if default_retrieval is None or default_metrics is None:
            default_metrics = all_metrics[-1]
            default_retrieval = retrieval
        if args.verbose:
            print_scheme_report(
                context=context,
                scheme_output=scheme_output,
                retrieval=default_retrieval,
                metrics=default_metrics,
                max_text_chars=args.visual_text_chars,
            )

    csv_path = RESULTS_DIR / "comparison_results.csv"
    save_metrics_csv(all_metrics, csv_path)
    figure_paths = plot_comparison_figures(
        all_metrics,
        PICTURE_DIR,
        default_ef_search=int(args.ef_search),
    )

    if args.verbose:
        print("\nComparison Metrics Summary")
        print_table(
            [
                "Scheme",
                "Backend",
                "ef_search",
                "Dim",
                "Mean Query Time",
                "Build Time",
                "NSR",
                "Mean Sigma",
                "Mean Epsilon",
                "Recall@5",
                "MRR@5",
            ],
            [
                [
                    item["scheme"],
                    item["backend_type"],
                    item["ef_search"],
                    item["vector_dim"],
                    f"{float(item['mean_query_time']):.8f}s",
                    f"{float(item['index_build_time']):.6f}s",
                    f"{float(item['mean_noise_signal_ratio']):.6f}",
                    f"{float(item['mean_sigma']):.6f}",
                    f"{float(item['mean_epsilon']):.6f}",
                    f"{float(item['hnsw_recall_at_5']):.6f}",
                    f"{float(item['hnsw_mrr_at_5']):.6f}",
                ]
                for item in all_metrics
            ],
        )
        print_ef_search_summary(all_metrics)

    print("\nSaved comparison results:")
    print(f"- {csv_path}")
    print("\nSaved comparison figures:")
    for path in figure_paths:
        print(f"- {path}")


def save_metrics_csv(metrics: Sequence[Dict[str, float | int | str]], output_path: Path) -> None:
    if not metrics:
        return
    fieldnames = sorted({key for row in metrics for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def run_scheme_retrieval(
    scheme_output: SchemeOutput,
    top_k: int,
    ef_search: int,
    M: int,
    ef_construction: int,
    default_space: str,
    random_seed: int,
) -> RetrievalResult:
    hnsw_space = str(scheme_output.metadata.get("hnsw_space", default_space))
    if scheme_output.backend_type == "hnsw_filter_refine":
        if scheme_output.reference_document_vectors is None or scheme_output.reference_query_vectors is None:
            raise ValueError(f"{scheme_output.name} requires reference vectors for refine")
        ratio_k = int(float(scheme_output.metadata.get("ratio_k", 4)))
        k_prime = max(top_k, ratio_k * top_k)
        return run_hnsw_filter_refine_retrieval(
            document_vectors=scheme_output.document_vectors,
            query_vectors=scheme_output.query_vectors,
            reference_document_vectors=scheme_output.reference_document_vectors,
            reference_query_vectors=scheme_output.reference_query_vectors,
            top_k=max(top_k, 5),
            k_prime=max(k_prime, 5),
            ef_search=ef_search,
            M=M,
            ef_construction=ef_construction,
            space=hnsw_space,
            random_seed=random_seed,
        )

    return run_hnsw_retrieval(
        document_vectors=scheme_output.document_vectors,
        query_vectors=scheme_output.query_vectors,
        top_k=max(top_k, 10, 5),
        ef_search=ef_search,
        M=M,
        ef_construction=ef_construction,
        space=hnsw_space,
        random_seed=random_seed,
    )


def parse_int_list(raw_value: str) -> list[int]:
    values: list[int] = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--ef-search-list values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("--ef-search-list must contain at least one positive integer")
    return sorted(set(values))


def apply_recommended_params(args: argparse.Namespace) -> None:
    if args.recommended_params is None:
        return
    with Path(args.recommended_params).open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    our_params = params.get("Our DP-RAG", {})
    if "utility_scale" in our_params:
        args.utility_scale = float(our_params["utility_scale"])

    dcpe_params = params.get("DCPE+DCE", {})
    if "beta" in dcpe_params:
        args.dcpe_beta = float(dcpe_params["beta"])
    if "ratio_k" in dcpe_params:
        args.dcpe_ratio_k = int(dcpe_params["ratio_k"])

    if getattr(args, "verbose", False):
        print(f"Loaded recommended params from {args.recommended_params}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
