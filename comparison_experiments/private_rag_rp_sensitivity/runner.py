"""Projection-dimension sensitivity experiment for Private RAG-RP."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[2]
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
from comparison_experiments.private_rag_rp_sensitivity.plotting import (  # noqa: E402
    plot_private_rag_rp_sensitivity,
)
from comparison_experiments.schemes.private_rag_random_projection import (  # noqa: E402
    PrivateRAGRandomProjectionScheme,
)
from comparison_experiments.shared.context import (  # noqa: E402
    add_context_args,
    prepare_comparison_context,
)
from comparison_experiments.shared.metrics import compute_scheme_metrics  # noqa: E402
from comparison_experiments.shared.report import print_table  # noqa: E402


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
RESULTS_CSV = RESULTS_DIR / "private_rag_rp_sensitivity_results.csv"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison" / "private_rag_rp_sensitivity"
DEFAULT_K_LIST = "64,128,256,512,768"
DEFAULT_PROJECTION_SIGMA = 0.1
DEFAULT_PROJECTION_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Private RAG-RP projection-dimension sensitivity experiments."
    )
    add_context_args(parser)
    parser.set_defaults(
        sample_chunks=DEFAULT_SAMPLE_CHUNKS,
        num_queries=DEFAULT_TEST_QUERIES,
        query_seed=TEST_QUERY_SEED,
    )
    parser.add_argument("--k-list", default=DEFAULT_K_LIST)
    parser.add_argument("--private-rag-rp-sigma", type=float, default=DEFAULT_PROJECTION_SIGMA)
    parser.add_argument("--private-rag-rp-seed", type=int, default=DEFAULT_PROJECTION_SEED)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    k_values = parse_k_list(args.k_list)
    if args.private_rag_rp_sigma <= 0.0:
        raise ValueError("--private-rag-rp-sigma must be greater than 0")
    if args.top_k <= 0 or args.ef_search <= 0 or args.M <= 0 or args.ef_construction <= 0:
        raise ValueError("HNSW and Top-K parameters must be positive")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    context = prepare_comparison_context(args)
    raw_dim = int(context.raw_embeddings.shape[1])
    if args.verbose and any(k > raw_dim for k in k_values):
        print(
            "Note: k larger than the raw embedding dimension is an expanding random "
            "mapping, not dimensionality reduction."
        )

    all_metrics: list[Dict[str, float | int | str]] = []
    for k in k_values:
        scheme = PrivateRAGRandomProjectionScheme(
            projection_dim=k,
            projection_sigma=args.private_rag_rp_sigma,
            random_seed=args.private_rag_rp_seed,
        )
        scheme_output = scheme.run(
            raw_embeddings=context.raw_embeddings,
            query_embeddings=context.query_embeddings,
            chunk_records=context.chunk_records,
        )
        retrieval = run_scheme_retrieval(
            scheme_output=scheme_output,
            top_k=args.top_k,
            ef_search=args.ef_search,
            M=args.M,
            ef_construction=args.ef_construction,
            default_space=args.hnsw_space,
            random_seed=args.hnsw_seed,
        )
        metrics = compute_scheme_metrics(
            scheme_output=scheme_output,
            retrieval=retrieval,
            top_k=args.top_k,
            ef_search=args.ef_search,
        )
        metrics.update(
            {
                "k": int(k),
                "projection_dim": int(k),
                "projection_sigma": float(args.private_rag_rp_sigma),
                "projection_seed": int(args.private_rag_rp_seed),
                "top_k": int(args.top_k),
                "sample_chunks": int(len(context.chunk_records)),
                "num_queries": int(len(context.queries)),
            }
        )
        all_metrics.append(metrics)

    save_metrics_csv(all_metrics, RESULTS_CSV)
    figure_paths = plot_private_rag_rp_sensitivity(all_metrics, PICTURE_DIR)
    print_summary(all_metrics)
    print("\nSaved Private RAG-RP sensitivity results:")
    print(f"- {RESULTS_CSV}")
    print("\nSaved Private RAG-RP sensitivity figures:")
    for path in figure_paths:
        print(f"- {path}")


def parse_k_list(raw_value: str) -> list[int]:
    values: list[int] = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError("--k-list values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError("--k-list must contain at least one positive integer")
    return sorted(set(values))


def save_metrics_csv(metrics: Sequence[Dict[str, float | int | str]], output_path: Path) -> None:
    if not metrics:
        return
    fieldnames = sorted({key for row in metrics for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def print_summary(metrics: Sequence[Dict[str, float | int | str]]) -> None:
    print("\nPrivate RAG-RP Projection-Dimension Sensitivity")
    print_table(
        ["k", "Recall@5", "MRR@5", "Mean Query Time", "Index Build Time", "Vector Dim"],
        [
            [
                row["k"],
                f"{float(row['hnsw_recall_at_5']):.6f}",
                f"{float(row['hnsw_mrr_at_5']):.6f}",
                f"{float(row['mean_query_time']):.8f}s",
                f"{float(row['index_build_time']):.6f}s",
                row["vector_dim"],
            ]
            for row in sorted(metrics, key=lambda row: int(row["k"]))
        ],
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
