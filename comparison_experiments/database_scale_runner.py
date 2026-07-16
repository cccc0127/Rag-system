"""Database scale experiment for comparison schemes."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comparison_experiments.comparison_config import (  # noqa: E402
    DEFAULT_EF_SEARCH,
    DEFAULT_TEST_QUERIES,
    DEFAULT_TOP_K,
    TEST_QUERY_SEED,
)
from comparison_experiments.comparison_runner import (  # noqa: E402
    DEFAULT_RECOMMENDED_PARAMS,
    DEFAULT_OUR_VARIANTS,
    apply_recommended_params,
    build_schemes,
    run_scheme_retrieval,
)
from comparison_experiments.plotting import plot_database_scale_figures  # noqa: E402
from comparison_experiments.shared.context import (  # noqa: E402
    add_context_args,
    prepare_comparison_context,
)
from comparison_experiments.shared.metrics import compute_scheme_metrics  # noqa: E402
from comparison_experiments.shared.report import print_table  # noqa: E402
from config import config  # noqa: E402


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
RESULTS_CSV = RESULTS_DIR / "database_scale_results.csv"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison" / "database_scale"
DEFAULT_SAMPLE_CHUNKS_LIST = "100,300,500,1000"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run database scale experiments for DP-RAG comparison schemes."
    )
    add_context_args(parser)
    parser.set_defaults(
        num_queries=DEFAULT_TEST_QUERIES,
        query_seed=TEST_QUERY_SEED,
    )
    parser.add_argument("--sample-chunks-list", default=DEFAULT_SAMPLE_CHUNKS_LIST)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--utility-scale", type=float, default=getattr(config, "DP_UTILITY_SCALE", 0.01))
    parser.add_argument("--dp-delta", type=float, default=getattr(config, "DP_DELTA", 1e-5))
    parser.add_argument("--noise-seed", type=int, default=getattr(config, "DP_RANDOM_SEED", 42))
    parser.add_argument("--jl-target-dim", type=int, default=getattr(config, "JL_TARGET_DIM", 256))
    parser.add_argument("--jl-epsilon", type=float, default=getattr(config, "JL_EPSILON", 0.3))
    parser.add_argument("--jl-seed", type=int, default=getattr(config, "JL_RANDOM_SEED", 42))
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument("--dcpe-beta", type=float, default=0.5)
    parser.add_argument("--dcpe-ratio-k", type=int, default=4)
    parser.add_argument("--dcpe-seed", type=int, default=42)
    parser.add_argument("--recommended-params", type=Path, default=DEFAULT_RECOMMENDED_PARAMS)
    parser.add_argument("--no-recommended-params", action="store_true")
    parser.add_argument("--our-variants", default=DEFAULT_OUR_VARIANTS)
    parser.add_argument("--disable-dcpe-dce", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    recommended_params = apply_recommended_params(args)
    sample_chunks_list = parse_int_list(args.sample_chunks_list, "--sample-chunks-list")

    all_metrics: list[Dict[str, float | int | str]] = []
    for sample_chunks in sample_chunks_list:
        if args.verbose:
            print(f"\nRunning database scale stage: sample_chunks={sample_chunks}")
        scale_args = argparse.Namespace(**vars(args))
        scale_args.sample_chunks = int(sample_chunks)
        context = prepare_comparison_context(scale_args)
        schemes = build_schemes(scale_args, recommended_params)

        stage_metrics: list[Dict[str, float | int | str]] = []
        for scheme in schemes:
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
                    "sample_chunks": int(sample_chunks),
                    "num_queries": int(len(context.queries)),
                    "dataset_size_label": f"{sample_chunks} chunks",
                    "scale_stage": int(sample_chunks),
                    "representation_mode": scheme_output.metadata.get("representation_mode", ""),
                    "uses_jl": scheme_output.metadata.get("uses_jl", ""),
                }
            )
            stage_metrics.append(metrics)
            all_metrics.append(metrics)

        if args.verbose:
            print_scale_table(stage_metrics)

    save_metrics_csv(all_metrics, RESULTS_CSV)
    figure_paths = plot_database_scale_figures(all_metrics, PICTURE_DIR)

    print("\nSaved database scale results:")
    print(f"- {RESULTS_CSV}")
    print("\nSaved database scale figures:")
    for path in figure_paths:
        print(f"- {path}")


def parse_int_list(raw_value: str, arg_name: str) -> list[int]:
    values: list[int] = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value <= 0:
            raise ValueError(f"{arg_name} values must be positive integers")
        values.append(value)
    if not values:
        raise ValueError(f"{arg_name} must contain at least one positive integer")
    return sorted(set(values))


def save_metrics_csv(metrics: Sequence[Dict[str, float | int | str]], output_path: Path) -> None:
    if not metrics:
        return
    fieldnames = sorted({key for row in metrics for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def print_scale_table(metrics: Sequence[Dict[str, float | int | str]]) -> None:
    print_table(
        ["Scheme", "Recall@5", "MRR@5", "Query Time", "Build Time"],
        [
            [
                row["scheme"],
                f"{float(row['hnsw_recall_at_5']):.6f}",
                f"{float(row['hnsw_mrr_at_5']):.6f}",
                f"{float(row['mean_query_time']):.8f}s",
                f"{float(row['index_build_time']):.6f}s",
            ]
            for row in metrics
        ],
    )


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
