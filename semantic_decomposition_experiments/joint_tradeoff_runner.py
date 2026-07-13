"""Run JL dimension and utility-scale joint tradeoff experiments."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comparison_experiments.comparison_config import (  # noqa: E402
    DEFAULT_EF_SEARCH,
    DEFAULT_SAMPLE_CHUNKS,
    DEFAULT_TEST_QUERIES,
    TEST_QUERY_SEED,
)
from comparison_experiments.shared.context import (  # noqa: E402
    add_context_args,
    prepare_comparison_context,
)
from comparison_experiments.shared.retrievers import run_hnsw_retrieval  # noqa: E402
from config import config  # noqa: E402
from dimension_reduction import JLProjector, l2_normalize  # noqa: E402
from gaussian_noise import AnalyticGaussianCalibrator, NoiseApplication  # noqa: E402
from semantic_decomposition_experiments.metrics import (  # noqa: E402
    exact_retrieval,
    mean_direction_cosine,
    mean_mrr_at_5,
    mean_overlap,
)
from semantic_decomposition_experiments.plotting import plot_joint_tradeoff  # noqa: E402


RESULTS_DIR = Path("semantic_decomposition_experiments") / "results"
RESULTS_CSV = RESULTS_DIR / "joint_tradeoff_results.csv"
DEFAULT_JL_TARGET_DIM_LIST = "128,256,384,512,768"
DEFAULT_UTILITY_SCALE_LIST = "0.001,0.005,0.01,0.05,0.1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run JL target dimension x utility_scale tradeoff for Our DP-RAG."
    )
    add_context_args(parser)
    parser.set_defaults(
        sample_chunks=DEFAULT_SAMPLE_CHUNKS,
        num_queries=DEFAULT_TEST_QUERIES,
        query_seed=TEST_QUERY_SEED,
    )
    parser.add_argument("--jl-target-dim-list", default=DEFAULT_JL_TARGET_DIM_LIST)
    parser.add_argument("--utility-scale-list", default=DEFAULT_UTILITY_SCALE_LIST)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--jl-epsilon", type=float, default=float(config.JL_EPSILON))
    parser.add_argument("--jl-seed", type=int, default=int(config.JL_RANDOM_SEED))
    parser.add_argument("--noise-seed", type=int, default=int(config.DP_RANDOM_SEED))
    parser.add_argument("--dp-delta", type=float, default=float(config.DP_DELTA))
    parser.add_argument(
        "--include-no-jl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include Raw 1024d -> dimension-aware DP -> HNSW without JL projection.",
    )
    parser.add_argument("--no-jl-label", default="No-JL-1024d")
    parser.add_argument("--max-dp-loss-for-recommendation", type=float, default=0.02)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jl_target_dims = parse_int_list(args.jl_target_dim_list)
    utility_scales = parse_float_list(args.utility_scale_list)
    retrieval_depth = max(10, int(args.top_k))

    context = prepare_comparison_context(args)
    raw_doc_vectors = l2_normalize(context.raw_embeddings)
    raw_query_vectors = l2_normalize(context.query_embeddings)
    raw_topk, _ = exact_retrieval(raw_doc_vectors, raw_query_vectors, retrieval_depth)

    rows: List[Dict[str, float | int | str]] = []
    for jl_target_dim in jl_target_dims:
        if args.verbose:
            print(f"Running JL target dim {jl_target_dim}...")
        rows.extend(
            run_dimension_grid(
                context=context,
                args=args,
                jl_target_dim=jl_target_dim,
                utility_scales=utility_scales,
                raw_topk=raw_topk,
                retrieval_depth=retrieval_depth,
            )
        )
    if args.include_no_jl:
        if args.verbose:
            print(f"Running {args.no_jl_label}...")
        rows.extend(
            run_no_jl_grid(
                context=context,
                args=args,
                utility_scales=utility_scales,
                raw_doc_vectors=raw_doc_vectors,
                raw_query_vectors=raw_query_vectors,
                raw_topk=raw_topk,
                retrieval_depth=retrieval_depth,
            )
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_results_csv(rows, RESULTS_CSV)
    figure_paths = plot_joint_tradeoff(
        rows,
        max_dp_loss_for_recommendation=args.max_dp_loss_for_recommendation,
    )

    if args.verbose:
        print_joint_table(rows)
        print()

    print("Saved joint tradeoff results:")
    print(f"- {RESULTS_CSV}")
    print()
    print("Saved joint tradeoff figures:")
    for path in figure_paths:
        print(f"- {path}")


def run_dimension_grid(
    context,
    args: argparse.Namespace,
    jl_target_dim: int,
    utility_scales: List[float],
    raw_topk: np.ndarray,
    retrieval_depth: int,
) -> List[Dict[str, float | int | str]]:
    projector = JLProjector(
        target_dim=jl_target_dim,
        eps=args.jl_epsilon,
        random_state=args.jl_seed,
    )
    jl_doc_vectors = projector.fit_transform(context.raw_embeddings)
    jl_query_vectors = projector.transform(context.query_embeddings)
    jl_topk, jl_query_times = exact_retrieval(jl_doc_vectors, jl_query_vectors, retrieval_depth)

    jl_recall_at_5 = mean_overlap(raw_topk, jl_topk, 5)
    jl_mrr_at_5 = mean_mrr_at_5(raw_topk, jl_topk)

    rows: List[Dict[str, float | int | str]] = []
    for utility_scale in utility_scales:
        if args.verbose:
            print(f"  utility_scale={utility_scale:g}")
        dp_doc_vectors, applications = apply_dp_noise(
            jl_doc_vectors=jl_doc_vectors,
            chunk_records=context.chunk_records,
            utility_scale=utility_scale,
            dp_delta=args.dp_delta,
            noise_seed=args.noise_seed,
        )
        dp_topk, dp_query_times = exact_retrieval(dp_doc_vectors, jl_query_vectors, retrieval_depth)
        hnsw_result = run_hnsw_retrieval(
            document_vectors=dp_doc_vectors,
            query_vectors=jl_query_vectors,
            top_k=retrieval_depth,
            ef_search=args.ef_search,
            M=args.M,
            ef_construction=args.ef_construction,
            space="cosine",
            random_seed=args.jl_seed,
        )

        rows.append(
            {
                "jl_target_dim": int(jl_target_dim),
                "representation_type": "JL",
                "representation_label": f"{int(jl_target_dim)}d JL",
                "is_no_jl": False,
                "utility_scale": float(utility_scale),
                "sampled_chunks": int(len(context.chunk_records)),
                "num_queries": int(len(context.queries)),
                "ef_search": int(args.ef_search),
                "jl_recall_at_5": jl_recall_at_5,
                "jl_mrr_at_5": jl_mrr_at_5,
                "final_exact_recall_at_5": mean_overlap(raw_topk, dp_topk, 5),
                "final_hnsw_recall_at_5": mean_overlap(raw_topk, hnsw_result.topk_indices, 5),
                "final_hnsw_mrr_at_5": mean_mrr_at_5(raw_topk, hnsw_result.topk_indices),
                "dp_recall_at_5": mean_overlap(jl_topk, dp_topk, 5),
                "dp_mrr_at_5": mean_mrr_at_5(jl_topk, dp_topk),
                "dp_loss_at_5": 1.0 - mean_overlap(jl_topk, dp_topk, 5),
                "mean_noise_signal_ratio": compute_mean_noise_signal_ratio(applications),
                "mean_direction_cosine": mean_direction_cosine(jl_doc_vectors, dp_doc_vectors),
                "hnsw_recall_at_5": mean_overlap(dp_topk, hnsw_result.topk_indices, 5),
                "hnsw_mrr_at_5": mean_mrr_at_5(dp_topk, hnsw_result.topk_indices),
                "mean_hnsw_query_time": float(np.mean(hnsw_result.query_times)),
                "hnsw_index_build_time": float(hnsw_result.index_build_time),
                "vector_dim": int(dp_doc_vectors.shape[1]),
                "estimated_storage_mb": estimate_storage_mb(
                    n_vectors=len(context.chunk_records),
                    vector_dim=dp_doc_vectors.shape[1],
                ),
                "mean_exact_query_time": float(np.mean(dp_query_times)),
                "mean_jl_exact_query_time": float(np.mean(jl_query_times)),
            }
        )
    return rows


def run_no_jl_grid(
    context,
    args: argparse.Namespace,
    utility_scales: List[float],
    raw_doc_vectors: np.ndarray,
    raw_query_vectors: np.ndarray,
    raw_topk: np.ndarray,
    retrieval_depth: int,
) -> List[Dict[str, float | int | str]]:
    rows: List[Dict[str, float | int | str]] = []
    raw_dim = int(raw_doc_vectors.shape[1])
    for utility_scale in utility_scales:
        if args.verbose:
            print(f"  utility_scale={utility_scale:g}")
        no_jl_dp_doc_vectors, applications = apply_dp_noise(
            jl_doc_vectors=raw_doc_vectors,
            chunk_records=context.chunk_records,
            utility_scale=utility_scale,
            dp_delta=args.dp_delta,
            noise_seed=args.noise_seed,
        )
        no_jl_dp_topk, exact_query_times = exact_retrieval(
            no_jl_dp_doc_vectors,
            raw_query_vectors,
            retrieval_depth,
        )
        hnsw_result = run_hnsw_retrieval(
            document_vectors=no_jl_dp_doc_vectors,
            query_vectors=raw_query_vectors,
            top_k=retrieval_depth,
            ef_search=args.ef_search,
            M=args.M,
            ef_construction=args.ef_construction,
            space="cosine",
            random_seed=args.jl_seed,
        )
        dp_recall_at_5 = mean_overlap(raw_topk, no_jl_dp_topk, 5)

        rows.append(
            {
                "jl_target_dim": raw_dim,
                "representation_type": "No-JL",
                "representation_label": args.no_jl_label,
                "is_no_jl": True,
                "utility_scale": float(utility_scale),
                "sampled_chunks": int(len(context.chunk_records)),
                "num_queries": int(len(context.queries)),
                "ef_search": int(args.ef_search),
                "jl_recall_at_5": 1.0,
                "jl_mrr_at_5": 1.0,
                "final_exact_recall_at_5": dp_recall_at_5,
                "final_hnsw_recall_at_5": mean_overlap(raw_topk, hnsw_result.topk_indices, 5),
                "final_hnsw_mrr_at_5": mean_mrr_at_5(raw_topk, hnsw_result.topk_indices),
                "dp_recall_at_5": dp_recall_at_5,
                "dp_mrr_at_5": mean_mrr_at_5(raw_topk, no_jl_dp_topk),
                "dp_loss_at_5": 1.0 - dp_recall_at_5,
                "mean_noise_signal_ratio": compute_mean_noise_signal_ratio(applications),
                "mean_direction_cosine": mean_direction_cosine(raw_doc_vectors, no_jl_dp_doc_vectors),
                "hnsw_recall_at_5": mean_overlap(no_jl_dp_topk, hnsw_result.topk_indices, 5),
                "hnsw_mrr_at_5": mean_mrr_at_5(no_jl_dp_topk, hnsw_result.topk_indices),
                "mean_hnsw_query_time": float(np.mean(hnsw_result.query_times)),
                "hnsw_index_build_time": float(hnsw_result.index_build_time),
                "vector_dim": raw_dim,
                "estimated_storage_mb": estimate_storage_mb(
                    n_vectors=len(context.chunk_records),
                    vector_dim=raw_dim,
                ),
                "mean_exact_query_time": float(np.mean(exact_query_times)),
                "mean_jl_exact_query_time": float("nan"),
            }
        )
    return rows


def apply_dp_noise(
    jl_doc_vectors: np.ndarray,
    chunk_records: List[Dict[str, object]],
    utility_scale: float,
    dp_delta: float,
    noise_seed: int,
) -> tuple[np.ndarray, List[NoiseApplication]]:
    calibrator = AnalyticGaussianCalibrator(
        delta=dp_delta,
        l2_clip_norm=1.0,
        utility_scale=utility_scale,
        random_state=noise_seed,
    )

    applications: List[NoiseApplication] = []
    noised_rows: List[np.ndarray] = []
    for vector, record in zip(jl_doc_vectors, chunk_records):
        application = calibrator.apply_noise_with_diagnostics(vector, extract_raw_score(record))
        applications.append(application)
        noised_rows.append(application.noised_vector)
    return l2_normalize(np.vstack(noised_rows)).astype(np.float32), applications


def extract_raw_score(record: Dict[str, object]) -> float:
    for key in ("raw_sensitivity_score", "raw_score", "sensitivity_score"):
        if key in record:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                break
    return 0.1


def compute_mean_noise_signal_ratio(applications: List[NoiseApplication]) -> float:
    ratios: List[float] = []
    for application in applications:
        signal_norm = float(np.linalg.norm(application.clipped_vector, ord=2))
        noise_norm = float(np.linalg.norm(application.noise_vector, ord=2))
        ratios.append(noise_norm / max(signal_norm, 1e-12))
    return float(np.mean(ratios)) if ratios else float("nan")


def estimate_storage_mb(n_vectors: int, vector_dim: int) -> float:
    return float(n_vectors * vector_dim * 4 / 1024 / 1024)


def parse_int_list(value: str) -> List[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one integer")
    return items


def parse_float_list(value: str) -> List[float]:
    items = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not items:
        raise ValueError("Expected at least one float")
    return items


def write_results_csv(rows: List[Dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def print_joint_table(rows: List[Dict[str, float | int | str]]) -> None:
    columns = [
        "representation_type",
        "jl_target_dim",
        "utility_scale",
        "final_hnsw_recall_at_5",
        "dp_loss_at_5",
        "mean_noise_signal_ratio",
        "mean_direction_cosine",
        "mean_hnsw_query_time",
    ]
    widths = {column: len(column) for column in columns}
    rendered_rows: List[Dict[str, str]] = []
    for row in rows:
        rendered = {}
        for column in columns:
            value = row[column]
            rendered[column] = f"{value:.6f}" if isinstance(value, float) else str(value)
            widths[column] = max(widths[column], len(rendered[column]))
        rendered_rows.append(rendered)

    header = " | ".join(column.ljust(widths[column]) for column in columns)
    rule = "-+-".join("-" * widths[column] for column in columns)
    print(header)
    print(rule)
    for row in rendered_rows:
        print(" | ".join(row[column].ljust(widths[column]) for column in columns))


if __name__ == "__main__":
    main()
