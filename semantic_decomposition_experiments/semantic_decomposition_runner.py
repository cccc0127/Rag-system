"""Run semantic retrieval loss decomposition for Our DP-RAG."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from comparison_experiments.comparison_config import (  # noqa: E402
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
    RetrievalStage,
    evaluate_comparison,
    exact_retrieval,
    mean_direction_cosine,
    rows_to_table,
)
from semantic_decomposition_experiments.plotting import (  # noqa: E402
    plot_semantic_decomposition,
)


RESULTS_DIR = Path("semantic_decomposition_experiments") / "results"
RESULTS_CSV = RESULTS_DIR / "semantic_decomposition_results.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Decompose Our DP-RAG semantic retrieval loss into JL, DP, and HNSW stages."
    )
    add_context_args(parser)
    parser.set_defaults(
        sample_chunks=DEFAULT_SAMPLE_CHUNKS,
        num_queries=DEFAULT_TEST_QUERIES,
        query_seed=TEST_QUERY_SEED,
    )
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--utility-scale", type=float, default=float(config.DP_UTILITY_SCALE))
    parser.add_argument(
        "--recommended-params",
        type=Path,
        default=None,
        help="Optional JSON file. If present, Our DP-RAG.utility_scale overrides --utility-scale.",
    )
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--jl-target-dim", type=int, default=int(config.JL_TARGET_DIM))
    parser.add_argument("--jl-epsilon", type=float, default=float(config.JL_EPSILON))
    parser.add_argument("--jl-seed", type=int, default=int(config.JL_RANDOM_SEED))
    parser.add_argument("--noise-seed", type=int, default=int(config.DP_RANDOM_SEED))
    parser.add_argument("--dp-delta", type=float, default=float(config.DP_DELTA))
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.utility_scale = resolve_utility_scale(args)
    retrieval_depth = max(10, int(args.top_k))

    context = prepare_comparison_context(args)
    stages, diagnostics = build_stages(context, args, retrieval_depth)
    rows = build_comparison_rows(stages, diagnostics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    write_results_csv(rows, RESULTS_CSV)
    figure_paths = plot_semantic_decomposition(rows)

    if args.verbose:
        print(rows_to_table(rows))
        print()
        print(f"utility_scale={args.utility_scale}")
        print(f"mean_noise_signal_ratio={diagnostics['mean_noise_signal_ratio']:.6f}")
        print(f"mean_direction_cosine_c_vs_b={diagnostics['mean_direction_cosine_c_vs_b']:.6f}")
        print()

    print("Saved semantic decomposition results:")
    print(f"- {RESULTS_CSV}")
    print()
    print("Saved semantic decomposition figures:")
    for path in figure_paths:
        print(f"- {path}")


def resolve_utility_scale(args: argparse.Namespace) -> float:
    if args.recommended_params is None:
        return float(args.utility_scale)
    with args.recommended_params.open("r", encoding="utf-8") as handle:
        params = json.load(handle)

    our_params = params.get("Our DP-RAG", {})
    if isinstance(our_params, dict) and "utility_scale" in our_params:
        return float(our_params["utility_scale"])
    return float(args.utility_scale)


def build_stages(
    context,
    args: argparse.Namespace,
    retrieval_depth: int,
) -> tuple[Dict[str, RetrievalStage], Dict[str, float]]:
    raw_doc_vectors = l2_normalize(context.raw_embeddings)
    raw_query_vectors = l2_normalize(context.query_embeddings)
    raw_topk, raw_times = exact_retrieval(raw_doc_vectors, raw_query_vectors, retrieval_depth)
    stage_a = RetrievalStage(
        name="Raw-Exact",
        vectors=raw_doc_vectors,
        query_vectors=raw_query_vectors,
        topk_indices=raw_topk,
        query_times=raw_times,
        vector_dim=int(raw_doc_vectors.shape[1]),
    )

    projector = JLProjector(
        target_dim=args.jl_target_dim,
        eps=args.jl_epsilon,
        random_state=args.jl_seed,
    )
    jl_doc_vectors = projector.fit_transform(context.raw_embeddings)
    jl_query_vectors = projector.transform(context.query_embeddings)
    jl_topk, jl_times = exact_retrieval(jl_doc_vectors, jl_query_vectors, retrieval_depth)
    stage_b = RetrievalStage(
        name="JL-Exact",
        vectors=jl_doc_vectors,
        query_vectors=jl_query_vectors,
        topk_indices=jl_topk,
        query_times=jl_times,
        vector_dim=int(jl_doc_vectors.shape[1]),
    )

    dp_doc_vectors, applications = apply_dp_noise(
        jl_doc_vectors=jl_doc_vectors,
        chunk_records=context.chunk_records,
        args=args,
    )
    dp_topk, dp_times = exact_retrieval(dp_doc_vectors, jl_query_vectors, retrieval_depth)
    stage_c = RetrievalStage(
        name="JL-DP-Exact",
        vectors=dp_doc_vectors,
        query_vectors=jl_query_vectors,
        topk_indices=dp_topk,
        query_times=dp_times,
        vector_dim=int(dp_doc_vectors.shape[1]),
    )

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
    stage_d = RetrievalStage(
        name="JL-DP-HNSW",
        vectors=dp_doc_vectors,
        query_vectors=jl_query_vectors,
        topk_indices=hnsw_result.topk_indices,
        query_times=hnsw_result.query_times,
        vector_dim=int(dp_doc_vectors.shape[1]),
    )

    mean_nsr = compute_mean_noise_signal_ratio(applications)
    direction_c_vs_b = mean_direction_cosine(jl_doc_vectors, dp_doc_vectors)
    diagnostics = {
        "mean_noise_signal_ratio": mean_nsr,
        "mean_direction_cosine_c_vs_b": direction_c_vs_b,
        "hnsw_index_build_time": float(hnsw_result.index_build_time),
    }
    return (
        {
            "A": stage_a,
            "B": stage_b,
            "C": stage_c,
            "D": stage_d,
        },
        diagnostics,
    )


def apply_dp_noise(
    jl_doc_vectors: np.ndarray,
    chunk_records: List[Dict[str, object]],
    args: argparse.Namespace,
) -> tuple[np.ndarray, List[NoiseApplication]]:
    calibrator = AnalyticGaussianCalibrator(
        delta=args.dp_delta,
        l2_clip_norm=1.0,
        utility_scale=args.utility_scale,
        random_state=args.noise_seed,
    )

    applications: List[NoiseApplication] = []
    noised_rows: List[np.ndarray] = []
    for vector, record in zip(jl_doc_vectors, chunk_records):
        raw_score = extract_raw_score(record)
        application = calibrator.apply_noise_with_diagnostics(vector, raw_score)
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


def build_comparison_rows(
    stages: Dict[str, RetrievalStage],
    diagnostics: Dict[str, float],
) -> List[Dict[str, float | int | str]]:
    mean_nsr = diagnostics["mean_noise_signal_ratio"]
    direction_c_vs_b = diagnostics["mean_direction_cosine_c_vs_b"]

    return [
        evaluate_comparison(
            comparison_name="JL-Exact vs Raw-Exact",
            reference_stage=stages["A"],
            candidate_stage=stages["B"],
        ),
        evaluate_comparison(
            comparison_name="JL-DP-Exact vs JL-Exact",
            reference_stage=stages["B"],
            candidate_stage=stages["C"],
            mean_noise_signal_ratio=mean_nsr,
            mean_direction_cosine=direction_c_vs_b,
        ),
        evaluate_comparison(
            comparison_name="JL-DP-HNSW vs JL-DP-Exact",
            reference_stage=stages["C"],
            candidate_stage=stages["D"],
            mean_noise_signal_ratio=mean_nsr,
        ),
        evaluate_comparison(
            comparison_name="JL-DP-Exact vs Raw-Exact",
            reference_stage=stages["A"],
            candidate_stage=stages["C"],
            mean_noise_signal_ratio=mean_nsr,
        ),
        evaluate_comparison(
            comparison_name="JL-DP-HNSW vs Raw-Exact",
            reference_stage=stages["A"],
            candidate_stage=stages["D"],
            mean_noise_signal_ratio=mean_nsr,
        ),
    ]


def write_results_csv(rows: List[Dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
