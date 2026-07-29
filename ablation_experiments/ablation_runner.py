"""Run internal DP-RAG ablation experiments."""

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

from config import config
from evaluator import (
    count_iterated_documents,
    generate_random_queries,
    iter_documents_recursive,
    load_embedding_model,
    print_table,
    sample_chunks,
)
from dimension_reduction import JLProjector
from gaussian_noise import AnalyticGaussianCalibrator
from ablation_experiments.metrics import evaluate_scheme_metrics, metric_rows
from ablation_experiments.no_jl_schemes import (
    matched_dynamic_calibration,
    run_fixed_dp_calibration_no_jl,
    run_full_current_no_jl,
    run_no_dimension_aware_scaling_no_jl,
    run_no_dp_baseline_no_jl,
)
from ablation_experiments.plotting import plot_all, plot_no_jl_main
from ablation_experiments.schemes import SCHEMES


UTILITY_SCALE_LIST = [0.001, 0.01, 0.05, 0.1, 0.2, 0.5]
RESULTS_DIR = ROOT_DIR / "ablation_experiments" / "results"   #csv文件
PICTURE_DIR = ROOT_DIR / "Result_picture" / "ablation"      #实验结果图片
LEGACY_PICTURE_DIR = ROOT_DIR / "Result_picture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DP-RAG internal ablation experiments.")
    parser.add_argument("--knowledge-base", type=Path, default=config.REFERENCE_FOLDER)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--sample-chunks", type=int, default=100)
    parser.add_argument("--num-queries", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=getattr(config, "CHUNK_SIZE", 1000))
    parser.add_argument("--overlap", type=int, default=getattr(config, "OVERLAP", 200))
    parser.add_argument("--jl-target-dim", type=int, default=getattr(config, "JL_TARGET_DIM", 256))
    parser.add_argument("--jl-epsilon", type=float, default=getattr(config, "JL_EPSILON", 0.3))
    parser.add_argument("--jl-seed", type=int, default=getattr(config, "JL_RANDOM_SEED", 42))
    parser.add_argument("--noise-seed", type=int, default=getattr(config, "DP_RANDOM_SEED", 42))
    parser.add_argument("--dp-delta", type=float, default=getattr(config, "DP_DELTA", 1e-5))
    parser.add_argument("--query-seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--ef-search", type=int, default=64)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument("--enable-nlp-privacy", action="store_true")
    parser.add_argument("--representation-mode", choices=["jl", "no_jl"], default="jl")
    return parser.parse_args()


def prepare_context(args: argparse.Namespace) -> Dict[str, object]:
    doc_counter = {"count": 0}
    docs = count_iterated_documents(iter_documents_recursive(args.knowledge_base), doc_counter)
    chunk_records = sample_chunks(
        docs,
        max_chunks=args.sample_chunks,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        enable_nlp_privacy=args.enable_nlp_privacy,
    )
    if not chunk_records:
        raise RuntimeError("No chunks were sampled from the knowledge base.")

    texts = [str(record["content"]) for record in chunk_records]
    embedding_model = load_embedding_model(str(args.embedding_model))
    raw_embeddings = np.asarray(
        embedding_model.encode(texts, batch_size=16, show_progress_bar=True),
        dtype=np.float32,
    )

    queries = generate_random_queries(chunk_records, args.num_queries, args.query_seed)
    query_embeddings = np.asarray(
        embedding_model.encode(queries, batch_size=args.num_queries, show_progress_bar=False),
        dtype=np.float32,
    )
    projector = None
    if args.representation_mode == "no_jl":
        representation_embeddings = raw_embeddings
        query_representation = query_embeddings
    else:
        projector = JLProjector(
            target_dim=args.jl_target_dim,
            eps=args.jl_epsilon,
            random_state=args.jl_seed,
        )
        representation_embeddings = projector.fit_transform(raw_embeddings)
        query_representation = projector.transform(query_embeddings)

    print("\nAblation context")
    print("=" * 78)
    print(f"Knowledge base:             {args.knowledge_base}")
    print(f"Scanned readable documents: {doc_counter['count']}")
    print(f"Sampled chunks:             {len(chunk_records)}")
    print(f"Raw embedding shape:        {raw_embeddings.shape}")
    print(f"Representation mode:        {args.representation_mode}")
    print(f"Representation shape:       {representation_embeddings.shape}")
    print(f"Queries:                    {len(queries)}")

    return {
        "chunk_records": chunk_records,
        "raw_embeddings": raw_embeddings,
        "reduced_embeddings": representation_embeddings,
        "projector": projector,
        "query_reduced": query_representation,
    }


def run_ablation(args: argparse.Namespace) -> List[Dict[str, float | str]]:
    context = prepare_context(args)
    results: List[Dict[str, float | str]] = []

    for utility_scale in UTILITY_SCALE_LIST:
        config.DP_UTILITY_SCALE = float(utility_scale)
        print(f"\nUtility scale = {utility_scale:g}")
        print("-" * 78)
        scale_results: List[Dict[str, float | str]] = []

        if args.representation_mode == "no_jl":
            scale_results = _run_no_jl_scale(context, args, float(utility_scale))
            results.extend(scale_results)
            print_table(
                ["Scheme", "Scale", "NSR", "Overlap@5", "MRR@5", "Pearson", "MeanDrift", "DirCos"],
                metric_rows(scale_results),
            )
            continue

        for scheme_name, scheme_fn in SCHEMES:
            calibrator = AnalyticGaussianCalibrator(
                delta=args.dp_delta,
                utility_scale=float(utility_scale),
                random_state=args.noise_seed,
            )
            scheme_output = scheme_fn(
                context["raw_embeddings"],
                context["reduced_embeddings"],
                context["chunk_records"],
                context["projector"],
                calibrator,
                float(utility_scale),
            )
            metrics = evaluate_scheme_metrics(
                scheme_output,
                context["reduced_embeddings"],
                context["query_reduced"],
                float(utility_scale),
            )
            scale_results.append(metrics)
            results.append(metrics)

        print_table(
            ["Scheme", "Scale", "NSR", "Overlap@5", "MRR@5", "Pearson", "MeanDrift", "DirCos"],
            metric_rows(scale_results),
        )

    return results


def _run_no_jl_scale(
    context: Dict[str, object], args: argparse.Namespace, utility_scale: float,
) -> List[Dict[str, float | str]]:
    calibrator = AnalyticGaussianCalibrator(delta=args.dp_delta, utility_scale=utility_scale, random_state=args.noise_seed)
    raw_embeddings = context["raw_embeddings"]
    chunk_records = context["chunk_records"]
    if not isinstance(raw_embeddings, np.ndarray) or not isinstance(chunk_records, list):
        raise TypeError("Invalid No-JL ablation context")
    matched_epsilon, matched_sensitivity = matched_dynamic_calibration(chunk_records, calibrator)
    outputs = [
        run_full_current_no_jl(raw_embeddings, chunk_records, calibrator),
        run_no_dp_baseline_no_jl(raw_embeddings),
        run_no_dimension_aware_scaling_no_jl(
            raw_embeddings,
            chunk_records,
            AnalyticGaussianCalibrator(delta=args.dp_delta, utility_scale=utility_scale, random_state=args.noise_seed),
            utility_scale,
        ),
        run_fixed_dp_calibration_no_jl(
            raw_embeddings,
            AnalyticGaussianCalibrator(delta=args.dp_delta, utility_scale=utility_scale, random_state=args.noise_seed),
            utility_scale,
            matched_epsilon,
            matched_sensitivity,
        ),
    ]
    metadata: Dict[str, float | str | bool] = {
        "representation_mode": "no_jl",
        "uses_jl": False,
        "vector_dim": int(raw_embeddings.shape[1]),
        "sample_chunks": int(raw_embeddings.shape[0]),
        "num_queries": int(np.asarray(context["query_reduced"]).shape[0]),
        "query_seed": int(args.query_seed),
        "noise_seed": int(args.noise_seed),
        "dp_delta": float(args.dp_delta),
        "top_k": int(args.top_k),
        "ef_search": int(args.ef_search),
        "hnsw_M": int(args.M),
        "hnsw_ef_construction": int(args.ef_construction),
        "hnsw_space": str(args.hnsw_space),
        "hnsw_seed": int(args.hnsw_seed),
    }
    hnsw_config: Dict[str, float | int | str] = {
        "ef_search": int(args.ef_search),
        "M": int(args.M),
        "ef_construction": int(args.ef_construction),
        "space": str(args.hnsw_space),
        "random_seed": int(args.hnsw_seed),
    }
    return [
        evaluate_scheme_metrics(
            output,
            raw_embeddings,
            np.asarray(context["query_reduced"]),
            utility_scale,
            metadata,
            hnsw_config,
        )
        for output in outputs
    ]


def save_csv(results: List[Dict[str, float | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys()) if results else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def cleanup_legacy_ablation_figures() -> None:
    """Remove ablation figures left in Result_picture before path reorganization."""
    legacy_filenames = [
        "ablation_noise_signal_ratio_curve.png",
        "ablation_overlap_at_1_curve.png",
        "ablation_overlap_at_3_curve.png",
        "ablation_overlap_at_5_curve.png",
        "ablation_overlap_at_10_curve.png",
        "ablation_pearson_correlation_curve.png",
        "ablation_mean_absolute_drift_curve.png",
        "ablation_max_absolute_drift_curve.png",
        "ablation_mrr_at_5_curve.png",
        "ablation_direction_cosine_curve.png",
        "ablation_retrieval_time_curve.png",
        "ablation_dp_noise_time_curve.png",
    ]
    for filename in legacy_filenames:
        path = LEGACY_PICTURE_DIR / filename
        if path.is_file():
            path.unlink()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)
    if args.representation_mode == "jl":
        cleanup_legacy_ablation_figures()

    results = run_ablation(args)
    csv_path = RESULTS_DIR / ("no_jl_ablation_results.csv" if args.representation_mode == "no_jl" else "ablation_results.csv")
    save_csv(results, csv_path)
    figure_paths = (
        plot_no_jl_main(results, PICTURE_DIR / "no_jl")
        if args.representation_mode == "no_jl"
        else plot_all(results, PICTURE_DIR)
    )

    print("\nSaved ablation results:")
    print(f"  {csv_path}")
    print("\nSaved ablation figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
