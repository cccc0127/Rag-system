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
from ablation_experiments.plotting import plot_all
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
    parser.add_argument("--enable-nlp-privacy", action="store_true")
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

    projector = JLProjector(
        target_dim=args.jl_target_dim,
        eps=args.jl_epsilon,
        random_state=args.jl_seed,
    )
    reduced_embeddings = projector.fit_transform(raw_embeddings)

    queries = generate_random_queries(chunk_records, args.num_queries, args.query_seed)
    query_embeddings = np.asarray(
        embedding_model.encode(queries, batch_size=args.num_queries, show_progress_bar=False),
        dtype=np.float32,
    )
    query_reduced = projector.transform(query_embeddings)

    print("\nAblation context")
    print("=" * 78)
    print(f"Knowledge base:             {args.knowledge_base}")
    print(f"Scanned readable documents: {doc_counter['count']}")
    print(f"Sampled chunks:             {len(chunk_records)}")
    print(f"Raw embedding shape:        {raw_embeddings.shape}")
    print(f"JL embedding shape:         {reduced_embeddings.shape}")
    print(f"Queries:                    {len(queries)}")

    return {
        "chunk_records": chunk_records,
        "raw_embeddings": raw_embeddings,
        "reduced_embeddings": reduced_embeddings,
        "projector": projector,
        "query_reduced": query_reduced,
    }


def run_ablation(args: argparse.Namespace) -> List[Dict[str, float | str]]:
    context = prepare_context(args)
    results: List[Dict[str, float | str]] = []

    for utility_scale in UTILITY_SCALE_LIST:
        config.DP_UTILITY_SCALE = float(utility_scale)
        print(f"\nUtility scale = {utility_scale:g}")
        print("-" * 78)
        scale_results: List[Dict[str, float | str]] = []

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
    cleanup_legacy_ablation_figures()

    results = run_ablation(args)
    csv_path = RESULTS_DIR / "ablation_results.csv"
    save_csv(results, csv_path)
    figure_paths = plot_all(results, PICTURE_DIR)

    print("\nSaved ablation results:")
    print(f"  {csv_path}")
    print("\nSaved ablation figures:")
    for path in figure_paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
