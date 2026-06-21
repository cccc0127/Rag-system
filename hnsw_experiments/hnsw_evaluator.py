"""Evaluate whether DP-RAG private embeddings transfer to HNSW retrieval."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from config import config
from dimension_reduction import JLProjector, l2_normalize
from evaluator import (
    apply_dp_noise,
    clean_for_panel,
    cosine_scores,
    count_iterated_documents,
    generate_random_queries,
    iter_documents_recursive,
    load_embedding_model,
    print_table,
    sample_chunks,
    top_k_indices,
    truncate,
)
from gaussian_noise import AnalyticGaussianCalibrator
from hnsw_experiments.hnsw_index import HNSWRetriever
from hnsw_experiments.plotting import plot_hnsw_curves


RESULT_DIR = ROOT_DIR / "Result_picture" / "hnsw"
EF_SEARCH_LIST = [16, 32, 64, 128, 256]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare exact cosine retrieval with HNSW on DP-RAG private vectors."
    )
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
    parser.add_argument("--utility-scale", type=float, default=getattr(config, "DP_UTILITY_SCALE", 0.01))
    parser.add_argument("--query-seed", type=int, default=2026)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--visual-text-chars", type=int, default=400)
    parser.add_argument(
        "--enable-nlp-privacy",
        action="store_true",
        help="Enable optional zero-shot privacy scoring. Disabled by default for offline evaluation.",
    )
    return parser.parse_args()


def prepare_context(args: argparse.Namespace) -> Dict[str, object]:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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
    reduced_raw_embeddings = projector.fit_transform(raw_embeddings)

    calibrator = AnalyticGaussianCalibrator(
        delta=args.dp_delta,
        utility_scale=args.utility_scale,
        random_state=args.noise_seed,
    )
    (
        final_noised_embeddings,
        clipped_embeddings,
        noise_vectors,
        sigmas,
        sigma_per_dim,
        epsilons,
    ) = apply_dp_noise(reduced_raw_embeddings, chunk_records, calibrator)

    queries = generate_random_queries(chunk_records, args.num_queries, args.query_seed)
    query_embeddings = np.asarray(
        embedding_model.encode(queries, batch_size=args.num_queries, show_progress_bar=False),
        dtype=np.float32,
    )
    query_reduced = projector.transform(query_embeddings)

    return {
        "doc_counter": doc_counter,
        "chunk_records": chunk_records,
        "raw_embeddings": raw_embeddings,
        "reduced_raw_embeddings": reduced_raw_embeddings,
        "final_noised_embeddings": l2_normalize(final_noised_embeddings),
        "clipped_embeddings": clipped_embeddings,
        "noise_vectors": noise_vectors,
        "sigmas": sigmas,
        "sigma_per_dim": sigma_per_dim,
        "epsilons": epsilons,
        "queries": queries,
        "query_reduced": l2_normalize(query_reduced),
    }


def exact_search(query_vector: np.ndarray, vectors: np.ndarray, top_k: int) -> np.ndarray:
    scores = cosine_scores(query_vector, vectors)
    return top_k_indices(scores, top_k)


def overlap_at_k(left: Sequence[int], right: Sequence[int], k: int) -> float:
    left_set = set(int(idx) for idx in list(left)[:k])
    right_set = set(int(idx) for idx in list(right)[:k])
    return len(left_set & right_set) / max(1, min(k, len(left_set)))


def evaluate_ef_search(
    args: argparse.Namespace,
    context: Dict[str, object],
    ef_search: int,
) -> Dict[str, float]:
    chunk_records = context["chunk_records"]
    final_noised_embeddings = context["final_noised_embeddings"]
    queries = context["queries"]
    query_reduced = context["query_reduced"]

    assert isinstance(chunk_records, list)
    assert isinstance(final_noised_embeddings, np.ndarray)
    assert isinstance(queries, list)
    assert isinstance(query_reduced, np.ndarray)

    build_start = time.perf_counter()
    retriever = HNSWRetriever(
        dim=final_noised_embeddings.shape[1],
        space=args.space,
        M=args.M,
        ef_construction=args.ef_construction,
        ef_search=ef_search,
        random_seed=args.random_seed,
    ).build(final_noised_embeddings)
    index_build_time = time.perf_counter() - build_start

    recall_values: Dict[int, List[float]] = {1: [], 3: [], 5: [], 10: []}
    exact_times: List[float] = []
    hnsw_times: List[float] = []
    rows: List[List[object]] = []
    top_records: List[Dict[str, object]] = []
    search_depth = min(max(args.top_k, 10), final_noised_embeddings.shape[0])

    for query_id, (query, query_vec) in enumerate(zip(queries, query_reduced), start=1):
        exact_start = time.perf_counter()
        exact_top = exact_search(query_vec, final_noised_embeddings, search_depth)
        exact_times.append(time.perf_counter() - exact_start)

        hnsw_start = time.perf_counter()
        hnsw_top, _ = retriever.search(query_vec, top_k=search_depth)
        hnsw_times.append(time.perf_counter() - hnsw_start)

        for k in recall_values:
            recall_values[k].append(overlap_at_k(exact_top, hnsw_top, k))

        if query_id == 1 and exact_top.size > 0 and hnsw_top.size > 0:
            exact_top1 = int(exact_top[0])
            hnsw_top1 = int(hnsw_top[0])
            top_records.append(
                {
                    "query_id": query_id,
                    "query": query,
                    "exact_top1": exact_top1,
                    "hnsw_top1": hnsw_top1,
                    "exact_text": str(chunk_records[exact_top1]["content"]),
                    "hnsw_text": str(chunk_records[hnsw_top1]["content"]),
                }
            )

        rows.append(
            [
                query_id,
                truncate(query, 42),
                ",".join(str(idx) for idx in exact_top[:5].tolist()),
                ",".join(str(idx) for idx in hnsw_top[:5].tolist()),
                f"{overlap_at_k(exact_top, hnsw_top, 5):.3f}",
            ]
        )

    mean_exact_time = float(np.mean(exact_times))
    mean_hnsw_time = float(np.mean(hnsw_times))
    speedup = mean_exact_time / max(mean_hnsw_time, 1e-12)
    result = {
        "ef_search": float(ef_search),
        "recall_at_1": float(np.mean(recall_values[1])),
        "recall_at_3": float(np.mean(recall_values[3])),
        "recall_at_5": float(np.mean(recall_values[5])),
        "recall_at_10": float(np.mean(recall_values[10])),
        "overlap_at_5": float(np.mean(recall_values[5])),
        "mean_exact_query_time": mean_exact_time,
        "mean_hnsw_query_time": mean_hnsw_time,
        "speedup_ratio": float(speedup),
        "index_build_time": float(index_build_time),
    }

    print(f"\nHNSW Retrieval Evaluation | ef_search={ef_search}")
    print("=" * 78)
    print_table(["Q", "Query", "Exact Top-5", "HNSW Top-5", "Overlap@5"], rows)
    print("\nAggregate HNSW Metrics")
    print_table(
        ["Metric", "Value"],
        [
            ["HNSW Recall@1 vs Exact", f"{result['recall_at_1']:.6f}"],
            ["HNSW Recall@3 vs Exact", f"{result['recall_at_3']:.6f}"],
            ["HNSW Recall@5 vs Exact", f"{result['recall_at_5']:.6f}"],
            ["HNSW Recall@10 vs Exact", f"{result['recall_at_10']:.6f}"],
            ["HNSW Overlap@5 vs Exact", f"{result['overlap_at_5']:.6f}"],
            ["Mean Query Time Exact", f"{result['mean_exact_query_time']:.8f}s"],
            ["Mean Query Time HNSW", f"{result['mean_hnsw_query_time']:.8f}s"],
            ["Speedup Ratio", f"{result['speedup_ratio']:.4f}x"],
            ["Index Build Time", f"{result['index_build_time']:.6f}s"],
        ],
    )
    if top_records:
        print_top1_panel(top_records[0], args.visual_text_chars)
    return result


def print_top1_panel(record: Dict[str, object], max_chars: int, width: int = 100) -> None:
    inner_width = width - 4
    border = "+" + "-" * (width - 2) + "+"
    title = f"Top-1 Semantic Alignment Panel | Query {record['query_id']}"
    fields = [
        ("Query", str(record["query"])),
        ("Exact Top-1 Chunk ID", str(record["exact_top1"])),
        ("HNSW Top-1 Chunk ID", str(record["hnsw_top1"])),
        ("Exact Top-1 Text", clean_for_panel(str(record["exact_text"]), max_chars)),
        ("HNSW Top-1 Text", clean_for_panel(str(record["hnsw_text"]), max_chars)),
    ]
    print("\nTop-1 Semantic Alignment Panel")
    print(border)
    print("| " + title.center(inner_width) + " |")
    print(border)
    for idx, (label, value) in enumerate(fields):
        line = f"{label}: {value}"
        for wrapped in _wrap_text(line, inner_width):
            print("| " + wrapped[:inner_width].ljust(inner_width) + " |")
        if idx < len(fields) - 1:
            print("| " + "-" * inner_width + " |")
    print(border)


def _wrap_text(text: str, width: int) -> List[str]:
    import textwrap

    lines: List[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return lines


def print_context_summary(args: argparse.Namespace, context: Dict[str, object]) -> None:
    doc_counter = context["doc_counter"]
    chunk_records = context["chunk_records"]
    raw_embeddings = context["raw_embeddings"]
    final_noised_embeddings = context["final_noised_embeddings"]
    queries = context["queries"]

    assert isinstance(doc_counter, dict)
    assert isinstance(chunk_records, list)
    assert isinstance(raw_embeddings, np.ndarray)
    assert isinstance(final_noised_embeddings, np.ndarray)
    assert isinstance(queries, list)

    print("\nDP-RAG HNSW Experiment Context")
    print("=" * 78)
    print(f"Knowledge base:             {args.knowledge_base}")
    print(f"Scanned readable documents: {doc_counter['count']}")
    print(f"Sampled chunks:             {len(chunk_records)}")
    print(f"Raw embedding shape:        {raw_embeddings.shape}")
    print(f"Private embedding shape:    {final_noised_embeddings.shape}")
    print(f"Queries:                    {len(queries)}")
    print(f"Utility scale:              {args.utility_scale}")
    print(f"HNSW space:                 {args.space}")
    print(f"HNSW M:                     {args.M}")
    print(f"HNSW ef_construction:       {args.ef_construction}")


def main() -> None:
    args = parse_args()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    context = prepare_context(args)
    print_context_summary(args, context)

    results = [evaluate_ef_search(args, context, ef_search) for ef_search in EF_SEARCH_LIST]

    print("\nHNSW ef_search Summary")
    print_table(
        [
            "ef_search",
            "Recall@1",
            "Recall@3",
            "Recall@5",
            "Recall@10",
            "Exact Time",
            "HNSW Time",
            "Speedup",
            "Build Time",
        ],
        [
            [
                f"{item['ef_search']:.0f}",
                f"{item['recall_at_1']:.6f}",
                f"{item['recall_at_3']:.6f}",
                f"{item['recall_at_5']:.6f}",
                f"{item['recall_at_10']:.6f}",
                f"{item['mean_exact_query_time']:.8f}s",
                f"{item['mean_hnsw_query_time']:.8f}s",
                f"{item['speedup_ratio']:.4f}x",
                f"{item['index_build_time']:.6f}s",
            ]
            for item in results
        ],
    )

    saved_paths = plot_hnsw_curves(results, RESULT_DIR)
    print("\nSaved HNSW figures:")
    for path in saved_paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
