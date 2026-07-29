"""Shared context preparation for comparison experiments."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import numpy as np

from comparison_experiments.comparison_config import (
    DEFAULT_SAMPLE_CHUNKS,
    DEFAULT_TEST_QUERIES,
    TEST_QUERY_SEED,
)
from config import config
from evaluator import (
    count_iterated_documents,
    generate_random_queries,
    iter_documents_recursive,
    load_embedding_model,
    sample_chunks,
)


@dataclass
class ComparisonContext:
    chunk_records: List[Dict[str, object]]
    raw_embeddings: np.ndarray
    queries: List[str]
    query_embeddings: np.ndarray
    metadata: Dict[str, object]
    additional_queries: List[str] | None = None
    additional_query_embeddings: np.ndarray | None = None


def add_context_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--knowledge-base", type=Path, default=config.REFERENCE_FOLDER)
    parser.add_argument("--embedding-model", default=config.EMBEDDING_MODEL)
    parser.add_argument("--sample-chunks", type=int, default=DEFAULT_SAMPLE_CHUNKS)
    parser.add_argument("--num-queries", type=int, default=DEFAULT_TEST_QUERIES)
    parser.add_argument("--chunk-size", type=int, default=getattr(config, "CHUNK_SIZE", 1000))
    parser.add_argument("--overlap", type=int, default=getattr(config, "OVERLAP", 200))
    parser.add_argument("--query-seed", type=int, default=TEST_QUERY_SEED)
    parser.add_argument(
        "--enable-nlp-privacy",
        action="store_true",
        help="Enable optional zero-shot privacy scoring. Disabled by default.",
    )


def prepare_comparison_context(
    args: argparse.Namespace,
    additional_query_builder: Callable[[List[Dict[str, object]]], List[str]] | None = None,
) -> ComparisonContext:
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
    model = load_embedding_model(str(args.embedding_model))
    raw_embeddings = np.asarray(
        model.encode(texts, batch_size=16, show_progress_bar=True),
        dtype=np.float32,
    )

    queries = generate_random_queries(chunk_records, args.num_queries, args.query_seed)
    additional_queries = additional_query_builder(chunk_records) if additional_query_builder else []
    all_queries = queries + additional_queries
    all_query_embeddings = np.asarray(
        model.encode(all_queries, batch_size=max(1, len(all_queries)), show_progress_bar=False),
        dtype=np.float32,
    )
    query_embeddings = all_query_embeddings[: len(queries)]
    additional_query_embeddings = all_query_embeddings[len(queries):] if additional_queries else None

    return ComparisonContext(
        chunk_records=chunk_records,
        raw_embeddings=raw_embeddings,
        queries=queries,
        query_embeddings=query_embeddings,
        metadata={
            "knowledge_base": str(args.knowledge_base),
            "embedding_model": str(args.embedding_model),
            "sampled_chunks": len(chunk_records),
            "scanned_readable_documents": int(doc_counter["count"]),
            "num_queries": len(queries),
            "num_additional_queries": len(additional_queries),
            "raw_embedding_dim": int(raw_embeddings.shape[1]),
        },
        additional_queries=additional_queries,
        additional_query_embeddings=additional_query_embeddings,
    )
