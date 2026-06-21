"""Console reporting for comparison experiments."""

from __future__ import annotations

import textwrap
from typing import Dict, Sequence

import numpy as np

from comparison_experiments.shared.context import ComparisonContext
from comparison_experiments.shared.retrievers import RetrievalResult
from comparison_experiments.schemes.our_dp_rag import SchemeOutput


def print_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    table = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in table)) if table else len(str(header))
        for idx, header in enumerate(headers)
    ]
    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(border)
    print("| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |")
    print(border)
    for row in table:
        print("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    print(border)


def print_context_summary(context: ComparisonContext) -> None:
    print("\nComparison Experiment Context")
    print("=" * 78)
    print_table(
        ["Field", "Value"],
        [
            ["Knowledge Base", context.metadata["knowledge_base"]],
            ["Embedding Model", context.metadata["embedding_model"]],
            ["Scanned Readable Documents", context.metadata["scanned_readable_documents"]],
            ["Sampled Chunks", context.metadata["sampled_chunks"]],
            ["Raw Embedding Shape", context.raw_embeddings.shape],
            ["Queries", context.metadata["num_queries"]],
        ],
    )


def print_scheme_report(
    context: ComparisonContext,
    scheme_output: SchemeOutput,
    retrieval: RetrievalResult,
    metrics: Dict[str, float | int | str],
    max_text_chars: int = 400,
) -> None:
    print("\nScheme Summary")
    print("=" * 78)
    print_table(
        ["Field", "Value"],
        [
            ["Scheme", scheme_output.name],
            ["Backend Type", scheme_output.backend_type],
            ["Document Vector Shape", scheme_output.document_vectors.shape],
            ["Query Vector Shape", scheme_output.query_vectors.shape],
            ["Vector Dim", scheme_output.vector_dim],
        ],
    )

    print("\nRetrieval Summary")
    print_table(
        ["Metric", "Value"],
        [
            ["Mean Query Time", f"{float(metrics['mean_query_time']):.8f}s"],
            ["Index Build Time", f"{float(metrics['index_build_time']):.6f}s"],
            ["HNSW Recall@1 vs Exact", f"{float(metrics['hnsw_recall_at_1']):.6f}"],
            ["HNSW Recall@3 vs Exact", f"{float(metrics['hnsw_recall_at_3']):.6f}"],
            ["HNSW Recall@5 vs Exact", f"{float(metrics['hnsw_recall_at_5']):.6f}"],
            ["HNSW Recall@10 vs Exact", f"{float(metrics['hnsw_recall_at_10']):.6f}"],
        ],
    )

    print("\nMetadata Summary")
    print_table(
        ["Metric", "Value"],
        [
            ["Mean Noise/Signal Ratio", f"{float(metrics['mean_noise_signal_ratio']):.6f}"],
            ["Mean Sigma", f"{float(metrics['mean_sigma']):.6f}"],
            ["Mean Epsilon", f"{float(metrics['mean_epsilon']):.6f}"],
            ["Utility Scale", scheme_output.metadata.get("utility_scale", "")],
            ["DP Delta", scheme_output.metadata.get("dp_delta", "")],
            ["JL Target Dim", scheme_output.metadata.get("jl_target_dim", "")],
        ],
    )

    print_top1_alignment_panel(context, retrieval, max_text_chars=max_text_chars)


def print_ef_search_summary(metrics_rows: Sequence[Dict[str, float | int | str]]) -> None:
    print("\nef_search Summary")
    print_table(
        [
            "Scheme",
            "ef_search",
            "Recall@5",
            "MRR@5",
            "Mean Query Time",
            "Index Build Time",
        ],
        [
            [
                item["scheme"],
                item["ef_search"],
                f"{float(item['hnsw_recall_at_5']):.6f}",
                f"{float(item['hnsw_mrr_at_5']):.6f}",
                f"{float(item['mean_query_time']):.8f}s",
                f"{float(item['index_build_time']):.6f}s",
            ]
            for item in metrics_rows
        ],
    )


def print_top1_alignment_panel(
    context: ComparisonContext,
    retrieval: RetrievalResult,
    max_text_chars: int = 400,
    width: int = 96,
) -> None:
    if retrieval.topk_indices.size == 0 or not context.queries:
        return
    query = context.queries[0]
    top1_id = int(retrieval.topk_indices[0, 0])
    raw_text = str(context.chunk_records[top1_id]["content"])
    title = "Top-1 Semantic Alignment Panel | Query 1"
    inner_width = width - 4
    border = "+" + "-" * (width - 2) + "+"

    print("\nTop-1 Semantic Alignment Panel")
    print(border)
    print("| " + title.center(inner_width) + " |")
    print(border)
    fields = [
        ("Query", query),
        ("HNSW Top-1 Chunk ID", str(top1_id)),
        ("Raw Text", _truncate_text(raw_text, max_text_chars)),
    ]
    for idx, (label, value) in enumerate(fields):
        for line in _wrap_field(label, value, inner_width):
            print("| " + line[:inner_width].ljust(inner_width) + " |")
        if idx < len(fields) - 1:
            print("| " + "-" * inner_width + " |")
    print(border)


def _truncate_text(text: str, max_chars: int) -> str:
    text = "\n".join(line.rstrip() for line in str(text).strip().splitlines())
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)].rstrip() + "..."


def _wrap_field(label: str, value: str, width: int) -> list[str]:
    prefix = f"{label}: "
    available = max(20, width - len(prefix))
    lines: list[str] = []
    for paragraph_idx, paragraph in enumerate(str(value).splitlines() or [""]):
        wrapped = textwrap.wrap(
            paragraph,
            width=available if paragraph_idx == 0 and not lines else width,
            replace_whitespace=False,
            drop_whitespace=False,
        ) or [""]
        for line_idx, line in enumerate(wrapped):
            if paragraph_idx == 0 and line_idx == 0 and not lines:
                lines.append(prefix + line)
            else:
                lines.append(line)
    return lines
