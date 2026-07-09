"""Validation tuning for comparison experiment default parameters."""

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

from comparison_experiments.comparison_runner import run_scheme_retrieval
from comparison_experiments.comparison_config import (
    DEFAULT_EF_SEARCH,
    DEFAULT_SAMPLE_CHUNKS,
    DEFAULT_TOP_K,
    DEFAULT_VALIDATION_QUERIES,
    VALIDATION_QUERY_SEED,
)
from comparison_experiments.plotting import plot_validation_tuning_figures
from comparison_experiments.shared.context import add_context_args, prepare_comparison_context
from comparison_experiments.shared.metrics import compute_scheme_metrics
from comparison_experiments.schemes.dcpe_dce import DCPEDCEScheme
from comparison_experiments.schemes.our_dp_rag import OurDPRAGScheme
from config import config


RESULTS_DIR = ROOT_DIR / "comparison_experiments" / "results"
PICTURE_DIR = ROOT_DIR / "Result_picture" / "comparison" / "validation_tuning"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune medium/default parameters for comparison schemes."
    )
    add_context_args(parser)
    parser.set_defaults(
        sample_chunks=DEFAULT_SAMPLE_CHUNKS,
        num_queries=DEFAULT_VALIDATION_QUERIES,
        query_seed=VALIDATION_QUERY_SEED,
    )
    parser.add_argument("--utility-scale-list", default="0.001,0.01,0.05,0.1")
    parser.add_argument("--dcpe-beta-list", default="0.1,0.25,0.5,1.0,2.0")
    parser.add_argument("--dcpe-ratio-k", type=int, default=4)
    parser.add_argument("--ef-search", type=int, default=DEFAULT_EF_SEARCH)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--M", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-space", choices=["cosine", "ip", "l2"], default="cosine")
    parser.add_argument("--hnsw-seed", type=int, default=42)
    parser.add_argument("--dp-delta", type=float, default=getattr(config, "DP_DELTA", 1e-5))
    parser.add_argument("--noise-seed", type=int, default=getattr(config, "DP_RANDOM_SEED", 42))
    parser.add_argument("--jl-target-dim", type=int, default=getattr(config, "JL_TARGET_DIM", 256))
    parser.add_argument("--jl-epsilon", type=float, default=getattr(config, "JL_EPSILON", 0.3))
    parser.add_argument("--jl-seed", type=int, default=getattr(config, "JL_RANDOM_SEED", 42))
    parser.add_argument("--dcpe-seed", type=int, default=42)
    parser.add_argument("--nsr-min-threshold", type=float, default=0.003)
    parser.add_argument("--sap-nsr-min-threshold", type=float, default=0.003)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PICTURE_DIR.mkdir(parents=True, exist_ok=True)

    context = prepare_comparison_context(args)
    all_metrics: list[Dict[str, float | int | str]] = []

    for utility_scale in parse_float_list(args.utility_scale_list):
        scheme = OurDPRAGScheme(
            jl_target_dim=args.jl_target_dim,
            jl_epsilon=args.jl_epsilon,
            jl_seed=args.jl_seed,
            dp_delta=args.dp_delta,
            utility_scale=utility_scale,
            noise_seed=args.noise_seed,
        )
        output = scheme.run(
            raw_embeddings=context.raw_embeddings,
            query_embeddings=context.query_embeddings,
            chunk_records=context.chunk_records,
        )
        retrieval = run_scheme_retrieval(
            scheme_output=output,
            top_k=args.top_k,
            ef_search=args.ef_search,
            M=args.M,
            ef_construction=args.ef_construction,
            default_space=args.hnsw_space,
            random_seed=args.hnsw_seed,
        )
        metrics = compute_scheme_metrics(output, retrieval, top_k=args.top_k, ef_search=args.ef_search)
        metrics["utility_scale"] = float(utility_scale)
        metrics["tuned_parameter"] = "utility_scale"
        all_metrics.append(metrics)

    for beta in parse_float_list(args.dcpe_beta_list):
        scheme = DCPEDCEScheme(
            beta=beta,
            ratio_k=args.dcpe_ratio_k,
            random_seed=args.dcpe_seed,
        )
        output = scheme.run(
            raw_embeddings=context.raw_embeddings,
            query_embeddings=context.query_embeddings,
            chunk_records=context.chunk_records,
        )
        retrieval = run_scheme_retrieval(
            scheme_output=output,
            top_k=args.top_k,
            ef_search=args.ef_search,
            M=args.M,
            ef_construction=args.ef_construction,
            default_space=args.hnsw_space,
            random_seed=args.hnsw_seed,
        )
        metrics = compute_scheme_metrics(output, retrieval, top_k=args.top_k, ef_search=args.ef_search)
        metrics["beta"] = float(beta)
        metrics["ratio_k"] = int(args.dcpe_ratio_k)
        metrics["tuned_parameter"] = "beta"
        all_metrics.append(metrics)

    recommended = build_recommended_params(
        all_metrics,
        nsr_min_threshold=args.nsr_min_threshold,
        sap_nsr_min_threshold=args.sap_nsr_min_threshold,
    )
    csv_path = RESULTS_DIR / "validation_tuning_results.csv"
    json_path = RESULTS_DIR / "recommended_params.json"
    save_metrics_csv(all_metrics, csv_path)
    save_recommended_params(recommended, json_path)
    figure_paths = plot_validation_tuning_figures(
        all_metrics,
        PICTURE_DIR,
        nsr_min_threshold=args.nsr_min_threshold,
        sap_nsr_min_threshold=args.sap_nsr_min_threshold,
    )

    if args.verbose:
        print(f"Tuned {len(all_metrics)} validation configurations.")

    print("\nSaved validation tuning results:")
    print(f"- {csv_path}")
    print(f"- {json_path}")
    print("\nSaved validation tuning figures:")
    for path in figure_paths:
        print(f"- {path}")


def build_recommended_params(
    metrics: Sequence[Dict[str, float | int | str]],
    nsr_min_threshold: float,
    sap_nsr_min_threshold: float,
) -> Dict[str, object]:
    our_rows = [row for row in metrics if str(row["scheme"]) == "Our DP-RAG"]
    dcpe_rows = [row for row in metrics if str(row["scheme"]) == "DCPE+DCE"]
    our_selected, our_fallback = select_by_threshold(
        our_rows,
        threshold_key="mean_noise_signal_ratio",
        threshold=nsr_min_threshold,
    )
    dcpe_selected, dcpe_fallback = select_by_threshold(
        dcpe_rows,
        threshold_key="sap_noise_signal_ratio",
        threshold=sap_nsr_min_threshold,
    )

    recommended: Dict[str, object] = {}
    if our_selected is not None:
        entry: Dict[str, object] = {
            "utility_scale": float(our_selected["utility_scale"]),
            "selection_rule": "max Recall@5 subject to mean_noise_signal_ratio >= threshold",
            "nsr_min_threshold": float(nsr_min_threshold),
            "selected_metrics": selected_metrics(
                our_selected,
                ["hnsw_recall_at_5", "hnsw_mrr_at_5", "mean_noise_signal_ratio"],
            ),
        }
        if our_fallback:
            entry["fallback_reason"] = "no candidate satisfied mean_noise_signal_ratio threshold"
        recommended["Our DP-RAG"] = entry

    if dcpe_selected is not None:
        entry = {
            "beta": float(dcpe_selected["beta"]),
            "ratio_k": int(float(dcpe_selected["ratio_k"])),
            "selection_rule": "max Recall@5 subject to sap_noise_signal_ratio >= threshold",
            "sap_nsr_min_threshold": float(sap_nsr_min_threshold),
            "selected_metrics": selected_metrics(
                dcpe_selected,
                ["hnsw_recall_at_5", "hnsw_mrr_at_5", "sap_noise_signal_ratio"],
            ),
        }
        if dcpe_fallback:
            entry["fallback_reason"] = "no candidate satisfied sap_noise_signal_ratio threshold"
        recommended["DCPE+DCE"] = entry

    return recommended


def select_by_threshold(
    rows: Sequence[Dict[str, float | int | str]],
    threshold_key: str,
    threshold: float,
) -> tuple[Dict[str, float | int | str] | None, bool]:
    if not rows:
        return None, False
    valid = [
        row for row in rows
        if is_number(row.get(threshold_key)) and float(row[threshold_key]) >= threshold
    ]
    if valid:
        return max(valid, key=lambda row: (float(row["hnsw_recall_at_5"]), float(row["hnsw_mrr_at_5"]))), False
    fallback_rows = [row for row in rows if is_number(row.get(threshold_key))]
    if fallback_rows:
        return max(fallback_rows, key=lambda row: (float(row[threshold_key]), float(row["hnsw_recall_at_5"]))), True
    return max(rows, key=lambda row: float(row["hnsw_recall_at_5"])), True


def selected_metrics(row: Dict[str, float | int | str], keys: Sequence[str]) -> Dict[str, float]:
    return {key: float(row[key]) for key in keys if is_number(row.get(key))}


def is_number(value: object) -> bool:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return False
    return not np_is_nan(number)


def np_is_nan(value: float) -> bool:
    return value != value


def parse_float_list(raw_value: str) -> list[float]:
    values: list[float] = []
    for part in str(raw_value).split(","):
        part = part.strip()
        if not part:
            continue
        value = float(part)
        if value <= 0.0:
            raise ValueError("list values must be positive")
        values.append(value)
    if not values:
        raise ValueError("list must contain at least one positive value")
    return sorted(set(values))


def save_metrics_csv(metrics: Sequence[Dict[str, float | int | str]], output_path: Path) -> None:
    if not metrics:
        return
    fieldnames = sorted({key for row in metrics for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metrics)


def save_recommended_params(params: Dict[str, object], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(params, handle, indent=2, sort_keys=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
