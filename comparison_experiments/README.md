# Comparison Experiments

This folder is the external comparison framework for the DP-RAG project.
It is separate from:

- `evaluator.py`, which evaluates the current full method.
- `ablation_experiments/`, which studies internal component ablations.
- `hnsw_experiments/`, which validates HNSW retrieval migration.

The current implementation provides two schemes:

```text
Our DP-RAG + HNSW
DCPE+DCE + HNSW filter-refine
```

Future baselines should be added as new scheme adapters without changing the
shared data context or core DP modules.

## Current Pipeline

```text
shared context:
same chunks
same raw embeddings
same queries
same query embeddings

Our DP-RAG:
raw 1024d embeddings
-> JL projection 256d
-> L2 clipping
-> dynamic analytic Gaussian DP noise
-> final L2 normalization
-> HNSW retrieval

DCPE+DCE:
raw 1024d embeddings
-> L2 normalization
-> SAP/DCPE dense perturbation
-> HNSW filter over SAP vectors
-> exact-distance-equivalent refine over normalized raw vectors
```

The current DCPE+DCE adapter is a retrieval-behavior reproduction. It
implements SAP/DCPE filtering and emulates DCE refine with exact Euclidean
distance. It does not implement the full cryptographic DCE KeyGen, Enc,
TrapGen, or DistanceComp pipeline.

## Run

Recommended workflow:

```bash
python3 comparison_experiments/validation_tuner.py
python3 comparison_experiments/comparison_runner.py \
  --recommended-params comparison_experiments/results/recommended_params.json
```

The tuning stage selects medium/default parameters for each scheme on
validation queries. The formal comparison then fixes those method-internal
parameters and only varies shared comparison variables such as `ef_search`.

Run comparison directly:

```bash
python3 comparison_experiments/comparison_runner.py
```

Default settings:

- `sample_chunks=100`
- `validation num_queries=30`
- `test num_queries=30`
- `validation query_seed=2026`
- `test query_seed=2027`
- `top_k=5`
- `utility_scale=0.01`
- `ef_search=64`
- `ef_search_list=16,32,64,128,256`
- `M=16`
- `ef_construction=200`
- `dcpe_beta=0.5`
- `dcpe_ratio_k=4`

`ef_search` is the HNSW query-time candidate pool size. Larger values search
more graph candidates and usually improve recall at the cost of more query
time. It only changes the HNSW retrieval backend; it does not alter any
scheme's internal privacy or vector-transformation logic.

`dcpe_beta` and `dcpe_ratio_k` are DCPE+DCE internal parameters. They are not
used as cross-scheme x-axis variables in the main comparison figures.

## Outputs

CSV:

```text
comparison_experiments/results/comparison_results.csv
comparison_experiments/results/validation_tuning_results.csv
comparison_experiments/results/recommended_params.json
```

Figures:

```text
Result_picture/comparison/
├── default_config/
│   ├── comparison_recall_at_5.png
│   ├── comparison_mrr_at_5.png
│   ├── comparison_query_time.png
│   └── comparison_vector_dim.png
└── ef_search/
    ├── comparison_ef_search_recall_at_5.png
    ├── comparison_ef_search_mrr_at_5.png
    └── comparison_ef_search_query_time.png
└── validation_tuning/
    ├── tuning_our_dprag_recall_vs_utility_scale.png
    ├── tuning_our_dprag_nsr_vs_utility_scale.png
    ├── tuning_dcpe_dce_recall_vs_beta.png
    └── tuning_dcpe_dce_sap_nsr_vs_beta.png
```

The default-configuration bar charts compare schemes at the default
`ef_search`. The `ef_search` line charts show how each scheme behaves as the
HNSW search effort changes.

Recall@5 and MRR@5 use a unified semantic reference: exact Top-K retrieval over
raw L2-normalized embeddings. Each scheme is evaluated by how closely its final
retrieval results preserve the original semantic retrieval result, while HNSW
remains only the shared retrieval backend.

`comparison_runner.py` uses concise output by default and prints only saved
result paths. Pass `--verbose` to show context summaries, scheme reports,
ef_search tables, and the Top-1 semantic alignment panel.

`validation_tuner.py` also uses concise output by default. It tunes
`utility_scale` for `Our DP-RAG` and `beta` for `DCPE+DCE`; these are not main
comparison x-axis variables.

## Extension Rule

Do not force JL projection or DP noise into future baselines unless those steps
belong to the baseline itself. The shared framework standardizes the data,
queries, metrics, and retrieval protocol; each scheme keeps its own internal
method.

## Future Extensions

- `database_scale`: compare scalability as the number of chunks grows.
- `protection_level`: compare low/medium/high protection settings using
  scheme-specific parameter mappings.
- `security attack evaluation`: compare reconstruction, membership inference,
  and nearest-neighbor recovery risks.
