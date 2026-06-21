# Comparison Experiments

This folder is the external comparison framework for the DP-RAG project.
It is separate from:

- `evaluator.py`, which evaluates the current full method.
- `ablation_experiments/`, which studies internal component ablations.
- `hnsw_experiments/`, which validates HNSW retrieval migration.

The current implementation provides the anchor scheme:

```text
Our DP-RAG + HNSW
```

Future baselines such as `DCPE+DCE` and other comparison schemes should be
added as new scheme adapters without changing the shared data context or core
DP modules.

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
```

## Run

```bash
python3 comparison_experiments/comparison_runner.py
```

Default settings:

- `sample_chunks=100`
- `num_queries=5`
- `top_k=5`
- `utility_scale=0.01`
- `ef_search=64`
- `ef_search_list=16,32,64,128,256`
- `M=16`
- `ef_construction=200`

`ef_search` is the HNSW query-time candidate pool size. Larger values search
more graph candidates and usually improve recall at the cost of more query
time. It only changes the HNSW retrieval backend; it does not alter any
scheme's internal privacy or vector-transformation logic.

## Outputs

CSV:

```text
comparison_experiments/results/comparison_results.csv
```

Figures:

```text
Result_picture/comparison/
├── default_config/
│   ├── comparison_query_time.png
│   └── comparison_vector_dim.png
└── ef_search/
    ├── comparison_ef_search_recall_at_1.png
    ├── comparison_ef_search_recall_at_3.png
    ├── comparison_ef_search_recall_at_5.png
    ├── comparison_ef_search_recall_at_10.png
    ├── comparison_ef_search_mrr_at_5.png
    ├── comparison_ef_search_query_time.png
    └── comparison_ef_search_index_build_time.png
```

The default-configuration bar charts compare schemes at the default
`ef_search`. The `ef_search` line charts show how each scheme behaves as the
HNSW search effort changes. With only `Our DP-RAG` implemented, each line chart
has one curve; adding future baselines will automatically add more curves.

## Extension Rule

Do not force JL projection or DP noise into future baselines unless those steps
belong to the baseline itself. The shared framework standardizes the data,
queries, metrics, and retrieval protocol; each scheme keeps its own internal
method.
