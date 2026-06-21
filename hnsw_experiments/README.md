# HNSW Retrieval Experiments

This folder adds an ANN retrieval branch for the DP-RAG pipeline without
changing the core privacy, chunking, embedding, or ablation modules.

## Pipeline

```text
Raw documents
-> chunk.py records with raw sensitivity score
-> BGE-M3 embeddings, 1024d
-> JL projection, 256d
-> L2 clipping
-> dynamic analytic Gaussian DP noise
-> final L2 normalization
-> exact retrieval and HNSW retrieval comparison
```

## Run

```bash
python3 hnsw_experiments/hnsw_evaluator.py
```

The script uses exact NumPy cosine retrieval as the reference result and HNSW
as the approximate index. If `hnswlib` is not installed, install it with:

```bash
python3 -m pip install hnswlib
```

## Outputs

Figures are written to:

```text
Result_picture/hnsw/
├── hnsw_recall_curve.png
├── hnsw_latency_curve.png
└── hnsw_speedup_curve.png
```

## Metrics

- `Recall@K`: fraction of exact Top-K ids recovered by HNSW Top-K.
- `Overlap@5`: same as Recall@5, kept as a retrieval-stability label.
- `Mean Query Time Exact`: brute-force exact cosine retrieval time.
- `Mean Query Time HNSW`: HNSW query time.
- `Speedup Ratio`: exact query time divided by HNSW query time.
- `Index Build Time`: time to build the HNSW graph index.
