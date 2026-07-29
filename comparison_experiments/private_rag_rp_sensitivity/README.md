# Private RAG-RP Projection-Dimension Sensitivity

This module studies how the Private RAG-RP baseline changes as its random
projection dimension `k` changes. It is a parameter-sensitivity experiment,
not a new privacy mechanism.

For every setting, the experiment uses the same sampled chunks, queries, raw
embeddings, HNSW configuration, projection-matrix scale (`sigma=0.1`), and
random seed. Only `k` changes. The default sweep is:

```text
64,128,256,512,768
```

It reports Recall@5, MRR@5, mean query time, index build time, and vector
dimension. No extraction, leakage, membership-inference, reconstruction, or
generation-quality evaluation is performed; these plots cannot establish a
privacy-protection level.

The main comparison experiment remains a faithful paper-default baseline at
`k=64`. This sensitivity sweep must not replace that main-comparison setting or
be used to automatically select a new default k.

## Run

Default experiment:

```bash
python3 comparison_experiments/private_rag_rp_sensitivity/runner.py
```

Small validation run:

```bash
python3 comparison_experiments/private_rag_rp_sensitivity/runner.py \
  --sample-chunks 20 \
  --num-queries 5 \
  --k-list 64,128 \
  --ef-search 16 \
  --verbose
```

Custom projection-dimension sweep:

```bash
python3 comparison_experiments/private_rag_rp_sensitivity/runner.py \
  --k-list 64,128,256,512,768 \
  --private-rag-rp-sigma 0.1 \
  --private-rag-rp-seed 42
```

## Outputs

```text
comparison_experiments/results/private_rag_rp_sensitivity_results.csv
Result_picture/comparison/private_rag_rp_sensitivity/
```
