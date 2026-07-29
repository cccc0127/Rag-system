# Targeted Sensitive Retrieval Exposure

This experiment evaluates retrieval-layer exposure: an attacker submits a
semantic query and observes the returned Top-K chunk identifiers. The attacker
targets chunks containing an email address, URL, or phone number, but the
attack query is built only after all enabled entity strings have been removed
from the source text.

The experiment never invokes an LLM, prints sensitive entities, writes chunk
text, or stores attack-query text in results. It is not an end-to-end RAG
output-leakage evaluation and does not establish an `(epsilon, delta)`-DP
guarantee.

## Threat model

- The attacker can issue retrieval queries and observe final Top-K chunk IDs.
- The attacker knows non-sensitive semantic context about a target document.
- The attacker does not include the target email, URL, or phone string in the
  query.
- This module evaluates only whether a sensitive chunk is retrieved.

CKKS is excluded here because it protects a different boundary: encrypted
distance computation. If a CKKS pipeline returns raw chunk IDs or text, its
retrieval exposure must be evaluated in a later CKKS-specific experiment that
also accounts for candidate-stage leakage.

## Schemes

By default the experiment evaluates:

```text
Vanilla Raw HNSW
Our DP-RAG-JL256
Private RAG-RP
DCPE+DCE
```

All schemes use the same sampled chunks, document embeddings, attack-query
embeddings, HNSW settings, and query seed. `Vanilla Raw HNSW` is the
unprotected exposure reference.

## Metrics

- `sensitive_target_recall_at_1`: the target sensitive chunk is ranked first.
- `sensitive_target_recall_at_5`: the target sensitive chunk appears in Top-5.
- `sensitive_top1_exposure_rate`: identical to target Recall@1 for this
  targeted setting; retained as an explicit exposure label.
- `mean_sensitive_chunks_at_5`: mean number of any sensitive chunks in Top-5.
- `mean_target_rank_when_retrieved`: target rank conditional on Top-5 retrieval.
- Normal-query Recall@5, MRR@5, query time, and index build time provide the
  security–utility comparison.

Lower sensitive retrieval/exposure values are better; higher normal Recall@5
is better.

## Run

Default:

```bash
python3 comparison_experiments/security_experiments/sensitive_retrieval/runner.py
```

Small run:

```bash
python3 comparison_experiments/security_experiments/sensitive_retrieval/runner.py \
  --sample-chunks 100 \
  --num-queries 20 \
  --top-k 5 \
  --ef-search 64 \
  --min-sensitive-chunks 5 \
  --verbose
```

Example using only email and URL targets:

```bash
python3 comparison_experiments/security_experiments/sensitive_retrieval/runner.py \
  --sensitive-types email,url \
  --max-sensitive-targets 50
```

## Outputs

```text
comparison_experiments/results/security_sensitive_retrieval_results.csv
Result_picture/comparison/security/sensitive_retrieval/
```

The CSV contains aggregate scheme statistics only. It must not contain source
text, sensitive entities, or generated attack queries.
