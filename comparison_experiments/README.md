# Comparison Experiments

This folder is the external comparison framework for the DP-RAG project.
It is separate from:

- `evaluator.py`, which evaluates the current full method.
- `ablation_experiments/`, which studies internal component ablations.
- `hnsw_experiments/`, which validates HNSW retrieval migration.

The current implementation provides several scheme adapters:

```text
Our DP-RAG-NoJL + HNSW
Our DP-RAG-JL768 + HNSW
Our DP-RAG-JL256 + HNSW
Private RAG-RP + HNSW
DCPE+DCE + HNSW filter-refine
PartialHE-CKKS-FullScan
HNSW+PartialHE-CKKS-Refine
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

Our DP-RAG variants:
raw 1024d embeddings
-> representation layer:
   - NoJL: raw normalized 1024d, high-fidelity mode
   - JL768: JL projection 768d, balanced mode
   - JL256: JL projection 256d, high-efficiency / historical mode
-> L2 clipping
-> dynamic analytic Gaussian DP noise
-> final L2 normalization
-> HNSW retrieval

Private RAG-RP:
raw 1024d embeddings
-> row-wise L2 normalization (satisfies paper gamma=1, Delta=2 defaults)
-> shared Gaussian random projection R, R_ij ~ N(0, 0.1^2), 1024d -> 64d
-> no post-projection L2 normalization and no DP noise
-> HNSW L2 retrieval

DCPE+DCE:
raw 1024d embeddings
-> L2 normalization
-> SAP/DCPE dense perturbation
-> HNSW filter over SAP vectors
-> exact-distance-equivalent refine over normalized raw vectors

PartialHE-CKKS-FullScan:
raw 1024d embeddings
-> L2 normalization
-> CKKS encrypted squared-L2 distance against every document vector
-> decrypt distances
-> sort exact full-scan candidates

HNSW+PartialHE-CKKS-Refine:
raw 1024d embeddings
-> L2 normalization
-> plaintext HNSW candidate generation over normalized embeddings
-> CKKS encrypted squared-L2 distance over Top-K' candidates
-> decrypt distances
-> refine to final Top-K
```

The current DCPE+DCE adapter is a retrieval-behavior reproduction. It
implements SAP/DCPE filtering and emulates DCE refine with exact Euclidean
distance. It does not implement the full cryptographic DCE KeyGen, Enc,
TrapGen, or DistanceComp pipeline.

The CKKS adapters reuse the usable TenSEAL logic from
`external_baselines/reward`: CKKS context setup, encrypted `(A-B)^2` distance,
absolute/relative HE error, and ciphertext/plaintext communication size. CKKS
does not provide DP noise, so `mean_noise_signal_ratio`, `mean_sigma`, and
`mean_epsilon` are intentionally left as `NaN`.

## Run

Recommended workflow:

```bash
python3 comparison_experiments/validation_tuner.py
python3 comparison_experiments/comparison_runner.py
```

The tuning stage selects medium/default parameters for each scheme on
validation queries. The formal comparison then fixes those method-internal
parameters and only varies shared comparison variables such as `ef_search`.
By default, `comparison_runner.py` automatically reads:

```text
comparison_experiments/results/recommended_params.json
```

If the file does not exist, the runner falls back to CLI/default parameters.

Disable recommended parameters:

```bash
python3 comparison_experiments/comparison_runner.py --no-recommended-params
```

Use a different recommendation file:

```bash
python3 comparison_experiments/comparison_runner.py \
  --recommended-params path/to/recommended_params.json
```

Run comparison directly:

```bash
python3 comparison_experiments/comparison_runner.py
```

By default, the runner evaluates:

```text
Our DP-RAG-NoJL
Our DP-RAG-JL768
Our DP-RAG-JL256
Private RAG-RP
DCPE+DCE
```

CKKS baselines are disabled by default because they are much slower than the
standard HNSW baselines.

Run a small HNSW+CKKS refine experiment:

```bash
python3 comparison_experiments/comparison_runner.py \
  --sample-chunks 100 \
  --num-queries 10 \
  --top-k 5 \
  --ef-search 64 \
  --ef-search-list 64 \
  --enable-ckks-refine \
  --ckks-ratio-k 4
```

Run an extreme small CKKS full-scan experiment:

```bash
python3 comparison_experiments/comparison_runner.py \
  --sample-chunks 10 \
  --num-queries 2 \
  --top-k 3 \
  --ef-search 16 \
  --ef-search-list 16 \
  --enable-ckks-fullscan \
  --no-recommended-params
```

Run only Our DP-RAG variants:

```bash
python3 comparison_experiments/comparison_runner.py \
  --disable-private-rag-rp \
  --disable-dcpe-dce
```

Disable the Private RAG-RP baseline:

```bash
python3 comparison_experiments/comparison_runner.py --disable-private-rag-rp
```

Use different Private RAG-RP projection parameters (the paper defaults are
`k=64`, `sigma=0.1`, and seed `42`):

```bash
python3 comparison_experiments/comparison_runner.py \
  --private-rag-rp-dim 128 \
  --private-rag-rp-sigma 0.1 \
  --private-rag-rp-seed 42
```

Customize Our DP-RAG variants:

```bash
python3 comparison_experiments/comparison_runner.py --our-variants no_jl,jl768
```

Default settings:

- `sample_chunks=100`
- `validation num_queries=30`
- `test num_queries=30`
- `validation query_seed=2026`
- `test query_seed=2027`
- `top_k=5`
- `utility_scale=0.01`
- `our_variants=no_jl,jl768,jl256`
- `ef_search=64`
- `ef_search_list=16,32,64,128,256`
- `M=16`
- `ef_construction=200`
- `dcpe_beta=0.5`
- `dcpe_ratio_k=4`
- `ckks_poly_modulus_degree=8192`
- `ckks_coeff_mod_bit_sizes=60,40,40,60`
- `ckks_global_scale=2**40`
- `ckks_ratio_k=4`
- `private_rag_rp_dim=64`
- `private_rag_rp_sigma=0.1`
- `private_rag_rp_seed=42`

`ef_search` is the HNSW query-time candidate pool size. Larger values search
more graph candidates and usually improve recall at the cost of more query
time. It only changes the HNSW retrieval backend; it does not alter any
scheme's internal privacy or vector-transformation logic.

`dcpe_beta` and `dcpe_ratio_k` are DCPE+DCE internal parameters. They are not
used as cross-scheme x-axis variables in the main comparison figures.

`ckks_ratio_k` controls the HNSW candidate pool for
`HNSW+PartialHE-CKKS-Refine`: `Top-K' = ckks_ratio_k * top_k`. FullScan ignores
`ef_search` for retrieval because it compares against every document vector.

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
└── ckks/
    ├── ckks_he_relative_error.png
    ├── ckks_he_absolute_error.png
    ├── ckks_cipher_expansion_ratio.png
    ├── ckks_ciphertext_size.png
    └── ckks_he_time.png
└── validation_tuning/
    ├── tuning_our_dprag_recall_vs_utility_scale.png
    ├── tuning_our_dprag_nsr_vs_utility_scale.png
    ├── tuning_dcpe_dce_recall_vs_beta.png
    └── tuning_dcpe_dce_sap_nsr_vs_beta.png
└── database_scale/
    ├── database_scale_recall_at_5.png
    ├── database_scale_mrr_at_5.png
    ├── database_scale_query_time.png
    ├── database_scale_index_build_time.png
    └── database_scale_vector_dim.png
    └── ckks/
        ├── ckks_scale_query_time.png
        ├── ckks_scale_he_time.png
        ├── ckks_scale_cipher_expansion_ratio.png
        └── ckks_scale_relative_error.png
```

The default-configuration bar charts compare schemes at the default
`ef_search`. The `ef_search` line charts show how each scheme behaves as the
HNSW search effort changes. CKKS baselines are excluded from the generic
`ef_search` curves because FullScan does not use HNSW retrieval, and Refine is
primarily reported through CKKS-specific figures.

Recall@5 and MRR@5 use a unified semantic reference: exact Top-K retrieval over
raw L2-normalized embeddings. Each scheme is evaluated by how closely its final
retrieval results preserve the original semantic retrieval result, while HNSW
remains only the shared retrieval backend.

CKKS rows share the same Recall@5, MRR@5, mean query time, and vector dimension
metrics. CKKS-specific metrics are:

- `he_relative_error_mean`: mean relative error between plaintext squared-L2
  distance and decrypted CKKS squared-L2 distance.
- `he_absolute_error_mean`: mean absolute error between plaintext and CKKS
  squared-L2 distance.
- `he_scan_time`: mean CKKS full-scan time per query.
- `he_refine_time`: mean CKKS refine time per query.
- `ciphertext_size_kb`: mean serialized ciphertext size.
- `plain_size_kb`: mean plaintext vector size.
- `cipher_expansion_ratio`: `ciphertext_size_kb / plain_size_kb`.

Security boundary:

- `PartialHE-CKKS-FullScan` performs CKKS distance over every document and does
  not use HNSW. It has a stronger encrypted-distance boundary but is suitable
  only for very small experiments.
- `HNSW+PartialHE-CKKS-Refine` uses plaintext HNSW to generate candidates, then
  CKKS only for candidate refinement. It is a scalable hybrid baseline, not
  fully encrypted HNSW.

The three Our DP-RAG variants share the same dynamic analytic Gaussian DP
mechanism and differ only in the representation layer before clipping/noise:

- `Our DP-RAG-NoJL`: highest semantic fidelity, highest vector dimension.
- `Our DP-RAG-JL768`: balanced compression mode.
- `Our DP-RAG-JL256`: historical high-compression mode.

`Private RAG-RP` is an empirical random-projection privacy baseline based on
the ICLR 2025 Building Trust Workshop paper *Private Retrieval Augmented
Generation with Random Projection*. It projects normalized document and query
embeddings with one shared Gaussian matrix and performs L2 HNSW retrieval. It
does not use this project's dynamic DP noise, sensitivity scoring, encryption,
or filter-refine processing. The paper does not provide a directly comparable
formal epsilon/delta calibration, so its `mean_epsilon`, `mean_sigma`, and NSR
fields are intentionally `NaN`; its projection sigma is only the random-matrix
scale. The fixed paper defaults are not included in `validation_tuner.py` or
`recommended_params.json` selection.

Current experiments compare retrieval utility, index build time, query time,
vector dimension, and database-scale behavior only. Attack, sensitive-data
leakage, membership-inference, and reconstruction evaluations are not yet
implemented; therefore these experiments must not be used to claim an attack
defense advantage for any scheme.

`comparison_runner.py` uses concise output by default and prints only saved
result paths. Pass `--verbose` to show context summaries, scheme reports,
ef_search tables, and the Top-1 semantic alignment panel.

`validation_tuner.py` also uses concise output by default. It tunes
`utility_scale` for `Our DP-RAG` and `beta` for `DCPE+DCE`; these are not main
comparison x-axis variables.

## Database Scale Experiment

The database scale experiment evaluates whether small-scale results remain
stable as the number of chunks grows. This is useful because at 100 chunks many
schemes can have similar Recall/MRR, while query time, index build time, and
filter-refine costs may only separate clearly at larger scales.

Run the default scale sweep:

```bash
python3 comparison_experiments/database_scale_runner.py
```

`database_scale_runner.py` uses the same automatic recommended-parameter
loading behavior. Disable it with:

```bash
python3 comparison_experiments/database_scale_runner.py --no-recommended-params
```

Default scale list:

```text
100,300,500,1000
```

Run a larger scale sweep:

```bash
python3 comparison_experiments/database_scale_runner.py \
  --sample-chunks-list 100,500,1000,2000,5000
```

Run a CKKS refine scale sweep:

```bash
python3 comparison_experiments/database_scale_runner.py \
  --sample-chunks-list 100,300,500 \
  --num-queries 10 \
  --top-k 5 \
  --ef-search 64 \
  --enable-ckks-refine \
  --ckks-ratio-k 4
```

Run CKKS full scan only at very small scale:

```bash
python3 comparison_experiments/database_scale_runner.py \
  --sample-chunks-list 10,20,50 \
  --num-queries 2 \
  --top-k 3 \
  --ef-search 16 \
  --enable-ckks-fullscan \
  --no-recommended-params
```

Outputs:

```text
comparison_experiments/results/database_scale_results.csv
Result_picture/comparison/database_scale/
```

Interpretation:

- Recall@5 and MRR@5 show whether semantic preservation degrades with more
  chunks.
- Mean query time measures online retrieval cost.
- Index build time measures offline indexing cost.
- If `Our DP-RAG-NoJL` keeps high recall but query/build time grows faster, it
  is a high-fidelity high-cost mode.
- If `Our DP-RAG-JL768` approaches NoJL recall at lower cost, it is the better
  deployment balance.
- If `DCPE+DCE` query time grows faster, the HNSW filter-refine cost is becoming
  visible at scale.

## Private RAG-RP Projection-Dimension Sensitivity

The main comparison keeps the paper-default Private RAG-RP setting at `k=64`.
To measure how this baseline's projection dimension affects retrieval utility
and runtime without changing any other experiment variable, run:

```bash
python3 comparison_experiments/private_rag_rp_sensitivity/runner.py
```

See `private_rag_rp_sensitivity/README.md` for the fixed variables, the
default `k=64,128,256,512,768` sweep, and output locations. This is a
performance sensitivity experiment only; it does not evaluate privacy attacks
or leakage.

## Security Experiments

The first retrieval-layer security evaluation measures whether de-identified
semantic attack queries retrieve chunks containing email addresses, URLs, or
phone numbers. It is separate from the normal utility experiments and does not
evaluate final LLM answer leakage:

```bash
python3 comparison_experiments/security_experiments/sensitive_retrieval/runner.py
```

See `security_experiments/sensitive_retrieval/README.md` for the threat model,
metrics, and output boundaries.

Known-candidate vector linkage is the second retrieval-layer security test:

```bash
python3 comparison_experiments/security_experiments/vector_linkage/runner.py
```

See `security_experiments/vector_linkage/README.md` for its white-box threat
model and recovery metrics.

The third retrieval-layer security test evaluates sensitive attribute inference
from protected document vectors using a disjoint auxiliary-data classifier:

```bash
python3 comparison_experiments/security_experiments/attribute_inference/runner.py
```

See `security_experiments/attribute_inference/README.md` for the strict
cross-validation threat model, ROC-AUC / TPR@1% FPR metrics, and the small
sample low-FPR limitation.

The fourth security test measures source-document-disjoint membership inference
against protected document vectors:

```bash
python3 comparison_experiments/security_experiments/membership_inference/runner.py
```

See `security_experiments/membership_inference/README.md` for its shadow-data
threat model, two attacks, bootstrap intervals, and low-FPR resolution limit.

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
