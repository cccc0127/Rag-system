# Our DP-RAG Semantic Decomposition Experiments

This module decomposes where Our DP-RAG loses semantic retrieval fidelity after
the comparison framework switched Recall/MRR to a unified raw semantic reference.

It is not an ablation experiment. Ablation removes or changes internal design
choices to compare schemes. This experiment keeps the final Our DP-RAG pipeline
intact and measures the retrieval loss contributed by each transition.

## Stages

- **Stage A: Raw-Exact**  
  Raw 1024d embeddings and raw query embeddings are L2-normalized and searched
  with exact cosine similarity. This is the semantic ground truth.

- **Stage B: JL-Exact**  
  Raw embeddings and queries are projected by the same `JLProjector` into 256d
  and searched exactly. `B vs A` estimates JL semantic preservation.

- **Stage C: JL-DP-Exact**  
  JL document vectors are clipped, noised by the dynamic analytic Gaussian
  mechanism, and normalized. Queries stay in the same JL space and are not
  noised. `C vs B` estimates DP noise preservation inside JL space.

- **Stage D: JL-DP-HNSW**  
  Stage C document vectors and JL query vectors are searched by HNSW. `D vs C`
  estimates ANN backend fidelity.

## Comparisons

- `JL-Exact vs Raw-Exact`: JL loss.
- `JL-DP-Exact vs JL-Exact`: DP noise loss.
- `JL-DP-HNSW vs JL-DP-Exact`: HNSW approximation loss.
- `JL-DP-Exact vs Raw-Exact`: exact final semantic preservation.
- `JL-DP-HNSW vs Raw-Exact`: deployed final semantic preservation.

## Run

```bash
python3 semantic_decomposition_experiments/semantic_decomposition_runner.py
```

Use `--verbose` to print a detailed table.

## Outputs

- CSV results: `semantic_decomposition_experiments/results/semantic_decomposition_results.csv`
- Figures: `Result_picture/semantic_decomposition/`

## Joint JL Dimension x Utility Scale Tradeoff

The single-point decomposition above identifies where the loss comes from for
one operating point. The joint tradeoff experiment searches for a better
operating region by sweeping both `jl_target_dim` and `utility_scale`.

This is necessary because increasing JL dimension can preserve more raw
semantic topology, but it also increases vector storage, HNSW cost, and the
space where DP perturbation can affect local ranking. The goal is not simply to
maximize dimension or minimize noise. The goal is to find a practical balance
between semantic recall, privacy perturbation, and retrieval efficiency.

Run:

```bash
python3 semantic_decomposition_experiments/joint_tradeoff_runner.py
```

Default sweep:

- `jl_target_dim`: `128,256,384,512,768`
- `utility_scale`: `0.001,0.005,0.01,0.05,0.1`
- No-JL 1024d is included by default. Disable it with `--no-include-no-jl`.

Outputs:

- CSV results: `semantic_decomposition_experiments/results/joint_tradeoff_results.csv`
- Figures: `Result_picture/semantic_decomposition/joint_tradeoff/`

Interpretation:

- `final_hnsw_recall_at_5` high means the deployed private HNSW retrieval still
  matches raw semantic exact retrieval.
- `dp_loss_at_5` high means DP noise is starting to disrupt the JL-space
  retrieval order.
- `mean_noise_signal_ratio` measures perturbation strength.
- `mean_direction_cosine` close to 1 means the DP-noised vectors preserve the JL
  direction.
- `mean_hnsw_query_time` and `estimated_storage_mb` show deployment cost.
- Pareto scatter plots help choose an operating point instead of optimizing one
  metric in isolation.

### No-JL 1024d + Dimension-Aware DP

The joint tradeoff experiment also includes a special `No-JL` representation:

```text
Raw 1024d embeddings
-> L2 clipping
-> dynamic analytic Gaussian DP with sigma_per_dim = sigma * utility_scale / sqrt(1024)
-> final L2 normalization
-> HNSW retrieval
```

Queries stay in raw normalized 1024d space and are not noised.

This is not the same as setting `jl_target_dim=1024`. A 1024d JL projector would
still apply a random projection and change the geometry. `No-JL` completely
skips `JLProjector` and directly tests whether the dimension-aware noise scaling
alone can prevent high-dimensional Gaussian energy explosion while preserving
raw semantic topology.

This also differs from the old pipeline ablation:

- Old pipeline: `Raw 1024d -> DP -> JL 256d -> retrieval`
- No-JL test: `Raw 1024d -> DP -> retrieval`

Use the results as follows:

- If No-JL has high recall and low DP loss but much higher query/storage cost,
  JL is mainly an efficiency module.
- If No-JL also has high DP loss, the raw 1024d space itself is sensitive to
  perturbation despite dimension-aware scaling.
- If 768d JL approaches No-JL recall at lower cost, 768d is likely the better
  deployment balance.

The figure `representation_comparison_best_by_dim.png` selects the best
`utility_scale` for each representation under the configured DP-loss threshold
and is intended as the quickest operating-point summary.
