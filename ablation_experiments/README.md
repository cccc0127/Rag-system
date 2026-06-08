# DP-RAG Ablation Experiments

This folder contains internal ablation experiments for the current DP-RAG
pipeline. It does not modify the production pipeline modules.

Run:

```bash
python3 ablation_experiments/ablation_runner.py
```

Schemes:

- `Full Current`: JL -> clipping -> dynamic analytic Gaussian DP -> dimension-aware noise scaling -> final normalization.
- `No DP Baseline`: JL-only retrieval without privacy noise.
- `Old Pipeline DP Before JL`: applies DP noise in the original 1024d embedding space before JL projection.
- `No Dimension-Aware Scaling`: removes the `sqrt(dim)` correction and uses `sigma * utility_scale` per dimension.
- `Fixed DP Calibration`: replaces dynamic raw-score calibration with fixed epsilon and local sensitivity.

Outputs:

- `ablation_experiments/results/ablation_results.csv`
- `Result_picture/ablation/ablation_*.png`
