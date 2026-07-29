# Protected-Vector Sensitive Attribute Inference

This third security experiment tests whether an attacker can infer a binary
sensitive attribute from a document vector. It is distinct from the first
experiment's retrieval exposure and the second experiment's known-candidate
identity linkage.

## Threat model

- The attacker obtains protected document vectors for a sampled collection.
- The attacker knows the scheme and its public configuration.
- The attacker has labeled auxiliary data from the same distribution, modeled
  by the training folds of a stratified cross-validation split.
- Each held-out target vector is never used to train the classifier that
  predicts it.
- The attacker does **not** receive raw target text, raw target embeddings, a
  known candidate identity database, a per-vector noise realization, or final
  LLM outputs.

The attacker trains `StandardScaler + LogisticRegression(class_weight="balanced")`
on each auxiliary fold and predicts the held-out fold. This is a vector-space
attribute inference evaluation; it is not membership inference, text
reconstruction, retrieval-interface exposure, or a formal DP proof.

## Attributes and metrics

By default, labels are computed in memory from the same email/URL/phone regex
definitions used by the first security experiment:

- `has_email`, `has_url`, `has_phone`
- `has_any_sensitive` (the primary aggregate target)

Only aggregate labels and results are written. Raw chunks, entity values,
chunk identifiers, record-level predictions, and classifier scores are never
printed or saved.

- `roc_auc`: discrimination across all thresholds. `0.5` is random guessing;
  lower is better for protection.
- `tpr_at_fpr_1pct`: attacker detection rate while allowing at most 1% false
  positives; lower is better.
- `macro_f1`, `macro_precision`, `macro_recall`: balanced binary prediction
  quality; lower values usually indicate less attribute leakage, but must be
  interpreted alongside class imbalance.
- `*_fold_mean` / `*_fold_std`: variation across the 5 strict hold-out folds.

With fewer than 100 negative chunks, 1% FPR cannot be resolved precisely. The
CSV marks such rows with `low_fpr_resolution_limited=true`; use a larger sample
for a paper-quality low-FPR claim.

CKKS is intentionally excluded: its ciphertext/confidentiality boundary is
not the plaintext protected-vector interface evaluated here.

## Run

Default smoke-scale run:

```bash
python3 comparison_experiments/security_experiments/attribute_inference/runner.py
```

Larger, more meaningful run:

```bash
python3 comparison_experiments/security_experiments/attribute_inference/runner.py \
  --sample-chunks 1000 \
  --num-queries 30 \
  --cv-folds 5 \
  --min-attribute-positives 10 \
  --our-variant jl256
```

Use a subset of label types when a corpus lacks enough positives:

```bash
python3 comparison_experiments/security_experiments/attribute_inference/runner.py \
  --sensitive-types email,url \
  --sample-chunks 500
```

## Outputs

```text
comparison_experiments/results/security_attribute_inference_results.csv
Result_picture/comparison/security/attribute_inference/
```

The primary plots show any-sensitive ROC-AUC, TPR@1% FPR, Macro F1, and a
normal Recall@5 versus ROC-AUC security–utility trade-off. A separate grouped
plot reports ROC-AUC for each eligible sensitive attribute type.
