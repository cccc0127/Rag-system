# Protected-Vector Membership Inference

This experiment asks whether protected document vectors reveal whether a chunk
belongs to the target private index. It is separate from sensitive retrieval,
known-candidate linkage, attribute inference, text reconstruction, and
end-to-end RAG output leakage.

## Threat model

The attacker sees protected target vectors, knows public scheme parameters and
seeds, and has a source-document-disjoint auxiliary (shadow) corpus from the
same knowledge-base distribution. The attacker never receives target raw text,
raw embeddings, paths, record identities, target member labels, or per-vector
random-noise realizations. Target labels are used only after scoring to measure
attack success.

Documents are split before chunking into four source-document-disjoint groups:
`target_member`, `target_nonmember`, `shadow_member`, and
`shadow_nonmember`. Member/nonmember class sizes are balanced in both target
and shadow data. No source document may occur in more than one group.

CKKS is excluded because its security boundary is encrypted distance
computation, not the common plaintext protected-vector observation interface.
This experiment does not decrypt or export CKKS vectors.

## Attacks and metrics

- `shadow_logistic_regression`: `StandardScaler + LogisticRegression` trained
  only on shadow protected vectors. It is the primary attack.
- `shadow_density_knn`: exact L2-neighbor density score in L2-normalized
  protected-vector space. Its threshold is calibrated only on shadow data; it
  is a supporting distribution attack, not a replacement for the primary one.

Members are the positive class. Lower ROC-AUC, TPR at low FPR, and attack
advantage are better. The CSV reports ROC-AUC, TPR at the configured low FPR,
attack advantage `max(TPR-FPR)`, balanced accuracy, precision, recall, F1, and
reproducible 95% stratified bootstrap intervals for the first three metrics.

If fewer than 100 target nonmembers are used at the default FPR=1%, the CSV
sets `low_fpr_resolution_limited=true`. Such a TPR@1% value is not suitable for
a strong low-FPR claim.

An AUC close to 0.5 only means these attacks failed under this threat model; it
does not establish a formal differential-privacy guarantee. This experiment
also does not assess membership leakage through final RAG-generated answers.

## Run

Default:

```bash
python3 comparison_experiments/security_experiments/membership_inference/runner.py
```

Small smoke run:

```bash
python3 comparison_experiments/security_experiments/membership_inference/runner.py \
  --membership-samples-per-class 20 \
  --shadow-samples-per-class 20 \
  --bootstrap-samples 100 \
  --num-queries 10
```

Larger reporting run:

```bash
python3 comparison_experiments/security_experiments/membership_inference/runner.py \
  --membership-samples-per-class 500 \
  --shadow-samples-per-class 500 \
  --bootstrap-samples 1000 \
  --our-variant jl256
```

If the runner reports insufficient source-document-disjoint chunks, the
knowledge base lacks enough independent documents for the requested split. Do
not reduce this requirement by reusing documents across groups; reduce the
sample count or use a larger knowledge base.

## Outputs

```text
comparison_experiments/results/security_membership_inference_results.csv
Result_picture/comparison/security/membership_inference/
```

Only aggregate rows and figures are written. No content, path, chunk ID,
per-record label, vector, score, or prediction is persisted.
