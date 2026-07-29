# Known-Candidate Protected-Vector Linkage

This experiment measures whether an attacker who obtains a protected vector
index can link each vector back to a candidate original document identity.
It is an index-vector identity-linkage evaluation, not a text-inversion,
membership-inference, final-answer leakage, or formal DP evaluation.

## Threat model

- The attacker has protected document vectors and a candidate raw-document
  embedding collection.
- The attacker knows the embedding model, algorithms, public parameters, and
  random seeds.
- The attacker does not know the per-vector random noise realizations in Our
  DP-RAG or DCPE+DCE.
- For Private RAG-RP, the shared Gaussian projection matrix is public because
  its seed and parameters are public. A high recovery rate in this setting is
  an important result: the baseline must not rely on projection secrecy.
- The candidate pool is the sampled raw chunk collection. The attacker knows
  possible document identities but not the protected-index correspondence.

The attacker reconstructs each scheme's public deterministic pre-noise
representation, then uses exact cosine or L2 nearest-neighbor matching. Exact
matching avoids HNSW approximation from affecting the security conclusion.

## Metrics

- `linkage_top1_recovery_rate`: correct chunk identity is the first candidate.
- `linkage_recall_at_5`: correct identity occurs within Top-5 candidates.
- `linkage_mrr_at_5`: reciprocal rank of the correct identity within Top-5.
- `sensitive_*`: the same metrics restricted to chunks containing enabled
  email, URL, or phone patterns.

Lower recovery metrics indicate that the protected vectors are harder to link
to candidate original records. Low linkage does not itself imply formal DP.

The normal-query retrieval metrics in the output support a security–utility
trade-off plot. Its preferred region is lower-right: high normal Recall@5 and
low sensitive Top-1 linkage recovery.

CKKS is excluded because encrypted-distance security cannot be tested with the
same plaintext vector-linkage interface. A later CKKS experiment must instead
evaluate ciphertext visibility and, for hybrid schemes, plaintext candidate
leakage.

## Run

Default:

```bash
python3 comparison_experiments/security_experiments/vector_linkage/runner.py
```

Small run:

```bash
python3 comparison_experiments/security_experiments/vector_linkage/runner.py \
  --sample-chunks 100 \
  --num-queries 20 \
  --min-sensitive-chunks 5 \
  --verbose
```

## Outputs

```text
comparison_experiments/results/security_vector_linkage_results.csv
Result_picture/comparison/security/vector_linkage/
```

Only aggregate statistics are saved. The CSV, images, and console output must
not contain raw chunk text, sensitive entity values, or per-record mappings.
