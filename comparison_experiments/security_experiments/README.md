# Security Experiments

This directory contains security evaluations that are intentionally separate
from the standard retrieval-performance comparisons. Each subdirectory states
its own threat model and measurement boundary.

The first experiment, `sensitive_retrieval/`, measures targeted sensitive
chunk exposure at the retrieval interface. It does not evaluate final LLM
answers, membership inference, embedding reconstruction, or CKKS computation
confidentiality.

`vector_linkage/` measures whether protected vectors can be linked to a known
candidate pool of original record identities under public scheme parameters.
It is separate from retrieval-interface exposure and does not reconstruct text.

`attribute_inference/` measures whether a classifier trained on disjoint,
similarly distributed auxiliary vectors can infer whether a protected document
vector contains a configured sensitive attribute. It uses strict stratified
out-of-fold evaluation and neither exposes raw text nor attempts identity
linkage, membership inference, reconstruction, or final-answer leakage.

`membership_inference/` measures whether protected document vectors disclose
target-index membership under source-document-disjoint shadow data. It reports
aggregate attack metrics only and is distinct from final-answer membership
leakage.
