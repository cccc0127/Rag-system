"""Binary sensitive-attribute labels without exposing entity values."""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np

from comparison_experiments.security_experiments.sensitive_retrieval.sensitive_data import (
    ENTITY_PATTERNS,
    VALID_SENSITIVE_TYPES,
)


def build_sensitive_attribute_labels(
    chunk_records: Sequence[Dict[str, object]], sensitive_types: Sequence[str]
) -> dict[str, np.ndarray]:
    """Return one binary label vector per enabled type plus ``has_any_sensitive``.

    Labels are retained only in memory for the attack evaluation. Callers must
    report aggregate counts and metrics, never content, matched entity strings,
    or record-level labels.
    """
    enabled = tuple(sensitive_types)
    invalid = sorted(set(enabled) - set(VALID_SENSITIVE_TYPES))
    if invalid:
        raise ValueError(f"Unsupported sensitive types: {invalid}")
    labels: dict[str, np.ndarray] = {}
    for kind in enabled:
        labels[f"has_{kind}"] = np.asarray(
            [bool(ENTITY_PATTERNS[kind].search(str(record.get("content", "")))) for record in chunk_records],
            dtype=np.int64,
        )
    if labels:
        labels["has_any_sensitive"] = np.maximum.reduce(list(labels.values())).astype(np.int64)
    else:
        labels["has_any_sensitive"] = np.zeros(len(chunk_records), dtype=np.int64)
    return labels


def aggregate_label_counts(labels: dict[str, np.ndarray]) -> dict[str, int]:
    """Return only aggregate positive counts for safe console reporting."""
    return {name: int(np.asarray(values, dtype=np.int64).sum()) for name, values in labels.items()}
