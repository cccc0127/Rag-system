"""Safe sensitive-chunk labeling and de-identified attack-query construction."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Iterable, Sequence

import numpy as np


ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[\w.%+-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
    "url": re.compile(r"\b(?:https?://|www\.)[^\s<>()\[\]{}\"']+", re.IGNORECASE),
    "phone": re.compile(
        r"(?<![\d+])(?:\+?\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?)?\d{3,4}[\s.-]\d{4}(?!\d)"
    ),
}
WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9_'-]{2,}|[\u4e00-\u9fff]{2,}")
VALID_SENSITIVE_TYPES = tuple(ENTITY_PATTERNS)


@dataclass(frozen=True)
class SensitiveChunk:
    chunk_id: int
    entity_types: tuple[str, ...]
    entity_count: int


@dataclass(frozen=True)
class AttackQuery:
    target_chunk_id: int
    entity_types: tuple[str, ...]
    query: str


def parse_sensitive_types(raw_value: str) -> tuple[str, ...]:
    requested = tuple(item.strip().lower() for item in str(raw_value).split(",") if item.strip())
    if not requested:
        raise ValueError("--sensitive-types must contain at least one type")
    invalid = sorted(set(requested) - set(VALID_SENSITIVE_TYPES))
    if invalid:
        raise ValueError(
            f"Unsupported sensitive types: {invalid}. Expected: {list(VALID_SENSITIVE_TYPES)}"
        )
    return tuple(dict.fromkeys(requested))


def find_sensitive_chunks(
    chunk_records: Sequence[Dict[str, object]], sensitive_types: Iterable[str]
) -> list[SensitiveChunk]:
    enabled = tuple(sensitive_types)
    results: list[SensitiveChunk] = []
    for chunk_id, record in enumerate(chunk_records):
        text = str(record.get("content", ""))
        counts = {kind: len(ENTITY_PATTERNS[kind].findall(text)) for kind in enabled}
        entity_types = tuple(kind for kind, count in counts.items() if count > 0)
        entity_count = sum(counts.values())
        if entity_types:
            results.append(SensitiveChunk(int(chunk_id), entity_types, int(entity_count)))
    return results


def redact_sensitive_entities(text: str, sensitive_types: Iterable[str]) -> str:
    redacted = str(text)
    for kind in sensitive_types:
        # Remove entity values rather than leaving type markers that could make
        # the attack query explicitly signal the sensitive field being targeted.
        redacted = ENTITY_PATTERNS[kind].sub(" ", redacted)
    return redacted


def build_attack_queries(
    chunk_records: Sequence[Dict[str, object]],
    sensitive_chunks: Sequence[SensitiveChunk],
    sensitive_types: Iterable[str],
    seed: int,
    max_terms: int = 8,
) -> list[AttackQuery]:
    """Create semantic queries after removing all enabled sensitive entities."""
    rng = np.random.default_rng(seed)
    attacks: list[AttackQuery] = []
    for target in sensitive_chunks:
        # Target selection may be restricted (for example, email only), but
        # attack-query construction always removes every supported direct
        # identifier so a different entity type cannot leak into the query.
        deidentified = redact_sensitive_entities(
            str(chunk_records[target.chunk_id].get("content", "")), VALID_SENSITIVE_TYPES
        )
        terms = list(dict.fromkeys(term.lower() for term in WORD_RE.findall(deidentified)))
        if len(terms) < 4:
            continue
        selected = rng.choice(terms, size=min(max_terms, len(terms)), replace=False)
        attacks.append(
            AttackQuery(
                target_chunk_id=target.chunk_id,
                entity_types=target.entity_types,
                query=" ".join(str(term) for term in selected),
            )
        )
    return attacks


def safe_sensitive_summary(sensitive_chunks: Sequence[SensitiveChunk]) -> dict[str, int]:
    summary: dict[str, int] = {kind: 0 for kind in VALID_SENSITIVE_TYPES}
    for target in sensitive_chunks:
        for kind in target.entity_types:
            summary[kind] += 1
    summary["total_sensitive_chunks"] = len(sensitive_chunks)
    return summary
