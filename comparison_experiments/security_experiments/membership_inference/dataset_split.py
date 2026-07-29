"""Source-document-disjoint data preparation for membership inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

from evaluator import sample_chunks


GROUP_NAMES = ("target_member", "target_nonmember", "shadow_member", "shadow_nonmember")


@dataclass(frozen=True)
class MembershipDataSplit:
    groups: dict[str, list[Dict[str, object]]]

    def source_document_sets(self) -> dict[str, set[str]]:
        return {name: {str(record.get("filename", "")) for record in records} for name, records in self.groups.items()}


def split_source_documents(
    documents: Sequence[Dict[str, str]], seed: int
) -> dict[str, list[Dict[str, str]]]:
    """Partition source documents before chunking; no document appears twice."""
    if len(documents) < len(GROUP_NAMES):
        raise ValueError("Need at least four source documents for document-disjoint membership inference")
    filenames = [str(document.get("filename", "")) for document in documents]
    if len(set(filenames)) != len(filenames) or any(not item for item in filenames):
        raise ValueError("Source documents require unique non-empty filenames for safe splitting")
    order = np.random.default_rng(seed).permutation(len(documents))
    partitions = np.array_split(order, len(GROUP_NAMES))
    return {
        name: [dict(documents[int(index)]) for index in partition]
        for name, partition in zip(GROUP_NAMES, partitions)
    }


def build_membership_data_split(
    documents: Iterable[Dict[str, str]],
    membership_samples_per_class: int,
    shadow_samples_per_class: int,
    chunk_size: int,
    overlap: int,
    enable_nlp_privacy: bool,
    split_seed: int,
) -> MembershipDataSplit:
    """Chunk four source-disjoint document groups with balanced class sample caps."""
    if membership_samples_per_class <= 0 or shadow_samples_per_class <= 0:
        raise ValueError("Samples per class must be greater than zero")
    requested = {
        "target_member": membership_samples_per_class,
        "target_nonmember": membership_samples_per_class,
        "shadow_member": shadow_samples_per_class,
        "shadow_nonmember": shadow_samples_per_class,
    }
    # Assign each source document to one shuffled group in a streaming pass.
    # This avoids retaining a potentially very large knowledge base in memory,
    # while preserving deterministic source-document disjointness and stopping
    # as soon as every balanced class has enough chunks.
    groups: dict[str, list[Dict[str, object]]] = {name: [] for name in GROUP_NAMES}
    seen_sources: set[str] = set()
    rng = np.random.default_rng(split_seed)
    assignment_cycle: list[str] = []
    for document in documents:
        if not assignment_cycle:
            assignment_cycle = [GROUP_NAMES[int(index)] for index in rng.permutation(len(GROUP_NAMES))]
        name = assignment_cycle.pop()
        filename = str(document.get("filename", ""))
        if not filename or filename in seen_sources:
            raise ValueError("Source documents require unique non-empty filenames for safe splitting")
        seen_sources.add(filename)
        remaining = requested[name] - len(groups[name])
        if remaining > 0:
            groups[name].extend(
                sample_chunks(
                    iter((document,)), max_chunks=remaining, chunk_size=chunk_size,
                    overlap=overlap, enable_nlp_privacy=enable_nlp_privacy,
                )
            )
        if all(len(groups[group]) >= requested[group] for group in GROUP_NAMES):
            break
    for name in GROUP_NAMES:
        if len(groups[name]) < requested[name]:
            raise RuntimeError(
                "Insufficient source-document-disjoint chunks for membership inference: "
                f"{name} has {len(groups[name])}, requires {requested[name]}. "
                "Increase source documents, reduce samples per class, or use a larger knowledge base."
            )
    result = MembershipDataSplit(groups=groups)
    _validate_disjoint_sources(result)
    return result


def _validate_disjoint_sources(data_split: MembershipDataSplit) -> None:
    source_sets = data_split.source_document_sets()
    for index, left_name in enumerate(GROUP_NAMES):
        for right_name in GROUP_NAMES[index + 1:]:
            if source_sets[left_name] & source_sets[right_name]:
                raise RuntimeError("Membership data split reused a source document across groups")
