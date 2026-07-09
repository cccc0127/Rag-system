"""Shared data types for comparison experiment schemes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SchemeOutput:
    name: str
    backend_type: str
    document_vectors: np.ndarray
    query_vectors: np.ndarray
    vector_dim: int
    metadata: Dict[str, float | int | str | bool]
    reference_document_vectors: np.ndarray | None = None
    reference_query_vectors: np.ndarray | None = None
