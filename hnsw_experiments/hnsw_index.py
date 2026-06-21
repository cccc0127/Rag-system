"""HNSW retrieval adapter for DP-RAG private embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def _load_hnswlib():
    try:
        import hnswlib
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "hnswlib is required for HNSW experiments. Install it with: "
            "python3 -m pip install hnswlib"
        ) from exc
    return hnswlib


class HNSWRetriever:
    """Small hnswlib wrapper with chunk-id aligned labels."""

    def __init__(
        self,
        dim: int = 256,
        space: str = "cosine",
        M: int = 16,
        ef_construction: int = 200,
        ef_search: int = 64,
        random_seed: int = 42,
    ):
        if dim <= 0:
            raise ValueError("dim must be greater than 0")
        if space not in {"cosine", "ip", "l2"}:
            raise ValueError("space must be one of: cosine, ip, l2")
        if M <= 0:
            raise ValueError("M must be greater than 0")
        if ef_construction <= 0:
            raise ValueError("ef_construction must be greater than 0")
        if ef_search <= 0:
            raise ValueError("ef_search must be greater than 0")

        self.dim = int(dim)
        self.space = space
        self.M = int(M)
        self.ef_construction = int(ef_construction)
        self.ef_search = int(ef_search)
        self.random_seed = int(random_seed)
        self.index = None
        self.num_elements = 0

    def build(self, vectors: np.ndarray, ids: Iterable[int] | None = None) -> "HNSWRetriever":
        vectors = self._as_matrix(vectors)
        if ids is None:
            labels = np.arange(vectors.shape[0], dtype=np.int64)
        else:
            labels = np.asarray(list(ids), dtype=np.int64)
            if labels.shape[0] != vectors.shape[0]:
                raise ValueError("ids length must match number of vectors")

        hnswlib = _load_hnswlib()
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(
            max_elements=vectors.shape[0],
            ef_construction=self.ef_construction,
            M=self.M,
            random_seed=self.random_seed,
        )
        self.index.add_items(vectors, labels)
        self.index.set_ef(self.ef_search)
        self.num_elements = int(vectors.shape[0])
        return self

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        labels, distances = self.batch_search(np.asarray(query_vector, dtype=np.float32), top_k=top_k)
        return labels[0], distances[0]

    def batch_search(self, query_vectors: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        if self.index is None:
            raise ValueError("HNSWRetriever must be built or loaded before search")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        queries = self._as_matrix(query_vectors)
        k = min(int(top_k), max(1, self.num_elements))
        labels, distances = self.index.knn_query(queries, k=k)
        return labels.astype(np.int64), distances.astype(np.float32)

    def set_ef(self, ef_search: int) -> None:
        if ef_search <= 0:
            raise ValueError("ef_search must be greater than 0")
        self.ef_search = int(ef_search)
        if self.index is not None:
            self.index.set_ef(self.ef_search)

    def save(self, path: str | Path) -> None:
        if self.index is None:
            raise ValueError("Cannot save an unbuilt HNSW index")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.index.save_index(str(path))

    def load(self, path: str | Path, max_elements: int) -> "HNSWRetriever":
        if max_elements <= 0:
            raise ValueError("max_elements must be greater than 0")
        hnswlib = _load_hnswlib()
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(str(path), max_elements=int(max_elements))
        self.index.set_ef(self.ef_search)
        self.num_elements = int(max_elements)
        return self

    def _as_matrix(self, vectors: np.ndarray) -> np.ndarray:
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError(f"vectors must be 1D or 2D, got shape {array.shape}")
        if array.shape[1] != self.dim:
            raise ValueError(f"Vector dim mismatch: got {array.shape[1]}, expected {self.dim}")
        return np.ascontiguousarray(array, dtype=np.float32)
