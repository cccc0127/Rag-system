from pathlib import Path
from typing import Optional, Union

import numpy as np


ArrayLike = Union[np.ndarray, list]


def johnson_lindenstrauss_min_dim(n_samples: int, eps: float = 0.3) -> int:
    # 提供JL理论目标维护估算
    if n_samples <= 1:
        return 1
    if not 0.0 < eps < 1.0:
        raise ValueError("eps must be in (0, 1)")

    denominator = (eps**2 / 2.0) - (eps**3 / 3.0)
    return int(np.ceil(4.0 * np.log(n_samples) / denominator))


class JLProjector:
    """Gaussian JL projection with L2 normalization before and after projection."""

    def __init__(
        self,
        target_dim: Optional[int] = None,
        eps: float = 0.3,
        random_state: int = 42,
        dtype=np.float32,
    ):
        self.target_dim = target_dim
        self.eps = eps
        self.random_state = random_state
        self.dtype = dtype
        self.input_dim: Optional[int] = None
        self.projection_matrix: Optional[np.ndarray] = None

    def fit(self, vectors: ArrayLike) -> "JLProjector":
        vectors = _as_2d_float_array(vectors)
        n_samples, input_dim = vectors.shape
        target_dim = self._resolve_target_dim(n_samples, input_dim)

        rng = np.random.default_rng(self.random_state)
        self.projection_matrix = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(target_dim),
            size=(input_dim, target_dim),
        ).astype(self.dtype)
        self.input_dim = input_dim
        self.target_dim = target_dim
        return self

    def transform(self, vectors: ArrayLike) -> np.ndarray:
        if self.projection_matrix is None or self.input_dim is None:
            raise ValueError("JLProjector must be fitted or loaded before transform")

        vectors = l2_normalize(_as_2d_float_array(vectors))
        if vectors.shape[1] != self.input_dim:
            raise ValueError(
                f"Vector dim mismatch: got {vectors.shape[1]}, expected {self.input_dim}"
            )

        projected = (vectors @ self.projection_matrix).astype(self.dtype)
        return l2_normalize(projected).astype(self.dtype)

    def fit_transform(self, vectors: ArrayLike) -> np.ndarray:
        return self.fit(vectors).transform(vectors)

    def save(self, path: Union[str, Path]) -> None:
        if self.projection_matrix is None or self.input_dim is None:
            raise ValueError("Cannot save an unfitted JLProjector")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            projection_matrix=self.projection_matrix,
            input_dim=np.array(self.input_dim, dtype=np.int64),
            target_dim=np.array(self.target_dim, dtype=np.int64),
            eps=np.array(self.eps, dtype=np.float32),
            random_state=np.array(self.random_state, dtype=np.int64),
        )

    @classmethod
    def load(cls, path: Union[str, Path]) -> "JLProjector":
        data = np.load(Path(path), allow_pickle=False)
        projector = cls(
            target_dim=int(data["target_dim"]),
            eps=float(data["eps"]),
            random_state=int(data["random_state"]),
        )
        projector.input_dim = int(data["input_dim"])
        projector.projection_matrix = data["projection_matrix"].astype(np.float32)
        return projector

    def _resolve_target_dim(self, n_samples: int, input_dim: int) -> int:
        if self.target_dim is None:
            target_dim = johnson_lindenstrauss_min_dim(n_samples, self.eps)
        else:
            target_dim = int(self.target_dim)

        if target_dim <= 0:
            raise ValueError("target_dim must be greater than 0")
        return min(target_dim, input_dim)


def fit_reduce_embeddings(
    embeddings: ArrayLike,
    save_path: Union[str, Path],
    target_dim: Optional[int] = None,
    eps: float = 0.3,
    random_state: int = 42,
) -> np.ndarray:
    """Normalize, fit JL projection, reduce, normalize again, then save projector."""
    projector = JLProjector(target_dim=target_dim, eps=eps, random_state=random_state)
    reduced = projector.fit_transform(embeddings)
    projector.save(save_path)
    return reduced


def reduce_query_embedding(query_embedding: ArrayLike, projector_path: Union[str, Path]) -> np.ndarray:
    """Normalize, reduce, and re-normalize query embedding with the saved doc projector."""
    projector = JLProjector.load(projector_path)
    return projector.transform(query_embedding)


def l2_normalize(vectors: ArrayLike, eps: float = 1e-12) -> np.ndarray:
    """L2-normalize one vector or a batch of vectors row-wise."""
    array = _as_2d_float_array(vectors)
    norms = np.linalg.norm(array, ord=2, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return (array / norms).astype(np.float32)


def _as_2d_float_array(vectors: ArrayLike) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2:
        raise ValueError(f"vectors must be 1D or 2D, got shape {array.shape}")
    return array
