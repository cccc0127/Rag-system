"""CKKS helpers adapted from the external reward baseline scripts.

The implementation intentionally mirrors the usable TenSEAL steps in:

- external_baselines/reward/实验1.py
- external_baselines/reward/ckks与无同态作比较.py
- external_baselines/reward/通信开销.py
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


DEFAULT_CKKS_POLY_MODULUS_DEGREE = 8192
DEFAULT_CKKS_COEFF_MOD_BIT_SIZES = (60, 40, 40, 60)
DEFAULT_CKKS_GLOBAL_SCALE = 2**40


@dataclass
class CKKSDistanceResult:
    score: float
    plain_score: float
    absolute_error: float
    relative_error: float
    left_cipher_bytes: int
    right_cipher_bytes: int
    left_plain_bytes: int
    right_plain_bytes: int


def parse_coeff_mod_bit_sizes(raw_value: str | Sequence[int]) -> list[int]:
    if isinstance(raw_value, str):
        values = [int(part.strip()) for part in raw_value.split(",") if part.strip()]
    else:
        values = [int(value) for value in raw_value]
    if not values:
        raise ValueError("CKKS coeff_mod_bit_sizes must contain at least one value")
    return values


def create_ckks_context(
    poly_modulus_degree: int = DEFAULT_CKKS_POLY_MODULUS_DEGREE,
    coeff_mod_bit_sizes: Sequence[int] = DEFAULT_CKKS_COEFF_MOD_BIT_SIZES,
    global_scale: float = DEFAULT_CKKS_GLOBAL_SCALE,
):
    ts = _import_tenseal()
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=int(poly_modulus_degree),
        coeff_mod_bit_sizes=list(coeff_mod_bit_sizes),
    )
    context.global_scale = float(global_scale)
    context.generate_galois_keys()
    context.generate_relin_keys()
    try:
        context._comparison_poly_modulus_degree = int(poly_modulus_degree)
    except AttributeError:
        pass
    return context


def ckks_squared_l2(context, left: np.ndarray, right: np.ndarray) -> CKKSDistanceResult:
    ts = _import_tenseal()
    left_vector = np.asarray(left, dtype=np.float64).reshape(-1)
    right_vector = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_vector.shape != right_vector.shape:
        raise ValueError(f"CKKS vector shape mismatch: {left_vector.shape} vs {right_vector.shape}")

    max_slots = _ckks_max_slots(context)
    encrypted_score = None
    left_cipher_bytes = 0
    right_cipher_bytes = 0

    for start in range(0, left_vector.size, max_slots):
        enc_left = ts.ckks_vector(context, left_vector[start:start + max_slots])
        enc_right = ts.ckks_vector(context, right_vector[start:start + max_slots])
        left_cipher_bytes += len(enc_left.serialize())
        right_cipher_bytes += len(enc_right.serialize())

        diff = enc_left - enc_right
        chunk_score = diff.dot(diff)
        if encrypted_score is None:
            encrypted_score = chunk_score
        else:
            encrypted_score += chunk_score

    if encrypted_score is None:
        raise ValueError("CKKS distance requires non-empty vectors")

    score = float(encrypted_score.decrypt()[0])
    plain_score = float(np.sum((left_vector - right_vector) ** 2))
    absolute_error = abs(plain_score - score)
    if abs(plain_score) < 1e-12:
        relative_error = absolute_error
    else:
        relative_error = absolute_error / abs(plain_score)

    return CKKSDistanceResult(
        score=score,
        plain_score=plain_score,
        absolute_error=float(absolute_error),
        relative_error=float(relative_error),
        left_cipher_bytes=int(left_cipher_bytes),
        right_cipher_bytes=int(right_cipher_bytes),
        left_plain_bytes=int(left_vector.nbytes),
        right_plain_bytes=int(right_vector.nbytes),
    )


def _ckks_max_slots(context) -> int:
    configured_degree = getattr(context, "_comparison_poly_modulus_degree", None)
    if configured_degree is not None:
        return max(1, int(configured_degree) // 2)
    try:
        return max(1, int(context.poly_modulus_degree()) // 2)
    except AttributeError:
        return max(1, DEFAULT_CKKS_POLY_MODULUS_DEGREE // 2)


def _import_tenseal():
    try:
        import tenseal as ts
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TenSEAL is required for CKKS baselines. Install tenseal before "
            "running --enable-ckks-fullscan or --enable-ckks-refine."
        ) from exc
    return ts
