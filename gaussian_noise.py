import logging
from dataclasses import dataclass

import numpy as np
# 从scipy.optimize导入brentq函数，用于求解非线性方程
from scipy.optimize import brentq
# 从scipy.stats导入norm函数，用于计算正态分布的累计分布函数（CDF）
from scipy.stats import norm


logger = logging.getLogger(__name__)

# 定义一个数据类NoiseCalibration，用于存储隐私噪声参数
@dataclass(frozen=True)
class NoiseCalibration:
    """Resolved DP noise parameters for one embedding vector."""
    raw_score: float
    normalized_score: float
    epsilon: float
    local_sensitivity: float
    sigma: float
    solved_by_brentq: bool


@dataclass(frozen=True)
class NoiseApplication:
    """Diagnostic details for one clipped-and-noised embedding vector."""
    noised_vector: np.ndarray
    clipped_vector: np.ndarray
    noise_vector: np.ndarray
    calibration: NoiseCalibration
    sigma_per_dim: float


class AnalyticGaussianCalibrator:
    """Neighbourhood-aware analytic Gaussian noise calibrator for embeddings.

    The design follows the ACL Findings 2023 neighbourhood-aware idea at the
    engineering level: more sensitive text receives a smaller privacy budget
    and a larger local neighbourhood radius before Gaussian perturbation.
    """

    SCORE_MIN = 0.1
    SCORE_MAX = 10.0

    def __init__(
        self,
        delta: float = 1e-5,
        root_interval: tuple[float, float] = (0.0001, 100.0),
        l2_clip_norm: float = 1.0,
        utility_scale: float = 0.01,
        random_state: int | None = None,
    ):
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in the open interval (0, 1)")
        if root_interval[0] <= 0.0 or root_interval[0] >= root_interval[1]:
            raise ValueError("root_interval must be a positive increasing interval")
        if l2_clip_norm <= 0.0:
            raise ValueError("l2_clip_norm must be greater than 0")
        if utility_scale <= 0.0:
            raise ValueError("utility_scale must be greater than 0")

        self.delta = float(delta)
        self.root_interval = root_interval
        self.l2_clip_norm = float(l2_clip_norm)
        self.utility_scale = float(utility_scale)
        self.rng = np.random.default_rng(random_state)

    def normalize_score(self, raw_score: float) -> float:
        """Map raw sensitivity score [0.1, 10.0] linearly to S_norm in [0, 1]."""
        clipped_score = np.clip(float(raw_score), self.SCORE_MIN, self.SCORE_MAX)
        return float((clipped_score - self.SCORE_MIN) / 9.9)

    def compute_epsilon(self, normalized_score: float) -> float:
        """Dynamic privacy budget: higher sensitivity means smaller epsilon.

        The curve is intentionally smoother than a steep exponential decay so
        that high-sensitivity chunks remain in a retrieval-usable epsilon band.
        """
        s_norm = np.clip(float(normalized_score), 0.0, 1.0)
        epsilon = 1.25 + 8.75 * ((1.0 - s_norm) ** 1.5)
        return float(np.clip(epsilon, 0.5, 10.0))

    def compute_local_sensitivity(self, normalized_score: float) -> float:
        """Dynamic local sensitivity Delta_i as a retrieval-scale radius proxy."""
        s_norm = np.clip(float(normalized_score), 0.0, 1.0)
        return float(0.25 + 0.25 * s_norm)

    def solve_noise_multiplier(self, epsilon: float) -> tuple[float, bool]:
        """Solve g(u) - delta = 0 for the analytic Gaussian multiplier u*.

        g(u) = Phi(1/(2u) - epsilon*u)
             - exp(epsilon) * Phi(-1/(2u) - epsilon*u)

        If the requested bracket has no valid root, fall back to the classical
        Gaussian mechanism scale sqrt(2 ln(1.25/delta)) / epsilon.
        """
        epsilon = float(epsilon)

        def objective(u: float) -> float:
            return self._g(u, epsilon) - self.delta

        lower, upper = self.root_interval
        f_lower = objective(lower)
        f_upper = objective(upper)

        if np.isfinite(f_lower) and np.isfinite(f_upper):
            if f_lower == 0.0:
                return lower, True
            if f_upper == 0.0:
                return upper, True
            if f_lower * f_upper < 0.0:
                return float(brentq(objective, lower, upper)), True

        fallback_u = np.sqrt(2.0 * np.log(1.25 / self.delta)) / max(epsilon, 1e-12)
        logger.warning(
            "Analytic Gaussian root not bracketed in [%s, %s] for epsilon=%.6f; "
            "falling back to classical Gaussian multiplier %.6f.",
            lower,
            upper,
            epsilon,
            fallback_u,
        )
        return float(fallback_u), False

    def calibrate(self, raw_score: float) -> NoiseCalibration:
        """Resolve epsilon, Delta_i and sigma for one raw sensitivity score."""
        normalized_score = self.normalize_score(raw_score)
        epsilon = self.compute_epsilon(normalized_score)
        local_sensitivity = self.compute_local_sensitivity(normalized_score)
        u_star, solved_by_brentq = self.solve_noise_multiplier(epsilon)
        sigma = u_star * local_sensitivity

        return NoiseCalibration(
            raw_score=float(np.clip(raw_score, self.SCORE_MIN, self.SCORE_MAX)),
            normalized_score=normalized_score,
            epsilon=epsilon,
            local_sensitivity=local_sensitivity,
            sigma=float(sigma),
            solved_by_brentq=solved_by_brentq,
        )

    def apply_noise(self, vector: np.ndarray, raw_score: float) -> np.ndarray:
        """Clip an embedding and add calibrated isotropic Gaussian noise."""
        return self.apply_noise_with_diagnostics(vector, raw_score).noised_vector

    def apply_noise_with_diagnostics(
        self,
        vector: np.ndarray,
        raw_score: float,
    ) -> NoiseApplication:
        """Clip an embedding, add scaled Gaussian noise, and expose diagnostics."""
        vector = np.asarray(vector, dtype=np.float32)
        calibration = self.calibrate(raw_score)

        # L2 clipping enforces the mathematical sensitivity boundary assumed by
        # the Gaussian mechanism: v = v / max(1, ||v||_2).
        norm_value = float(np.linalg.norm(vector, ord=2))
        clipped = vector / max(1.0, norm_value / self.l2_clip_norm)
        dim = int(clipped.size)
        sigma_per_dim = self.compute_sigma_per_dim(calibration.sigma, dim)

        # z ~ N(0, sigma_per_dim^2 I). The sqrt(dim) correction keeps the
        # total noise energy aligned with the utility budget instead of letting
        # high-dimensional Gaussian energy dominate the embedding signal.
        noise = self.rng.normal(
            loc=0.0,
            scale=sigma_per_dim,
            size=clipped.shape,
        ).astype(np.float32)
        noised = clipped + noise
        return NoiseApplication(
            noised_vector=noised.astype(np.float32),
            clipped_vector=clipped.astype(np.float32),
            noise_vector=noise,
            calibration=calibration,
            sigma_per_dim=sigma_per_dim,
        )

    def compute_sigma_per_dim(self, sigma: float, dim: int) -> float:
        if dim <= 0:
            raise ValueError("dim must be greater than 0")
        return float((float(sigma) * self.utility_scale) / np.sqrt(dim))

    @staticmethod
    def _g(u: float, epsilon: float) -> float:
        return float(
            norm.cdf(1.0 / (2.0 * u) - epsilon * u)
            - np.exp(epsilon) * norm.cdf(-1.0 / (2.0 * u) - epsilon * u)
        )
