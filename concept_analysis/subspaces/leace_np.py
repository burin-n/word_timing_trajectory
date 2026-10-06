"""NumPy LEACE fitter and eraser, including the source CCA and orthogonal modes.

Ported from the original repository's ``spaces/leace.py`` and the inherited
fitter calculations in concept-erasure 0.2.4:
https://github.com/EleutherAI/concept-erasure

Shrinkage follows the user-supplied upstream version with ``inplace`` support.
Source variable names, comments, and calculation order are retained so the
NumPy port can be compared directly with its original source.

Inputs are feature matrices and numeric concept matrices (one-hot encoded for
categorical concepts). ``fit`` preserves the feature dtype. Direct fitter
construction defaults to float32, as in the PyTorch implementation. There is
no device or autograd API.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np


def optimal_linear_shrinkage(
    S_n: np.ndarray, n: int | np.ndarray, *, inplace: bool = False
) -> np.ndarray:
    """Optimal linear shrinkage for a sample covariance matrix or batch thereof.

    Given a sample covariance matrix `S_n` of shape (*, p, p) and a sample size `n`,
    this function computes the optimal shrinkage coefficients `alpha` and `beta`, then
    returns the covariance estimate `alpha * S_n + beta * Sigma0`, where `Sigma0` is
    an isotropic covariance matrix with the same trace as `S_n`.

    The formula is distribution-free and asymptotically optimal in the Frobenius norm
    among all linear shrinkage estimators as the dimensionality `p` and sample size `n`
    jointly tend to infinity, with the ratio `p / n` converging to a finite positive
    constant `c`. The derivation is based on Random Matrix Theory and assumes that the
    underlying distribution has finite moments up to 4 + eps, for some eps > 0.

    See "On the Strong Convergence of the Optimal Linear Shrinkage Estimator for Large
    Dimensional Covariance Matrix" <https://arxiv.org/abs/1308.2608> for details.

    Args:
        S_n: Sample covariance matrices of shape (*, p, p).
        n: Sample size, broadcastable with trace(S_n) of shape (*, 1, 1).
        inplace: Whether to modify and return S_n directly.
    """
    p = S_n.shape[-1]
    assert S_n.shape[-2:] == (p, p)

    trace_S = trace(S_n)

    # Since sigma0 is I * tr(S_n) / p, its squared Frobenius norm is tr(S_n) ** 2 / p.
    sigma0_norm_sq = trace_S**2 / p
    S_norm_sq = np.linalg.norm(S_n, axis=(-2, -1), keepdims=True) ** 2

    prod_trace = sigma0_norm_sq
    top = trace_S * trace_S.conj() * sigma0_norm_sq / n
    bottom = S_norm_sq * sigma0_norm_sq - prod_trace * prod_trace.conj()

    # Epsilon prevents dividing by zero for the zero matrix. In that case we end up
    # setting alpha = 0, beta = 1, but it doesn't matter since we're shrinking toward
    # tr(0)*I = 0, so it's a no-op.
    eps = np.finfo(S_n.dtype).eps
    alpha = 1 - (top + eps) / (bottom + eps)
    beta = (1 - alpha) * (prod_trace + eps) / (sigma0_norm_sq + eps)

    ret = np.multiply(S_n, alpha, out=S_n) if inplace else alpha * S_n
    diag = beta * trace_S / p
    # NumPy's diagonal view is read-only; assign through diagonal indices instead.
    diagonal_indices = np.arange(p)
    ret[..., diagonal_indices, diagonal_indices] += diag.squeeze(-1)
    return ret


def trace(matrices: np.ndarray) -> np.ndarray:
    """Version of `np.trace` that works for batches of matrices."""
    diag = np.linalg.diagonal(matrices)
    return diag.sum(axis=-1, keepdims=True)[..., None]


@dataclass(frozen=True)
class LeaceEraser:
    """Fitted LEACE eraser with optional CCA whitening."""

    proj_left: np.ndarray
    proj_right: np.ndarray
    bias: np.ndarray | None
    sigma_xx: np.ndarray | None
    sigma_xz: np.ndarray
    sigma_zz: np.ndarray | None
    W: np.ndarray
    W_inv: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, z: np.ndarray, **kwargs) -> LeaceEraser:
        return LeaceFitterCCA.fit(x, z, **kwargs).eraser

    def shape(self) -> tuple:
        if self.bias is not None:
            return self.proj_left.shape, self.proj_right.shape, self.bias.shape
        return self.proj_left.shape, self.proj_right.shape

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Erase concept from x by projecting out the concept subspace."""
        delta = x - self.bias if self.bias is not None else x
        x_erased = x - (delta @ self.proj_right.conj().T) @ self.proj_left.conj().T
        return x_erased.astype(x.dtype, copy=False)

    @property
    def P(self) -> np.ndarray:
        eye = np.eye(self.proj_left.shape[0], dtype=self.proj_left.dtype)
        return eye - self.proj_left @ self.proj_right


class LeaceFitterCCA:
    """Fit LEACE, orthogonal erasure, or CCA from batch or incremental statistics.

    Defaults and covariance normalization follow the source fitter. ``method``
    is one of ``leace``, ``orth`` (COV), or ``cca``. As in the source CCA branch,
    label whitening uses the unnormalized ``sigma_zz_`` accumulator.
    """

    def __init__(
        self,
        x_dim: int,
        z_dim: int,
        method: str = "leace",
        *,
        affine: bool = True,
        constrain_cov_trace: bool = True,
        dtype: np.dtype | type | None = None,
        shrinkage: bool = True,
        svd_tol: float = 0.01,
    ) -> None:
        """Initialize a `LeaceFitterCCA`.

        Args:
            x_dim: Dimensionality of the representation.
            z_dim: Dimensionality of the concept.
            method: Type of projection matrix to use.
            affine: Whether to use a bias term to ensure the unconditional mean of the
                features remains the same after erasure.
            constrain_cov_trace: Whether to constrain the trace of the covariance of X
                after erasure to be no greater than before erasure. This is especially
                useful when injecting the scrubbed features back into a model. Without
                this constraint, the norm of the model's hidden states may diverge in
                some cases.
            dtype: Data type to use for the statistics.
            shrinkage: Whether to use shrinkage to estimate the covariance matrix of X.
            svd_tol: Singular values under this threshold are truncated, both during
                the phase where we do SVD on the cross-covariance matrix, and at the
                phase where we compute the pseudoinverse of the projected covariance
                matrix. Higher values are more numerically stable and result in less
                damage to the representation, but may leave trace correlations intact.
        """
        # Inline the upstream initializer and the source CCA extension.
        dtype = np.float32 if dtype is None else dtype

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.affine = affine
        self.constrain_cov_trace = constrain_cov_trace
        self.method = method
        self.shrinkage = shrinkage

        assert svd_tol > 0.0, "`svd_tol` must be positive for numerical stability."
        self.svd_tol = svd_tol

        self.mean_x = np.zeros(x_dim, dtype=dtype)
        self.mean_z = np.zeros(z_dim, dtype=dtype)

        self.n = 0
        self.sigma_xz_ = np.zeros((x_dim, z_dim), dtype=dtype)

        if self.method in ("leace", "cca"):
            self.sigma_xx_ = np.zeros((x_dim, x_dim), dtype=dtype)
        elif self.method == "orth":
            self.sigma_xx_ = None
        else:
            raise ValueError(f"Unknown projection type {self.method}")

        if self.method == "cca":
            self.sigma_zz_ = np.zeros((z_dim, z_dim), dtype=dtype)
        else:
            self.sigma_zz_ = None

    @classmethod
    def fit(cls, x: np.ndarray, z: np.ndarray, **kwargs) -> "LeaceFitterCCA":
        n, d = x.shape
        _, k = z.reshape(n, -1).shape
        fitter = cls(d, k, dtype=x.dtype, **kwargs)
        return fitter.update(x, z)

    def update(self, x: np.ndarray, z: np.ndarray) -> "LeaceFitterCCA":
        # NumPy cached_property replaces the source cache decorator.
        self.__dict__.pop("eraser", None)
        d, c = self.sigma_xz_.shape
        x = x.reshape(-1, d).astype(self.mean_x.dtype, copy=False)
        n, d2 = x.shape
        assert d == d2, f"Unexpected number of features {d2}"
        self.n += n

        delta_x = x - self.mean_x
        self.mean_x += delta_x.sum(axis=0) / self.n
        delta_x2 = x - self.mean_x

        if self.method in ("leace", "cca"):
            assert self.sigma_xx_ is not None
            self.sigma_xx_ += delta_x.conj().T @ delta_x2

        z = z.reshape(n, -1).astype(x.dtype, copy=False)
        assert z.shape[-1] == c, f"Unexpected number of classes {z.shape[-1]}"

        delta_z = z - self.mean_z
        self.mean_z += delta_z.sum(axis=0) / self.n
        delta_z2 = z - self.mean_z

        self.sigma_xz_ += delta_x.conj().T @ delta_z2

        if self.method == "cca":
            assert self.sigma_zz_ is not None
            self.sigma_zz_ += delta_z.conj().T @ delta_z2

        return self

    @classmethod
    def compute_whitening(cls, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        L, V = np.linalg.eigh(sigma)

        # Threshold used by torch.linalg.pinv
        mask = L > (L[-1] * sigma.shape[-1] * np.finfo(L.dtype).eps)

        # Assuming PSD; account for numerical error
        np.clip(L, 0.0, None, out=L)
        # np.where evaluates both branches; avoid inverse square roots of zero.
        safe_L = np.where(mask, L, 1.0)
        W = V * np.where(mask, safe_L ** -0.5, 0.0) @ V.conj().T
        W_inv = V * np.where(mask, np.sqrt(L), 0.0) @ V.conj().T
        return W, W_inv

    @cached_property
    def eraser(self) -> LeaceEraser:
        eye = np.eye(self.x_dim, dtype=self.mean_x.dtype)

        if self.method in ("leace", "cca"):
            W, W_inv = self.compute_whitening(self.sigma_xx)
            if self.method == "cca":
                W_z, W_inv_z = self.compute_whitening(self.sigma_zz_)
            else:
                W_z, W_inv_z = (np.eye(self.z_dim, dtype=W.dtype),) * 2
        else:
            W = W_inv = eye

        svd_input = W @ self.sigma_xz @ W_z if self.method == "cca" else W @ self.sigma_xz
        u, s, _ = np.linalg.svd(svd_input, full_matrices=False)
        u *= s > self.svd_tol

        proj_left = W_inv @ u
        proj_right = u.conj().T @ W

        if self.constrain_cov_trace and self.method == "leace":
            P = eye - proj_left @ proj_right
            # Prevent the covariance trace from increasing
            sigma = self.sigma_xx
            old_trace = np.trace(sigma)
            new_trace = np.trace(P @ sigma @ P.conj().T)

            # If applying the projection matrix increases the variance, this might
            # cause instability, especially when erasure is applied multiple times.
            # We regularize toward the orthogonal projection matrix to avoid this.
            if new_trace.real > old_trace.real:
                Q = eye - u @ u.conj().T

                # Set up the variables for the quadratic equation
                x = new_trace
                y = 2 * np.trace(P @ sigma @ Q.conj().T)
                z = np.trace(Q @ sigma @ Q.conj().T)
                w = old_trace

                # Solve for the mixture of P and Q that makes the trace equal to the
                # trace of the original covariance matrix
                discr = np.sqrt(
                    4 * w * x - 4 * w * y + 4 * w * z - 4 * x * z + y**2
                )
                alpha1 = (-y / 2 + z - discr / 2) / (x - y + z)
                alpha2 = (-y / 2 + z + discr / 2) / (x - y + z)

                # Choose the positive root
                alpha = np.clip(np.where(alpha1.real > 0, alpha1, alpha2), 0, 1)
                P = alpha * P + (1 - alpha) * Q

                # TODO: Avoid using SVD here
                u, s, vh = np.linalg.svd(eye - P)
                proj_left = u * np.sqrt(s)
                proj_right = vh * np.sqrt(s)

        return LeaceEraser(
            proj_left=proj_left,
            proj_right=proj_right,
            bias=self.mean_x if self.affine else None,
            sigma_xx=None if self.method == "orth" else self.sigma_xx,
            sigma_xz=self.sigma_xz,
            sigma_zz=self.sigma_zz_ if self.method == "cca" else None,
            W=W,
            W_inv=W_inv,
        )

    @property
    def sigma_xx(self) -> np.ndarray:
        """The covariance matrix of X."""
        assert self.n > 1, "Call update() before accessing sigma_xx"
        assert (
            self.sigma_xx_ is not None
        ), "Covariance statistics are not being tracked for X"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_xx_ + self.sigma_xx_.conj().T) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / self.n, self.n)

        # Just apply Bessel's correction
        else:
            return S_hat / (self.n - 1)

    @property
    def sigma_xz(self) -> np.ndarray:
        """The cross-covariance matrix."""
        assert self.n > 1, "Call update() with labels before accessing sigma_xz"
        return self.sigma_xz_ / (self.n - 1)

    @property
    def sigma_zz(self) -> np.ndarray:
        """The covariance matrix of Z."""
        assert self.n > 1, "Call update() before accessing sigma_zz"
        assert (
            self.sigma_zz_ is not None
        ), "Covariance statistics are not being tracked for Z"

        # Accumulated numerical error may cause this to be slightly non-symmetric
        S_hat = (self.sigma_zz_ + self.sigma_zz_.conj().T) / 2

        # Apply Random Matrix Theory-based shrinkage
        if self.shrinkage:
            return optimal_linear_shrinkage(S_hat / self.n, self.n)

        # # Just apply Bessel's correction
        # else:
        return S_hat / (self.n - 1)
