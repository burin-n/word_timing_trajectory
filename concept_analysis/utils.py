"""Shared utility functions."""

from __future__ import annotations

import numpy as np


def cosine_sim_pairs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity: (n, dim) × (n, dim) → (n,)."""
    A_norm = A / np.linalg.norm(A, axis=-1, keepdims=True)
    B_norm = B / np.linalg.norm(B, axis=-1, keepdims=True)
    return np.multiply(A_norm, B_norm).sum(axis=1)


def get_mean_agg(
    features: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-class centroid vectors and the sorted unique label array."""
    unique = np.array(sorted(np.unique(labels)))
    centroids = np.array([features[labels == lab].mean(axis=0) for lab in unique])
    return centroids, unique
