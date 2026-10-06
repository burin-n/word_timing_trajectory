"""Numerical trajectory geometry on supplied arrays."""

from __future__ import annotations

import warnings

import numpy as np
from einops import repeat

from concept_analysis.utils import cosine_sim_pairs


###################
# Rotation and smoothing
###################

def rotate(x: np.ndarray, theta: float = 0.0) -> np.ndarray:
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
        ])
    return x @ R


def rotate_to_anchor(x: np.ndarray, anchor: np.ndarray | None = None) -> np.ndarray:
    if anchor is None:
        if len(x) > 1:
            a0, a1 = compute_point_angles(x[:2])
            anchor = np.array([np.cos(np.pi / 10), np.sin(np.pi / 10) * (1 if a0 >= a1 else -1)])
        else:
            anchor = np.array([1.0, 0.0])
    best_theta, best_dist = 0.0, np.linalg.norm(x[0] - anchor)
    for theta in np.arange(0, 2 * np.pi, 0.01):
        d = np.linalg.norm(rotate(x, theta)[0] - anchor)
        if d < best_dist:
            best_dist, best_theta = d, theta
    return rotate(x, best_theta)


def smoothing(x: np.ndarray, window: int | tuple[int, int] = 1) -> np.ndarray:
    """Symmetric or asymmetric sliding-window average."""
    if isinstance(window, int):
        left = window // 2
        right = window - left - 1
    else:
        left, right = window
    padded = np.concatenate([np.zeros(left), x, np.zeros(right)])
    return np.array([padded[i: i + left + right + 1].mean() for i in range(len(x))])


###################
# Relative angle to an anchor
###################

def _get_relative_angle_cosine(anchor: np.ndarray, points: np.ndarray, n_components: int | None = None) -> np.ndarray:
    n = n_components or anchor.shape[0]
    cos = cosine_sim_pairs(
        repeat(anchor[:n], "h -> (n) h", n=len(points)),
        points[:, :n],
    )
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _get_relative_angle_atan2(anchor: np.ndarray, points: np.ndarray, mode: str) -> np.ndarray:
    if anchor.shape[0] != 2 or points.shape[-1] != 2:
        raise ValueError("atan2 angle modes require n_components == 2")

    anchor_theta = np.arctan2(anchor[1], anchor[0])
    point_theta = np.arctan2(points[:, 1], points[:, 0])

    if mode == "atan2_unwrapped":
        theta = np.concatenate([[anchor_theta], point_theta])
        theta = np.unwrap(theta)
        return theta[1:] - theta[0]
    raise ValueError(f"Unknown atan2 angle mode: {mode!r}")


def compute_relative_angle(
    anchor: np.ndarray,
    points: np.ndarray,
    n_components: int | None = None,
    angle_mode: str = "cosine_arccos",
) -> np.ndarray:
    """Compute one angle per point relative to the anchor vector."""
    n = n_components or anchor.shape[0]
    if angle_mode == "cosine_arccos":
        return _get_relative_angle_cosine(anchor, points, n)
    if angle_mode.startswith("atan2_"):
        if n != 2:
            raise ValueError("atan2 angle modes require n_components == 2")
        return _get_relative_angle_atan2(anchor[:n], points[:, :n], angle_mode)
    raise ValueError(f"Unknown angle_mode: {angle_mode!r}")


def get_sequence_relative_angles(
    X: np.ndarray,
    n_components: int | None = None,
    relative_to: int | str = 0,
    angle_mode: str = "cosine_arccos",
) -> np.ndarray:
    """Return whole-sequence angles relative to an absolute frame index."""
    if relative_to == "first_frame":
        relative_to = 0
    if not isinstance(relative_to, (int, np.integer)):
        raise ValueError(f"whole-sequence relative_to must be a frame index, got {relative_to!r}")
    X = X.copy()
    n = n_components or X.shape[-1]
    X = X[:, :n]
    return compute_relative_angle(X[relative_to], X, n, angle_mode)


def get_segment_relative_angles(
    X: np.ndarray,
    segment_splice: list[tuple[int, int]],
    n_components: int | None = None,
    segment_offset_by_length: dict | None = None,
    relative_to: int | str = 0,
    angle_mode: str = "cosine_arccos",
) -> list[np.ndarray]:
    """Return angles per segment, optionally subtracting a length-specific offset."""
    if relative_to == "first_frame":
        relative_to = 0
    if relative_to != "label_onset" and not isinstance(relative_to, (int, np.integer)):
        raise ValueError(f"segment relative_to must be a frame index, 'first_frame', or 'label_onset', got {relative_to!r}")
    X = X.copy()
    n = n_components or X.shape[-1]
    X = X[:, :n]
    segment_offset_by_length = segment_offset_by_length or {}
    zero = np.zeros(n)
    angles = []
    for s, e in segment_splice:
        seg = X[s:e]
        seg = seg - segment_offset_by_length.get(len(seg), zero)[:n]
        anchor = seg[0] if relative_to == "label_onset" else X[relative_to]
        angles.append(compute_relative_angle(anchor, seg, n, angle_mode))
    return angles


###################
# Absolute polar angles, centering, and radii
###################

def compute_point_angles(points: np.ndarray) -> np.ndarray:
    """Return each 2D point's angle from the positive x-axis in [0, 2π)."""
    angles = np.arctan2(points[:, 1], points[:, 0])
    return angles % (2 * np.pi)


def normalize_trajectory(trajectory: np.array, mean_trajectory: np.array = None, centering_mode="centroid"):
    """
        centroid = mean_centroid: centroid of a mean trajectory
        circle = mean_circle: a circle center of a mean trajectory
        instance_centroid: a centroid of a single instance
        instance_circle: a circle center fitted using a sigle instance
    """
    assert centering_mode in ["centroid", "circle", "none", "instance_centroid", "instance_circle", "mean_centroid", "mean_circle"]

    if centering_mode in ["centroid", "circle", "mean_centroid", "mean_circle"]:
        assert mean_trajectory is not None

    if centering_mode in ["centroid", "mean_centroid"]:
        trajectory -= mean_trajectory.mean(axis=0)
    elif centering_mode in ["circle", "mean_circle"]:
        xc, yc, *_ = _fit_circle(mean_trajectory[:, 0], mean_trajectory[:, 1])
        trajectory -= np.array([xc, yc])
    elif centering_mode == "instance_centroid":
        trajectory -= trajectory.mean(axis=0)
    elif centering_mode == "instance_circle":
        xc, yc, *_ = _fit_circle(trajectory[:, 0], trajectory[:, 1])
        trajectory -= np.array([xc, yc])
    elif centering_mode == "none":
        pass

    return trajectory


def get_mean_trajectory_angles(subspace, X_group: dict, group_centering_method: str = "centroid") -> dict[int, np.ndarray]:
    """Return frame angles of each duration group's mean trajectory."""

    if type(group_centering_method) == bool:
        warnings.warn("boolean value for group_centering_method is deprecated. Please provide specific options: 'centroid', 'circle', 'none'")
    if group_centering_method == True:
        # backward compatibility
        group_centering_method = "centroid"
    elif group_centering_method == False:
        group_centering_method = "none"
    assert group_centering_method in ["centroid", "circle", "none"], "Invalid group_centering_method option. Choose from 'centroid', 'circle', 'none'."

    angles = {}
    for seq_len in sorted(X_group):
        sample = X_group[seq_len]
        # assume 3-dim sample (N, T, H)
        if sample.ndim == 3:
            # Source omitted the assignment; see migration decision 0003.
            mean_trajectory = sample.mean(axis=0)
        # assume 2-dim sample (T, H) is mean trajectory
        else:
            mean_trajectory = sample

        if subspace is not None:
            feat = subspace.transform(mean_trajectory, n_components=2)
        else:
            feat = mean_trajectory

        if feat.shape[1] < 2:
            raise ValueError(f"Polar angle plot requires at least 2 components, got {feat.shape[1]}")

        feat = normalize_trajectory(feat, mean_trajectory=feat, centering_mode=group_centering_method)
        angles[seq_len] = compute_point_angles(feat)
    # dict[int: seq_len, np.array(shape=(seq_len,)) ]
    return angles


def get_instance_trajectory_angles(subspace, X_group: dict, group_centering_method: str = "centroid") -> dict[int, np.ndarray]:
    """Return frame angles for every trajectory in each duration group."""
    assert group_centering_method in ["centroid", "circle", "none", "instance_centroid", "instance_circle", "mean_centroid", "mean_circle"], \
        "Invalid group_centering_method option. Choose from 'centroid', 'circle', 'none', 'instance_centroid', 'instance_circle', 'mean_centroid', 'mean_circle'."

    angles = {}
    for seq_len in sorted(X_group):
        angles[seq_len] = []
        # for normalization purpose
        mean_trajectory = subspace.transform(X_group[seq_len].mean(axis=0), n_components=2)
        for sample in X_group[seq_len]:
            feat = subspace.transform(sample, n_components=2)
            feat = normalize_trajectory(feat, mean_trajectory, centering_mode=group_centering_method)
            angles[seq_len].append(compute_point_angles(feat))
        angles[seq_len] = np.array(angles[seq_len])
    # dict[int: seq_len, np.array(shape=(N, seq_len)) ]
    return angles


def get_instance_trajectory_radii(subspace, X_group: dict, group_centering_method: str = "centroid") -> dict[int, np.ndarray]:
    """Return frame radii for every trajectory in each duration group."""
    assert group_centering_method in ["centroid", "circle", "none", "instance_centroid", "instance_circle", "mean_centroid", "mean_circle"], \
        "Invalid group_centering_method option. Choose from 'centroid', 'circle', 'none', 'instance_centroid', 'instance_circle', 'mean_centroid', 'mean_circle'."

    radii = {}
    for seq_len in sorted(X_group):
        radii[seq_len] = []
        mean_trajectory = subspace.transform(X_group[seq_len].mean(axis=0), n_components=2)
        for sample in X_group[seq_len]:
            feat = subspace.transform(sample, n_components=2)
            feat = normalize_trajectory(feat, mean_trajectory, centering_mode=group_centering_method)
            radii[seq_len].append(np.linalg.norm(feat, axis=1))
        radii[seq_len] = np.array(radii[seq_len])
    return radii


def get_mean_trajectory_radii(subspace, X_group: dict, group_centering_method: str = "centroid") -> dict[int, np.ndarray]:
    """Return frame radii of each duration group's mean trajectory."""
    assert group_centering_method in ["centroid", "circle", "none"]
    radii = {}
    for seq_len in sorted(X_group):
        mean_trajectory = X_group[seq_len].mean(axis=0)
        feat = subspace.transform(mean_trajectory, n_components=2)
        if feat.shape[1] < 2:
            raise ValueError(f"Polar radius plot requires at least 2 components, got {feat.shape[1]}")

        feat = normalize_trajectory(feat, mean_trajectory=feat, centering_mode=group_centering_method)
        radii[seq_len] = np.linalg.norm(feat, axis=1)
    return radii


def _fit_circle(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    M = np.column_stack([x, y, np.ones(len(x))])
    z = x ** 2 + y ** 2
    v, residuals, *_ = np.linalg.lstsq(M, z, rcond=None)
    A, B, C = v
    xc, yc = A / 2, B / 2
    R = np.sqrt(C + xc ** 2 + yc ** 2)
    return xc, yc, R, residuals
