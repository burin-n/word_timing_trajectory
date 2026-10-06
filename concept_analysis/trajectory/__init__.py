"""Temporal trajectory analysis on supplied arrays."""

from .grouping import group_and_preprocess_feature, group_feature_by_dur
from .estimation import get_temporal_subspace, get_mean_trajectories
from .segmentation import detect_boundaries
from .metrics import (
    explained_variance_scores, get_elapsed_angle_regression_features,
    optimal_line_mse, polar_scores,
)
from .geometry import (
    compute_point_angles, compute_relative_angle,
    get_mean_trajectory_angles, get_instance_trajectory_angles,
    get_mean_trajectory_radii, get_instance_trajectory_radii,
    get_segment_relative_angles, get_sequence_relative_angles,
    normalize_trajectory, rotate, rotate_to_anchor, smoothing,
)

__all__ = [
    "group_and_preprocess_feature", "group_feature_by_dur",
    "get_temporal_subspace", "get_mean_trajectories",
    "detect_boundaries",
    "get_elapsed_angle_regression_features", "polar_scores",
    "explained_variance_scores", "optimal_line_mse",
    "compute_point_angles", "compute_relative_angle",
    "get_mean_trajectory_angles", "get_instance_trajectory_angles",
    "get_mean_trajectory_radii", "get_instance_trajectory_radii",
    "get_segment_relative_angles", "get_sequence_relative_angles",
    "normalize_trajectory", "rotate", "rotate_to_anchor", "smoothing",
]
