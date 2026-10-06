"""Temporal trajectory metrics on supplied grouped features and frames."""

import numpy as np
from sklearn.linear_model import LinearRegression

from .geometry import get_instance_trajectory_angles, get_mean_trajectory_angles
from concept_analysis.projector import BaseProjection


def get_elapsed_angle_regression_features(
    group_angles: dict[int, np.ndarray],
    force_ccw: bool = False,
    flatten: bool = True,
):
    """
        Pair each relative, unwrapped angle with normalized elapsed time.
            group_angles: dict[seq_len, np.ndarray of shape (N, seq_len) or (seq_len,) for N samples]
            force_ccw – if True, unwrap angles to be counter-clockwise only (no negative angles)

        Return
            flatten=False
                elapsed_dict = dict[int: seq_len, np.array(shape=(sample_angle.shape)) ]
                angles_dict = dict[int: seq_len, np.array(shape=(sample_angle.shape)) ]
    """

    elapsed_all, angles_all = [], []
    elapsed_dict, angles_dict = {}, {}

    for seq_len in sorted(group_angles.keys()):
        angles = group_angles[seq_len] # (N, seq_len) or (seq_len, )
        if angles.ndim not in (1, 2):
            raise ValueError(f"Expected 1D or 2D angles, got shape {angles.shape}")

        # (seq_len, )
        elapsed = np.arange(seq_len) / max(seq_len - 1, 1)

        for sample_angle in np.atleast_2d(angles): # vector of angle: (seq_len,) -> one row; (N, seq_len) -> N rows
            unwp_angle = np.unwrap(sample_angle)
            relative_angle = unwp_angle - unwp_angle[0]

            if force_ccw and relative_angle[-1] < 0:
                relative_angle *= -1

            if flatten:
                elapsed_all.append(elapsed)
                angles_all.append(relative_angle)
            else:
                elapsed_dict.setdefault(seq_len, []).append(elapsed)
                angles_dict.setdefault(seq_len, []).append(relative_angle)

    if flatten:
        return (np.concatenate(elapsed_all).reshape(-1, 1),
                np.concatenate(angles_all).reshape(-1, 1))

    return elapsed_dict, angles_dict


def polar_scores(
    subspace: BaseProjection,
    X_group_train: dict[int, np.ndarray],
    X_group_test: dict[int, np.ndarray] = None,
    group_centering_method: str = "centroid",
    force_ccw: bool = False,
    level: str = "mean",
) -> tuple[dict, dict]:
    """Fit elapsed time from polar angle on train groups and score test groups

    level – "mean" (default) fits on one averaged trajectory per duration group
          - "instance" fits on every individual trajectory

    force_ccw, if True, reverses sequences with cw rotation to ensure ccw

    Returns (train_dict, test_dict), each with keys:
        r2                  – R² decoding elapsed time from elapsed angle
        slope               – fitted slope (elapsed unit per radian)
        endpoint_completion – decoded angle at elapsed=1, divided by 2π

    Test values are None when X_group_test is not provided.
    """

    assert group_centering_method in ("centroid", "circle", "none")
    assert level in ("mean", "instance")

    get_angle_fn = get_instance_trajectory_angles if level == "instance" \
                  else get_mean_trajectory_angles

    train_groups = {k: v for k, v in X_group_train.items() if len(v) > 0}

    # dict[int: seq_len, np.array(shape=(seq_len,)) ] for level=mean
    # dict[int: seq_len, np.array(shape=(N, seq_len)) ] for level=instance
    train_angles = get_angle_fn(subspace, train_groups, group_centering_method)

    elapsed_train, angle_train = get_elapsed_angle_regression_features(
        group_angles=train_angles, force_ccw=force_ccw, flatten=True)

    linear_model = LinearRegression(fit_intercept=False).fit(angle_train, elapsed_train)

    slope = float(linear_model.coef_[0, 0])
    intercept = float(np.asarray(linear_model.intercept_).reshape(-1)[0])
    if np.isclose(slope, 0.0):
        endpoint_completion = np.nan
    else:
        endpoint_angle = (1.0 - intercept) / slope
        endpoint_completion = abs(endpoint_angle) / (2.0 * np.pi)

    train_scores = {
        "r2": linear_model.score(angle_train, elapsed_train),
        "slope": slope,
        "endpoint_completion": endpoint_completion,
    }
    test_scores = {key: None for key in train_scores}
    if X_group_test is not None:
        test_groups = {k: v for k, v in X_group_test.items()
                       if len(v) > 0}
        test_angles = get_angle_fn(subspace, test_groups, group_centering_method)
        elapsed_test, angle_test = get_elapsed_angle_regression_features(
            test_angles, force_ccw=force_ccw, flatten=True)
        test_scores = {"r2": linear_model.score(angle_test, elapsed_test)}
    return train_scores, test_scores


def explained_variance_scores(
    subspace: BaseProjection,
    X_group_train: dict[int, np.ndarray],
    X_frames_train: np.ndarray,
    X_group_test: dict[int, np.ndarray] | None = None,
    X_frames_test: np.ndarray | None = None,
    n_components: int = 2,
    denominator: str = "ambient",
) -> tuple[dict, dict]:
    """Measure projected variance of group means and individual frames."""

    def variance_fraction(features: np.ndarray) -> float:
        selected_projection = subspace.project(features, n_components=n_components)
        numerator = float(np.sum(np.var(selected_projection, axis=0)))
        if denominator == "ambient":
            denominator_variance = float(np.sum(np.var(features, axis=0)))
        elif denominator == "subspace":
            full_projection = subspace.project(features, n_components=subspace.n_components)
            denominator_variance = float(np.sum(np.var(full_projection, axis=0)))
        else:
            raise ValueError(
                f"denominator must be 'ambient' or 'subspace', got {denominator!r}")
        return numerator / denominator_variance

    def scores(groups: dict[int, np.ndarray], frames: np.ndarray) -> dict:
        mean_frames = np.concatenate([groups[key].mean(axis=0) for key in sorted(groups)])
        return {
            "mean_explained_variance": variance_fraction(mean_frames),
            "frame_explained_variance": variance_fraction(frames),
        }

    train_scores = scores(X_group_train, X_frames_train)
    test_scores = {key: None for key in train_scores}
    if X_group_test is not None and X_frames_test is not None:
        test_scores = scores(X_group_test, X_frames_test)
    return train_scores, test_scores


def optimal_line_mse(
    subspace,
    X_group_train: dict[int, np.ndarray],
    X_group_test: dict[int, np.ndarray] = None,
    group_centering_method: str = "centroid",
    force_ccw: bool = False,
    level: str = "mean",
) -> tuple[float, float | None]:
    """Return MSE to the better clockwise or counterclockwise full-circle optimal reference line."""
    assert group_centering_method in ("centroid", "circle", "none")
    assert level in ("mean", "instance")
    get_angle_fn = (get_instance_trajectory_angles if level == "instance"
                  else get_mean_trajectory_angles)

    def group_mse(groups: dict[int, np.ndarray]) -> float:
        nonempty_groups = {k: v for k, v in groups.items()
                           if len(v) > 0}
        angles = get_angle_fn(subspace, nonempty_groups, group_centering_method)
        elapsed_positions, relative_angles = get_elapsed_angle_regression_features(
            angles, force_ccw=force_ccw, flatten=True)

        mse_ccw = np.mean((elapsed_positions - relative_angles / (2 * np.pi)) ** 2)
        mse_cw = np.mean((elapsed_positions + relative_angles / (2 * np.pi)) ** 2)

        return float(min(mse_ccw, mse_cw))

    train_mse = group_mse(X_group_train)
    test_mse = None if X_group_test is None else group_mse(X_group_test)
    return train_mse, test_mse
