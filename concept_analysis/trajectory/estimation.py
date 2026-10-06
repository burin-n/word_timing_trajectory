"""Fit temporal subspaces from token-position-labelled feature sequences."""

import numpy as np

from concept_analysis.projector import (
    BaseProjection, COVProjection, LDAProjection, LEACEProjection, PCAProjection,
    CPCAProjection, RandomProjection, 
)
from concept_analysis.trajectory.grouping import group_and_preprocess_feature


def get_mean_trajectories(X_group: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    """Return one unprojected (frames, features) template per duration."""
    return {k: X_group[k].mean(axis=0) for k in sorted(X_group)}


def get_temporal_subspace(
    X: list[np.ndarray],
    y: list[np.ndarray],
    estimator: str = "Z_CPCA",
    subspace_name: str = "temporal",
    *,
    min_dur: float = 0.0,
    max_dur: float = 5.0,
    low_freq_threshold: int = 50,
    feature_frame_rate: float = 50,
    label_type: str,
    feature_frame_rate_legacy_params: dict[str, int | float] | None = None,
) -> BaseProjection:
    """Group labelled sequences and fit the selected temporal estimator.

    ``label_type`` must declare token-position labels. Duration limits are in
    seconds; grouping keys are frame counts. LEACE and COV fit every grouped
    frame with its within-token position label. Z_LEACE instead assigns a
    distinct label to each (duration, within-token position) pair.
    A legacy parameter dictionary selects legacy duration-limit conversion.
    """
    if estimator not in (
        "Z_CPCA", "Z_LEACE", "LEACE", "CPCA", "COV", "LDA", "RANDOM",
    ):
        raise ValueError(f"Unsupported trajectory estimator: {estimator!r}")

    # Group feature segments by durations
    X_group, _ = group_and_preprocess_feature(
        X, y, min_dur=min_dur, max_dur=max_dur,
        low_freq_threshold=low_freq_threshold, label_type=label_type,
        feature_frame_rate=feature_frame_rate,
        feature_frame_rate_legacy_params=feature_frame_rate_legacy_params,
    )

    if estimator == "RANDOM":
        H = next(iter(X_group.values())).shape[-1]
        return RandomProjection(H, n_components=3)

    elif estimator == "Z_CPCA":
        X_mean = get_mean_trajectories(X_group)
        X = np.concatenate([X_mean[k] for k in sorted(X_mean)])
        return PCAProjection(name=f"{estimator}_{subspace_name}").fit(X)

    else:
        X_batches = []
        y_batches = []
        
        y_offset = 0
        for k in sorted(X_group):
            X = X_group[k]
            N, L, H = X.shape
            X_batches.append(X.reshape(-1, H))
            y = np.arange(L)
            if estimator in ["Z_LEACE"]:
                # y = position relative to segment onset and total segment length (timing_label)
                y += y_offset
                y_offset += L
            else:
                # y = position relative to segment onset
                pass
            y_batches.append(np.tile(y, N))

        proj_cls = {
            "LEACE": LEACEProjection, "Z_LEACE": LEACEProjection,
            "COV": COVProjection, "LDA": LDAProjection,
            "CPCA": CPCAProjection,
        }[estimator]
        X = np.concatenate(X_batches)
        y = np.concatenate(y_batches)
        return proj_cls(name=f"{estimator}_{subspace_name}").fit(X, y)
