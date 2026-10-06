"""Peak-based boundary detection on projected utterances."""

import numpy as np
from scipy.signal import find_peaks

from .geometry import get_sequence_relative_angles, smoothing


def detect_boundaries(
    projected_traj: np.ndarray,
    *,
    prominence: float,
    smooth_window_size: int = 1,
    threshold_max_angle: float | None = None,
) -> np.ndarray:
    """Return frame indices using the source exp06 prediction calculation.

    Angles are relative to the first frame. Smooth before inversion, retaining
    the source zero padding; apply the angle threshold to unsmoothed angles.
    No frames are removed and utterance endpoints are not inserted.
    """
    if projected_traj.ndim != 2 or not all(projected_traj.shape):
        raise ValueError("projected_traj must be a nonempty (frames, components) array")
    if (isinstance(smooth_window_size, bool)
            or not isinstance(smooth_window_size, int) or smooth_window_size < 1):
        raise ValueError("smooth_window_size must be a positive integer")
    if not np.isfinite(prominence) or prominence < 0:
        raise ValueError("prominence must be finite and nonnegative")
    if threshold_max_angle is not None and not np.isfinite(threshold_max_angle):
        raise ValueError("threshold_max_angle must be finite or None")

    angles = get_sequence_relative_angles(projected_traj, relative_to=0, angle_mode="cosine_arccos")
    smoothed = smoothing(angles, window=smooth_window_size)
    # inverted signal to detect troughs
    smoothed = smoothed.max() - smoothed
    peaks, _ = find_peaks(smoothed, prominence=prominence)
    if threshold_max_angle is not None:
        peaks = peaks[angles[peaks] <= threshold_max_angle]
    return peaks
