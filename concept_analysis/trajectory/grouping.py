"""Duration grouping of token-position-labelled sequences.

Pass labels from load_data(..., label_types=["token_pos"], preserve_sequence=True).
Integer arrays alone do not identify their semantics: the required label_type
argument is a caller declaration, not automatic provenance detection.
"""

import logging
import numpy as np
from concept_analysis.annotations import time_to_idx

log = logging.getLogger(__name__)


def group_and_preprocess_feature(
    X: list[np.ndarray],
    y: list[np.ndarray],
    min_dur: float = 0.0,
    max_dur: float = 5.0,
    low_freq_threshold: int = 50,
    feature_frame_rate: float = 50,
    *,
    label_type: str,
    feature_frame_rate_legacy_params: dict[str, int | float] | None = None,
) -> tuple[dict, dict]:
    """Group frame-level features by segment duration, then filter rare/short/long groups.

    Returns ``(X_group, y_group)`` dicts keyed by duration in frames.
    A legacy parameter dictionary selects legacy multiply-then-divide timing.
    """
    X_group, y_group = group_feature_by_dur(X, y, label_type=label_type)
    X_group, y_group = _filter_low_freq(X_group, y_group, low_freq_threshold)
    feature_frame_rate_legacy_params = feature_frame_rate_legacy_params or {}
    min_frames = time_to_idx(min_dur, feature_frame_rate=feature_frame_rate,
                             **feature_frame_rate_legacy_params)
    max_frames = time_to_idx(max_dur, feature_frame_rate=feature_frame_rate,
                             **feature_frame_rate_legacy_params)
    X_group, y_group = _filter_dur(X_group, y_group, min_frames, max_frames)
    n_seqs = sum(v.shape[0] for v in X_group.values())
    log.debug("Grouped features: %d duration buckets, %d sequences (dur=%.2f–%.2fs)",
              len(X_group), n_seqs, min_dur, max_dur)
    return X_group, y_group


def group_feature_by_dur(
    X: list[np.ndarray],
    y: list[np.ndarray],
    exclude_first_label: bool = False,
    exclude_last_label: bool = False,
    *,
    label_type: str,
) -> tuple[dict, dict]:
    """Split frame-level sequences into fixed-length segments grouped by duration.
     input
        X : list of features [(n_frames , dim )] x n_utterance
        y : list of labels [(n_frames)] x n_utterance
     return
        X_group : dict {label_len: (samples, len, dim)}
        y_group : dict {label_len: (samples,)}
    """
    if label_type != "token_pos":
        raise ValueError("Grouping requires label_type='token_pos', not class IDs")

    X_group: dict[int, list] = {}
    y_group: dict[int, list] = {}

    for x_utt, y_utt in zip(X, y):
        N = len(x_utt)
        if N == 0:
            continue
        onset = 0
        for t in range(1, N):
            if y_utt[t] != y_utt[onset]:
                if onset > 0 or not exclude_first_label:
                    _append_segment(X_group, y_group, x_utt, y_utt, onset, t)
                onset = t
        if not exclude_last_label:
            _append_segment(X_group, y_group, x_utt, y_utt, onset, N)

    for k in X_group:
        X_group[k] = np.asarray(X_group[k])
        y_group[k] = np.asarray(y_group[k])
    return X_group, y_group


def _append_segment(X_group, y_group, x_utt, y_utt, onset, offset):
    length = offset - onset
    X_group.setdefault(length, []).append(x_utt[onset:offset])
    y_group.setdefault(length, []).append(y_utt[onset])


def _filter_low_freq(X_group, y_group=None, threshold: int = 100):
    filtered_X = {k: v for k, v in X_group.items() if v.shape[0] >= threshold}
    if y_group is None:
        return filtered_X
    filtered_y = {k: y_group[k] for k in filtered_X}
    return filtered_X, filtered_y


def _filter_dur(X_group, y_group, min_frames: int, max_frames: int):
    valid = {k for k in X_group if min_frames <= X_group[k].shape[1] <= max_frames}
    return {k: X_group[k] for k in valid}, {k: y_group[k] for k in valid}
