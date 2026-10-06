"""Phone, syllable, and word alignment conversion from source data.py.

Callers own model timing and train/test splits.
Alignments are pandas DataFrames. Frame intervals
are half-open; later rows overwrite earlier rows where intervals intersect.
"""

import math
from pathlib import Path
import re
from typing import Literal

import pandas as pd
import numpy as np

SILENCE_PHONES = {"<eps>", "<sil>", "<SIL>", "SIL", "SPN"}
EMPTY_LABEL = -1


def time_to_idx(
    t: float,
    *,
    sr: int | float | None = None,
    subsampling_factor: int | float | None = None,
    rounding: Literal["floor", "ceiling"] = "floor",
    feature_frame_rate: float = 50,
) -> int:
    """Convert seconds to frames; an explicit sampling pair overrides the default rate."""
    if sr is None and subsampling_factor is None:
        if not math.isfinite(feature_frame_rate) or feature_frame_rate <= 0:
            raise ValueError("feature_frame_rate must be finite and positive")
        # Remove tiny boundary errors, e.g. 0.58 * 50 = 28.999999999999996.
        # Keep floor/ceiling below; this precision is in frames, not seconds.
        frames = round(t * feature_frame_rate, 10)
    else:
        if sr is None or subsampling_factor is None:
            raise ValueError("sr and subsampling_factor must be supplied together")
        frames = t * sr / subsampling_factor
    if rounding == "floor":
        return int(math.floor(frames))
    return int(math.ceil(frames))


def idx_to_time(
    idx: int, *, sr: int | float | None = None,
    subsampling_factor: int | float | None = None,
    feature_frame_rate: float = 50,
) -> float:
    """Convert frames to seconds; an explicit sampling pair overrides the default rate."""
    if sr is None and subsampling_factor is None:
        if not math.isfinite(feature_frame_rate) or feature_frame_rate <= 0:
            raise ValueError("feature_frame_rate must be finite and positive")
        return idx / feature_frame_rate
    if sr is None or subsampling_factor is None:
        raise ValueError("sr and subsampling_factor must be supplied together")
    return idx * subsampling_factor / sr


def is_silence(phone: str, cd_phone_sep: str = "_", extra_silence: list[str] | None = None) -> bool:
    silence = SILENCE_PHONES | set(extra_silence or [])
    return any(p in silence for p in phone.split(cd_phone_sep))


def generate_class_index(classes: pd.Series) -> tuple[dict, list]:
    unique = np.unique(classes)
    class2id = {c: i for i, c in enumerate(sorted(unique))}
    id2class = list(class2id)
    return class2id, id2class


def get_phone_alignment(
    alignment_path: str | Path,
    class2id: dict | None = None,
    drop_silence: bool = True,
    feature_frame_rate: float = 50,
    *,
    feature_frame_rate_legacy_params: dict[str, int | float] | None = None,
) -> pd.DataFrame:
    """Load phone rows in seconds and return a frame-aligned DataFrame.

    The space-separated file needs utt_id, phone, start_time, and phone_dur.
    Phone suffixes and stress digits are removed as in the source loader.
    End frames follow
    the source convention: floor(start) + ceil(duration), not ceil(end).
    Supply a shared class2id for training and held-out data. Unknown retained
    phones raise KeyError. A legacy parameter dictionary uses the source's
    multiply-then-divide timing and takes precedence over feature_frame_rate.
    """
    feature_frame_rate_legacy_params = feature_frame_rate_legacy_params or {}
    alignment = pd.read_csv(alignment_path, sep=" ")
    alignment["utt_id"] = alignment["utt_id"].str.strip("lbi-")
    alignment["phone"] = alignment["phone"].map(
        lambda phone: re.sub(r"[0-9]", "", phone.split("_")[0])
    )
    if drop_silence:
        alignment = alignment[~alignment["phone"].map(is_silence)]

    if class2id is None:
        class2id, _ = generate_class_index(alignment["phone"])
    alignment["phone_id"] = alignment["phone"].map(lambda phone: class2id[phone])
    alignment["start_time"] = alignment["start_time"].map(
        lambda t: time_to_idx(t, feature_frame_rate=feature_frame_rate, **feature_frame_rate_legacy_params)
    )
    alignment["end_time"] = alignment["start_time"] + alignment["phone_dur"].map(
        lambda t: time_to_idx(t, rounding="ceiling", feature_frame_rate=feature_frame_rate,
                              **feature_frame_rate_legacy_params)
    )
    return alignment



def get_syllable_alignment(
    alignment_path: str | Path,
    class2id: dict | None = None,
    drop_silence: bool = True,
    feature_frame_rate: float = 50,
    *,
    feature_frame_rate_legacy_params: dict[str, int | float] | None = None,
) -> pd.DataFrame:
    """Load syllable rows in seconds and convert them to frame boundaries.

    The space-separated file needs utt_id, syllable, start_time, and
    syllable_dur. Stress digits are removed before silence filtering and
    vocabulary assignment. End frames are floor(start) + ceil(duration).
    A supplied class2id must use stress-free tokens; unknown tokens raise KeyError.
    A legacy parameter dictionary takes precedence over feature_frame_rate.
    Use label_types=["token_pos"] with load_data for trajectories.
    """
    feature_frame_rate_legacy_params = feature_frame_rate_legacy_params or {}
    alignment = pd.read_csv(alignment_path, sep=" ")
    # normalise to CMU-dict phone set
    alignment["syllable"] = alignment["syllable"].str.replace(r"[0-9]", "", regex=True)
    if drop_silence:
        alignment = alignment[~alignment["syllable"].map(is_silence)]
    if class2id is None:
        class2id, _ = generate_class_index(alignment["syllable"])
    alignment["syllable_id"] = alignment["syllable"].map(lambda token: class2id[token])
    alignment["start_time"] = alignment["start_time"].map(
        lambda t: time_to_idx(t, feature_frame_rate=feature_frame_rate, **feature_frame_rate_legacy_params)
    )
    alignment["end_time"] = alignment["start_time"] + alignment["syllable_dur"].map(
        lambda t: time_to_idx(t, rounding="ceiling", feature_frame_rate=feature_frame_rate,
                              **feature_frame_rate_legacy_params)
    )
    return alignment


def get_word_alignment(
    alignment_path: str | Path,
    class2id: dict | None = None,
    drop_silence: bool = True,
    feature_frame_rate: float = 50,
    *,
    feature_frame_rate_legacy_params: dict[str, int | float] | None = None,
) -> pd.DataFrame:
    """Load word rows in seconds and convert them to frame boundaries.

    The space-separated file needs utt_id, word, start_time, and word_dur.
    Word labels are left as supplied. End frames are floor(start) +
    ceil(duration). A supplied class2id is reused; unknown tokens raise KeyError.
    A legacy parameter dictionary takes precedence over feature_frame_rate.
    Use label_types=["token_pos"] with load_data for trajectories.
    """
    feature_frame_rate_legacy_params = feature_frame_rate_legacy_params or {}
    alignment = pd.read_csv(alignment_path, sep=" ")
    alignment["utt_id"] = alignment["utt_id"].str.strip("lbi-")
    if drop_silence:
        alignment = alignment[~alignment["word"].map(is_silence)]
    if class2id is None:
        class2id, _ = generate_class_index(alignment["word"])
    alignment["word_id"] = alignment["word"].map(lambda token: class2id[token])
    alignment["start_time"] = alignment["start_time"].map(
        lambda t: time_to_idx(t, feature_frame_rate=feature_frame_rate, **feature_frame_rate_legacy_params)
    )
    alignment["end_time"] = alignment["start_time"] + alignment["word_dur"].map(
        lambda t: time_to_idx(t, rounding="ceiling", feature_frame_rate=feature_frame_rate,
                              **feature_frame_rate_legacy_params)
    )
    return alignment
