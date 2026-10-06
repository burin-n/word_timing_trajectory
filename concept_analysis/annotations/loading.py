"""Expand alignment rows into frame labels for cached feature arrays."""

import logging
import time

from joblib import Parallel, delayed
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
TOKEN_ID_COLUMNS = ("phone_id", "syllable_id", "word_id")


def _process_utterance(features, rows, label_types):
    mask = np.zeros(len(features), dtype=bool)
    labels = []
    for label_type in label_types:
        labels.append( np.empty(len(features), dtype=int) )

    for row_index, (start, end) in enumerate(zip(rows["start_time"], rows["end_time"])):
        mask[start:end] = True
        for label_index, label_type in enumerate(label_types):
            if label_type == "token_pos":
                label_ = row_index  
            else:
                label_ = rows[label_type].iloc[row_index]
            labels[label_index][start:end] = label_

    # return frames with non-empty labels
    return (features[mask], *(label[mask] for label in labels))


def load_data(id2features: dict[str, np.ndarray], alignment: pd.DataFrame, *, label_types: list[str], preserve_sequence=False, njobs=4):
    """
    This function maps the features from id2features and the labels from alignemnt.
    Return ``(X, *labels)`` in alignment utterance order and requested label order.
        
        Given that the phone_alignment has `phone_id` column
        X, y_phone, y_phone_pos = load_data(id2features, phone_alignment, label_types=["phone_id", "token_pos"])

        Given that the phone_speaker_alignment has `phone_id` and `speaker_id` columns
        X, y_phone, y_spk = load_data(id2features, phone_speaker_alignment, label_types=["phone_id", "speaker_id"])
        
        Given that the word_alignment has `word_id` column
        X, y_word_pos = load_data(id2features, word_alignment, label_types=["word_id"])

    preserve_sequence=True; Return X [ np.array(utt_len, hdim) ] * n_uniq_utt_id and y_ [utt_len]* n_uniq_utt_id
    preserve_sequence=False; Return flatten features X (total_frames, hdim) and y_ (total_frames, )
    """

    if not isinstance(label_types, list) or not label_types or any(
        not isinstance(label_type, str) for label_type in label_types
    ):
        raise ValueError("label_types must be a nonempty list of column names or 'token_pos'")

    if not isinstance(alignment, pd.DataFrame):
        raise TypeError("alignment must be a pandas DataFrame")
    
    required = {"utt_id", "start_time", "end_time"} | (set(label_types) - {"token_pos"})
    missing = required - set(alignment.columns)

    if missing:
        raise ValueError(f"Missing alignment columns: {sorted(missing)}")
    if "token_pos" in label_types:
        token_columns = set(TOKEN_ID_COLUMNS) & set(alignment.columns)
        if len(token_columns) != 1:
            raise ValueError("token_pos requires exactly one of phone_id, syllable_id, or word_id in alignment")
        required |= token_columns
    null_columns = [column for column in required if alignment[column].isna().any()]
    if null_columns:
        raise ValueError(f"Null alignment values in columns: {sorted(null_columns)}")
    if njobs < 1:
        raise ValueError("njobs must be positive")


    ids = alignment["utt_id"].drop_duplicates().tolist()
    if not ids:
        raise ValueError("alignment selects no utterances")
    missing_ids = [utt_id for utt_id in ids if utt_id not in id2features]
    if missing_ids:
        raise KeyError(f"Missing feature arrays for alignment utterances: {missing_ids!r}")

    start = time.time()
    groups = [alignment[alignment["utt_id"] == utt_id] for utt_id in ids]
    results = Parallel(n_jobs=njobs)(
        delayed(_process_utterance)(id2features[utt_id], rows, label_types)
        for utt_id, rows in zip(ids, groups)
    )
    if preserve_sequence:
        output = tuple([result[i] for result in results] for i in range(len(label_types) + 1))
    else:
        output = tuple(np.concatenate([result[i] for result in results]) for i in range(len(label_types) + 1))
    log.debug("Alignment data loaded in %.1fs", time.time() - start)
    return output
