"""Fit temporal subspaces from feature caches and alignments."""

import argparse
import json
from pathlib import Path

import numpy as np

from concept_analysis.annotations import (
    get_phone_alignment, get_syllable_alignment, get_word_alignment, load_data,
)
from concept_analysis.features import NpzFeatureReader
from concept_analysis.trajectory import (
    get_mean_trajectories, get_temporal_subspace, group_and_preprocess_feature,
)

ALIGNMENT_LOADERS = {
    "phone": get_phone_alignment,
    "word": get_word_alignment,
    "syllable": get_syllable_alignment,
}


def parse_layer_ids(value: str) -> list[int]:
    try:
        layer_ids = [int(part.strip()) for part in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated integer layer IDs") from exc
    if len(layer_ids) != len(set(layer_ids)):
        raise argparse.ArgumentTypeError("layer IDs must be unique")
    return layer_ids


def fit_temporal_subspaces(
    store: NpzFeatureReader,
    alignment_path: Path,
    output_dir: Path,
    *,
    layer_id: int,
    annotation_unit: str,
    estimator: str = "Z_CPCA",
    min_dur: float = 0.0,
    max_dur: float = 5.0,
    low_freq_threshold: int = 50,
    drop_silence: bool = True,
    legacy_feature_frame_rate_mode: bool = False,
) -> None:
    """Fit using the selected layer's manifest frame rate."""
    effective_frame_rate = store.feature_frame_rate_for_layer(layer_id)
    feature_frame_rate_legacy_params = {}
    if legacy_feature_frame_rate_mode:
        feature_frame_rate_legacy_params = store.feature_frame_rate_for_layer(layer_id, legacy=True)

    alignment = ALIGNMENT_LOADERS[annotation_unit](
        alignment_path, drop_silence=drop_silence,
        feature_frame_rate=effective_frame_rate,
        feature_frame_rate_legacy_params=feature_frame_rate_legacy_params,
    )

    ids = alignment["utt_id"].drop_duplicates().tolist()
    if not ids:
        raise ValueError("alignment selects no utterances")
    features = store.read_layer(ids, layer_id)

    X, y = load_data(
        id2features=dict(zip(ids, features)), alignment=alignment, label_types=["token_pos"],
        njobs=1, preserve_sequence=True,
    )

    subspace = get_temporal_subspace(
        X, y, estimator=estimator, min_dur=min_dur, max_dur=max_dur,
        low_freq_threshold=low_freq_threshold, label_type="token_pos",
        feature_frame_rate=effective_frame_rate,
        feature_frame_rate_legacy_params=feature_frame_rate_legacy_params,
    )

    X_group, y_group = group_and_preprocess_feature(
        X, y, min_dur=min_dur, max_dur=max_dur,
        low_freq_threshold=low_freq_threshold, label_type="token_pos",
        feature_frame_rate=effective_frame_rate,
        feature_frame_rate_legacy_params=feature_frame_rate_legacy_params,
    )

    templates = get_mean_trajectories(X_group)
    output_dir.mkdir(parents=True, exist_ok=True)
    subspace.save(output_dir / "temporal_subspace.pkl")
    np.savez_compressed(output_dir / f"mean_trajectories.npz",
                        **{str(k): value for k, value in templates.items()})
    metadata = {
        "estimator": estimator,
        "layer_id": layer_id,
        "annotation_unit": f"{annotation_unit}_token_pos",
        "feature_frame_rate": effective_frame_rate,
        "feature_frame_rate_mode": "legacy" if legacy_feature_frame_rate_mode else "feature_frame_rate",
        "min_dur": min_dur,
        "max_dur": max_dur,
        "low_freq_threshold": low_freq_threshold,
        "drop_silence": drop_silence,
    }
    if legacy_feature_frame_rate_mode:
        metadata.update(sampling_rate=feature_frame_rate_legacy_params["sr"],
                        subsampling_factor=feature_frame_rate_legacy_params["subsampling_factor"])
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--alignment-path", type=Path, required=True)

    parser.add_argument("--layer-id", type=parse_layer_ids, required=True,
                        help="one layer ID or comma-separated IDs, for example 1,2,3")
    parser.add_argument("--estimator", choices=(
        "Z_CPCA", "CPCA", "LEACE", "Z_LEACE", "COV", "RANDOM", "LDA",
    ),
                        default="Z_CPCA")
    
    parser.add_argument("--annotation-unit", choices=ALIGNMENT_LOADERS, required=True)
    parser.add_argument("--keep-silence", action="store_true")
    parser.add_argument("--legacy-feature-frame-rate-mode", action="store_true",
                        help="use the manifest sampling pair for source-style alignment and duration rounding")
    parser.add_argument("--min-dur", type=float, default=0.1)
    parser.add_argument("--max-dur", type=float, default=0.7)
    parser.add_argument("--low-freq-threshold", type=int, default=50)
    args = parser.parse_args()
    try:
        store = NpzFeatureReader(cache_dir=args.input_feature_dir)
        for layer_id in args.layer_id:
            store.feature_frame_rate_for_layer(layer_id)
            if args.legacy_feature_frame_rate_mode:
                store.feature_frame_rate_for_layer(layer_id, legacy=True)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    for layer_id in args.layer_id:
        output_dir = args.output_dir if len(args.layer_id) == 1 else args.output_dir / str(layer_id)
        fit_temporal_subspaces(store, args.alignment_path, output_dir,
                     layer_id=layer_id, annotation_unit=args.annotation_unit,
                     estimator=args.estimator,
                     min_dur=args.min_dur, max_dur=args.max_dur,
                     low_freq_threshold=args.low_freq_threshold,
                     drop_silence=not args.keep_silence,
                     legacy_feature_frame_rate_mode=args.legacy_feature_frame_rate_mode)


if __name__ == "__main__":
    main()
