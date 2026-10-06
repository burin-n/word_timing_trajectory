"""Write exp06 boundary predictions using selected YAML parameters."""

import argparse
import json
from pathlib import Path
import numpy as np

from joblib import Parallel, delayed

from concept_analysis.annotations import idx_to_time
from concept_analysis.features import NpzFeatureReader
from concept_analysis.projector import (
    COVProjection, CPCAProjection, LDAProjection, LEACEProjection, PCAProjection,
    RandomProjection, BaseProjection
)
from concept_analysis.trajectory import detect_boundaries


SAVED_INPUT_FILES = ("temporal_subspace.pkl", "metadata.json")
PROJECTOR_TYPES = {
    "Z_CPCA": PCAProjection,
    "CPCA": CPCAProjection,
    "LEACE": LEACEProjection,
    "Z_LEACE": LEACEProjection,
    "RANDOM": RandomProjection,
    "LDA": LDAProjection,
    "COV": COVProjection,
}


def prediction_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--subspace-dir", type=Path, required=True,
                        help="root containing numeric saved layer directories")
    parser.add_argument("--input-feature-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="model/estimator output directory; layer and split directories are added")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--annotation-unit", required=True,
                        help="annotation unit used for configuration lookup and output naming")
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--legacy-feature-frame-rate-mode", action="store_true",
                        help="convert timestamps with the prediction cache's sampling rate and subsampling factor")
    return parser


def save_segment_predictions(config: dict, subspace: BaseProjection, X: np.ndarray, utt_ids: list[str], output_dir: Path,
                             *, feature_frame_rate_legacy_params: dict[str, int | float] | None = None) -> Path:
    """Write source-style config and timestamp lists, skipping existing lists."""
    feature_frame_rate_legacy_params = feature_frame_rate_legacy_params or {}
    output_dir = output_dir / str(config["param_id"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # write segmentation config
    (output_dir / "config").write_text(str(config) + "\n")

    for utt_id, x_utt in zip(utt_ids, X, strict=True):
        out_file = output_dir / f"{utt_id}.list"
        if out_file.exists():
            continue

        X_proj = subspace.transform(x_utt, n_components=config["n_components"])
        peaks = detect_boundaries(
            X_proj, prominence=config["prominence"],
            smooth_window_size=config["smooth_window_size"],
            threshold_max_angle=config.get("threshold_max_angle"),
        )
        with out_file.open("w") as stream:
            for peak in peaks:
                t = idx_to_time(peak, feature_frame_rate=config["feature_frame_rate"],
                                **feature_frame_rate_legacy_params)
                stream.write(f"{t}\n")
    return output_dir


def predict_layer(layer_dir, input_feature_dir, output_dir, split, annotation_unit,
                  configs, n_jobs=1, *, legacy_feature_frame_rate_mode=False):
    """Load one layer once and run parameter configurations in parallel."""
    for name in SAVED_INPUT_FILES:
        if not (layer_dir / name).is_file():
            raise FileNotFoundError(f"missing saved input: {layer_dir / name}")

    metadata = json.loads((layer_dir / "metadata.json").read_text())
    layer_id = metadata.get("layer_id")
    if isinstance(layer_id, bool) or not isinstance(layer_id, int) or layer_id < 0:
        raise ValueError(f"invalid layer_id in {layer_dir / 'metadata.json'}")
    if layer_id != int(layer_dir.name):
        raise ValueError(f"layer_id does not match directory {layer_dir}")
    estimator = metadata.get("estimator")
    if estimator not in PROJECTOR_TYPES:
        raise ValueError(f"Unsupported saved trajectory estimator: {estimator!r}")

    subspace = PROJECTOR_TYPES[estimator].load(layer_dir / "temporal_subspace.pkl")
    store = NpzFeatureReader(input_feature_dir)
    feature_frame_rate = store.feature_frame_rate_for_layer(layer_id)
    feature_frame_rate_legacy_params = {}
    timing_config = {
        "feature_frame_rate": feature_frame_rate,
        "feature_frame_rate_mode": "legacy" if legacy_feature_frame_rate_mode else "feature_frame_rate",
    }
    if legacy_feature_frame_rate_mode:
        feature_frame_rate_legacy_params = store.feature_frame_rate_for_layer(layer_id, legacy=True)
        timing_config.update(
            sampling_rate=feature_frame_rate_legacy_params["sr"],
            subsampling_factor=feature_frame_rate_legacy_params["subsampling_factor"],
        )
    utt_ids = store.list_ids(layer_id)
    n_jobs = min(len(configs), n_jobs)

    if not utt_ids:
        raise ValueError(f"no utterances in layer {layer_id} of {input_feature_dir}")

    X = store.read_layer(utt_ids, layer_id)
    for utt_id, x_utt in zip(utt_ids, X, strict=True):
        if x_utt.ndim != 2 or len(x_utt) == 0:
            raise ValueError(f"empty or invalid feature sequence for {utt_id} in layer {layer_id}")
        if x_utt.shape[1] != subspace.components.shape[1]:
            raise ValueError(f"incompatible feature dimensions for {utt_id} in layer {layer_id}")

    output_dir = output_dir / f"{layer_id:02d}" / f"{split}_{annotation_unit}-space"
    return Parallel(n_jobs=n_jobs)(delayed(save_segment_predictions)(
        {**config, **timing_config}, subspace, X, utt_ids, output_dir,
        feature_frame_rate_legacy_params=feature_frame_rate_legacy_params,
    ) for config in configs)


def main() -> None:
    parser = prediction_parser(__doc__)
    parser.add_argument("--layer-id", type=int, required=True,
                        help="single saved layer ID to predict")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--estimator", required=True,
                        help="estimator key used to select the YAML configuration")
    args = parser.parse_args()

    import yaml

    with args.config.open() as stream:
        configs = yaml.safe_load(stream)

    if not isinstance(configs, dict):
        parser.error("config must contain model/estimator/unit parameter mappings")
    try:
        config = configs[args.model_name][args.estimator][f"{args.annotation_unit}_space"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing selected configuration for {args.model_name}/{args.estimator}/{args.annotation_unit}_space") from exc
    if not isinstance(config, dict) or not config:
        raise ValueError("prediction configurations must be nonempty dictionaries")
    layer_dir = args.subspace_dir / str(args.layer_id)
    paths = predict_layer(
        layer_dir, args.input_feature_dir, args.output_dir, args.split, args.annotation_unit,
        configs=[config],
        legacy_feature_frame_rate_mode=args.legacy_feature_frame_rate_mode,
    )
    print(f"Wrote or resumed {len(paths)} layer/parameter configurations in {args.output_dir}")


if __name__ == "__main__":
    main()
