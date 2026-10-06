"""Score cached trajectories with subspaces saved by fit_temporal_subspace.py."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed

from concept_analysis.annotations import (
    get_phone_alignment, get_syllable_alignment, get_word_alignment, load_data,
)
from concept_analysis.features import NpzFeatureReader
from concept_analysis.projector import (
    COVProjection, CPCAProjection, LDAProjection, LEACEProjection, PCAProjection,
    RandomProjection,
)
from concept_analysis.trajectory import (
    explained_variance_scores, group_and_preprocess_feature,
    optimal_line_mse, polar_scores,
)


ALIGNMENT_LOADERS = {
    "phone": get_phone_alignment,
    "word": get_word_alignment,
    "syllable": get_syllable_alignment,
}
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


def load_metric_inputs(
    data: NpzFeatureReader,
    alignment_path: Path,
    *,
    layer_id: int,
    metadata: dict,
    feature_frame_rate_legacy_params: dict[str, int | float] | None = None,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """Load labelled frames and group sequences with the saved fit settings."""

    annotation_unit = metadata["annotation_unit"].removesuffix("_token_pos")
    
    if annotation_unit not in ALIGNMENT_LOADERS:
        raise ValueError(f"Unsupported annotation unit in metadata: {annotation_unit!r}")
    
    alignment = ALIGNMENT_LOADERS[annotation_unit](
        alignment_path,
        drop_silence=metadata["drop_silence"],
        feature_frame_rate=metadata["feature_frame_rate"],
        feature_frame_rate_legacy_params=feature_frame_rate_legacy_params,
    )
    utterance_ids = alignment["utt_id"].drop_duplicates().tolist()

    if not utterance_ids:
        raise ValueError(f"alignment selects no utterances: {alignment_path}")
    
    feature_sequences = data.read_layer(utterance_ids, layer_id)
    feature_sequences, token_positions = load_data(
        id2features=dict(zip(utterance_ids, feature_sequences)), alignment=alignment,
        label_types=["token_pos"], njobs=1, preserve_sequence=True,
    )
    
    grouped_sequences, _ = group_and_preprocess_feature(
        feature_sequences, token_positions,
        label_type="token_pos",
        min_dur=metadata["min_dur"],
        max_dur=metadata["max_dur"],
        low_freq_threshold=metadata["low_freq_threshold"],
        feature_frame_rate=metadata["feature_frame_rate"],
        feature_frame_rate_legacy_params=feature_frame_rate_legacy_params,
    )

    if not grouped_sequences:
        raise ValueError(f"no trajectory groups remain for {alignment_path}")
    all_frames = np.concatenate(feature_sequences, axis=0)
    
    return grouped_sequences, all_frames


def validate_test_inputs(
    test_data: NpzFeatureReader | None,
    test_alignment_path: Path | None,
) -> None:
    if (test_data is None) != (test_alignment_path is None):
        raise ValueError("test feature directory and test alignment path must be supplied together")


def score_subspace(
    subspace_dir: Path,
    train_data: NpzFeatureReader,
    train_alignment_path: Path,
    test_data: NpzFeatureReader | None = None,
    test_alignment_path: Path | None = None,
    *,
    evaluation_level: str = "mean",
    group_centering_method: str = "centroid",
    force_ccw: bool = False,
    n_components: int = 2,
    legacy_feature_frame_rate_mode: bool = False,
) -> dict:
    """Compute the three migrated metric families for one saved layer."""
    validate_test_inputs(test_data, test_alignment_path)

    metadata = json.loads((subspace_dir / "metadata.json").read_text())
    estimator = metadata.get("estimator")
    if estimator not in PROJECTOR_TYPES:
        raise ValueError(f"Unsupported saved trajectory estimator: {estimator!r}")
    subspace = PROJECTOR_TYPES[estimator].load(subspace_dir / "temporal_subspace.pkl")
    available_components = subspace.components.shape[0]
    
    if available_components < 2:
        raise ValueError("polar metrics require at least two fitted components")
    if not 1 <= n_components <= available_components:
        raise ValueError(
            f"n_components must be between 1 and {available_components}, got {n_components}")
    
    layer_id = metadata["layer_id"]
    train_feature_frame_rate_legacy_params = {}
    test_feature_frame_rate_legacy_params = {}
    if legacy_feature_frame_rate_mode:
        train_feature_frame_rate_legacy_params = train_data.feature_frame_rate_for_layer(layer_id, legacy=True)
        if test_data is not None:
            test_feature_frame_rate_legacy_params = test_data.feature_frame_rate_for_layer(layer_id, legacy=True)
            assert train_feature_frame_rate_legacy_params == test_feature_frame_rate_legacy_params

    train_groups, train_frames = load_metric_inputs(
        train_data, train_alignment_path, layer_id=layer_id, metadata=metadata,
        feature_frame_rate_legacy_params=train_feature_frame_rate_legacy_params)

    test_groups = test_frames = None
    if test_data is not None:
        test_groups, test_frames = load_metric_inputs(
            test_data, test_alignment_path, layer_id=layer_id, metadata=metadata,
            feature_frame_rate_legacy_params=test_feature_frame_rate_legacy_params)

    polar_train, polar_test = polar_scores(
        subspace, train_groups, test_groups,
        group_centering_method=group_centering_method, force_ccw=force_ccw,
        level=evaluation_level,
    )
    ambient_train, ambient_test = explained_variance_scores(
        subspace, train_groups, train_frames, test_groups, test_frames,
        n_components=n_components, denominator="ambient",
    )
    subspace_train, subspace_test = explained_variance_scores(
        subspace, train_groups, train_frames, test_groups, test_frames,
        n_components=n_components, denominator="subspace",
    )
    line_train, line_test = optimal_line_mse(
        subspace, train_groups, test_groups,
        group_centering_method=group_centering_method, force_ccw=force_ccw,
        level=evaluation_level,
    )

    return {
        "layer_id": layer_id,
        "annotation_unit": metadata["annotation_unit"],
        "evaluation_level": evaluation_level,
        "group_centering_method": group_centering_method,
        "force_ccw": force_ccw,
        "n_components": n_components,
        "subspace_dir": str(subspace_dir),
        "feature_frame_rate_mode": "legacy" if legacy_feature_frame_rate_mode else "feature_frame_rate",
        "train": {
            "data_dir": str(train_data.cache_dir),
            "alignment_path": str(train_alignment_path),
            "feature_frame_rate_legacy_params": train_feature_frame_rate_legacy_params,
            "duration_counts": {str(key): len(values)
                                for key, values in sorted(train_groups.items())},
            "frame_count": len(train_frames),
            "polar_scores": polar_train,
            "explained_variance_ambient": ambient_train,
            "explained_variance_subspace": subspace_train,
            "optimal_line_mse": line_train,
        },
        "test": None if test_data is None else {
            "data_dir": str(test_data.cache_dir),
            "alignment_path": str(test_alignment_path),
            "feature_frame_rate_legacy_params": test_feature_frame_rate_legacy_params,
            "duration_counts": {str(key): len(values)
                                for key, values in sorted(test_groups.items())},
            "frame_count": len(test_frames),
            "polar_scores": polar_test,
            "explained_variance_ambient": ambient_test,
            "explained_variance_subspace": subspace_test,
            "optimal_line_mse": line_test,
        },
    }


def evaluate_all_layers(
    input_subspace_dir: Path,
    output_dir: Path,
    train_data: NpzFeatureReader,
    train_alignment_path: Path,
    test_data: NpzFeatureReader | None = None,
    test_alignment_path: Path | None = None,
    *,
    evaluation_level: str = "mean",
    group_centering_method: str = "centroid",
    force_ccw: bool = False,
    n_components: int = 2,
    n_jobs: int = 1,
    legacy_feature_frame_rate_mode: bool = False,
) -> list[Path]:

    """Score a flat saved subspace or every numeric layer directory.

    Use n_jobs joblib workers while preserving numeric layer order.
    """
    validate_test_inputs(test_data, test_alignment_path)
    if n_jobs == 0:
        raise ValueError("n_jobs must not be zero")

    layer_dirs = sorted(
        (child for child in input_subspace_dir.iterdir() if child.is_dir() and child.name.isdecimal()),
        key=lambda layer_dir: int(layer_dir.name),
    )

    if not layer_dirs:
        layer_dirs = [input_subspace_dir]

    for layer_dir in layer_dirs:
        missing_files = [filename for filename in SAVED_INPUT_FILES
                         if not (layer_dir / filename).is_file()]
        if missing_files:
            raise FileNotFoundError(f"missing saved inputs in {layer_dir}: {', '.join(missing_files)}")
        metadata = json.loads((layer_dir / "metadata.json").read_text())
        if layer_dir != input_subspace_dir and metadata.get("layer_id") != int(layer_dir.name):
            raise ValueError(
                f"layer_id in {layer_dir / 'metadata.json'} does not match directory {layer_dir.name}")

    scored_layers = Parallel(n_jobs=n_jobs)(
        delayed(score_subspace)(
            layer_dir, train_data, train_alignment_path, test_data, test_alignment_path,
            evaluation_level=evaluation_level, group_centering_method=group_centering_method,
            force_ccw=force_ccw, n_components=n_components,
            legacy_feature_frame_rate_mode=legacy_feature_frame_rate_mode,
        )
        for layer_dir in layer_dirs
    )

    output_paths = []
    for layer_dir, scores in zip(layer_dirs, scored_layers, strict=True):
        output_path = (output_dir / "metrics.json" if layer_dir == input_subspace_dir else
                       output_dir / f"metrics_{layer_dir.name}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(scores, indent=2) + "\n")
        output_paths.append(output_path)
    return output_paths


def write_scores_csv(output_paths: list[Path], output_csv: Path, model_name: str) -> None:
    """Write the supported exp20 columns for this invocation's scored layers."""
    rows = []
    for path in output_paths:
        scores = json.loads(path.read_text())
        metadata = json.loads((Path(scores['subspace_dir']) / 'metadata.json').read_text())
        train, test = scores['train'], scores['test']
        row = {
            'model_name': model_name,
            'layer_id': scores['layer_id'],
            'label_name': scores['annotation_unit'].removesuffix('_token_pos'),
            'estimator': metadata['estimator'],
            'r2_train': train['polar_scores']['r2'],
            'r2_test': test['polar_scores']['r2'] if test is not None else None,
            'slope': train['polar_scores']['slope'],
            'endpoint_completion': train['polar_scores']['endpoint_completion'],
            'optimal_line_mse_train': train['optimal_line_mse'],
            'optimal_line_mse_test': test['optimal_line_mse'] if test is not None else None,
        }
        for column, family, metric in (
            ('mean_explained_variance', 'explained_variance_ambient', 'mean_explained_variance'),
            ('mean_explained_variance_subspace', 'explained_variance_subspace', 'mean_explained_variance'),
            ('frame_explained_variance', 'explained_variance_ambient', 'frame_explained_variance'),
        ):
            row[f'{column}_train'] = train[family][metric]
            row[f'{column}_test'] = test[family][metric] if test is not None else None
        rows.append(row)
    if not rows:
        raise ValueError('No temporal scores to write')
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """
        This script evaluates temporal subspaces in `--subspace-dir`
        by training a linear regressor on `--train-feature-dir` and
        (if provided) report metrics for `--test-feature-dir`.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subspace-dir", type=Path, required=True,
                        help="fit_temporal_subspace.py output, flat or with numeric layer directories")
    parser.add_argument("--train-feature-dir", type=Path, required=True)
    parser.add_argument("--test-feature-dir", type=Path)
    parser.add_argument("--train-alignment-path", type=Path, required=True)
    parser.add_argument("--test-alignment-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path,
                        help="also write supported exp20 score columns for these layers")
    parser.add_argument("--model-name", help="model identifier for --output-csv")
    parser.add_argument("--evaluation-level", choices=("mean", "instance"), default="mean")
    parser.add_argument("--group-centering-method", choices=("centroid", "circle", "none"),
                        default="centroid")
    parser.add_argument("--force-ccw", action="store_true")
    parser.add_argument("--legacy-feature-frame-rate-mode", action="store_true",
                        help="use each cache's sampling pair for source-style alignment and duration rounding")
    parser.add_argument("--n-components", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="parallel layer workers (default: 1; -1 uses all CPUs)")

    args = parser.parse_args()
    if args.output_csv is not None and not args.model_name:
        parser.error("--model-name is required with --output-csv")
    if args.n_components < 1:
        parser.error("--n-components must be positive")
    if args.n_jobs == 0:
        parser.error("--n-jobs must not be zero")
    if (args.test_feature_dir is None) != (args.test_alignment_path is None):
        parser.error("--test-feature-dir and --test-alignment-path must be supplied together")

    train_data = NpzFeatureReader(cache_dir=args.train_feature_dir)
    test_data = (NpzFeatureReader(cache_dir=args.test_feature_dir)
                 if args.test_feature_dir is not None else None)

    output_paths = evaluate_all_layers(
        args.subspace_dir, args.output_dir, train_data, args.train_alignment_path,
        test_data, args.test_alignment_path,
        evaluation_level=args.evaluation_level,
        group_centering_method=args.group_centering_method,
        force_ccw=args.force_ccw, n_components=args.n_components, n_jobs=args.n_jobs,
        legacy_feature_frame_rate_mode=args.legacy_feature_frame_rate_mode,
    )
    if args.output_csv is not None:
        write_scores_csv(output_paths, args.output_csv, args.model_name)


if __name__ == "__main__":
    main()
