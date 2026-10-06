"""Write the exp06 parameter grid's predictions from saved subspaces and caches."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np

from predict_boundaries import predict_layer, prediction_parser


def parse_layer_ids(value: str) -> list[int]:
    return [int(part) for part in value.split(",")]


def generate_inference_param_grid() -> Iterator[dict]:
    param_id = 0
    for prominence in np.linspace(0, np.pi / 4, 10):
        for threshold_max_angle in np.linspace(np.pi / 4, np.pi, 10):
            for smooth_window_size in [1, 2, 3]:
                for n_components in [2]:
                    yield {
                        "param_id": param_id,
                        "prominence": round(float(prominence), 3),
                        "smooth_window_size": smooth_window_size,
                        "n_components": n_components,
                        "preprocessor": None,
                        "threshold_max_angle": float(threshold_max_angle),
                    }
                    param_id += 1


def predict_all_layers(
    input_subspace_dir: Path,
    input_feature_dir: Path,
    output_dir: Path,
    *,
    split: str,
    annotation_unit: str,
    layer_ids: list[int],
    n_jobs: int = 1,
    configs: list[dict],
    legacy_feature_frame_rate_mode: bool = False,
) -> list[Path]:
    """Visit layers serially in numeric order, parallelizing each parameter grid."""

    if split not in ("dev", "test"):
        raise ValueError("split must be dev or test")
    if not configs or any(not isinstance(config, dict) for config in configs):
        raise ValueError("prediction configurations must be nonempty dictionaries")

    results = [
        predict_layer(input_subspace_dir / str(layer_id), input_feature_dir, output_dir,
                      split, annotation_unit, configs, n_jobs=n_jobs,
                      legacy_feature_frame_rate_mode=legacy_feature_frame_rate_mode)
        for layer_id in sorted(set(layer_ids))
    ]
    return [path for paths in results for path in paths]


def main() -> None:
    parser = prediction_parser(__doc__)
    parser.add_argument("--layer-ids", type=parse_layer_ids, required=True,
                        help="comma-separated saved layer IDs to tune sequentially")
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="parallel parameter workers; negative values follow joblib conventions")
    args = parser.parse_args()
    paths = predict_all_layers(
        args.subspace_dir, args.input_feature_dir, args.output_dir,
        split=args.split, annotation_unit=args.annotation_unit, layer_ids=args.layer_ids, n_jobs=args.n_jobs,
        configs=list(generate_inference_param_grid()),
        legacy_feature_frame_rate_mode=args.legacy_feature_frame_rate_mode,
    )
    print(f"Wrote or resumed {len(paths)} layer/parameter configurations in {args.output_dir}")


if __name__ == "__main__":
    main()
