"""Plot the mean trajectories and subspace saved by fit_temporal_subspace.py."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from concept_analysis.plotting import group_gradient_plot, relative_polar_angle_plot
from concept_analysis.projector import (
    COVProjection, CPCAProjection, LDAProjection, LEACEProjection, PCAProjection,
    RandomProjection,
)


SAVED_INPUT_FILES = ("mean_trajectories.npz", "temporal_subspace.pkl", "metadata.json")
PROJECTOR_TYPES = {
    "Z_CPCA": PCAProjection,
    "CPCA": CPCAProjection,
    "LEACE": LEACEProjection,
    "Z_LEACE": LEACEProjection,
    "RANDOM": RandomProjection,
    "LDA": LDAProjection,
    "COV": COVProjection,
}


def load_and_project_mean_trajectories(input_dir: Path):
    """Load `fit_temporal_subspace.py` output directory and project each saved mean once."""
    metadata = json.loads((input_dir / "metadata.json").read_text())
    estimator = metadata.get("estimator")
    if estimator not in PROJECTOR_TYPES:
        raise ValueError(f"Unsupported saved trajectory estimator: {estimator!r}")
    subspace = PROJECTOR_TYPES[estimator].load(input_dir / "temporal_subspace.pkl")

    with np.load(input_dir / "mean_trajectories.npz", allow_pickle=False) as archive:
        mean_trajectories = {int(duration_frames): archive[duration_frames]
                             for duration_frames in archive.files}

    if not mean_trajectories:
        raise ValueError("mean_trajectories.npz contains no trajectories")
    group_mean_trajectories = {
        duration_frames: subspace.transform(mean_trajectory)
        for duration_frames, mean_trajectory in sorted(mean_trajectories.items())
    }
    return group_mean_trajectories, metadata


def plot_trajectories(
    group_mean_trajectories: np.ndarray,
    output_dir: Path,
    *,
    n_planes: int = 1,
    group_centering_method: str = "centroid",
    feature_frame_rate: int | float = 50,
    disable_ticks: bool = False,
    save_fig_name_suffix: str = "",
    label_name: str = "label",
    figure_title: str | None = None,
):
    projected_component_count = group_mean_trajectories[
        min(group_mean_trajectories.keys())
    ].shape[1]

    for first_component_index in range(n_planes):
        if projected_component_count <= first_component_index + 1:
            break

        # plot 2d
        plot_2d_suffix = save_fig_name_suffix
        if n_planes > 1:
            plot_2d_suffix += f"_pc{first_component_index}-{first_component_index + 1}"
        fig, _ = group_gradient_plot(
            group_mean_trajectories, label_name, "2d", feature_frame_rate=feature_frame_rate,
            save_fig_dir=output_dir, save_fig_name_suffix=plot_2d_suffix,
            components=(first_component_index, first_component_index + 1),
            disable_ticks=disable_ticks, figure_title=figure_title,
        )
        plt.close(fig)

        # plot 3d
        if projected_component_count > first_component_index + 2:
            plot_3d_suffix = save_fig_name_suffix
            if n_planes > 1:
                plot_3d_suffix += (
                    f"_pc{first_component_index}-{first_component_index + 1}"
                    f"-{first_component_index + 2}"
                )
            fig, _ = group_gradient_plot(
                group_mean_trajectories, label_name, "3d", feature_frame_rate=feature_frame_rate,
                save_fig_dir=output_dir, save_fig_name_suffix=plot_3d_suffix,
                components=(first_component_index, first_component_index + 1,
                            first_component_index + 2), disable_ticks=disable_ticks,
                figure_title=figure_title,
            )
            plt.close(fig)

    fig, _ = relative_polar_angle_plot(
        group_mean_trajectories, label_name, feature_frame_rate=feature_frame_rate,
        save_fig_dir=output_dir, save_fig_name_suffix=save_fig_name_suffix,
        group_centering_method=group_centering_method,
        figure_title=figure_title,
    )
    plt.close(fig)



def plot_saved_trajectories(
    input_dir: Path,
    output_dir: Path,
    *,
    n_planes: int = 1,
    group_centering_method: str = "centroid",
    disable_ticks: bool = False,
    save_fig_name_suffix: str = "",
    model_name: str | None = None,
):
    group_mean_trajectories, metadata = load_and_project_mean_trajectories(input_dir)
    label_name = metadata["annotation_unit"].removesuffix("_token_pos")
    feature_frame_rate = metadata["feature_frame_rate"]
    figure_title = None
    if model_name is not None:
        if "layer_id" not in metadata:
            raise ValueError(f"layer_id is required in {input_dir / 'metadata.json'} for model titles")
        figure_title = f"{model_name} L{metadata['layer_id']}"

    plot_trajectories(group_mean_trajectories, output_dir,
                n_planes=n_planes, group_centering_method=group_centering_method,
                disable_ticks=disable_ticks, save_fig_name_suffix=save_fig_name_suffix,
                feature_frame_rate=feature_frame_rate,
                label_name=label_name, figure_title=figure_title)

    return group_mean_trajectories


def plot_saved_layers(
    input_dir: Path,
    output_dir: Path,
    *,
    n_planes: int = 1,
    group_centering_method: str = "centroid",
    disable_ticks: bool = False,
    model_name: str | None = None,
) -> None:
    """Render every saved layer, or one flat slice-08 output directory."""
    layer_dirs = sorted(
        (child for child in input_dir.iterdir() if child.is_dir() and child.name.isdecimal()),
        key=lambda layer_dir: int(layer_dir.name),
    )
    root_artifacts = [filename for filename in SAVED_INPUT_FILES
                      if (input_dir / filename).exists()]
    if layer_dirs and root_artifacts:
        raise ValueError("input directory mixes flat saved inputs and layer directories")
    if not layer_dirs:
        if not root_artifacts:
            raise FileNotFoundError(f"no saved trajectory inputs or numeric layer directories in {input_dir}")
        missing_files = [filename for filename in SAVED_INPUT_FILES
                         if filename not in root_artifacts]
        if missing_files:
            raise FileNotFoundError(f"missing saved inputs in {input_dir}: {', '.join(missing_files)}")
        plot_saved_trajectories(
            input_dir, output_dir, n_planes=n_planes, group_centering_method=group_centering_method,
            disable_ticks=disable_ticks, model_name=model_name,
        )
        return

    layer_ids = [int(layer_dir.name) for layer_dir in layer_dirs]

    for layer_dir, layer_id in zip(layer_dirs, layer_ids, strict=True):

        missing_files = [filename for filename in SAVED_INPUT_FILES
                         if not (layer_dir / filename).is_file()]
        if missing_files:
            raise FileNotFoundError(f"missing saved inputs in {layer_dir}: {', '.join(missing_files)}")

        metadata = json.loads((layer_dir / "metadata.json").read_text())
        if metadata.get("layer_id") != layer_id:
            raise ValueError(f"layer_id in {layer_dir / 'metadata.json'} does not match directory {layer_id}")

    for layer_dir, layer_id in zip(layer_dirs, layer_ids, strict=True):
        plot_saved_trajectories(
            layer_dir, output_dir, n_planes=n_planes,
            group_centering_method=group_centering_method,
            disable_ticks=disable_ticks,
            save_fig_name_suffix=f"_layer{layer_id:02d}" if model_name is None else f"_{model_name.lower()}_layer{layer_id:02d}",
            model_name=model_name,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True,
                        help="`fit_temporal_subspace.py` output directory, flat or with numeric layer subdirectories")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-planes", type=int, default=1)
    parser.add_argument("--group-centering-method", choices=("centroid", "circle", "none"),
                        default="centroid")
    parser.add_argument("--disable-ticks", action="store_true")
    parser.add_argument("--model-name", help="display name for titles such as 'HuBERT L0'")
    args = parser.parse_args()
    if args.n_planes < 1:
        parser.error("--n-planes must be positive")
    plot_saved_layers(
        args.input_dir, args.output_dir, n_planes=args.n_planes,
        group_centering_method=args.group_centering_method,
        disable_ticks=args.disable_ticks,
        model_name=args.model_name,
    )


if __name__ == "__main__":
    main()
