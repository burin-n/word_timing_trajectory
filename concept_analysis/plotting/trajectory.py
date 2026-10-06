"""Render projected duration-group mean trajectories.

Import this module only with the ``plot`` extra installed. Numerical angle
calculations remain in :mod:`concept_analysis.trajectory.geometry`.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from concept_analysis.trajectory.geometry import (
    get_sequence_relative_angles, normalize_trajectory, rotate_to_anchor,
)


def _get_duration_color_norm(trajectories, feature_frame_rate, ms_range, cmap):
    if not np.isfinite(feature_frame_rate) or feature_frame_rate <= 0:
        raise ValueError("feature_frame_rate must be finite and positive")
    if not trajectories:
        raise ValueError("at least one trajectory is required")
    if ms_range is None:
        durations_ms = [duration_frames / feature_frame_rate * 1000
                        for duration_frames in sorted(trajectories.keys())]
        ms_range = (durations_ms[0], durations_ms[-1])
    duration_norm = mcolors.Normalize(vmin=ms_range[0], vmax=ms_range[1])
    return duration_norm, plt.get_cmap(cmap)


def group_gradient_plot(
    group_trajectories: dict[int, np.ndarray],
    label_name: str = "label",
    plot_projection: str = "2d",
    *,
    feature_frame_rate: float = 50,
    save_fig_dir: str | Path | None = None,
    save_fig_name_suffix: str = "",
    components: tuple[int, ...] | None = None,
    ms_range: tuple[float, float] | None = None,
    normalize_axis: bool = True,
    alpha: float = 0.5,
    rotate_anchor: bool = False,
    disable_ticks: bool = False,
    figure_title: str | None = None,
    cmap: str = "viridis",
):
    """Plot 2D or 3D projected means, colored by duration in milliseconds.

    ``trajectories`` maps frame-count keys to projected ``(frames, components)``
    arrays. The caller owns projection so the same inputs can be reused by all
    three figure types.
    """
    if plot_projection not in ("2d", "3d"):
        raise ValueError("plot_projection must be '2d' or '3d'")
    if rotate_anchor and plot_projection != "2d":
        raise ValueError("rotate_anchor requires a 2d plot")

    plotted_component_count = 2 if plot_projection == "2d" else 3
    selected_components = (tuple(range(plotted_component_count))
                           if components is None else tuple(components))
    if (len(selected_components) != plotted_component_count
            or min(selected_components) < 0):
        raise ValueError(f"{plot_projection} requires {plotted_component_count} component indices")

    duration_norm, color_map = _get_duration_color_norm(group_trajectories, feature_frame_rate, ms_range, cmap)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d") if plotted_component_count == 3 else fig.add_subplot()
    ax.set_box_aspect((1, 1, 1) if plotted_component_count == 3 else 1)

    xy_min = xy_max = 0.0
    for duration_frames in sorted(group_trajectories):

        projected_traj = group_trajectories[duration_frames]

        if (projected_traj.ndim != 2 or projected_traj.shape[0] != duration_frames
                or projected_traj.shape[1] <= max(selected_components)):
            plt.close(fig)
            raise ValueError(f"trajectory {duration_frames} has insufficient frames or components")

        plotting_points = projected_traj[:, selected_components]

        if rotate_anchor:
            plotting_points = rotate_to_anchor(plotting_points)

        duration_color = color_map(duration_norm(duration_frames / feature_frame_rate * 1000))

        if plotted_component_count == 2:
            ax.plot(plotting_points[:, 0], plotting_points[:, 1], "-x", alpha=alpha, color=duration_color)
            ax.scatter(plotting_points[0, 0], plotting_points[0, 1], marker="o", color=duration_color, zorder=10, s=100)
            ax.scatter(plotting_points[-1, 0], plotting_points[-1, 1], marker="+", color=duration_color, zorder=10, s=100)
        else:
            ax.plot(plotting_points[:, 0], plotting_points[:, 1], plotting_points[:, 2], "-x", alpha=alpha, color=duration_color)
            ax.scatter(*plotting_points[0], marker="o", color=duration_color, zorder=10, s=100)
            ax.scatter(*plotting_points[-1], marker="+", color=duration_color, zorder=10, s=100)
        xy_min = min(xy_min, plotting_points[:, :2].min())
        xy_max = max(xy_max, plotting_points[:, :2].max())

    if disable_ticks:
        ax.set_xticks([])
        ax.set_yticks([])

        if plotted_component_count == 3:
            ax.set_zticks([])
    else:
        axis_label_setters = [ax.set_xlabel, ax.set_ylabel]

        if plotted_component_count == 3:
            axis_label_setters.append(ax.set_zlabel)

        for component_index, set_axis_label in enumerate(axis_label_setters):
            set_axis_label(f"PC{selected_components[component_index] + 1}", fontsize=18)

        ax.tick_params(axis="both", which="major", labelsize=16)

    if normalize_axis:
        axis_limit = max(abs(xy_min), abs(xy_max)) * 1.05
        ax.set_xlim(-axis_limit, axis_limit)
        ax.set_ylim(-axis_limit, axis_limit)
        if plotted_component_count == 3:
            ax.set_zlim(-axis_limit, axis_limit)

    if figure_title is not None:
        ax.set_title(figure_title, fontsize=30)

    if save_fig_dir is not None:
        output_path = Path(save_fig_dir) / f"{plot_projection}_trajectory" / f"{label_name}{save_fig_name_suffix}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, ax


def _get_group_relative_angles(group_trajectories: dict[int, np.ndarray], group_centering_method: str = "none"):
    assert group_centering_method in ("centroid", "circle", "none"), (
        "Invalid group_centering_method option. Choose from 'centroid', 'circle', 'none'."
    )

    relative_angles = {}
    for duration_frames in sorted(group_trajectories.keys()):
        projected_traj = group_trajectories[duration_frames]
        # use the first two dimensions
        projected_traj = projected_traj[:, :2].copy()
        projected_traj = normalize_trajectory(
            projected_traj,
            mean_trajectory=projected_traj,
            centering_mode=group_centering_method,
        )
        relative_angles[duration_frames] = get_sequence_relative_angles(
            projected_traj,
            n_components=2,
            relative_to=0,
            angle_mode="atan2_unwrapped",
        )
    return relative_angles


def _pi_label(angle_rad):
    half_pi_multiple = round(angle_rad / (0.5 * np.pi))
    if half_pi_multiple == 0:
        return r"$0$"
    sign = "-" if half_pi_multiple < 0 else ""
    absolute_multiple = abs(half_pi_multiple)
    if absolute_multiple % 2 == 0:
        pi_multiple = absolute_multiple // 2
        return rf"${sign}{pi_multiple}\pi$" if pi_multiple > 1 else rf"${sign}\pi$"
    numerator = "" if absolute_multiple == 1 else str(absolute_multiple)
    return rf"${sign}\frac{{{numerator}\pi}}{{2}}$"


def relative_polar_angle_plot(
    group_trajectories: dict[int, np.ndarray],
    label_name: str = "label",
    *,
    feature_frame_rate: float = 50,
    save_fig_dir: str | Path | None = None,
    save_fig_name_suffix: str = "",
    group_centering_method: str = "centroid",
    ms_range: tuple[float, float] | None = None,
    figure_title: str | None = None,
    cmap: str = "viridis",
):
    """Plot relative polar angle against normalized elapsed duration."""
    duration_norm, color_map = _get_duration_color_norm(group_trajectories, feature_frame_rate, ms_range, cmap)
    relative_angles = _get_group_relative_angles(group_trajectories, group_centering_method)
    fig, ax = plt.subplots()
    ax.set_box_aspect(1)
    angle_min, angle_max = np.inf, -np.inf

    for duration_frames in sorted(relative_angles):

        relative_angle_values = relative_angles[duration_frames]
        elapsed_percent = np.arange(duration_frames) / max(duration_frames - 1, 1) * 100
        duration_color = color_map(duration_norm(duration_frames / feature_frame_rate * 1000))

        ax.plot(elapsed_percent, relative_angle_values, "-x", color=duration_color, alpha=0.3)
        ax.scatter(elapsed_percent[0], relative_angle_values[0], marker="o", color=duration_color, zorder=10, s=100)
        ax.scatter(elapsed_percent[-1], relative_angle_values[-1], marker="+", color=duration_color, zorder=10, s=100)

        angle_min = min(angle_min, relative_angle_values.min())
        angle_max = max(angle_max, relative_angle_values.max())

    half_pi = 0.5 * np.pi
    angle_ticks = np.arange(round(angle_min / half_pi) * half_pi,
                            round(angle_max / half_pi) * half_pi + half_pi * 0.1, half_pi)

    ax.set_xlabel(f"elapsed {label_name} proportion (%)", fontsize=18)
    ax.set_ylabel("rotation angle (rad)", fontsize=18)
    ax.set_xticks([0, 20, 40, 60, 80, 100], ["onset", 20, 40, 60, 80, "offset"], fontsize=16)
    ax.set_yticks(angle_ticks, [_pi_label(angle_rad) for angle_rad in angle_ticks], fontsize=16)

    if figure_title is not None:
        ax.set_title(figure_title, fontsize=30)

    if save_fig_dir is not None:
        output_path = Path(save_fig_dir) / "polar" / f"{label_name}{save_fig_name_suffix}.png"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")

    return fig, ax
