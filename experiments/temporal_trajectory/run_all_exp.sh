#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export LEGACY_FEATURE_FRAME_RATE_MODE=0
unset OUTPUT_DIR

#### variables for plot and temporal metric experiments
BASE_OUTPUT=OUTPUT/concept_trajectory_latest
SPACE_PLOT_DIR=${BASE_OUTPUT}/save_spaces
MEAN_TRAJ_PLOT_DIR=${BASE_OUTPUT}/plot_mean_trajectories
TEMPORAL_METRICS_DIR=${BASE_OUTPUT}/temporal_metrics
FEATURE_DIR="FEATURE_CACHE"
# feature_dir is assumed to have following structure
# FEATURE_CACHE/
# ├── <model_name>, e.g. hubert-base-ls960
# │   ├── Librispeech
# │   │   ├── dev-clean
# |   |   |   └── <layer_id>.npz
# |   |   |   └── manifest.toml

#### variables for word segmentation
SPACE_SEGMENT_DIR=${BASE_OUTPUT}/save_spaces_train100

#######################################################

# mean trajectory plotting and evaluation
OUTPUT_DIR="${SPACE_PLOT_DIR}" FEATURE_DIR="${FEATURE_DIR}" \
  bash "${script_dir}/exp0_save_temporal_subspaces.sh" || exit 1;

SUBSPACE_DIR="${SPACE_PLOT_DIR}" OUTPUT_DIR="${MEAN_TRAJ_PLOT_DIR}" \
  bash "${script_dir}/exp1_plot_mean_trajectories.sh" || exit 1;

SUBSPACE_DIR="${SPACE_PLOT_DIR}" OUTPUT_DIR="${TEMPORAL_METRICS_DIR}" \
FEATURE_DIR="${FEATURE_DIR}" \
  bash "${script_dir}/exp2a_temporal_metrics.sh" || exit 1;

INPUT_CSV="${TEMPORAL_METRICS_DIR}/score_all.csv" \
  bash "${script_dir}/exp2b_plot_temporal.sh" || exit 1;

# segmentation subspace estimation
OUTPUT_DIR="${SPACE_SEGMENT_DIR}" FEATURE_DIR="${FEATURE_DIR}" \
  bash "${script_dir}/exp3a_save_temporal_subspace.sh" || exit 1;

# tune prominence on dev-clean
SUBSPACE_DIR="${SPACE_SEGMENT_DIR}" \
FEATURE_DIR="${FEATURE_DIR}" \
OUTPUT_DIR="${BASE_OUTPUT}/peak_detection_tune_params" \
  bash "${script_dir}/exp3b_peak_detection_tune.sh" || exit 1;

SEGMENTATION_DIR="${BASE_OUTPUT}/peak_detection_tune_params" \
EVALUATION_SCRIPT="Simon_scripts/evaluation/boundary_eval.py" \
PYTHON_BIN="Simon_scripts/evaluation/venv/bin/python" \
TESTSET=dev \
REFERENCE_DIR="alignments/Librispeech-Simon-flatten" \
  bash "${script_dir}/exp3c_evaluate_segmentation.sh" || exit 1;

SEGMENTATION_DIR="${BASE_OUTPUT}/peak_detection_tune_params" \
TESTSET="dev" EXPORT_YAML=1 \
GATHER_MODE=paired-threshold \
  bash "${script_dir}/exp3d_gather_segmentation_results.sh" || exit 1;

# inference test set using the tuned prominence with absoluate threshold
SUBSPACE_DIR="${SPACE_SEGMENT_DIR}" \
OUTPUT_DIR="${BASE_OUTPUT}/peak_detection_threshold_train100" \
FEATURE_DIR="${FEATURE_DIR}" \
CONFIG_PATH="${BASE_OUTPUT}/peak_detection_tune_params/results/dev_paired-threshold_layer09_word_R-value_eval_threshold.yaml" \
  bash "${script_dir}/exp3e_peak_detection.sh" || exit 1;

# This step requires downloading Simon evaluation script from https://github.com/s-malan/evaluation/blob/main/boundary_eval.py
# EVALUATION_SCRIPT and PYTHON_BIN are needed to be set accordingly.
SEGMENTATION_DIR="${BASE_OUTPUT}/peak_detection_threshold_train100" \
EVALUATION_SCRIPT="Simon_scripts/evaluation/boundary_eval.py" \
PYTHON_BIN="Simon_scripts/evaluation/venv/bin/python" \
REFERENCE_DIR="alignments/Librispeech-Simon-flatten" \
  bash "${script_dir}/exp3c_evaluate_segmentation.sh" || exit 1;

SEGMENTATION_DIR="${BASE_OUTPUT}/peak_detection_threshold_train100" \
GATHER_MODE=best EXPORT_YAML=0 \
  bash "${script_dir}/exp3d_gather_segmentation_results.sh" || exit 1;

# inference test set with no absoluate threshold
SUBSPACE_DIR="${SPACE_SEGMENT_DIR}" \
OUTPUT_DIR="${BASE_OUTPUT}/peak_detection_nothreshold_train100" \
FEATURE_DIR="${FEATURE_DIR}" \
CONFIG_PATH="${BASE_OUTPUT}/peak_detection_tune_params/results/dev_paired-threshold_layer09_word_R-value_eval_nothreshold.yaml" \
  bash "${script_dir}/exp3e_peak_detection.sh" || exit 1;

# This step requires downloading Simon evaluation script from https://github.com/s-malan/evaluation/blob/main/boundary_eval.py
# EVALUATION_SCRIPT and PYTHON_BIN are needed to be set accordingly.
SEGMENTATION_DIR="${BASE_OUTPUT}/peak_detection_nothreshold_train100" \
EVALUATION_SCRIPT="Simon_scripts/evaluation/boundary_eval.py" \
PYTHON_BIN="Simon_scripts/evaluation/venv/bin/python" \
REFERENCE_DIR="alignments/Librispeech-Simon-flatten" \
  bash "${script_dir}/exp3c_evaluate_segmentation.sh" || exit 1;

SEGMENTATION_DIR="${BASE_OUTPUT}/peak_detection_nothreshold_train100" \
GATHER_MODE=best EXPORT_YAML=0 \
  bash "${script_dir}/exp3d_gather_segmentation_results.sh" || exit 1;
