#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
# Use separate output roots for each mode; existing timestamp lists are resumed.
timing_args=()
if [[ "${LEGACY_FEATURE_FRAME_RATE_MODE:-0}" == "1" ]]; then
    timing_args+=(--legacy-feature-frame-rate-mode)
fi
subspace_dir="${SUBSPACE_DIR:-${repo_root}/OUTPUT/save_spaces_train100}"
output_dir="${OUTPUT_DIR:-${repo_root}/OUTPUT/peak_detection_threshold}"
feature_dir="${FEATURE_DIR:-${repo_root}/cache}"
config_path="${CONFIG_PATH:-${repo_root}/experiments/temporal_trajectory/configs/dev_paired-threshold_layer09_word_R-value_eval_threshold.yaml}"

echo "exp 3e run peak detection on dev and test set"
echo "subspace dir: ${subspace_dir}"
echo "feature dir: ${feature_dir}"
echo "config path: ${config_path}"
echo "output dir: ${output_dir}"


model_names=("hubert-base-ls960" "hubert-base-random")
annotation_units=("word")
subspace_estimation_methods=("Z_CPCA" "LEACE")
layer_id=9
prediction_splits=("dev" "test")

if [[ ! -f "${config_path}" ]]; then
    printf 'Missing parameter configuration: %s\n' "${config_path}" >&2
    exit 1
fi

predicted_count=0
skipped_count=0

for subspace_estimator in "${subspace_estimation_methods[@]}"; do
    for model_name in "${model_names[@]}"; do
        for annotation_unit in "${annotation_units[@]}"; do
            input_subspace_dir="${subspace_dir}/${subspace_estimator}/${model_name}/${annotation_unit}_dev"
            output_prediction_dir="${output_dir}/${subspace_estimator}/${model_name}"
            if [[ ! -d "${input_subspace_dir}" ]]; then
                printf 'Skipping missing saved fit: %s\n' "${input_subspace_dir}" >&2
                ((skipped_count += 1))
                continue
            fi
            for prediction_split in "${prediction_splits[@]}"; do
                input_feature_dir="${feature_dir}/${model_name}/Librispeech/${prediction_split}-clean"
                printf 'Predicting %s %s %s on %s from %s\n' \
                    "${subspace_estimator}" "${model_name}" "${annotation_unit}" \
                    "${prediction_split}" "${input_subspace_dir}"
                "${python_bin}" "${repo_root}/scripts/predict_boundaries.py" \
                    --subspace-dir "${input_subspace_dir}" \
                    --input-feature-dir "${input_feature_dir}" \
                    --output-dir "${output_prediction_dir}" \
                    --model-name "${model_name}" \
                    --estimator "${subspace_estimator}" \
                    --annotation-unit "${annotation_unit}" \
                    --split "${prediction_split}" \
                    --config "${config_path}" \
                    --layer-id "${layer_id}" "${timing_args[@]}"
                ((predicted_count += 1))
            done
        done
    done
done

if (( predicted_count == 0 )); then
    printf 'No saved fits processed; skipped %d missing fits.\n' "${skipped_count}" >&2
    exit 1
fi
printf 'Processed %d saved-fit/split prediction jobs; skipped %d missing fits.\n' \
    "${predicted_count}" "${skipped_count}"
