#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
# Use separate output roots for each mode; existing timestamp lists are resumed.
timing_args=()
if [[ "${LEGACY_FEATURE_FRAME_RATE_MODE:-0}" == "1" ]]; then
    timing_args+=(--legacy-feature-frame-rate-mode)
fi
n_jobs="${N_JOBS:-8}"
if [[ ! "${n_jobs}" =~ ^-?[1-9][0-9]*$ ]]; then
    printf 'N_JOBS must be a nonzero integer: %s\n' "${n_jobs}" >&2
    exit 1
fi

subspace_dir="${SUBSPACE_DIR:-${repo_root}/OUTPUT/save_spaces_train100}"
output_dir="${OUTPUT_DIR:-${repo_root}/OUTPUT/peak_detection_tune}"
feature_dir="${FEATURE_DIR:-${repo_root}/cache}"

echo "exp 3b tune prominence for peak detection"
echo "subspace dir: ${subspace_dir}"
echo "feature dir: ${feature_dir}"
echo "output dir: ${output_dir}"

model_names=("hubert-base-ls960" "hubert-base-random")
annotation_units=("word")
subspace_estimation_methods=("Z_CPCA" "LEACE")
layer_ids="9" # Comma-separated IDs, visited sequentially; N_JOBS controls parameter workers.


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

            input_feature_dir="${feature_dir}/${model_name}/Librispeech/dev-clean"
            printf 'Tuning %s %s %s on %s from %s\n' \
                "${subspace_estimator}" "${model_name}" "${annotation_unit}" \
                "dev-clean" "${input_subspace_dir}"
            "${python_bin}" "${repo_root}/scripts/tune_peak_detection.py" \
                --subspace-dir "${input_subspace_dir}" \
                --input-feature-dir "${input_feature_dir}" \
                --output-dir "${output_prediction_dir}" \
                --model-name "${model_name}" \
                --annotation-unit "${annotation_unit}" \
                --split dev \
                --layer-ids "${layer_ids}" \
                --n-jobs "${n_jobs}" "${timing_args[@]}"
            ((predicted_count += 1))

        done
    done
done

if (( predicted_count == 0 )); then
    printf 'No saved fits processed; skipped %d missing fits.\n' "${skipped_count}" >&2
    exit 1
fi
printf 'Processed %d saved-fit/split tuning jobs; skipped %d missing fits.\n' \
    "${predicted_count}" "${skipped_count}"
