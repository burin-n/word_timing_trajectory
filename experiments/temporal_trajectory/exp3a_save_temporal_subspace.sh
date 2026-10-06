#!/usr/bin/env bash
set -euo pipefail

###
### In segmentation experiment, we estimate temporal subspaces on 5h audios sampled from train-clean-100,
###    unlike prior experiments that estimate temporal subspaces on dev-clean.
### We then use this subspace to tune RoSeg on dev-clean, and test RoSeg on test-clean. 
### This script is dedicated to estimate such temporal subspaces.
###

### START setup and hyperparameters
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/experiments/temporal_trajectory/common.sh"
python_bin="${PYTHON_BIN:-python}"
# Use separate output directories when switching timing modes.
timing_args=()
if [[ "${LEGACY_FEATURE_FRAME_RATE_MODE:-0}" == "1" ]]; then
    timing_args+=(--legacy-feature-frame-rate-mode)
fi
n_jobs="${N_JOBS:-8}"
if [[ ! "${n_jobs}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'N_JOBS must be a positive integer: %s\n' "${n_jobs}" >&2
    exit 1
fi

output_dir="${OUTPUT_DIR:-${repo_root}/OUTPUT/save_spaces_train100}"
feature_dir="${FEATURE_DIR:-${repo_root}/cache}"
alignment_dir="${ALIGNMENT_DIR:-${repo_root}/alignments/Librispeech}"
train_data_set=train-clean-100

echo "exp 3a estimating temporal subspaces for segmentation"
echo "train data set: ${train_data_set}"
echo "feature dir: ${feature_dir}"
echo "alignment dir: ${alignment_dir}"
echo "output dir: ${output_dir}"

# We only use layer 9 for segmentation experiments.
model_layer_specs=(
    "hubert-base-ls960|9"
    "hubert-base-random|9"
)

annotation_units=("word")
min_dur_word=0.1
max_dur_word=0.7
min_dur_syllable=0.1
max_dur_syllable=0.7
min_dur_phone=0.06
max_dur_phone=0.5

subspace_estimation_methods=("Z_CPCA" "LEACE")

### END setup and hyperparameters


fitted_count=0
skipped_count=0

for model_layer_spec in "${model_layer_specs[@]}"; do
    # expect "<model_name>|<layer_range>"
    IFS='|' read -r model_name layer_spec <<< "${model_layer_spec}"
    
    input_feature_dir="${feature_dir}/${model_name}/Librispeech/${train_data_set}"
    if [[ ! -d "${input_feature_dir}" ]]; then
        printf 'Skipping missing feature directory: %s\n' "${input_feature_dir}" >&2
        ((skipped_count += 1))
        continue
    fi

    layer_ids=() #  filled by parse_layer_ids
    parse_layer_ids "${model_name}" "${layer_spec}" || exit 1

    for subspace_estimator in "${subspace_estimation_methods[@]}"; do
        for annotation_unit in "${annotation_units[@]}"; do
            alignment_path="${alignment_dir}/${train_data_set}-${annotation_unit}.ali"
            if [[ ! -f "${alignment_path}" ]]; then
                printf 'Missing alignment: %s\n' "${alignment_path}" >&2
                exit 1
            fi

            if [[ "${annotation_unit}" == "phone" ]]; then
                min_dur=$min_dur_phone
                max_dur=$max_dur_phone
            elif [[ "${annotation_unit}" == "syllable" ]]; then
                min_dur=$min_dur_syllable
                max_dur=$max_dur_syllable
            else
                min_dur=$min_dur_word
                max_dur=$max_dur_word
            fi

            job_pids=()
            job_labels=()

            for layer_id in "${layer_ids[@]}"; do
                output_layer_dir="${output_dir}/${subspace_estimator}/${model_name}/${annotation_unit}_dev/${layer_id}"
                printf 'Fitting %s %s %s layer %s\n' \
                    "${subspace_estimator}" "${model_name}" \
                    "${annotation_unit}" "${layer_id}"

                "${python_bin}" "${repo_root}/scripts/fit_temporal_subspace.py" \
                    --output-dir "${output_layer_dir}" \
                    --input-feature-dir "${input_feature_dir}" \
                    --alignment-path "${alignment_path}" \
                    --layer-id "${layer_id}" \
                    --estimator "${subspace_estimator}" \
                    --annotation-unit "${annotation_unit}" \
                    --min-dur "${min_dur}" \
                    --max-dur "${max_dur}" \
                    --low-freq-threshold 50 "${timing_args[@]}" &

                job_pids+=("$!")
                job_labels+=("${subspace_estimator} ${model_name} ${annotation_unit} layer ${layer_id}")
                ((fitted_count += 1))
                if (( ${#job_pids[@]} >= n_jobs )); then
                    wait_for_jobs "Fit" || exit 1
                fi
            done

            wait_for_jobs "Fit" || exit 1
        done
    done
done

printf 'Fitted %d estimator/model/unit/layer combinations; skipped %d missing model feature directories.\n' \
    "${fitted_count}" "${skipped_count}"
if (( fitted_count == 0 )); then
    exit 1
fi
