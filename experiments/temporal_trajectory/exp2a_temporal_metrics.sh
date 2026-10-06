#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
# Use separate output directories when switching timing modes.
timing_args=()
if [[ "${LEGACY_FEATURE_FRAME_RATE_MODE:-0}" == "1" ]]; then
    timing_args+=(--legacy-feature-frame-rate-mode)
fi
n_jobs="${N_JOBS:-8}"
if [[ ! "${n_jobs}" =~ ^-?[1-9][0-9]*$ ]]; then
    printf 'N_JOBS must be a nonzero integer: %s\n' "${n_jobs}" >&2
    exit 1
fi

subspace_dir="${SUBSPACE_DIR:-${repo_root}/OUTPUT/save_spaces}"
output_dir="${OUTPUT_DIR:-${repo_root}/OUTPUT/temporal_metrics}"
feature_dir="${FEATURE_DIR:-${repo_root}/cache}"
alignment_dir="${ALIGNMENT_DIR:-${repo_root}/alignments/Librispeech}"
evaluation_level="mean"

echo "exp 2a temporal evaluation"
echo "subspace dir: ${subspace_dir}"
echo "feature dir: ${feature_dir}"
echo "alignment dir: ${alignment_dir}"
echo "output dir: ${output_dir}"

model_names=(
    # "melhubert-360h"
    # "spidr"
    # "dinosr-reproduced"
    # # "dinosr-original"
    # "mimi"
    "hubert-base-ls960"
    "hubert-base-random"
    # "wavlm-base"
    # "wav2vec2-base"
    # # "spectrogram"
    # "VibeVoice-ASR-HF_acoustic"
    # "VibeVoice-ASR-HF_semantic"
    # "whisper-base"
    # "whisper-random"
)

annotation_units=("word")
subspace_estimation_methods=("Z_CPCA" "LEACE")

scored_count=0
skipped_count=0
score_csvs=()


for subspace_estimator in "${subspace_estimation_methods[@]}"; do
    for model_name in "${model_names[@]}"; do
        train_feature_dir="${feature_dir}/${model_name}/Librispeech/dev-clean"
        test_feature_dir="${feature_dir}/${model_name}/Librispeech/test-clean"
        for annotation_unit in "${annotation_units[@]}"; do
            input_subspace_dir="${subspace_dir}/${subspace_estimator}/${model_name}/${annotation_unit}_dev"
            output_metrics_dir="${output_dir}/${subspace_estimator}/${model_name}/${annotation_unit}"
            if [[ ! -d "${input_subspace_dir}" ]]; then
                printf 'Skipping missing saved fit: %s\n' "${input_subspace_dir}" >&2
                ((skipped_count += 1))
                continue
            fi
            printf 'Scoring %s %s %s from %s\n' \
                "${subspace_estimator}" "${model_name}" "${annotation_unit}" "${input_subspace_dir}"
            "${python_bin}" "${repo_root}/scripts/evaluate_temporal_metrics.py" \
                --subspace-dir "${input_subspace_dir}" \
                --train-feature-dir "${train_feature_dir}" \
                --test-feature-dir "${test_feature_dir}" \
                --train-alignment-path "${alignment_dir}/dev-clean-${annotation_unit}.ali" \
                --test-alignment-path "${alignment_dir}/test-clean-${annotation_unit}.ali" \
                --output-dir "${output_metrics_dir}" \
                --output-csv "${output_metrics_dir}/score.csv" \
                --model-name "${model_name}" \
                --evaluation-level "${evaluation_level}" \
                --n-jobs "${n_jobs}" "${timing_args[@]}"
            score_csvs+=("${output_metrics_dir}/score.csv")
            ((scored_count += 1))
        done
    done
done

if (( scored_count == 0 )); then
    printf 'No saved fits scored; skipped %d missing fits.\n' "${skipped_count}" >&2
    exit 1
fi

"${python_bin}" "${repo_root}/scripts/gather_temporal_metrics.py" \
    --output-path "${output_dir}/score_all.csv" "${score_csvs[@]}"
printf 'Scored %d saved fits; skipped %d missing fits. Wrote %s\n' \
    "${scored_count}" "${skipped_count}" "${output_dir}/score_all.csv"
