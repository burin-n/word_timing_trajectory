#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${repo_root}/experiments/temporal_trajectory/common.sh"
python_bin="${PYTHON_BIN:-python}"
n_jobs="${N_JOBS:-8}"
if [[ ! "${n_jobs}" =~ ^[1-9][0-9]*$ ]]; then
    printf 'N_JOBS must be a positive integer: %s\n' "${n_jobs}" >&2
    exit 1
fi

subspace_dir="${SUBSPACE_DIR:-${repo_root}/OUTPUT/save_spaces}"
output_dir="${OUTPUT_DIR:-${repo_root}/OUTPUT/plot_mean_trajectories}"


echo "exp 1 plot mean trajectories"
echo "subspace dir: ${subspace_dir}"
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

model_plot_names=(
    # "MelHuBERT"
    # "SpidR"
    # "DinoSR"
    # # "DinoSR original"
    # "Mimi"
    "HuBERT"
    "Unt. HuBERT"
    # "WavLM"
    # "Wav2vec"
    # # "log Mel features"
    # "VibeVoice ac."
    # "VibeVoice sem."
    # "Whisper enc."
    # "Unt. Whisper"
)

annotation_units=("word")
subspace_estimation_methods=("Z_CPCA" "LEACE")

if (( ${#model_names[@]} != ${#model_plot_names[@]} )); then
    printf 'Model names and plot names must have the same length.\n' >&2
    exit 1
fi

plotted_count=0
skipped_count=0
job_pids=()
job_labels=()


for subspace_estimator in "${subspace_estimation_methods[@]}"; do
    for model_index in "${!model_names[@]}"; do
        model_name="${model_names[model_index]}"
        model_plot_name="${model_plot_names[model_index]}"
        for annotation_unit in "${annotation_units[@]}"; do
            input_subspace_dir="${subspace_dir}/${subspace_estimator}/${model_name}/${annotation_unit}_dev"
            output_plot_dir="${output_dir}/${subspace_estimator}/${model_name}/${annotation_unit}_dev"
            if [[ ! -d "${input_subspace_dir}" ]]; then
                printf 'Skipping missing saved fit: %s\n' "${input_subspace_dir}" >&2
                ((skipped_count += 1))
                continue
            fi
            printf 'Plotting %s %s %s from %s\n' \
                "${subspace_estimator}" "${model_name}" "${annotation_unit}" "${input_subspace_dir}"
            "${python_bin}" "${repo_root}/scripts/plot_temporal_trajectories.py" \
                --input-dir "${input_subspace_dir}" \
                --output-dir "${output_plot_dir}" \
                --model-name "${model_plot_name}" &
            job_pids+=("$!")
            job_labels+=("${subspace_estimator} ${model_name} ${annotation_unit}")
            ((plotted_count += 1))
            if (( ${#job_pids[@]} >= n_jobs )); then
                wait_for_jobs "Plot" || exit 1
            fi
        done
    done
done

wait_for_jobs "Plot" || exit 1

printf 'Plotted %d saved fits; skipped %d missing fits.\n' "${plotted_count}" "${skipped_count}"
if (( plotted_count == 0 )); then
    exit 1
fi
