#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

# Evaluation must already have written .res files beside the predictions.
segmentation_dir="${SEGMENTATION_DIR:-${repo_root}/OUTPUT/peak_detection_tune}"
output_dir="${OUTPUT_DIR:-${segmentation_dir}/results}"

echo "exp 3d gather segmentation results"
echo "segment dir: ${segmentation_dir}"
echo "output dir: ${output_dir}"

model_names=("hubert-base-ls960" "hubert-base-random")
subspace_estimation_methods=("Z_CPCA" "LEACE")
layer_ids=("09") # Use the directory names, including leading zeroes.
testspaces=("word")
testunits=("word")
tuned_metrics=("R-value")
eval_folder_suffix="eval"
testsets=(dev test)
if [[ -n "${TESTSET:-}" ]]; then
    testsets=("$TESTSET")
fi
mode="${GATHER_MODE:-paired-threshold}" # Or best.
paired_direction="threshold-to-no-threshold" # Or no-threshold-to-threshold.
export_yaml="${EXPORT_YAML:-1}"

gather_args=(
    --n-components 2
    --preprocessor none
    --use-absolute-threshold # Best mode: search every angle threshold.
    # --percentage
    # --decimals 1
    # --verbose
)

for testset in "${testsets[@]}"; do
    csv_path="${output_dir}/${testset}_${mode}_${eval_folder_suffix}.csv"
    "${python_bin}" "${repo_root}/scripts/gather_segmentation_results_v2.py" \
        --segmentation-dir "${segmentation_dir}" \
        --models "${model_names[@]}" \
        --estimators "${subspace_estimation_methods[@]}" \
        --layer-ids "${layer_ids[@]}" \
        --testspaces "${testspaces[@]}" \
        --testunits "${testunits[@]}" \
        --tuned-metrics "${tuned_metrics[@]}" \
        --testset "${testset}" \
        --eval-folder-suffix "${eval_folder_suffix}" \
        --mode "${mode}" \
        --paired-direction "${paired_direction}" \
        --save-csv "${csv_path}" \
        "${gather_args[@]}"

    if [[ "${export_yaml}" != "1" ]]; then
        continue
    fi

    if [[ "${mode}" == "paired-threshold" ]]; then
        threshold_modes=("threshold" "nothreshold")
    elif [[ "${mode}" == "best" ]]; then
        threshold_modes=("all")
    fi

    for layer_id in "${layer_ids[@]}"; do
        for testunit in "${testunits[@]}"; do
            for tuned_metric in "${tuned_metrics[@]}"; do
                for threshold_mode in "${threshold_modes[@]}"; do
                    yaml_name="${testset}_${mode}_layer${layer_id}_${testunit}_${tuned_metric}_${eval_folder_suffix}"
                    if [[ "${threshold_mode}" != "all" ]]; then
                        yaml_name+="_${threshold_mode}"
                    fi
                    "${python_bin}" "${repo_root}/scripts/export_segmentation_parameters.py" \
                        --input-csv "${csv_path}" \
                        --segmentation-dir "${segmentation_dir}" \
                        --output-yaml "${output_dir}/${yaml_name}.yaml" \
                        --layer-id "${layer_id}" \
                        --testset "${testset}" \
                        --test-unit "${testunit}" \
                        --tuned-metric "${tuned_metric}" \
                        --threshold-mode "${threshold_mode}"
                done
            done
        done
    done
done
