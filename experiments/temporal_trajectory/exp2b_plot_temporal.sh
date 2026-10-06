#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

input_csv="${INPUT_CSV:-${repo_root}/OUTPUT/temporal_metrics/score_all.csv}"
output_dir="${OUTPUT_DIR:-$(dirname "${input_csv}")/plot}"

echo "exp 2b plot temporal evaluation"
echo "input csv: ${input_csv}"
echo "output dir: ${output_dir}"

plot_args=(
    --label word
    # --estimator Z_CPCA LEACE
    # --metric r2_train
    # --layer 3-6
    # --testset
    # --transformer-only
)

"${python_bin}" "${repo_root}/scripts/plot_temporal_metrics.py" \
    --input-csv "${input_csv}" \
    --save-dir "${output_dir}" \
    "${plot_args[@]}"
