parse_layer_ids() {
    # expect "<model_name>|<layer_range>"
    # "melhubert-360h|0-12"
    # "spidr|0,1-12"
    # "model3|0,1,2,3,4"
    local model_name="$1"
    local layer_spec="$2"
    local layer_ranges layer_range first_layer last_layer layer_id

    # Parse layer id into the caller's layer_ids array.
    layer_ids=()
    IFS=',' read -r -a layer_ranges <<< "${layer_spec}"
    for layer_range in "${layer_ranges[@]}"; do
        if [[ "${layer_range}" =~ ^(0|[1-9][0-9]*)-(0|[1-9][0-9]*)$ ]]; then
            first_layer="${BASH_REMATCH[1]}"
            last_layer="${BASH_REMATCH[2]}"
            if (( first_layer > last_layer )); then
                printf 'Invalid layer range for %s: %s\n' "${model_name}" "${layer_range}" >&2
                return 1
            fi
            for ((layer_id = first_layer; layer_id <= last_layer; layer_id++)); do
                layer_ids+=("${layer_id}")
            done
        elif [[ "${layer_range}" =~ ^(0|[1-9][0-9]*)$ ]]; then
            layer_ids+=("${layer_range}")
        else
            printf 'Invalid layer selection for %s: %s\n' "${model_name}" "${layer_range}" >&2
            return 1
        fi
    done
    if (( ${#layer_ids[@]} == 0 )); then
        printf 'No layers selected for %s\n' "${model_name}" >&2
        return 1
    fi
}

wait_for_jobs() {
    local operation="$1"
    local job_failed=0
    local job_index
    for job_index in "${!job_pids[@]}"; do
        if ! wait "${job_pids[job_index]}"; then
            printf '%s failed for %s\n' "${operation}" "${job_labels[job_index]}" >&2
            job_failed=1
        fi
    done
    job_pids=()
    job_labels=()
    return "${job_failed}"
}
