#!/usr/bin/env bash
set -euo pipefail

dataset_path="${1:-./dataset/agentx/traces.jsonl}"
result_root="${2:-./results/agentx}"
concurrencies="${AGENTX_CONCURRENCIES:-1 2 4 8}"
trace_count="${AGENTX_TRACE_COUNT:-8}"
profile_duration="${AGENTX_PROFILE_DURATION:-60}"
max_requests="${AGENTX_MAX_REQUESTS:-}"
warmup="${AGENTX_WARMUP:-1}"

case "${warmup,,}" in
    1|true|yes)
        warmup_arg="--agentx_warmup"
        ;;
    0|false|no)
        warmup_arg="--no-agentx_warmup"
        ;;
    *)
        echo "AGENTX_WARMUP must be one of: 1, 0, true, false, yes, no" >&2
        exit 1
        ;;
esac

if [[ ! -f "${dataset_path}" ]]; then
    echo "AgentX dataset not found: ${dataset_path}" >&2
    echo "Run ./scripts/download_agentx.sh first." >&2
    exit 1
fi

run_case() {
    local topology="$1"
    local cluster="$2"
    local offload_name="$3"
    local kv_config="$4"
    local concurrency
    for concurrency in ${concurrencies}; do
        command=(
            ./scripts/run_python.sh ./benchmark.py
            --batching paged-attn
            --block_size 64
            --cluster "${cluster}"
            --dataset_path "${dataset_path}"
            --workload_type agentx_weka
            --agentx_trace_count "${trace_count}"
            --agentx_concurrency "${concurrency}"
            --agentx_profile_duration "${profile_duration}"
            "${warmup_arg}"
            --model ./data/psla/deepseek-v4-proxy.json
            --max_parallem_sum 2000000
            --verbose none
            --results_path "${result_root}/${topology}/${offload_name}"
        )
        if [[ -n "${kv_config}" ]]; then
            command+=(--kv_transfer_config "${kv_config}")
        fi
        if [[ -n "${max_requests}" ]]; then
            command+=(--agentx_max_requests "${max_requests}")
        fi
        "${command[@]}"
    done
}

run_case b200-tp8 ./data/clusters/8_b200/tp8_ep.json hbm-only ""
run_case b200-tp8 ./data/clusters/8_b200/tp8_ep.json dram-offload \
    ./data/kv_transfer/mooncake_store_dram_agentx.json
run_case b200-tp4-dp2 ./data/clusters/8_b200/tp4_dp2_ep.json hbm-only ""
run_case b200-tp4-dp2 ./data/clusters/8_b200/tp4_dp2_ep.json dram-offload \
    ./data/kv_transfer/mooncake_store_dram_agentx.json
run_case b300-tp8 ./data/clusters/8_b300/tp8_ep.json hbm-only ""
run_case b300-tp8 ./data/clusters/8_b300/tp8_ep.json dram-offload \
    ./data/kv_transfer/mooncake_store_dram_agentx.json
run_case b300-tp4-dp2 ./data/clusters/8_b300/tp4_dp2_ep.json hbm-only ""
run_case b300-tp4-dp2 ./data/clusters/8_b300/tp4_dp2_ep.json dram-offload \
    ./data/kv_transfer/mooncake_store_dram_agentx.json

./scripts/run_python.sh ./scripts/summarize_agentx.py "${result_root}"
