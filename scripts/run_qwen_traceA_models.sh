#!/usr/bin/env bash
set -euo pipefail

TRACE_PATH="${TRACE_PATH:-qwen-bailian-usagetraces-anon/qwen_traceA_blksz_16.jsonl}"
CLUSTER_PATH="${CLUSTER_PATH:-data/clusters/8_a100/h8_diy7.json}"
REQUEST_COUNT="${REQUEST_COUNT:-20}"
TRACE_TARGET_QPS="${TRACE_TARGET_QPS:-50}"
QPS="${QPS:-50}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/tokensim-matplotlib}"
export MPLCONFIGDIR

mkdir -p "${MPLCONFIGDIR}"

models=(
    "data/psla/deepseek-v3.2-685b.json"
    "data/psla/llama-3.1-405b.json"
    "data/psla/glm-5.2-753b.json"
    "data/psla/qwen3-4b.json"
    "data/psla/qwen3-32b.json"
)

for model in "${models[@]}"; do
    echo "[run] ${model}"
    python benchmark.py \
        --batching paged-attn \
        --block_size 16 \
        --cluster "${CLUSTER_PATH}" \
        --model "${model}" \
        --dataset_path "${TRACE_PATH}" \
        --workload_type qwen_jsonl \
        --request_count "${REQUEST_COUNT}" \
        --qps "${QPS}" \
        --trace_target_qps "${TRACE_TARGET_QPS}" \
        --max_parallem_sum 100 \
        --verbose none
done
