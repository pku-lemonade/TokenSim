#!/usr/bin/env bash
set -uo pipefail

: "${WORKDIR:?WORKDIR is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${TRACE_NAME:?TRACE_NAME is required}"
: "${SLURM_PROCID:?SLURM_PROCID is required}"

rank=$SLURM_PROCID
node=${SLURMD_NODENAME:-unknown}
queue="$RUN_ROOT/queues/${TRACE_NAME}.tsv"
progress="$RUN_ROOT/progress/${TRACE_NAME}_rank${rank}.tsv"
failures=0

printf 'trace\trank\tsequence\tmatrix_row\tstatus\tjson_count\tnode\tseconds\tresult_dir\tstdout\tstderr\n' > "$progress"
while IFS=$'\t' read -r core_rank sequence matrix_row trace gpu qps variant cluster model_config kv_config dataset request_count; do
    safe_qps=${qps//./p}
    name="${trace}_${gpu}_${variant}_${safe_qps}_row${matrix_row}"
    final="$RUN_ROOT/results/1-latency-sweep/${gpu}/minimax-m2.7-229b/${variant}/${trace}/qps_${safe_qps}"
    stdout="$RUN_ROOT/logs/${name}.log"
    stderr="$RUN_ROOT/stderr/${name}.err"
    scratch="/tmp/minimax_m2_7_${SLURM_JOB_ID}_${rank}_${sequence}"
    mkdir -p "$final"

    shopt -s nullglob
    existing=("$final"/result_*.json)
    shopt -u nullglob
    if ((${#existing[@]})); then
        printf '%s\t%s\t%s\t%s\tSKIPPED\t%s\t%s\t0\t%s\t%s\t%s\n' \
            "$trace" "$rank" "$sequence" "$matrix_row" "${#existing[@]}" "$node" "$final" "$stdout" "$stderr" \
            >> "$progress"
        continue
    fi

    rm -rf "$scratch"
    mkdir -p "$scratch/results"
    start_epoch=$(date +%s)
    printf 'trace=%s gpu=%s node=%s rank=%s sequence=%s matrix_row=%s variant=%s qps=%s\n' \
        "$trace" "$gpu" "$node" "$rank" "$sequence" "$matrix_row" "$variant" "$qps" > "$stdout"
    case "$trace" in
        traceA) trace_offset=0 ;;
        traceB) trace_offset=1000 ;;
        thinking) trace_offset=2000 ;;
        coder) trace_offset=3000 ;;
        *) echo "unknown trace: $trace" >>"$stderr"; exit 2 ;;
    esac
    program_id=$((7600000 + trace_offset + matrix_row))

    cd "$WORKDIR"
    set +e
    python benchmark.py \
        --batching paged-attn \
        --block_size 16 \
        --request_count "$request_count" \
        --cluster "$cluster" \
        --dataset_path "$dataset" \
        --workload_type qwen_jsonl \
        --qps "$qps" \
        --trace_target_qps "$qps" \
        --max_parallem_sum 100 \
        --verbose simple \
        --program_id "$program_id" \
        --model "$model_config" \
        --tensor_parallel_size 8 \
        --pipeline_parallel_size 1 \
        --data_parallel_size 1 \
        --enable_expert_parallel \
        --kv_transfer_config "$kv_config" \
        --results_path "$scratch/results" \
        >> "$stdout" 2>"$stderr"
    status=$?
    set -e

    shopt -s nullglob
    files=("$scratch/results"/*.json)
    json_count=${#files[@]}
    if ((json_count)); then
        cp -f "${files[@]}" "$final/"
    fi
    shopt -u nullglob
    if ((status == 0 && json_count == 0)); then
        status=3
    fi
    if ((status != 0)); then
        failures=$((failures + 1))
    fi
    end_epoch=$(date +%s)
    seconds=$((end_epoch - start_epoch))
    rm -rf "$scratch"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$trace" "$rank" "$sequence" "$matrix_row" "$status" "$json_count" "$node" "$seconds" "$final" "$stdout" "$stderr" \
        >> "$progress"
done < <(tail -n +2 "$queue" | awk -F '\t' -v rank="$rank" '$1 == rank')

exit "$failures"
