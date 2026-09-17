#!/usr/bin/env bash
# Collect collective latency curves with nccl-tests and import them.
#
#   NCCL_TESTS=/path/to/nccl-tests/build DEVICE=a100_sxm_80g ./scripts/profiling/run_nccl_tests.sh 8 1
#
# Arguments: <gpus per node> <nodes>. Multi-node runs need mpirun; adapt the
# launcher below. Each binary writes one text file that operator_data.cli
# import-nccl converts into rows of the `collective` table.
set -euo pipefail
GPUS=${1:-8}
NODES=${2:-1}
NCCL_TESTS=${NCCL_TESTS:-./nccl-tests/build}
DEVICE=${DEVICE:?set DEVICE to the catalog device_id, e.g. a100_sxm_80g}
OUT=${OUT:-./tmp/nccl/${DEVICE}}
mkdir -p "$OUT"
for pair in all_reduce_perf:all_reduce all_gather_perf:all_gather reduce_scatter_perf:reduce_scatter alltoall_perf:all_to_all; do
  bin=${pair%%:*}; op=${pair##*:}
  echo "== $bin on $GPUS GPUs x $NODES nodes"
  "$NCCL_TESTS/$bin" -b 1K -e 1G -f 2 -g "$GPUS" -d half | tee "$OUT/${bin}_g${GPUS}_n${NODES}.txt"
  python -m TokenSim.operator_data.cli import-nccl --device "$DEVICE" --backend nccl \
    --file "$OUT/${bin}_g${GPUS}_n${NODES}.txt" --operation "$op" --group-size $((GPUS * NODES)) --nodes "$NODES" \
    --reference "nccl-tests $bin -b 1K -e 1G -f 2 -g $GPUS -d half"
done
