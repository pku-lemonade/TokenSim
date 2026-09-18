#!/usr/bin/env bash
# Collect collective latency curves with nccl-tests and import them into a
# TokenSim operator package. Covers what the AIConfigurator collector does not:
# multi-node groups and NIC fabrics.
#
#   DEVICE=a100_sxm_80g NCCL_TESTS=/opt/nccl-tests/build ./scripts/collect/run_nccl_tests.sh 8 1
#   DEVICE=a100_sxm_80g NODES=2 LAUNCHER="mpirun -np 16 -H nodeA:8,nodeB:8" ./scripts/collect/run_nccl_tests.sh 8 2
#
# Arguments: <gpus per node> <nodes>. For a single node the binaries are run
# with -g <gpus>; for several nodes set LAUNCHER to the mpirun/srun command that
# starts one process per GPU. Each binary writes a text file that
# operator_data.cli import-nccl converts into rows of the `collective` table.
set -euo pipefail
GPUS=${1:-8}
NODES=${2:-1}
NCCL_TESTS=${NCCL_TESTS:-./nccl-tests/build}
DEVICE=${DEVICE:?set DEVICE to the catalog device_id, e.g. a100_sxm_80g}
BACKEND=${BACKEND:-trtllm}
OUT=${OUT:-./tmp/nccl/${DEVICE}}
PACKAGE=${PACKAGE:-./data/operator_data/${DEVICE}/${BACKEND}}
DTYPE=${DTYPE:-half}
LAUNCHER=${LAUNCHER:-}
mkdir -p "$OUT"
GROUP=$((GPUS * NODES))
for pair in all_reduce_perf:all_reduce all_gather_perf:all_gather reduce_scatter_perf:reduce_scatter alltoall_perf:all_to_all; do
  bin=${pair%%:*}; op=${pair##*:}
  echo "== $bin on $GPUS GPUs x $NODES nodes (dtype $DTYPE)"
  log="$OUT/${bin}_g${GPUS}_n${NODES}_${DTYPE}.txt"
  if [ -z "$LAUNCHER" ]; then
    "$NCCL_TESTS/$bin" -b 1K -e 1G -f 2 -g "$GPUS" -d "$DTYPE" -w 20 -n 50 | tee "$log"
  else
    $LAUNCHER "$NCCL_TESTS/$bin" -b 1K -e 1G -f 2 -g 1 -d "$DTYPE" -w 20 -n 50 | tee "$log"
  fi
  python -m TokenSim.operator_data.cli import-nccl --device "$DEVICE" --backend "$BACKEND" \
    --file "$log" --operation "$op" --group-size "$GROUP" --nodes "$NODES" --out "$PACKAGE" \
    --reference "nccl-tests $bin -b 1K -e 1G -f 2 -d $DTYPE on $GROUP ranks / $NODES nodes" \
    --notes "NCCL $(python -c 'import torch;print(".".join(map(str,torch.cuda.nccl.version())))' 2>/dev/null || echo unknown); NCCL_DEBUG=${NCCL_DEBUG:-unset}; launcher: ${LAUNCHER:-local}"
done
echo "collective rows imported into $PACKAGE"
