#!/usr/bin/env bash
# Measure GEMM / attention / MoE / intra-node collective latencies on the
# current node with the NVIDIA AIConfigurator collector, then import the result
# into a TokenSim operator package.
#
#   DEVICE=a100_sxm_80g ./scripts/collect/run_aiconfigurator.sh
#   DEVICE=rtx_4090 OPS="gemm attention" MODEL_PATH=meta-llama/Llama-3.1-8B ./scripts/collect/run_aiconfigurator.sh
#   DEVICE=a100_sxm_80g ./scripts/collect/run_aiconfigurator.sh --import-only   # reuse an earlier run directory
#
# Environment:
#   DEVICE        TokenSim device_id (data/devices/<id>.yaml)          required
#   BACKEND       trtllm | vllm | sglang                               default trtllm
#   AIC_DIR       AIConfigurator checkout                              default ./tmp/aiconfigurator
#   AIC_COMMIT    upstream commit to pin                               default f254959eb89e2f206b8f9a77051644d7c1cbdb89
#   RUN_DIR       where the collector writes *_perf.parquet            default ./tmp/collect/<DEVICE>/<BACKEND>
#   OPS           collector ops, space separated                       default "gemm attention moe"
#   MODEL_PATH    HF model id for a model-centric ("healing") run      default: full base grid
#   GPU_TYPE      AIConfigurator system name for capability floors     default: local SM detection
#   SKIP_COMM     1 = skip network/collect_comm.sh                     default 0
#   PACKAGE       import target                                        default ./data/operator_data/<DEVICE>/<BACKEND>
set -euo pipefail
DEVICE=${DEVICE:?set DEVICE to the catalog device_id, e.g. a100_sxm_80g}
BACKEND=${BACKEND:-trtllm}
AIC_DIR=${AIC_DIR:-./tmp/aiconfigurator}
AIC_COMMIT=${AIC_COMMIT:-f254959eb89e2f206b8f9a77051644d7c1cbdb89}
RUN_DIR=${RUN_DIR:-./tmp/collect/${DEVICE}/${BACKEND}}
OPS=${OPS:-"gemm attention moe"}
MODEL_PATH=${MODEL_PATH:-}
GPU_TYPE=${GPU_TYPE:-}
SKIP_COMM=${SKIP_COMM:-0}
PACKAGE=${PACKAGE:-./data/operator_data/${DEVICE}/${BACKEND}}
REPO_ROOT=$(cd "$(dirname "$0")/../.." && pwd)
IMPORT_ONLY=0
[ "${1:-}" = "--import-only" ] && IMPORT_ONLY=1

if [ "$IMPORT_ONLY" -eq 0 ]; then
  # 1. Pin the collector source.
  if [ ! -d "$AIC_DIR/collector" ]; then
    git clone --filter=blob:none https://github.com/ai-dynamo/aiconfigurator.git "$AIC_DIR"
  fi
  git -C "$AIC_DIR" fetch --quiet origin "$AIC_COMMIT" 2>/dev/null || true
  git -C "$AIC_DIR" checkout --quiet "$AIC_COMMIT"

  # 2. Sanity checks: the framework must be importable and clocks should be locked.
  python - <<PY
import importlib, sys
mod = {"trtllm": "tensorrt_llm", "vllm": "vllm", "sglang": "sglang"}["$BACKEND"]
try:
    m = importlib.import_module(mod)
    print(f"{mod} {getattr(m, '__version__', '?')} importable")
except Exception as exc:  # pragma: no cover
    sys.exit(f"{mod} is not importable in this environment: {exc}")
PY
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,persistence_mode,clocks.sm,clocks.max.sm,clocks.mem --format=csv | tee "$RUN_DIR.env.txt" 2>/dev/null || true
    echo "If clocks.sm != clocks.max.sm consider: sudo nvidia-smi -pm 1 && sudo nvidia-smi -lgc <max_sm>"
  fi

  # 3. Run the collector from RUN_DIR (it writes *_perf.parquet into the cwd).
  mkdir -p "$RUN_DIR"
  RUN_DIR_ABS=$(cd "$RUN_DIR" && pwd)
  pushd "$RUN_DIR_ABS" >/dev/null
  COLLECT_ARGS=(--backend "$BACKEND" --resume)
  [ -n "$OPS" ] && COLLECT_ARGS+=(--ops $OPS)
  [ -n "$MODEL_PATH" ] && COLLECT_ARGS+=(--model-path "$MODEL_PATH")
  [ -n "$GPU_TYPE" ] && COLLECT_ARGS+=(--gpu "$GPU_TYPE")
  echo "== python3 $AIC_DIR/collector/collect.py ${COLLECT_ARGS[*]}"
  PYTHONPATH="$AIC_DIR:${PYTHONPATH:-}" python3 "$AIC_DIR/collector/collect.py" "${COLLECT_ARGS[@]}"
  if [ "$SKIP_COMM" -eq 0 ]; then
    echo "== network/collect_comm.sh --all_reduce_backend $BACKEND"
    PYTHONPATH="$AIC_DIR:${PYTHONPATH:-}" bash "$AIC_DIR/collector/network/collect_comm.sh" --all_reduce_backend "$BACKEND" || \
      echo "collect_comm.sh failed (nccl-tests binaries on PATH? mpirun available?); continuing with compute tables"
    # collect_comm.sh leaves CSV staging files; the importer reads *_perf.txt directly.
  fi
  popd >/dev/null
fi

# 4. Convert into a TokenSim package (flat collector layout is auto-detected).
cd "$REPO_ROOT"
python -m TokenSim.operator_data.cli import-aiconfigurator \
  --system-dir "$RUN_DIR" --device "$DEVICE" --backend "$BACKEND" \
  --upstream-commit "$AIC_COMMIT" --out "$PACKAGE"
python -m TokenSim.operator_data.cli validate "$PACKAGE"
echo "package written to $PACKAGE"
