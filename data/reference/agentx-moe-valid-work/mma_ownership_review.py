"""Record pinned dynamic MMA rows and the unresolved mixed-input compute pipe."""

import hashlib
import json
import math
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
SOURCE = Path("/tmp/tokensim-deepgemm-nonuniform-audit")
SCHEDULER = Path("/tmp/tokensim-deepgemm-buffer-audit/scheduler.cuh")
kernel = (SOURCE / "fp8-fp4-kernel.cuh").read_text()
scheduler = SCHEDULER.read_text()
opcode = (SOURCE / "tcgen05.cuh").read_text()
assert "update_instr_desc_with_umma_n(instr_desc, task_info.get_umma_aligned_valid_m())" in kernel
assert "return math::align(valid_m, 16u)" in scheduler
assert "SM100_MMA_MXF8F6F4_2x1SM_SS::fma" in kernel
assert "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale" in opcode
assert "constexpr uint32_t UMMA_K = 32" in kernel
assert "num_tokens - owner_m_block_idx * BLOCK_M" in scheduler

cases = []
for block in (16, 32, 64, 128, 192, 240):
    for received in (0, 1, 16, 17, 63, 64, 65, 128, 129, 200, 240, 241, 4096):
        tiles = math.ceil(received / block)
        valid_m = [min(block, received - index * block) for index in range(tiles)]
        executed_m = sum(16 * math.ceil(rows / 16) for rows in valid_m)
        assert executed_m == 16 * math.ceil(received / 16)
        cases.append({"block_m": block, "received_rows": received, "m_tiles": tiles,
                      "current_full_tile_mma_rows": tiles * block,
                      "pinned_dynamic_mma_rows": executed_m,
                      "weight_loads_per_m_tile_unchanged": True})

paths = [SOURCE / "fp8-fp4-kernel.cuh", SCHEDULER, SOURCE / "tcgen05.cuh",
         ROOT / "TokenSim/latency/deepseek_v4_moe_work.py",
         ROOT / "TokenSim/hardware/device/dtypes.py",
         ROOT / "data/devices/b200_sxm.yaml", ROOT / "data/devices/b300_sxm.yaml"]
report = {
    "scope": "offline pinned MMA instruction and dynamic valid-tail ownership; runtime frozen",
    "runtime_change": False, "formal_accuracy_evidence": False,
    "pinned_deepgemm_commit": "e1f418c2a4f20818221f6b0e578b4c2f634d4c3f",
    "source_checks": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    "source_urls": {
        "opcode": "https://raw.githubusercontent.com/vllm-project/DeepGEMM/e1f418c2a4f20818221f6b0e578b4c2f634d4c3f/deep_gemm/include/deep_gemm/ptx/tcgen05.cuh",
        "ptx_matrix_shapes": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-matrix-shape",
        "ptx_scale_kinds": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-valid-comb",
    },
    "confirmed_facts": [
        "expert tasks keep valid_m = min(expert_count - expert_M_block * BLOCK_M, BLOCK_M)",
        "kernel updates MMA descriptor N from valid_m aligned to16, after swapping A/B; FLOP rows are not always a complete BLOCK_M",
        "sum of issued MMA rows per integer-count expert is16*ceil(count/16) for all current candidate BLOCK_M; weight/scales TMA still executes each M task",
        "pinned routed opcode is tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale with FP8/FP4 operands and UMMA_K32",
        "catalog storage dtype mxfp4 currently maps to the nvfp4 compute pipe; native mixed opcode must be checked independently",
    ],
    "compute_pipe_pending": {
        "current_catalog_peak_flops_per_s": {"b200": 9e15, "b300": 14e15},
        "fp8_catalog_peak_flops_per_s": {"b200": 4.5e15, "b300": 4.5e15},
        "ptx_evidence": "MXF8F6F4 accepts mixed E4M3/E2M1 and scale_vec1X; MXF4/MXF4NVF4 uses E2M1 pairs with different K/scale shapes",
        "hardware_issue_throughput_verified": False,
        "rule": "opcode and K shape distinguish the path but alone do not establish issue cycles; do not assert an exact mixed throughput measurement",
    },
    "integer_expert_tail_cases": cases,
    "fractional_spectrum_limit": "runtime expected spectrum remains grade D; do not present fractional expected counts as measured integer route histograms",
    "next_checks": [
        "correct source-proven dynamic16-aligned MMA tail math in an independent runtime stage while retaining per-M-task traffic",
        "verify the mixed-input pipe from architecture/performance evidence before choosing a peak; no SKU-throughput fitting",
        "repeat full repository tests, uniform and nonuniform growth audits, both original full-window protocols, and paired source-scope checks",
    ],
}
destination = EVIDENCE / "mma-ownership-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
print("Integer tail cases checked:", len(cases))
