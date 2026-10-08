"""Reproduce target row ownership gaps while the runtime remains frozen."""

import hashlib
import json
import math
from pathlib import Path
from unittest.mock import patch

from TokenSim.config.config import ParallelConfig, ParallelRankInfo
from TokenSim.hardware.context import HardwareContext
from TokenSim.latency.deepseek_v4_moe import MoEPass, RankShape, uniform_decode_rows
from TokenSim.latency.operator_table import OperatorTableLatencyBackend
from TokenSim.llm.llm_request import Request


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
VLLM = Path("/home/zhuohang/Serving/vllm-note")
context = HardwareContext.load(ROOT / "data")
model = context.model("DeepSeek-V4-Pro")
cases = []
for sku in ("b200", "b300"):
    for tp in (1, 4, 8):
        dp = 8 // tp
        parallel = ParallelConfig(tensor_parallel_size=tp, data_parallel_size=dp,
                                  enable_expert_parallel=True)
        device = context.device(sku)
        backend = OperatorTableLatencyBackend(
            device=device, model=model, parallel_config=parallel,
            rank_info=ParallelRankInfo.from_global_rank(0, parallel),
            package=context.operator_package(device.device_id, "vllm"),
            attention_backend="native_dsv4", mhc_backend="native_dsv4",
            moe_execution={"backend": "deep_gemm_amxf4_mega_moe"},
            expected_decode_query_len=4,
        )
        for eager in (False, True):
            local_rows = 20
            width = 1 if eager and dp == 1 else 4
            step = MoEPass({rank: RankShape(
                local_rows if rank == 0 else (512 if eager else 64),
                width if rank == 0 else (None if eager else 4),
                eager and rank != 0,
                local_rows // width if rank == 0 else (1 if eager else 16),
            ) for rank in range(dp)})
            backend.set_moe_step(step)
            requests = []
            for identifier in range(local_rows // width):
                request = Request(identifier, 8192, 20, 256)
                request.generation_idx = 1
                request.num_computed_tokens = 8192
                request.scheduled_tokens = min(2, width)
                request.scheduled_query_tokens = width
                requests.append(request)
            with patch.object(backend.native_mhc, "estimate", wraps=backend.native_mhc.estimate) as mhc, \
                    patch.object(backend, "_gemm", wraps=backend._gemm) as gemm, \
                    patch.object(backend.native_attention, "estimate_batch", wraps=backend.native_attention.estimate_batch) as attention, \
                    patch.object(backend.native_mtp, "estimate", return_value={}):
                backend.estimate_step_latency(requests)
            capture = uniform_decode_rows(step, 4, backend.max_num_seqs)
            full = capture if capture is not None else local_rows
            source_shard = math.ceil(full / tp)
            router_calls = [call.args[0] for call in gemm.call_args_list
                            if call.args[1:3] == (model.moe.num_experts, model.hidden_size)]
            assert len(router_calls) == 1
            sparse_buckets = attention.call_args.args[1]
            actual_queries = sum(width * count for (_length, width), count in sparse_buckets.items())
            assert actual_queries == local_rows
            parts = []
            index_fraction = math.ceil(width / 2) / width
            for kind in ("csa", "hca"):
                estimator = backend.native_attention
                latency, key = estimator._decode_layer(kind, local_rows, 8192,
                                                       indexer_kv_fraction=index_fraction)
                shared, phase = estimator.sol.shared_phase_us(key, indexer_kv_fraction=index_fraction)
                floor_rows = key["batch_size"]
                captured_key = {**key, "batch_size": max(full, floor_rows)}
                captured_shared = estimator.sol.shared_phase_us(captured_key, indexer_kv_fraction=index_fraction)[0]
                increment = max(0, captured_shared - shared) * latency / (shared + phase)
                parts.append({"kind": kind, "source_floor_rows": floor_rows,
                              "candidate_shared_rows": captured_key["batch_size"],
                              "source_shared_bound_us": shared, "source_phase_bound_us": phase,
                              "candidate_projection_increment_per_layer_us": increment})
            cases.append({
                "hardware": sku, "tp": tp, "dp": dp,
                "regime": "remote-context-forced-eager" if eager and dp > 1 else ("staged-q1-eager" if eager else "eligible-decode"),
                "logical_rows": local_rows, "capture_rows": capture,
                "source_full_rows": full, "source_sp_rows": source_shard,
                "current_main_mhc_rows": mhc.call_args.args[0],
                "current_main_router_rows": router_calls[0],
                "current_attention_logical_rows": actual_queries,
                "source_main_mhc_and_router_rows": source_shard,
                "current_target_head_rows": backend.target_logits_counts["output_forward_rows"],
                "projection_allocation_diagnostic": parts,
            })
            backend.reset_profile_stats()

paths = [
    VLLM / "vllm/v1/worker/gpu/model_runner.py",
    VLLM / "vllm/v1/worker/gpu/cudagraph_utils.py",
    VLLM / "vllm/models/deepseek_v4/nvidia/model.py",
    VLLM / "vllm/models/deepseek_v4/attention.py",
    VLLM / "vllm/models/deepseek_v4/nvidia/flashinfer_sparse.py",
    ROOT / "TokenSim/latency/operator_table.py",
    ROOT / "TokenSim/latency/deepseek_v4_attention.py",
]
report = {
    "scope": "source and current target row ownership; offline follow-up, no runtime edits",
    "runtime_change": False, "formal_accuracy_evidence": False,
    "vllm_revision": "30f5c01d4ebb5c493ed86195db24d0ec5b3ce3d7",
    "source_checks": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths],
    "facts": [
        "FULL graph input/output projections consume descriptor rows, while sparse query metadata keeps real query ownership",
        "main mHC/norm operates on contiguous SP shards before attention all-gather and after its reduce-scatter",
        "main FFN gate/router receives the SP-sharded tensor, including graph and SP tail padding",
        "current target output head already owns descriptor rows; do not add its cost again",
        "source minimum query floors can already cover graph rows; only positive incremental shared work is eligible",
    ],
    "cases": cases,
    "limits": [
        "whole-module shared/phase allocation is source-derived SOL, not measured component timing",
        "candidate projection increments are diagnostics only and have not been added to serving runtime",
        "historical image padding/fusion defaults are unrecorded; current read-only source is the implementation basis",
    ],
    "next_stage": "independently align eligible main capture/SP row owners; retain valid sparse queries and source floors; then full repository tests and both full windows",
}
destination = EVIDENCE / "capture-ownership-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for case in cases:
    print(case["hardware"], "TP", case["tp"], case["regime"],
          "mhc", case["current_main_mhc_rows"], "router", case["current_main_router_rows"],
          "source", case["source_main_mhc_and_router_rows"])
