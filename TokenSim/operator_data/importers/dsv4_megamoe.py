"""Normalize fused expert-parallel MoE (DeepGEMM MegaMoE) module measurements.

The timed scope is fixed by the collector and checked per row: routed
dispatch, expert compute and combine under CUDA graphs, excluding gate/top-k,
buffer initialization and shared experts.
"""

import math

from TokenSim.operator_data.package import OperatorDataValidationError


def normalize(rows, source_id, *, source_hardware, collection_version):
    result = []
    for row in rows:
        if (
            min(
                int(row[name])
                for name in (
                    "hidden_size",
                    "inter_size",
                    "topk",
                    "num_experts",
                    "moe_ep_size",
                )
            )
            <= 0
        ):
            raise OperatorDataValidationError("invalid expert module geometry")
        if row["moe_tp_size"] != 1 or row["num_experts"] % row["moe_ep_size"]:
            raise OperatorDataValidationError(
                "expert module requires divisible EP geometry and TP1"
            )
        required = {
            "used_cuda_graph": True,
            "includes_gate_topk": False,
            "includes_buffer_init": False,
            "includes_routed_scale": True,
        }
        for name, expected in required.items():
            if row.get(name) is not expected:
                raise OperatorDataValidationError(
                    f"MegaMoE timing scope mismatch: {name}"
                )
        if row["num_fused_shared_experts"] != 0:
            raise OperatorDataValidationError(
                "collector appended shared MXFP4 experts cannot price native shared FP8 fusion"
            )
        if (
            row["kernel_source"],
            row["kernel_dtype"],
            row["moe_dtype"],
            row["pre_dispatch"],
            row["source_policy"],
        ) != (
            "deepgemm_megamoe",
            "fp8_fp4",
            "w4a8_mxfp4_mxfp8",
            "sglang_jit",
            "random",
        ):
            raise OperatorDataValidationError(
                "unsupported MegaMoE kernel/quantization/pre-dispatch scope"
            )
        if row["phase"] not in ("generation", "context"):
            raise OperatorDataValidationError("unknown MegaMoE collector regime")
        if (
            row["global_num_tokens"] != row["num_tokens"] * row["moe_ep_size"]
            or row["num_tokens"] < 1
        ):
            raise OperatorDataValidationError(
                "MegaMoE local/global token identity mismatch"
            )
        if not math.isfinite(row["latency"]) or row["latency"] <= 0:
            raise OperatorDataValidationError("invalid MegaMoE latency")
        result.append(
            {
                "dtype": "mxfp4",
                "phase": row["phase"],
                "kernel_source": row["kernel_source"],
                "kernel_dtype": row["kernel_dtype"],
                "distribution": row["distribution"],
                "pre_dispatch": row["pre_dispatch"],
                "source_policy": row["source_policy"],
                "num_tokens": row["num_tokens"],
                "hidden_size": row["hidden_size"],
                "inter_size": row["inter_size"],
                "top_k": row["topk"],
                "num_experts": row["num_experts"],
                "num_fused_shared_experts": 0,
                "tp_size": 1,
                "ep_size": row["moe_ep_size"],
                "latency_us": float(row["latency"]) * 1000,
                "source_id": source_id,
                "kernel": row["kernel_source"],
                "grade": "B",
                "num_max_tokens_per_rank": row["num_max_tokens_per_rank"],
                "effective_num_max_tokens_per_rank": row[
                    "effective_num_max_tokens_per_rank"
                ],
                "global_num_tokens": row["global_num_tokens"],
                "routed_scaling_factor": row["routed_scaling_factor"],
                **{name: row[name] for name in required},
                "recorded_framework_version": row["version"],
                "collection_version": collection_version,
                "source_hardware": source_hardware,
            }
        )
    return result
