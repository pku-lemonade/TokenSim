"""Normalize complete compressed sparse-attention module measurements.

Each AIConfigurator row times one attention layer end to end: input
projections, compressor/indexer, sparse attention and output projection.
Model identity and head geometry stay in the row.
"""

import math

from TokenSim.operator_data.package import OperatorDataValidationError


def normalize(
    rows,
    *,
    layer_kind,
    phase,
    source_id,
    source_hardware,
    source_version,
):
    """Convert one raw collector table into the package's stable key space."""
    if layer_kind not in ("csa", "hca") or phase not in ("context", "generation"):
        raise ValueError("unsupported compressed-attention layer kind or phase")
    result = []
    for row in rows:
        if row.get("kernel_source") != "FLASHMLA_SPARSE_DSV4":
            continue
        if row.get("kv_cache_dtype") != "fp8" or row.get("gemm_type") != "fp8_block":
            raise OperatorDataValidationError(
                "compressed attention rows must use the collected FP8 KV/fp8_block scope"
            )
        if int(row.get("compress_ratio", 0)) <= 1:
            raise OperatorDataValidationError(
                "compressed attention rows need a compression ratio above one"
            )
        tp = int(row["tp_size"])
        heads = int(row["num_heads"])
        if tp < 1 or heads < 1:
            raise OperatorDataValidationError(
                "compressed attention rows need positive TP size and local heads"
            )
        if int(row["isl"]) < 1 or int(row["step"]) < 0:
            raise OperatorDataValidationError(
                "compressed attention lengths must be non-negative"
            )
        latency = float(row["latency"])
        if not math.isfinite(latency) or latency <= 0:
            raise OperatorDataValidationError("invalid compressed attention latency")
        result.append(
            {
                "layer_kind": layer_kind,
                "phase": phase,
                "mla_dtype": row["mla_dtype"],
                "kv_cache_dtype": row["kv_cache_dtype"],
                "kernel_source": row["kernel_source"],
                "kernel_dtype": row["gemm_type"],
                "model": row["model"],
                "architecture": row["architecture"],
                "num_heads": heads,
                "tp_size": tp,
                "batch_size": int(row["batch_size"]),
                "input_seq_len": int(row["isl"]) if phase == "context" else 1,
                "context_len": int(row["step"]),
                "compress_ratio": int(row["compress_ratio"]),
                "latency_us": latency * 1000.0,
                "source_id": source_id,
                "kernel": row["kernel_source"],
                "grade": "B",
                "source_hardware": source_hardware,
                "source_version": source_version,
                "source_model": row["model"],
            }
        )
    if not result:
        raise OperatorDataValidationError(
            f"no {layer_kind}/{phase} compressed attention rows in source"
        )
    return result
