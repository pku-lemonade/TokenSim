"""Preserve the bespoke AIConfigurator MegaMoE scope when borrowing data."""

from dataclasses import replace
import math
from pathlib import Path

from TokenSim.operator_data.package import (
    PackageMeta,
    SourceRecord,
    validate_rows,
    _write_table,
    META_FILENAME,
    OperatorDataValidationError,
)
from TokenSim.hardware._yaml import load_yaml_mapping
from TokenSim.operator_data.schema import TABLE_SPECS


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
    if not result:
        raise OperatorDataValidationError("no supported MegaMoE measurements in source")
    return result


def attach(
    target,
    rows,
    *,
    source_hardware,
    collection_version,
    upstream_commit,
    relative_source,
):
    """Add only this table and its metadata; leave all existing data files intact."""
    import yaml

    target = Path(target)
    meta_path = target / META_FILENAME
    meta = PackageMeta.from_mapping(load_yaml_mapping(meta_path), meta_path)
    source_id = f"aiconfigurator-borrowed:{source_hardware}:dsv4_megamoe:sglang:{collection_version}"
    normalized = normalize(
        rows,
        source_id,
        source_hardware=source_hardware,
        collection_version=collection_version,
    )
    source = SourceRecord(
        source_id,
        "B",
        "imported",
        reference=f"https://github.com/ai-dynamo/aiconfigurator/blob/{upstream_commit}/{relative_source}",
        device=source_hardware,
        backend=f"sglang:{collection_version}",
        notes="Cross-hardware/framework borrowing at coefficient 1.0. Routed module includes dispatch/combine; gate/top-k and shared FP8 expert excluded. Buffer capacity and recorded versions retained per row.",
    )
    updated = replace(
        meta,
        sources={**meta.sources, source_id: source},
        dataset_version=meta.dataset_version.split("+megamoe")[0]
        + "+megamoe-"
        + upstream_commit[:7],
    )
    checked = validate_rows("dsv4_megamoe", normalized, updated)
    spec = TABLE_SPECS["dsv4_megamoe"]
    _write_table(target / spec.perf_filename, checked, spec.required_fields)
    metadata = updated.to_dict()
    metadata["tables"]["dsv4_megamoe"] = {"rows": len(checked)}
    metadata["dsv4_megamoe_scope"] = {
        "source_hardware": source_hardware,
        "target_hardware": meta.device_id,
        "source_framework": "sglang",
        "target_framework": meta.backend,
        "dispatch_combine_included": True,
        "gate_topk_included": False,
        "shared_experts_included": 0,
        "initial_latency_scale": 1.0,
        "single_nvlink_node_ep_sizes": [4, 8],
        "buffer_capacity_match": "approximation; collector and serving capacities differ",
    }
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True))
    return checked
