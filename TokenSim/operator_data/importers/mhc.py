"""Import the measured vLLM mHC pre/post module for DSV4-Pro."""

from dataclasses import replace
import math
from pathlib import Path

from TokenSim.hardware._yaml import load_yaml_mapping
from TokenSim.operator_data.package import (
    META_FILENAME,
    OperatorDataValidationError,
    PackageMeta,
    SourceRecord,
    _write_table,
    validate_rows,
)
from TokenSim.operator_data.schema import TABLE_SPECS


def normalize(rows, *, source_id, source_hardware, source_version):
    result = []
    for row in rows:
        if (
            min(
                int(row.get(name, 0))
                for name in ("hidden_size", "hc_mult", "num_sites", "sinkhorn_iters")
            )
            <= 0
        ):
            raise OperatorDataValidationError("unsupported mHC measurement scope")
        if row.get("op_name") not in ("pre", "post"):
            raise OperatorDataValidationError("unknown mHC operation")
        latency = float(row["latency"])
        if not math.isfinite(latency) or latency <= 0 or row["num_tokens"] < 1:
            raise OperatorDataValidationError("invalid mHC measurement")
        result.append(
            {
                "op_name": row["op_name"],
                "architecture": row["architecture"],
                "kernel_source": row["kernel_source"],
                "hidden_size": int(row["hidden_size"]),
                "hc_mult": int(row["hc_mult"]),
                "num_sites": int(row["num_sites"]),
                "sinkhorn_iters": int(row["sinkhorn_iters"]),
                "num_tokens": int(row["num_tokens"]),
                "latency_us": latency * 1000.0,
                "source_id": source_id,
                "kernel": row["kernel_source"],
                "grade": "B",
                "source_hardware": source_hardware,
                "source_version": source_version,
                "source_framework": row["framework"],
            }
        )
    if not result:
        raise OperatorDataValidationError("no mHC rows in source")
    return result


def attach(
    target,
    rows,
    *,
    source_hardware,
    source_version,
    collection_version,
    upstream_commit,
    relative_source,
):
    import yaml

    target = Path(target)
    meta_path = target / META_FILENAME
    meta = PackageMeta.from_mapping(load_yaml_mapping(meta_path), meta_path)
    source_id = (
        f"aiconfigurator-borrowed:{source_hardware}:mhc:vllm:{collection_version}"
    )
    normalized = normalize(
        rows,
        source_id=source_id,
        source_hardware=source_hardware,
        source_version=source_version,
    )
    source = SourceRecord(
        source_id,
        "B",
        "imported",
        reference=f"https://github.com/ai-dynamo/aiconfigurator/blob/{upstream_commit}/{relative_source}",
        device=source_hardware,
        backend=f"vllm:{collection_version}",
        notes="vLLM TileLang mHC pre/post modules; num_sites=2 is preserved as collector scope and is not multiplied again.",
    )
    updated = replace(
        meta,
        sources={**meta.sources, source_id: source},
        dataset_version=meta.dataset_version.split("+mhc")[0]
        + "+mhc-"
        + upstream_commit[:7],
    )
    checked = validate_rows("mhc", normalized, updated)
    spec = TABLE_SPECS["mhc"]
    _write_table(target / spec.perf_filename, checked, spec.required_fields)
    metadata = updated.to_dict()
    metadata["tables"]["mhc"] = {"rows": len(checked)}
    metadata["mhc_scope"] = {
        "source_hardware": source_hardware,
        "target_hardware": meta.device_id,
        "module_scope": "mHC pre + post TileLang; num_sites=2",
        "hidden_size": 7168,
        "hc_mult": 4,
        "sinkhorn_iters": 20,
        "residual_dtype": "bf16",
        "parameter_dtype": "fp32",
        "initial_latency_scale": 1.0,
    }
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True))
    return checked
