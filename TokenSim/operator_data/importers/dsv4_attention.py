"""Import complete DSV4 CSA/HCA module measurements with their scope intact."""

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


MODEL = "deepseek-ai/DeepSeek-V4-Pro"
ARCHITECTURE = "DeepseekV4ForCausalLM"


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
        raise ValueError("unsupported DSV4 attention layer or phase")
    result = []
    for row in rows:
        if row.get("kernel_source") != "FLASHMLA_SPARSE_DSV4":
            continue
        if row.get("kv_cache_dtype") != "fp8" or row.get("gemm_type") != "fp8_block":
            raise OperatorDataValidationError(
                "DSV4 attention requires the collected FP8 KV/fp8_block scope"
            )
        if int(row.get("compress_ratio", 0)) <= 1:
            raise OperatorDataValidationError(
                "DSV4 attention compression ratio mismatch"
            )
        tp = int(row["tp_size"])
        heads = int(row["num_heads"])
        if tp < 1 or heads < 1:
            raise OperatorDataValidationError(
                "DSV4 attention rows must use local Pro heads for TP1/2/4/8"
            )
        if int(row["isl"]) < 1 or int(row["step"]) < 0:
            raise OperatorDataValidationError(
                "DSV4 attention lengths must be non-negative"
            )
        latency = float(row["latency"])
        if not math.isfinite(latency) or latency <= 0:
            raise OperatorDataValidationError("invalid DSV4 attention latency")
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
            f"no {MODEL} {layer_kind}/{phase} DSV4 attention rows in source"
        )
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
    """Attach normalized rows while preserving every existing operator table."""
    import yaml

    target = Path(target)
    meta_path = target / META_FILENAME
    meta = PackageMeta.from_mapping(load_yaml_mapping(meta_path), meta_path)
    source_id = (
        f"aiconfigurator-borrowed:{source_hardware}:dsv4_attention:"
        f"vllm:{collection_version}"
    )
    normalized = []
    for layer_kind, phase, source_rows in rows:
        normalized.extend(
            normalize(
                source_rows,
                layer_kind=layer_kind,
                phase=phase,
                source_id=source_id,
                source_hardware=source_hardware,
                source_version=source_version,
            )
        )
    source = SourceRecord(
        source_id,
        "B",
        "imported",
        reference=(
            "https://github.com/ai-dynamo/aiconfigurator/blob/"
            f"{upstream_commit}/{relative_source}"
        ),
        device=source_hardware,
        backend=f"vllm:{collection_version}",
        notes=(
            "Complete DSV4 CSA/HCA attention module: input projections, "
            "compressor/indexer, sparse attention and output projection. "
            "Collected kernel is FLASHMLA_SPARSE_DSV4 with FP8 KV; the AgentX "
            "reference selects FlashInfer and FP4 indexer cache, so this is a "
            "measured module baseline with an explicit backend/layout mismatch."
        ),
    )
    updated = replace(
        meta,
        sources={**meta.sources, source_id: source},
        dataset_version=meta.dataset_version.split("+dsv4attn")[0]
        + "+dsv4attn-"
        + upstream_commit[:7],
    )
    checked = validate_rows("dsv4_attention", normalized, updated)
    spec = TABLE_SPECS["dsv4_attention"]
    _write_table(target / spec.perf_filename, checked, spec.required_fields)
    metadata = updated.to_dict()
    metadata["tables"]["dsv4_attention"] = {"rows": len(checked)}
    metadata["dsv4_attention_scope"] = {
        "source_hardware": source_hardware,
        "target_hardware": meta.device_id,
        "source_framework": "vllm",
        "target_framework": meta.backend,
        "source_version": source_version,
        "kernel_source": "FLASHMLA_SPARSE_DSV4",
        "module_scope": "input projections + compressor/indexer + sparse attention + output projection",
        "kv_cache_dtype": "fp8 (source table; reference uses plain FP8)",
        "indexer_cache_dtype": "source collector default (not reference MXFP4)",
        "reference_backend": "FLASHINFER_MLA_SPARSE_DSV4",
        "initial_latency_scale": 1.0,
    }
    meta_path.write_text(yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True))
    return checked
