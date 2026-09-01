#!/usr/bin/env python3
import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

MODEL = "minimax-m2.7-229b"
MODEL_CONFIG = "data/psla/minimax-m2.7-229b.json"
QPS_FULL = (0.05, 0.1, 0.3, 0.5, 0.8, 1, 2, 4, 8, 16, 32)
QPS_STANDARD = tuple(value for value in QPS_FULL if value != 0.8)
TRACE_DATA = (
    ("traceA", "qwen_traceA_blksz_16.jsonl", 43058),
    ("traceB", "qwen_traceB_blksz_16.jsonl", 172800),
    ("thinking", "qwen_thinking_blksz_16.jsonl", 10812),
    ("coder", "qwen_coder_blksz_16.jsonl", 43011),
)
HBF1_48 = ((8, 80), (12, 120), (16, 160), (20, 200), (30, 300))
HBF2 = ((8, 80), (12, 120), (20, 200), (30, 300))
SSD_CAPACITIES_TIB = (12, 24, 48)
GPU_DATA = {
    "H100": {
        "cluster": "data/clusters/8_h100/h8.json",
        "hbf1_48_base": "data/kv_transfer/mooncake_store_h100_hbf1-48gib.json",
        "hbf2_base": "data/kv_transfer/mooncake_store_h100_hbf2.json",
        "ssd_base": "data/kv_transfer/mooncake_store_h100_ssd_hbf1.json",
    },
    "b200": {
        "cluster": "data/clusters/8_b200/h8.json",
        "hbf1_48_base": "data/kv_transfer/mooncake_store_b200_hbf1-48gib.json",
        "hbf2_base": "data/kv_transfer/mooncake_store_b200_hbf2.json",
        "ssd_base": "data/kv_transfer/mooncake_store_b200_ssd_hbf1.json",
    },
}


@dataclass(frozen=True)
class Variant:
    name: str
    family: str
    read_latency_us: int | None
    write_latency_us: int | None
    ssd_capacity_tib: int | None
    qps_values: tuple[float, ...]


def qps_directory(qps: float) -> str:
    return f"qps_{qps:g}".replace(".", "p")


def variants() -> tuple[Variant, ...]:
    values = [
        Variant(
            name=f"hbf1-48gib-r{read}-w{write}",
            family="hbf1-48gib",
            read_latency_us=read,
            write_latency_us=write,
            ssd_capacity_tib=None,
            qps_values=QPS_FULL,
        )
        for read, write in HBF1_48
    ]
    values.extend(
        Variant(
            name=f"hbf2-r{read}-w{write}",
            family="hbf2",
            read_latency_us=read,
            write_latency_us=write,
            ssd_capacity_tib=None,
            qps_values=QPS_STANDARD,
        )
        for read, write in HBF2
    )
    values.extend(
        Variant(
            name=f"ssd{capacity}",
            family="ssd",
            read_latency_us=None,
            write_latency_us=None,
            ssd_capacity_tib=capacity,
            qps_values=QPS_STANDARD,
        )
        for capacity in SSD_CAPACITIES_TIB
    )
    return tuple(values)


def write_config(workdir: Path, gpu: str, variant: Variant) -> str:
    gpu_data = GPU_DATA[gpu]
    if variant.family == "hbf1-48gib":
        base_path = workdir / gpu_data["hbf1_48_base"]
    elif variant.family == "hbf2":
        base_path = workdir / gpu_data["hbf2_base"]
    else:
        base_path = workdir / gpu_data["ssd_base"]

    payload = json.loads(base_path.read_text())
    extra = payload["kv_connector_extra_config"]
    output_dir = workdir / "data/kv_transfer_minimax_sweep"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{gpu.lower()}_{variant.name}.json"

    if variant.family == "ssd":
        extra["experiment_label"] = f"{gpu.lower()}-minimax-{variant.name}-legacy"
        extra["ssd_capacity_gb"] = variant.ssd_capacity_tib * 1024
        extra["ssd_io_model"] = "legacy"
        for field in (
            "ssd_bucket_size_mb",
            "ssd_bucket_max_keys",
            "ssd_offload_heartbeat_seconds",
            "ssd_offload_queue_limit",
            "ssd_offload_queue_cap_ratio",
            "ssd_high_watermark_ratio",
            "ssd_low_watermark_ratio",
            "ssd_offload_policy",
        ):
            extra.pop(field, None)
    else:
        extra["experiment_label"] = f"{gpu.lower()}-minimax-{variant.name}"
        extra["hbf_read_latency_us"] = variant.read_latency_us
        extra["hbf_write_latency_us"] = variant.write_latency_us

    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    return output_path.relative_to(workdir).as_posix()


def write_queue(run_root: Path, trace: str, rows: list[dict[str, object]]) -> None:
    queue_path = run_root / "queues" / f"{trace}.tsv"
    queue_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "core_rank",
        "sequence",
        "matrix_row",
        "trace",
        "gpu",
        "qps",
        "variant",
        "cluster",
        "model_config",
        "kv_config",
        "dataset",
        "request_count",
    )
    with queue_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for index, row in enumerate(rows):
            row["core_rank"] = index % 96
            row["sequence"] = index // 96
            row["matrix_row"] = index
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    arguments = parser.parse_args()
    workdir = arguments.workdir.resolve()
    run_root = arguments.run_root.resolve()

    if not (workdir / MODEL_CONFIG).is_file():
        raise FileNotFoundError(workdir / MODEL_CONFIG)
    for _, dataset, _ in TRACE_DATA:
        trace_path = workdir / "qwen-bailian-usagetraces-anon" / dataset
        if not trace_path.is_file():
            raise FileNotFoundError(trace_path)

    variant_values = variants()
    if any(variant.name.startswith("hbf1-r") for variant in variant_values):
        raise AssertionError("excluded HBF1 variants must not enter the manifest")
    if any("batched" in variant.name for variant in variant_values):
        raise AssertionError("Mooncake SSD-batching variants must not enter the manifest")

    generated_configs = {
        (gpu, variant.name): write_config(workdir, gpu, variant)
        for gpu in GPU_DATA
        for variant in variant_values
    }
    for config_path in generated_configs.values():
        config = json.loads((workdir / config_path).read_text())
        extra = config["kv_connector_extra_config"]
        if extra.get("offload_tier") == "ssd" and extra.get("ssd_io_model") != "legacy":
            raise AssertionError(f"SSD config is not legacy/non-batched: {config_path}")

    all_rows: dict[str, list[dict[str, object]]] = {trace: [] for trace, _, _ in TRACE_DATA}
    for trace, dataset, request_count in TRACE_DATA:
        for gpu, gpu_data in GPU_DATA.items():
            for variant in variant_values:
                for qps in variant.qps_values:
                    all_rows[trace].append(
                        {
                            "trace": trace,
                            "gpu": gpu,
                            "qps": f"{qps:g}",
                            "variant": variant.name,
                            "cluster": gpu_data["cluster"],
                            "model_config": MODEL_CONFIG,
                            "kv_config": generated_configs[(gpu, variant.name)],
                            "dataset": f"qwen-bailian-usagetraces-anon/{dataset}",
                            "request_count": request_count,
                        }
                    )
        write_queue(run_root, trace, all_rows[trace])

    counts = {trace: len(rows) for trace, rows in all_rows.items()}
    if counts != {trace: 250 for trace, _, _ in TRACE_DATA}:
        raise AssertionError(counts)
    if sum(counts.values()) != 1000:
        raise AssertionError(counts)

    metadata = {
        "model": MODEL,
        "total_simulations": 1000,
        "per_trace": counts,
        "gpus": list(GPU_DATA),
        "traces": [trace for trace, _, _ in TRACE_DATA],
        "variants": [variant.name for variant in variant_values],
        "excluded_variants": ["hbf1-r8-w80", "hbf1-r12-w120", "hbf1-r20-w200", "hbf1-r30-w300"],
        "ssd_io_model": "legacy",
        "mooncake_ssd_batching_enabled": False,
        "qps_full": list(QPS_FULL),
        "qps_standard": list(QPS_STANDARD),
    }
    metadata_path = run_root / "meta" / "manifest.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
