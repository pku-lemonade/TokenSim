#!/usr/bin/env python3
"""Check native model geometry, rank weight storage, and full trace constraints.

This is a model/data audit. It does not run a benchmark or replace the required
one-hour simulations. Legacy replay/cache limitations are reported explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.parallel_config import ParallelRankInfo
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.config.weight_memory import deepseek_v4_weight_memory
from TokenSim.hardware.context import HardwareContext
from util.agentx_reference import file_identity, load_experiments


def audit_dataset(path: Path) -> dict:
    digest = hashlib.sha256()
    trace_ids = set()
    block_sizes = Counter()
    request_count = max_input = max_context = 0
    max_source = None
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                continue
            trace = json.loads(raw)
            if trace["id"] in trace_ids:
                raise ValueError(f"duplicate trace: {trace['id']}")
            trace_ids.add(trace["id"])
            block_sizes[trace["block_size"]] += 1
            for outer, entry in enumerate(trace["requests"]):
                requests = entry["requests"] if "requests" in entry else [entry]
                for inner, request in enumerate(requests):
                    request_count += 1
                    max_input = max(max_input, request["in"])
                    context = request["in"] + max(1, request["out"])
                    if context > max_context:
                        max_context = context
                        max_source = (
                            trace["id"],
                            outer,
                            inner if "requests" in entry else None,
                        )
    return {
        "path": str(path.resolve()),
        "sha256": digest.hexdigest(),
        "trace_count": len(trace_ids),
        "original_request_count": request_count,
        "native_block_sizes": dict(block_sizes),
        "max_input_tokens": max_input,
        "max_requested_context_tokens": max_context,
        "max_context_source": max_source,
    }


def validate(matrix: Path, dataset: Path, output: Path) -> dict:
    dataset_report = audit_dataset(dataset)
    hardware = HardwareContext.load("data")
    references = json.loads(Path("data/reference/agentx/summary.json").read_text())
    references = {e["key_point"]["experiment_id"]: e for e in references["experiments"]}
    report = {
        "schema_version": 1,
        "kind": "complete-model-and-dataset-audit",
        "simulation_performed": False,
        "dataset": dataset_report,
        "matrix": file_identity(matrix),
        "experiments": [],
    }
    output.mkdir(parents=True, exist_ok=False)
    for exp in load_experiments(matrix):
        psla = PSLAConfig.from_file(exp["model_config_path"])
        model = hardware.model(psla.model).with_moe_override(psla.moe_config)
        if (
            model.deepseek_v4 is None
            or dataset_report["trace_count"] != exp["trace_count"]
        ):
            raise ValueError("full model and full dataset pool required")
        if (
            dataset_report["max_requested_context_tokens"]
            > model.max_position_embeddings
        ):
            raise ValueError("original requested context exceeds full model limit")
        cluster = ClusterConfig.from_file(exp["cluster"])
        parallel = cluster.effective_parallel_config()
        if (
            parallel.world_size != exp["num_gpus"]
            or len(cluster.workers(parallel)) != exp["num_gpus"]
        ):
            raise ValueError("actual cluster size differs from experiment GPU count")
        target = output / exp["id"]
        target.mkdir()
        rank_reports = []
        for dp in range(parallel.data_parallel_size):
            rank = ParallelRankInfo(dp_rank=dp)
            weights = deepseek_v4_weight_memory(model, parallel, rank)
            logged_gib = references[exp["id"]]["server_log"]["weight_gib_per_rank"][
                str(dp)
            ]
            delta_pct = (weights["total_bytes"] / 2**30 / logged_gib - 1) * 100
            if abs(delta_pct) > 2:
                raise ValueError(
                    f"{exp['id']} rank {dp} weight difference exceeds 2%: {delta_pct}"
                )
            cache = CacheConfig(
                exp["block_size"],
                hardware.device(exp["hardware"]),
                model,
                parallel,
                rank,
                usable_memory_fraction=exp["gpu_memory_utilization"],
            )
            weights["reference_weight_gib"] = logged_gib
            weights["reference_delta_pct"] = delta_pct
            (target / f"rank-{dp}-weights.json").write_text(
                json.dumps(weights, indent=2) + "\n"
            )
            rank_reports.append(
                {
                    "dp_rank": dp,
                    "weight_bytes": weights["total_bytes"],
                    "weight_gib": weights["total_bytes"] / 2**30,
                    "local_experts": weights["local_experts"],
                    "reference_weight_gib": logged_gib,
                    "reference_delta_pct": delta_pct,
                    "legacy_kv_bytes_per_token": cache.size_per_token,
                    "legacy_kv_capacity_tokens": cache.num_gpu_blocks
                    * cache.block_size,
                }
            )
        blockers = [
            "native hybrid KV groups and shared block pool are not yet integrated",
            "AIPerf-equivalent warmup, handoff, grace and cancellation are not yet integrated",
            "native attention/MTP latency modules are not yet integrated",
        ]
        if dataset_report["native_block_sizes"] != {
            exp["block_size"]: dataset_report["trace_count"]
        }:
            blockers.append(
                "legacy replay requires trace and engine block sizes to match; 64-to-256 aggregation pending"
            )
        if (
            min(r["legacy_kv_capacity_tokens"] for r in rank_reports)
            < dataset_report["max_requested_context_tokens"]
        ):
            blockers.append(
                "legacy uniform KV accounting cannot admit the largest original context"
            )
        result = {
            "experiment_id": exp["id"],
            "model": model.to_dict(),
            "rank_weights": rank_reports,
            "cluster": file_identity(Path(exp["cluster"])),
            "psla": file_identity(Path(exp["model_config_path"])),
            "weights_validated": True,
            "formal_simulation_ready": False,
            "pending_runtime_work": blockers,
        }
        report["experiments"].append(result)
        print(
            f"{exp['id']}: {rank_reports[0]['weight_gib']:.6f} GiB/rank, "
            f"delta={rank_reports[0]['reference_delta_pct']:.3f}%; runtime pending",
            flush=True,
        )
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("dataset/agentx/traces.jsonl")
    )
    parser.add_argument(
        "--matrix", type=Path, default=Path("data/experiments/agentx.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.matrix, args.dataset, args.output)


if __name__ == "__main__":
    main()
