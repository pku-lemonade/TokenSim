#!/usr/bin/env python3
"""Run one configured full-model AgentX point with its complete trace pool."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def command(point, defaults, result, connector, *, random_seed=0):
    return [
        sys.executable,
        str(ROOT / "benchmark.py"),
        "--model",
        defaults["model_config_path"],
        "--cluster",
        point["cluster"],
        "--dataset_path",
        "dataset/agentx/traces.jsonl",
        "--workload_type",
        "agentx_weka",
        "--batching",
        "paged-attn",
        "--block_size",
        str(defaults["block_size"]),
        "--agentx_concurrency",
        str(point["concurrency"]),
        "--agentx_trace_count",
        str(defaults["trace_count"]),
        "--agentx_profile_duration",
        str(defaults["profiling_window_s"]),
        "--max_parallem_sum",
        str(point["max_num_seqs"]),
        "--max_num_batched_tokens",
        str(point["max_num_batched_tokens"]),
        "--gpu_memory_utilization",
        str(point["gpu_memory_utilization"]),
        "--router_policy",
        defaults["router"],
        "--dp_lockstep",
        "--dp_sync_interval",
        str(defaults["dp_sync_interval"]),
        "--long_prefill_token_threshold",
        str(defaults["long_prefill_token_threshold"]),
        "--prefill_schedule_interval",
        str(defaults["prefill_schedule_interval"]),
        "--random_seed",
        str(random_seed),
        "--operator_backend",
        "vllm",
        "--moe_backend",
        point.get("moe_backend", defaults.get("moe_backend", "generic")),
        "--moe_latency_scale",
        str(point.get("moe_latency_scale", defaults.get("moe_latency_scale", 1.0))),
        "--attention_backend",
        point.get("attention_backend", defaults.get("attention_backend", "generic")),
        "--mhc_backend",
        point.get("mhc_backend", defaults.get("mhc_backend", "generic")),
        "--mtp_acceptance_length",
        str(
            point.get(
                "mtp_acceptance_length",
                (defaults.get("speculative") or {}).get("acceptance_length", 1.0),
            )
        ),
        *(
            []
            if point.get("moe_shared_fusion", defaults.get("moe_shared_fusion", True))
            else ["--disable_moe_shared_fusion"]
        ),
        "--latency_backend",
        "operator_table",
        "--latency_fallback",
        "table_first",
        "--verbose",
        "none",
        "--results_path",
        str(result),
        *(["--kv_transfer_config", str(connector)] if connector else []),
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument(
        "--backend",
        choices=("simple_cpu", "mooncake", "hbm"),
        default="simple_cpu",
    )
    parser.add_argument(
        "--transfer-utilization", type=float, choices=(0.5, 0.75, 1.0), default=0.75
    )
    parser.add_argument("--random-seed", type=int, default=0)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / "data/experiments/agentx.yaml").read_text())
    point = next(p for p in config["experiments"] if p["id"] == args.case)
    result = args.result_root.resolve() / point["hardware"]
    result.mkdir(parents=True, exist_ok=False)
    connector = None
    if args.backend == "simple_cpu":
        host = point["host_memory"]
        connector = result / "kv_transfer_config.json"
        connector.write_text(
            json.dumps(
                {
                    "kv_connector": "SimpleCPUOffloadConnector",
                    "kv_role": "kv_both",
                    "kv_connector_extra_config": {
                        "cpu_bytes_to_use_per_rank": host["cpu_budget_per_rank_bytes"],
                        "host_available_bytes": host["available_cpu_dram_mib"] * 2**20,
                        "lazy_offload": False,
                        "transfer_utilization": args.transfer_utilization,
                    },
                },
                indent=2,
            )
            + "\n"
        )
    elif args.backend == "mooncake":
        connector = result / "kv_transfer_config.json"
        mooncake = json.loads(
            (ROOT / "data/kv_transfer/mooncake_store_dram_agentx.json").read_text()
        )
        # Use the shared DRAM service and the native asynchronous page adapter.
        extra = mooncake.setdefault("kv_connector_extra_config", {})
        extra["load_async"] = True
        extra["transfer_overlap"] = True
        connector.write_text(json.dumps(mooncake, indent=2) + "\n")
    argv = command(
        point, config["defaults"], result, connector, random_seed=args.random_seed
    )
    manifest = {
        "point": point,
        "defaults": config["defaults"],
        "backend": args.backend,
        "backend_semantics": (
            "mooncake_no_admission_control"
            if args.backend == "mooncake"
            else args.backend
        ),
        "simulation_seed": args.random_seed,
        "transfer_utilization_assumption": args.transfer_utilization,
        "operator_scope": "native low-rank GEMMs, borrowed MegaMoE incl fused dispatch/combine and explicit shared FP8 cost, measured DSV4 CSA/HCA attention modules, MTP q=4 verify with 2/3 accepted-token progress; mHC selectable",
        "dataset_sha256": hashlib.sha256(
            (ROOT / "dataset/agentx/traces.jsonl").read_bytes()
        ).hexdigest(),
        "parent_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "command": argv,
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in ("TokenSim/**/*.py", "util/*.py", "benchmark.py")
            for path in ROOT.glob(pattern)
        },
    }
    (result / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    started = time.time()
    with (result / "run.log").open("w") as log:
        process = subprocess.run(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    (result / "run_status.json").write_text(
        json.dumps(
            {
                "exit_code": process.returncode,
                "started_unix_s": started,
                "elapsed_wall_s": time.time() - started,
            }
        )
        + "\n"
    )
    raise SystemExit(process.returncode)


if __name__ == "__main__":
    main()
