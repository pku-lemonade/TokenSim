#!/usr/bin/env python3
"""Validate first-source statistics and compare unchanged execution against full logs."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_agentx_cpu import validate


EXECUTION_FIELDS = (
    "request_id",
    "phase",
    "outcome",
    "start_s",
    "end_s",
    "input_tokens",
    "requested_output_tokens",
    "output_tokens",
    "trace_id",
    "conversation_id",
    "outer_index",
    "inner_index",
    "turn_index",
    "lane",
    "dp_rank",
    "worker_id",
    "pass_index",
    "cancellation_acknowledged",
    "ttft_s",
    "latency_s",
)


def audit(directory, baseline, *, hardware=("b200", "b300")):
    result = validate(directory, hardware=hardware, verify_reservations=True)
    for sku, point in result["checkpoints"].items():
        current = directory / sku
        previous = baseline / sku
        now_manifest = json.loads((current / "run_manifest.json").read_text())
        old_manifest = json.loads((previous / "run_manifest.json").read_text())
        for field in (
            "point",
            "defaults",
            "backend",
            "dataset_sha256",
            "transfer_utilization_assumption",
        ):
            assert now_manifest[field] == old_manifest[field], field
        old_rows = [
            json.loads(line)
            for line in (previous / "native_replay/requests.jsonl")
            .read_text()
            .splitlines()
        ]
        new_rows = [
            json.loads(line)
            for line in (current / "native_replay/requests.jsonl")
            .read_text()
            .splitlines()
        ]
        old_by_id = {r["request_id"]: r for r in old_rows}
        new_by_id = {r["request_id"]: r for r in new_rows}
        assert len(old_rows) == len(old_by_id) == len(new_rows) == len(new_by_id)
        assert old_by_id.keys() == new_by_id.keys()
        for identifier, old in old_by_id.items():
            new = new_by_id[identifier]
            assert all(
                old[field] == new[field] for field in EXECUTION_FIELDS
            ), identifier
            # The old metric was the last admission. It remains available as
            # a diagnostic and must reproduce the prior observation exactly.
            for field, new_field in (
                ("gpu_hit_tokens", "latest_gpu_hit_tokens"),
                ("cpu_hit_tokens", "latest_cpu_hit_tokens"),
                ("computed_input_tokens", "latest_computed_input_tokens"),
            ):
                assert old[field] == new[new_field], (identifier, field)
        successes = [
            r
            for r in new_rows
            if r["phase"] == "profiling" and r["outcome"] == "successful"
        ]
        changed = [
            r
            for r in successes
            if (r["gpu_hit_tokens"], r["cpu_hit_tokens"])
            != (r["latest_gpu_hit_tokens"], r["latest_cpu_hit_tokens"])
        ]
        point["execution_comparison"] = {
            "baseline": str(previous),
            "records_compared": len(new_rows),
            "request_identity_output_outcome_routing_timing_equal": True,
            "changed_successful_source_records": len(changed),
            "readmitted_successful_requests": sum(
                r["prefill_admission_count"] > 1 for r in successes
            ),
            "gpu_first_tokens": sum(r["gpu_hit_tokens"] for r in successes),
            "gpu_latest_tokens": sum(r["latest_gpu_hit_tokens"] for r in successes),
            "cpu_first_tokens": sum(r["cpu_hit_tokens"] for r in successes),
            "cpu_latest_tokens": sum(r["latest_cpu_hit_tokens"] for r in successes),
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT
        / "results/agentx-dsv4-full/simple-cpu/20261001-inflight-reservation",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--hardware", choices=("b200", "b300"), nargs="+", default=("b200", "b300")
    )
    args = parser.parse_args()
    result = audit(args.directory, args.baseline, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                sku: {
                    "first_gpu_percent": p["gpu_hit_percent"],
                    "first_cpu_percent": p["cpu_hit_percent"],
                    "execution": p["execution_comparison"],
                }
                for sku, p in result["checkpoints"].items()
            },
            indent=2,
        )
    )
