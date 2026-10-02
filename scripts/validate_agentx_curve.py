#!/usr/bin/env python3
"""Validate every completed DEP8 curve point in a seeded matrix root."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_agentx_cpu import validate


def audit(root):
    root = Path(root)
    rows = []
    for point_dir in sorted(root.glob("seed*/b*-c*")):
        statuses = list(point_dir.glob("b*/run_status.json"))
        if len(statuses) != 1:
            rows.append({"run": str(point_dir), "status": "incomplete"})
            continue
        status = json.loads(statuses[0].read_text())
        if status.get("exit_code") != 0:
            rows.append({"run": str(point_dir), "status": "failed", "exit_code": status.get("exit_code")})
            continue
        hardware = statuses[0].parent.name
        result_file = next(point_dir.glob(f"{hardware}/agentx_c*.json"))
        result = json.loads(result_file.read_text())
        records = [json.loads(line) for line in (point_dir / hardware / "native_replay/requests.jsonl").read_text().splitlines()]
        windows = [json.loads(line) for line in (point_dir / hardware / "native_replay/cache_windows.jsonl").read_text().splitlines()]
        metrics = result["agentx_metrics"]
        successful = [r for r in records if r["phase"] == "profiling" and r["outcome"] == "successful"]
        assert len(successful) == metrics["request_count"]
        assert metrics["trace_count"] == 393 and metrics["issue_window_s"] == 3600
        assert metrics["cancel_unreturned"] == 0
        final = windows[-1]
        assert all(
            worker["used_slots"] == 1
            and worker["dma_pinned_slots"] == 0
            and worker["loading"] == 0
            and worker["inflight_prefills"] == 0
            and worker["inflight_prefill_reserved_slots"] == 0
            for worker in final["workers"]
        )
        dp = final["dp_step"]
        if dp is not None:
            compute = sum(dp["local_compute_seconds_per_rank"].values())
            wait = sum(dp["synchronization_wait_seconds_per_rank"].values())
            assert abs(compute + wait - len(final["workers"]) * dp["elapsed_step_seconds"]) < 1e-6
        rows.append(
            {
                "run": str(point_dir),
                "hardware": hardware,
                "status": "ok",
                "output_tps": result["agentx_metrics"]["output_token_throughput_tps"],
                "p90_ttft_s": result["agentx_metrics"]["ttft_s"]["p90"],
                "p90_itl_s": result["agentx_metrics"]["tpot_s"]["p90"],
                "gpu_cache_hit_rate": result["agentx_metrics"]["gpu_cache_hit_rate"],
                "cpu_cache_hit_rate": result["agentx_metrics"]["cpu_cache_hit_rate"],
                "request_count": result["agentx_metrics"]["request_count"],
                "result_sha256": __import__("hashlib").sha256(result_file.read_bytes()).hexdigest(),
            }
        )
    if not rows or any(row["status"] != "ok" for row in rows):
        raise AssertionError(rows)
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(audit(args.root), indent=2) + "\n")
    print(args.output)
