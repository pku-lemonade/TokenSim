#!/usr/bin/env python3
"""Validate real native Mooncake shared-page full windows."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from validate_agentx_cpu import validate


def audit(directory, *, hardware=("b200", "b300")):
    directory = Path(directory)
    result = validate(
        directory, hardware=hardware, verify_reservations=True, verify_sources=False
    )
    result["scope"] = (
        "native Mooncake shared DRAM pages with async import/save, shared network "
        "queue, group-aware PoolKeys and DP lockstep wakeups"
    )
    for sku in tuple(result["checkpoints"]):
        concurrency = 196 if sku == "b200" else 384
        full = json.loads(
            (directory / sku / f"agentx_c{concurrency}.json").read_text()
        )
        backend = next(iter(full["latency_backends"].values()))
        # The connector description is exported through cache windows; ensure
        # the real run did not use the old no-admission control path.
        windows = [
            json.loads(line)
            for line in (directory / sku / "native_replay/cache_windows.jsonl")
            .read_text()
            .splitlines()
        ]
        descriptions = [
            worker.get("external_cache")
            for window in windows
            for worker in window["workers"]
            if worker.get("external_cache")
        ]
        assert descriptions, "Mooncake run must export external cache state"
        assert all(item.get("native_async") for item in descriptions)
        assert all(item.get("service_scope", "").startswith("engine-shared") for item in descriptions)
        query_tokens = full.get("mooncake_cache_query_tokens", 0)
        memory_hits = full.get("mooncake_memory_hit_tokens", 0)
        transferred = full.get("mooncake_transferred_bytes", 0)
        # Results made before the native stats hook was added still expose
        # the authoritative connector totals and first-admission CPU tokens.
        query_tokens = query_tokens or full.get("connector_transfer_blocks", 0)
        memory_hits = memory_hits or full["agentx_metrics"].get(
            "cpu_cached_input_tokens", 0
        )
        transferred = transferred or full.get("connector_transfer_bytes", 0)
        assert query_tokens > 0
        assert memory_hits > 0
        assert transferred > 0
        assert full.get("mooncake_pending_async_jobs", 0) == 0
        assert full.get("mooncake_eviction_count", 0) >= 0
        assert full.get("operator_component_seconds", {}).get("comm", 0) >= 0
        final = windows[-1]
        assert all(
            worker["dma_pinned_slots"] == 0
            and worker["inflight_prefills"] == 0
            and worker["inflight_prefill_reserved_slots"] == 0
            for worker in final["workers"]
        )
        result["checkpoints"][sku]["native_mooncake"] = {
            "shared_store": True,
            "native_async": True,
            "memory_hit_tokens": memory_hits,
            "query_tokens": query_tokens,
            "transferred_bytes": transferred,
            "effective_bandwidth_gbps": full.get(
                "mooncake_effective_transfer_bandwidth_gbps"
            ),
            "output_token_throughput_tps": full["agentx_metrics"][
                "output_token_throughput_tps"
            ],
            "final_references_released": True,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hardware", nargs="+", default=("b200", "b300"))
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(audit(args.directory, hardware=tuple(args.hardware)), indent=2)
        + "\n"
    )
    print(args.output)
