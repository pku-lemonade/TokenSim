#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else "./results/agentx")
    rows = []
    for result_path in sorted(root.rglob("agentx_c*.json")):
        result = json.loads(result_path.read_text())
        metrics = result.get("agentx_metrics") or {}
        cache_query_tokens = result.get("mooncake_cache_query_tokens", 0)
        hbm_hit_rate = (
            result.get("mooncake_local_gpu_hit_tokens", 0) / cache_query_tokens
            if cache_query_tokens
            else result.get("prefix_cache_hit_rate", 0)
        )
        dram_hit_rate = (
            result.get("mooncake_memory_hit_tokens", 0) / cache_query_tokens
            if cache_query_tokens
            else 0
        )
        relative = result_path.relative_to(root)
        rows.append(
            {
                "topology": relative.parts[0] if len(relative.parts) > 2 else "",
                "cache_mode": relative.parts[1] if len(relative.parts) > 2 else "",
                "concurrency": metrics.get("concurrency", 0),
                "requests": metrics.get("request_count", 0),
                "output_tokens_per_s": metrics.get("output_token_throughput_tps", 0),
                "output_tokens_per_s_per_gpu": metrics.get(
                    "output_token_throughput_per_gpu_tps", 0
                ),
                "ttft_p50_s": (metrics.get("ttft_s") or {}).get("p50", 0),
                "ttft_p99_s": (metrics.get("ttft_s") or {}).get("p99", 0),
                "tpot_p50_s": (metrics.get("tpot_s") or {}).get("p50", 0),
                "e2e_p50_s": (metrics.get("e2e_s") or {}).get("p50", 0),
                "interactivity_p50_tps": (metrics.get("interactivity_tps") or {}).get(
                    "p50", 0
                ),
                "prefix_cache_hit_rate": result.get("prefix_cache_hit_rate", 0),
                "cache_query_tokens": cache_query_tokens,
                "hbm_hit_rate": hbm_hit_rate,
                "dram_hit_rate": dram_hit_rate,
                "kv_cache_capacity_tokens_per_dp_rank": result.get(
                    "kv_cache_capacity_tokens_per_dp_rank", 0
                ),
                "kv_cache_capacity_tokens_total": result.get(
                    "kv_cache_capacity_tokens_total", 0
                ),
                "connector_transfers": result.get("connector_transfer_count", 0),
                "connector_transfer_bytes": result.get("connector_transfer_bytes", 0),
                "connector_save_wait_s": result.get("connector_save_wait_time", 0),
                "mooncake_puts": result.get("mooncake_put_count", 0),
                "mooncake_store_hits": result.get("mooncake_store_hit_count", 0),
                "mooncake_memory_tier_hits": result.get(
                    "mooncake_memory_tier_hit_count", 0
                ),
                "dram_hit_tokens": result.get("mooncake_memory_hit_tokens", 0),
                "recomputed_tokens": result.get("recomputed_tokens", 0),
            }
        )
    output_path = root / "summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit(f"no AgentX result files found below {root}")
    with output_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output_path)


if __name__ == "__main__":
    main()
