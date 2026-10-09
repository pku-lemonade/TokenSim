#!/usr/bin/env python3
"""Audit full native checkpoints after request-local MTP indexer KV reuse."""

import argparse
import hashlib
import json
from pathlib import Path

from validate_agentx_decode_batch import audit as audit_batches

BASELINE = Path("results/agentx-dsv4-full/native-attention-sol/20261007-physical-growth")
EVIDENCE = Path("data/reference/agentx-attention-sol/pipeline-validation.json")
SOURCE_CHANGES = {
    "TokenSim/latency/deepseek_v4_attention.py",
    "TokenSim/latency/deepseek_v4_attention_sol.py",
    "TokenSim/config/psla_config.py",
    "util/results.py",
}


def audit(root, *, hardware=("b200", "b300"), hardware_roots=None):
    roots = {sku: Path((hardware_roots or {}).get(sku, root)) for sku in hardware}
    result = None
    for sku in hardware:
        point_report = audit_batches(roots[sku], hardware=(sku,))
        if result is None:
            result = {**point_report, "checkpoints": {}}
        else:
            for field in ("seed", "trace_pool", "issue_window_seconds"):
                assert result[field] == point_report[field], field
        result["checkpoints"].update(point_report["checkpoints"])
    result["result_roots"] = {sku: str(path) for sku, path in roots.items()}
    previous = json.loads(EVIDENCE.read_text())["checkpoints"]
    for sku, concurrency in (("b200", 196), ("b300", 384)):
        if sku not in hardware:
            continue
        relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
        old, new = BASELINE / relative, roots[sku] / relative
        old_path = old / f"agentx_c{concurrency}.json"
        assert hashlib.sha256(old_path.read_bytes()).hexdigest() == previous[sku]["result_sha256"]
        before = json.loads((old / "run_manifest.json").read_text())
        after = json.loads((new / "run_manifest.json").read_text())
        for field in ("point", "defaults", "backend", "simulation_seed", "dataset_sha256", "transfer_utilization_assumption"):
            assert before[field] == after[field], field
        a, b = before["source_sha256"], after["source_sha256"]
        changes = {name for name in a.keys() | b.keys() if a.get(name) != b.get(name)}
        assert changes == SOURCE_CHANGES, changes
        current = json.loads((new / f"agentx_c{concurrency}.json").read_text())
        old_result = json.loads(old_path.read_text())
        assert len(old_result["worker_cache_budgets"]) == len(current["worker_cache_budgets"]) == 8
        for first, second in zip(old_result["worker_cache_budgets"], current["worker_cache_budgets"]):
            assert first.keys() == second.keys()
            assert {k: v for k, v in first.items() if k not in ("peak_used_slots", "evictions")} == {
                k: v for k, v in second.items() if k not in ("peak_used_slots", "evictions")
            }
        stats = current["attention_module_mtp_kv"]
        batches = current["attention_module_decode_batches"]
        assert 0 < stats["batches"] <= batches["batches"]
        assert 0 < batches["requests"] < batches["query_rows"]
        assert batches["requests"] <= batches["indexer_query_tiles"] <= batches["query_rows"]
        assert 0 < stats["source_geometry_indexer_bytes_after"] < stats["source_geometry_indexer_bytes_before"]
        assert stats["estimated_seconds_saved"] > 0
        assert stats["source_geometry_indexer_bytes_after"] / stats["source_geometry_indexer_bytes_before"] >= 0.5 - 1e-9
        provenance = result["checkpoints"][sku]["attention_provenance"]["mtp_indexer_kv_reuse"]
        assert not provenance["layout_correction"]
        assert provenance["indexer_queries_per_tile"] == 2
        assert provenance["query_math"].startswith("unchanged")
        assert provenance["sparse_attention_kv_reads"].startswith("per query retained")
        result["checkpoints"][sku]["mtp_kv_stats"] = stats
        result["checkpoints"][sku]["paired_baseline"] = {
            "root": str(old), "output_tps": previous[sku]["output_tps"],
            "throughput_change_percent": 100 * (result["checkpoints"][sku]["output_tps"] / previous[sku]["output_tps"] - 1),
            "source_changes": sorted(changes), "configuration_and_capacity_unchanged": True,
        }
    result["scope"] = "MTP per-Q-block indexer KV reuse with full query compute and sparse-attention reads; SOL source layout"
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--b300-root", type=Path, help="Separate complete B300 batch after an interrupted run.")
    args = parser.parse_args()
    report = audit(args.batch_root, hardware=args.hardware,
                   hardware_roots={"b300": args.b300_root} if args.b300_root else None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
