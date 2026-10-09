#!/usr/bin/env python3
"""Strict full-window audit with source-derived capacity changes, not overrides."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.persistent_buffers import MIB
from TokenSim.config.parallel_config import ParallelConfig
from TokenSim.hardware.context import HardwareContext
from validate_agentx_capture_rows import audit as audit_capture


BASELINE = Path("results/agentx-dsv4-full/native-capture-rows/20261008-main-capture-sp")
EVIDENCE = Path("data/reference/agentx-capture-rows/pipeline-validation.json")
SOURCE_CHANGES = {"TokenSim/config/cache_config.py", "TokenSim/config/memory_budget.py",
                  "TokenSim/config/deepseek_v4_buffers.py", "TokenSim/llm/llm_engine.py"}


def capacity_auditor(context):
    model = context.model("DeepSeek-V4-Pro")
    parallel = ParallelConfig(data_parallel_size=8, data_parallel_size_local=8, enable_expert_parallel=True)
    expected = {}
    for sku, tokens, seqs, utilization in (("b200", 8192, 49, .9), ("b300", 16384, 96, .92)):
        expected[sku] = CacheConfig(256, context.device(sku), model, parallel,
            usable_memory_fraction=utilization, max_num_batched_tokens=tokens, max_num_seqs=seqs,
            moe_execution={"backend": "deep_gemm_amxf4_mega_moe"}, attention_backend="module")

    def compare(before, after, sku):
        dynamic = {"cache", "pool_slots", "peak_used_slots", "evictions"}
        assert {k: v for k, v in before.items() if k not in dynamic} == {
            k: v for k, v in after.items() if k not in dynamic}
        first, second = before["cache"], after["cache"]
        changed = {"num_gpu_blocks", "logged_equivalent_tokens_computed", "memory_budget"}
        assert {k: v for k, v in first.items() if k not in changed} == {
            k: v for k, v in second.items() if k not in changed}
        cache = expected[sku]
        assert second["memory_budget"] == cache.memory_budget
        assert second["num_gpu_blocks"] == after["pool_slots"] == cache.num_gpu_blocks
        assert second["logged_equivalent_tokens_computed"] == cache.to_dict()["logged_equivalent_tokens_computed"]
        assert 1 <= after["peak_used_slots"] <= after["pool_slots"]
        budget = second["memory_budget"]
        for key in ("hardware_capacity_bytes", "hardware_budget_bytes", "utilization"):
            assert budget[key] == first["memory_budget"][key]
        for key in ("weights", "runtime_and_nccl_reserve"):
            assert budget["deductions"][key] == first["memory_budget"]["deductions"][key]
        assert budget["deductions"]["runtime_and_nccl_reserve"] == 4169138176
        assert budget["available_kv_bytes"] == budget["hardware_budget_bytes"] - sum(budget["deductions"].values())
        assert budget["deductions"]["native_persistent_buffers"] == sum(budget["persistent_components"].values())
        assert budget["deductions"]["peak_activation_estimate"] == sum(budget["activation_components"].values())
        assert all(value >= 0 for value in budget["deductions"].values())
        remainder = budget["available_kv_bytes"] - after["pool_slots"] * second["physical_slot_bytes"]
        assert 0 <= remainder < second["physical_slot_bytes"]
        mega = budget["native_mega_storage"]
        assert mega["cache_instances_per_rank"] == 1 and not mega["sms_measured_for_reference"]
        assert mega["bytes"] == {"b200": 2246689664, "b300": 3950321024}[sku]
        assert mega["aliases"] == {"shared_l1_x": "input_x"}
        assert sum(mega["components"].values()) == mega["bytes"] == mega["intervals"][-1]["end_byte"]
        assert all(left["end_byte"] == right["start_byte"] for left, right in zip(mega["intervals"], mega["intervals"][1:]))
        for plan in budget["activation_live_plans"] + budget["graph_storage"]["plans"]:
            rows = plan["storages"]
            assert len({row["storage"] for row in rows}) == len(rows)
            for index, phase in enumerate(plan["phases"]):
                assert phase["bytes"] == sum(row["bytes"] for row in rows if row["begin_phase"] <= index < row["end_phase"])
            assert plan["peak_bytes"] == max(phase["bytes"] for phase in plan["phases"])
        graph = budget["graph_storage"]
        assert graph["shared_pool_live_peak_bytes"] == max(plan["peak_bytes"] for plan in graph["plans"])
        assert graph["driver_increment_proxy_bytes"] == (graph["total_descriptors"] - 1) * MIB
        assert graph["bytes"] == graph["shared_pool_live_peak_bytes"] + graph["persistent_target_output_bytes"] + graph["driver_increment_proxy_bytes"]
        assert graph["bytes"] == budget["deductions"]["decode_graph_estimate"]
        assert "unmeasured" in graph["driver_grade"]
    return compare


def audit(root, *, hardware=("b200", "b300")):
    result = audit_capture(root, hardware=hardware, baseline_root=BASELINE,
        baseline_evidence=EVIDENCE, source_changes=SOURCE_CHANGES,
        cache_budget_auditor=capacity_auditor(HardwareContext.load("data")))
    for sku, point in result["checkpoints"].items():
        concurrency = 196 if sku == "b200" else 384
        relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
        old = json.loads((BASELINE / relative / f"agentx_c{concurrency}.json").read_text())
        new = json.loads((Path(root) / relative / f"agentx_c{concurrency}.json").read_text())
        path = Path(root) / relative / "native_replay/cache_windows.jsonl"
        windows = [json.loads(line) for line in path.read_text().splitlines()]
        budgets = {row["worker_id"]: row for row in new["worker_cache_budgets"]}
        for window in windows:
            assert {row["worker_id"] for row in window["workers"]} == budgets.keys()
            for worker in window["workers"]:
                budget = budgets[worker["worker_id"]]
                slots = budget["pool_slots"]
                assert worker["used_slots"] + worker["free_slots"] == slots
                assert 1 <= worker["used_slots"] <= budget["peak_used_slots"] <= slots
                assert 0 <= worker["dma_pinned_slots"] <= worker["used_slots"]
                assert worker["inflight_prefill_reserved_slots"] >= 0
                assert worker["external_cache"]["storage"]["used_bytes"] <= worker["external_cache"]["storage"]["capacity_bytes"]
        point["native_memory"] = {
            "budget": new["worker_cache_budgets"][0]["cache"]["memory_budget"],
            "baseline_slots": old["worker_cache_budgets"][0]["pool_slots"],
            "current_slots": new["worker_cache_budgets"][0]["pool_slots"],
            "all_rank_capacity_changes_audited": True,
            "hardware_weights_reserve_layout_and_configuration_unchanged": True,
            "overrides_used": False, "allocator_and_driver_are_measured": False,
            "all_window_pool_conservation_audited": True,
            "audited_window_count": len(windows),
        }
    result["scope"] = "native persistent storage identities, source live-phase upper bounds and shared graph-pool geometry with strictly audited derived KV capacity"
    result["formal_model_accuracy_accepted"] = False
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.batch_root, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
