#!/usr/bin/env python3
"""Strict original full windows for explicit K3 and fully disabled K0 paths."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from TokenSim.config.cache_config import CacheConfig
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.hardware.context import HardwareContext
from validate_agentx_capture_rows import audit as audit_capture
from validate_agentx_cpu import validate as validate_lifecycle


BASELINE = Path("results/agentx-dsv4-full/native-memory/20261008-storage-live-phases")
EVIDENCE = Path("data/reference/agentx-native-memory/pipeline-validation.json")
SOURCE_CHANGES = {"TokenSim/config/cache_config.py", "TokenSim/config/memory_budget.py",
                  "TokenSim/config/speculative_execution.py", "TokenSim/llm/llm_engine.py",
                  "TokenSim/llm/llm_scheduler.py", "benchmark.py"}


def disabled_audit(root, hardware):
    context = HardwareContext.load("data")
    model = context.model("DeepSeek-V4-Pro")
    parallel = ParallelConfig(data_parallel_size=8, data_parallel_size_local=8, enable_expert_parallel=True)
    result = {}
    for sku, concurrency, tokens, seqs, utilization in (("b200", 196, 8192, 49, .9), ("b300", 384, 16384, 96, .92)):
        if sku not in hardware:
            continue
        directory = Path(root) / "disabled/seed0" / f"{sku}-c{concurrency}-dep8-simplecpu-nomtp"
        point = directory / sku
        base = validate_lifecycle(directory, hardware=(sku,), verify_reservations=True)["checkpoints"][sku]
        current = json.loads((point / f"agentx_c{concurrency}.json").read_text())
        manifest = json.loads((point / "run_manifest.json").read_text())
        command = manifest["command"]
        assert command[command.index("--mtp_num_speculative_tokens") + 1] == "0"
        assert command[command.index("--mtp_acceptance_length") + 1] == "1.0"
        before = json.loads((BASELINE / "seed0" / f"{sku}-c{concurrency}-dep8-simplecpu" / sku / "run_manifest.json").read_text())
        changed = {name for name in before["source_sha256"].keys() | manifest["source_sha256"].keys()
                   if before["source_sha256"].get(name) != manifest["source_sha256"].get(name)}
        assert changed == SOURCE_CHANGES
        for name in ("defaults", "backend", "simulation_seed", "dataset_sha256", "transfer_utilization_assumption"):
            assert before[name] == manifest[name], name
        first, second = dict(before["point"]), dict(manifest["point"])
        for name in ("cache_backend", "parallelism", "num_gpus"):
            first[name] = first.get(name, before["defaults"][name])
            second[name] = second.get(name, manifest["defaults"][name])
        # References/provenance describe the K3 ground truth; a K0 ablation
        # intentionally has none. Actual host capacities must still match.
        assert not second.get("reference")
        first.pop("reference", None)
        second.pop("reference", None)
        for point_config in (first, second):
            point_config["host_memory"] = dict(point_config["host_memory"])
            for name in ("hardware_source", "runtime_source"):
                point_config["host_memory"].pop(name, None)
        for name in ("id", "groups", "speculative"):
            first.pop(name, None)
            second.pop(name, None)
        assert first == second
        assert manifest["point"]["speculative"] == {"method": "mtp", "num_speculative_tokens": 0, "acceptance_length": 1.0}
        def stable_command(argv):
            result, index = [], 0
            intentional = {"--mtp_num_speculative_tokens", "--mtp_acceptance_length", "--results_path", "--kv_transfer_config"}
            while index < len(argv):
                if argv[index] in intentional:
                    index += 2
                else:
                    result.append(argv[index])
                    index += 1
            return result
        assert stable_command(before["command"]) == stable_command(command)
        before_transfer = json.loads(Path(before["command"][before["command"].index("--kv_transfer_config") + 1]).read_text())
        after_transfer = json.loads(Path(command[command.index("--kv_transfer_config") + 1]).read_text())
        assert before_transfer == after_transfer
        steps = current["operator_step_count"]
        assert steps > 0 and not current.get("native_dsv4_mtp_counts")
        assert not current.get("native_dsv4_mtp_component_seconds")
        assert not current.get("native_dsv4_mtp_seconds")
        counts = current["native_megamoe_counts"]
        assert counts["rank_estimates"] == steps
        assert counts["graph_pass"] + counts["eager_pass"] == steps
        queries = sum(worker["counts"]["scheduled_context_query_tokens"]
                      + worker["counts"]["scheduled_decode_query_tokens"] for worker in current["scheduler_stats"])
        assert counts["routed_valid_rows"] == queries
        assert counts["routed_valid_layer_rows"] == 61 * queries
        logits, main = current["dsv4_target_logits"], current["dsv4_main_forward_counts"]
        assert main["steps"] == steps and main["logical_rows"] == logits["forward_rows"]
        assert main["full_rows"] == logits["output_forward_rows"]
        assert main["full_rows"] == main["logical_rows"] + main["capture_padding_rows"]
        assert counts["full_input_rows"] == main["full_rows"]
        assert counts["full_input_layer_rows"] == 61 * main["full_rows"]
        assert main["norm_layer_rows"] == 122 * main["full_rows"]
        assert main["router_layer_rows"] == main["mhc_layer_rows"] == 61 * main["full_rows"]
        assert current["dsv4_hc_head_counts"].get("draft_rows", 0) == 0
        assert current["dsv4_hc_head_counts"]["target_rows"] == logits["final_norm_rows"] == main["full_rows"]
        assert current.get("native_mhc_draft_seconds", 0) == 0
        components = current["operator_component_seconds"]
        assert abs(current["native_dsv4_attention_seconds"] - components["attention"]) < 1e-6
        assert abs(current["native_megamoe_routed_seconds"] + current["native_megamoe_shared_seconds"] - components["moe"]) < 1e-6
        local = sum(current["dp_step_stats"]["local_compute_seconds_per_rank"].values())
        assert abs(sum(components.values()) - local) <= 1e-6 * max(1, local)
        budget_map = {row["worker_id"]: row for row in current["worker_cache_budgets"]}
        assert len(budget_map) == 8
        for worker_id, row in budget_map.items():
            expected = CacheConfig(256, context.device(sku), model, parallel,
                rank_info=ParallelRankInfo.from_global_rank(worker_id, parallel),
                usable_memory_fraction=utilization, max_num_batched_tokens=tokens, max_num_seqs=seqs,
                speculative_tokens=0, moe_execution={"backend": "deep_gemm_amxf4_mega_moe"}, attention_backend="native_dsv4")
            cache = row["cache"]
            assert cache == expected.to_dict()
            assert row["pool_slots"] == cache["num_gpu_blocks"]
            assert cache["kv_layout"]["speculative_tokens"] == 0
            assert not any(layer["draft"] for group in cache["kv_layout"]["groups"] for layer in group["layers"])
            assert not any(group["eagle"] for group in cache["kv_layout"]["groups"])
            assert not any(tensor["name"].startswith("mtp.") for tensor in cache["weight_memory"]["tensors"])
            persistent = cache["memory_budget"]["persistent_components"]
            assert all(persistent[name] == 0 for name in ("target_mtp_hidden", "drafter_hidden", "draft_topk", "swa_rope_cache"))
            assert cache["memory_budget"]["graph_storage"]["draft_prefill_descriptors"] == 0
            assert cache["memory_budget"]["graph_storage"]["draft_decode_descriptors"] == 0
        windows = [json.loads(line) for line in (point / "native_replay/cache_windows.jsonl").read_text().splitlines()]
        for window in windows:
            for worker in window["workers"]:
                assert worker["used_slots"] + worker["free_slots"] == budget_map[worker["worker_id"]]["pool_slots"]
                assert 1 <= worker["used_slots"] <= budget_map[worker["worker_id"]]["pool_slots"]
        result[sku] = {**base, "execution_k": 0, "complete_disabled_path_audited": True,
                       "all_window_pool_conservation_audited": True, "audited_window_count": len(windows),
                       "no_independent_disabled_reference_identity": True,
                       "derived_budget_per_rank": current["worker_cache_budgets"][0]["cache"]["memory_budget"]}
    return result


def audit(root, *, modes=("main", "disabled"), hardware=("b200", "b300")):
    result = {"scope": "explicit MTP execution; K3 reference preserved and K0 completely disabled",
              "formal_model_accuracy_accepted": False, "checkpoints": {}}
    if "main" in modes:
        report = audit_capture(Path(root) / "main", hardware=hardware, baseline_root=BASELINE,
                               baseline_evidence=EVIDENCE, source_changes=SOURCE_CHANGES)
        for sku, point in report["checkpoints"].items():
            concurrency = 196 if sku == "b200" else 384
            directory = Path(root) / "main/seed0" / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
            command = json.loads((directory / "run_manifest.json").read_text())["command"]
            assert command[command.index("--mtp_num_speculative_tokens") + 1] == "3"
            point["execution_k"] = 3
            result["checkpoints"][f"{sku}-main"] = point
    if "disabled" in modes:
        for sku, point in disabled_audit(root, hardware).items():
            result["checkpoints"][f"{sku}-disabled"] = point
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--mode", nargs="+", choices=("main", "disabled"), default=("main", "disabled"))
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.batch_root, modes=args.mode, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(args.output)
