#!/usr/bin/env python3
"""Strict original full-window checks for expert-phase efficiency transfer."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from validate_agentx_capture_rows import audit as audit_capture


BASELINE = Path("results/agentx-dsv4-full/native-mtp-execution/20261008-effective-K/main")
EVIDENCE = Path("data/reference/agentx-moe-phase-efficiency/baseline-validation.json")
SOURCE_CHANGES = {"TokenSim/latency/deepseek_v4_moe.py",
                  "TokenSim/latency/deepseek_v4_moe_efficiency.py"}


def stable_command(command):
    result, index = [], 0
    while index < len(command):
        if command[index] in {"--results_path", "--kv_transfer_config"}:
            index += 2
        else:
            result.append(command[index])
            index += 1
    return result


def audit(root, *, hardware=("b200", "b300")):
    report = audit_capture(root, hardware=hardware, baseline_root=BASELINE,
                           baseline_evidence=EVIDENCE, source_changes=SOURCE_CHANGES)
    for sku, point in report["checkpoints"].items():
        concurrency = 196 if sku == "b200" else 384
        relative = Path("seed0") / f"{sku}-c{concurrency}-dep8-simplecpu" / sku
        old, new = BASELINE / relative, Path(root) / relative
        before = json.loads((old / "run_manifest.json").read_text())
        after = json.loads((new / "run_manifest.json").read_text())
        assert stable_command(before["command"]) == stable_command(after["command"])
        for flag, expected in (("--mtp_num_speculative_tokens", "3"),
                               ("--mtp_acceptance_length", "2.49")):
            assert after["command"][after["command"].index(flag) + 1] == expected
        def connector(manifest):
            argv = manifest["command"]
            return json.loads(Path(argv[argv.index("--kv_transfer_config") + 1]).read_text())
        assert connector(before) == connector(after)

        current = json.loads((new / f"agentx_c{concurrency}.json").read_text())
        counts, ledger = current["native_megamoe_counts"], current["native_megamoe_composition"]
        common, expert = (ledger[name] for name in
                          ("transferred_common_seconds", "transferred_expert_seconds"))
        routed = current["native_megamoe_routed_seconds"]
        assert common > 0 and expert > 0
        assert abs(common + expert - routed) <= 1e-8 * max(1, routed)
        categories = {name: count for name, count in counts.items() if name.startswith("efficiency:")}
        assert sum(categories.values()) == counts["rank_estimates"]
        allowed = {"exact_efficiency", "interpolated_efficiency", "boundary_efficiency",
                   "zero_routed_expert_work", "unchanged_no_same_capacity_curve"}
        assert all(name.split(":", 1)[1] in allowed for name in categories)
        description = point["moe_description"]["expert_phase_efficiency"]
        assert description["grade"].startswith("D:")
        assert "component efficiency not measured" in description["grade"]
        assert not any(description[name] for name in
                       ("serving_fit", "cache_credit", "mean_rank_module_query"))
        expected = ([{"phase": "context", "effective_capacity": 32832, "points": 6},
                     {"phase": "generation", "effective_capacity": 576, "points": 10}]
                    if sku == "b200" else
                    [{"phase": "generation", "effective_capacity": 192, "points": 8}])
        assert description["capacity_curves"] == expected
        for backend in current["latency_backends"].values():
            assert backend["native_megamoe"]["expert_phase_efficiency"] == description

        budgets = {row["worker_id"]: row for row in current["worker_cache_budgets"]}
        windows = [json.loads(line) for line in
                   (new / "native_replay/cache_windows.jsonl").read_text().splitlines()]
        for window in windows:
            assert {row["worker_id"] for row in window["workers"]} == budgets.keys()
            for worker in window["workers"]:
                budget = budgets[worker["worker_id"]]
                assert worker["used_slots"] + worker["free_slots"] == budget["pool_slots"]
                assert 1 <= worker["used_slots"] <= budget["peak_used_slots"] <= budget["pool_slots"]
                assert 0 <= worker["dma_pinned_slots"] <= worker["used_slots"]
                assert worker["inflight_prefill_reserved_slots"] >= 0
                storage = worker["external_cache"]["storage"]
                assert storage["used_bytes"] <= storage["capacity_bytes"]
        point["expert_phase_transfer"] = {
            "common_seconds": common, "expert_seconds": expert,
            "counts": categories, "description": description,
            "component_and_rank_count_conservation": True,
            "all_window_pool_conservation_audited": True,
            "audited_window_count": len(windows),
            "real_nonuniform_timing_measured": False,
        }
    report["scope"] = "same-capacity measured shape efficiency transferred only to expert/fused-dispatch work with original full-input/common/shared scope preserved"
    report["formal_model_accuracy_accepted"] = False
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_root", type=Path)
    parser.add_argument("--hardware", nargs="+", choices=("b200", "b300"), default=("b200", "b300"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.batch_root, hardware=args.hardware)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(args.output)
