"""Offline integer/critical-rank candidate; never imported by serving runtime."""

import ast
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path

import torch

from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkCatalog
from TokenSim.latency.deepseek_v4_moe_work import MegaMoEWorkModel, block_m_for_input


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
HELPER = Path("/tmp/aiconfigurator/collector/helper.py")
names = {"sample_power_law", "_round_robin_adjust_per_rank", "_generate_power_law_distribution"}
nodes = [node for node in ast.parse(HELPER.read_text()).body
         if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(nodes) == len(names)
namespace = {"os": os, "_assign_experts_from_counts": lambda *_args: None}
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(HELPER), "exec"), namespace)
torch.set_num_threads(1)
torch.set_default_device("cpu")


@lru_cache(maxsize=4096)
def histograms(global_rows):
    if not global_rows:
        return (((0,) * 48,) * 8,) * 3
    result = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        counts, _ = namespace["_generate_power_law_distribution"](global_rows, 384, 6, 8, 1.01)
        assert int(counts.sum()) == global_rows * 6 and int(counts.max()) <= global_rows
        result.append(tuple(tuple(row) for row in counts.reshape(8, 48).tolist()))
    return tuple(result)


class IntegerCriticalCandidate:
    def __init__(self, physical):
        self.physical = physical

    def estimate(self, inputs, valid):
        physical = self.physical
        baseline = physical.estimate(inputs, valid)
        blocks = [block_m_for_input(inputs[rank], 8, 384, 6) for rank in range(8)]
        totals = []
        for histogram in histograms(sum(valid.values())):
            # The source collector swaps max-load to rank0. Uniform anchors
            # are insensitive to that label; heterogeneous inputs are not.
            # Average critical costs over every cyclic placement, never rows.
            costs = {}
            for index, counts in enumerate(histogram):
                mma_rows = sum(16 * math.ceil(n / 16) for n in counts if n > 0)
                flops = mma_rows * 6 * physical.hidden * physical.inter
                activations = sum(counts) * (physical.hidden * (1 + 1 / 32 + 2)
                                             + physical.inter * (2 + 2 / 32))
                for block in set(blocks):
                    tiles = sum(math.ceil(n / block) for n in counts if n > 0)
                    weights = tiles * 3 * physical.hidden * physical.inter * (0.5 + 1 / 32)
                    costs[index, block] = max(flops / physical.compute,
                                              (weights + activations) / physical.bandwidth) * 1e6
            for rotation in range(8):
                expert = max(costs[(rank + rotation) % 8, block]
                             for rank, block in enumerate(blocks))
                totals.append(baseline.startup_us + baseline.input_us
                              + max(expert, baseline.dispatch_us) + baseline.combine_us)
        return sum(totals) / len(totals)


def main():
    context = HardwareContext.load(ROOT / "data")
    model = context.model("DeepSeek-V4-Pro")
    links = LinkCatalog.load(ROOT / "data/topologies/links.yaml")
    previous = json.loads((ROOT / "data/reference/agentx-moe-mma/work-growth-followup.json").read_text())
    results = {}
    for sku in ("b200", "b300"):
        device = context.device(sku)
        physical = MegaMoEWorkModel(model, device, 8,
                                   nvlink_bytes_per_s=links.get(device.scale_up_link).bandwidth_per_direction_bytes_per_s)
        candidate = IntegerCriticalCandidate(physical)
        source = previous["hardware"][sku]
        samples = {}
        for row in source["uniform_samples"]:
            full = {rank: row["num_tokens"] for rank in range(8)}
            samples[row["phase"], row["num_tokens"]] = candidate.estimate(full, full)
        pairs = []
        for pair in source["directed_same_effective_capacity_holdouts_within8x"]:
            phase = pair["phase"]
            anchor = next(row for row in source["uniform_samples"]
                          if row["phase"] == phase and row["num_tokens"] == pair["anchor_rows"])
            prediction = anchor["latency_us"] * samples[phase, pair["holdout_rows"]] / samples[phase, pair["anchor_rows"]]
            pairs.append({**pair, "candidate_prediction_us": prediction,
                          "candidate_error_percent": 100 * (prediction / pair["measured_holdout_us"] - 1)})
        unequal = []
        for inputs, valid in (([64] * 8, [64] + [16] * 7),
                              ([4096] + [64] * 7, [4096] + [64] * 7),
                              ([512] + [64] * 7, [512] + [64] * 7)):
            full, routed = dict(enumerate(inputs)), dict(enumerate(valid))
            uniform = {rank: max(inputs) for rank in range(8)}
            unequal.append({"inputs": inputs, "valid": valid,
                            "current_work_ratio": physical.estimate(full, routed).latency_us / physical.estimate(uniform, uniform).latency_us,
                            "candidate_work_ratio": candidate.estimate(full, routed) / candidate.estimate(uniform, uniform)})
        results[sku] = {"original_rows_sha256": source["original_rows_sha256"], "pairs": pairs,
                        "old_failures": source["directed_same_effective_capacity_failures_over25percent"],
                        "candidate_failures": sum(abs(pair["candidate_error_percent"]) > 25 for pair in pairs),
                        "candidate_max_abs_error_percent": max(abs(pair["candidate_error_percent"]) for pair in pairs),
                        "unequal_shape_diagnostics": unequal}
    report = {"runtime_change": False, "formal_accuracy_evidence": False, "fitted_parameters": False,
              "method": "original CPU helper integer histograms, fixed diagnostic seeds0/1/2; average of critical-rank costs over cyclic placements; input/dispatch/combine and measured anchors preserved",
              "helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(), "hardware": results,
              "limits": ["new samples do not recover historical timing histograms or actual serving routes",
                         "histogram mean and critical cost mean are distinct; no mean-row module lookup",
                         "no cache credit, startup fit or serving-reference coefficient added"]}
    destination = STAGE / "prototype-validation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination)
    for sku, result in results.items():
        print(sku, "failures", result["old_failures"], "->", result["candidate_failures"],
              "max error", result["candidate_max_abs_error_percent"])


if __name__ == "__main__":
    main()
