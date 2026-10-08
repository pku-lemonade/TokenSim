"""Public-version geometry fingerprints and an unadopted fixed192 scenario."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from TokenSim.hardware.context import HardwareContext
from TokenSim.hardware.links import LinkCatalog
from TokenSim.latency.deepseek_v4_moe_work import MegaMoEWorkModel


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
HISTORY = Path("/tmp/tokensim-deepgemm-layout-history")
APRIL = "7f2a703ed51ac1f7af07f5e1453b2d3267d37d50"
JULY = "559d79fb6994a58b8a15b4b93bf13ccc16edf247"
PINNED = "e1f418c2a4f20818221f6b0e578b4c2f634d4c3f"
assert "return 192;" in (HISTORY / "april-heuristics.hpp").read_text()
assert "align(num_max_tokens_per_rank, block_m)" in (HISTORY / "april-wrapper.py").read_text()
assert "return layout::kLCMCandidateBlockM" in (HISTORY / "july-api.hpp").read_text()
assert "kLCMCandidateBlockM = 384" in (HISTORY / f"{JULY}-layout.cuh").read_text()
assert "update_instr_desc_with_umma_n" in (HISTORY / "april-kernel.cuh").read_text()

context = HardwareContext.load(ROOT / "data")
model = context.model("DeepSeek-V4-Pro")
links = LinkCatalog.load(ROOT / "data/topologies/links.yaml")
current = json.loads((STAGE / "work-growth-followup.json").read_text())
results = {}
for sku in ("b200", "b300"):
    source = current["hardware"][sku]
    fingerprints = []
    for row in source["uniform_samples"]:
        requested, effective = row["num_max_tokens_per_rank"], row["effective_num_max_tokens_per_rank"]
        fingerprints.append({"phase": row["phase"], "rows": row["num_tokens"],
                             "requested_capacity": requested, "recorded_effective_capacity": effective,
                             "fixed192_capacity": ((requested + 191) // 192) * 192,
                             "july384_capacity": ((requested + 383) // 384) * 384,
                             "pinned1920_capacity": ((requested + 1919) // 1920) * 1920})
    device = context.device(sku)
    physical = MegaMoEWorkModel(model, device, 8,
                               nvlink_bytes_per_s=links.get(device.scale_up_link).bandwidth_per_direction_bytes_per_s)
    work = {}
    with patch("TokenSim.latency.deepseek_v4_moe_work.block_m_for_input", return_value=192):
        for row in source["uniform_samples"]:
            inputs = {rank: row["num_tokens"] for rank in range(8)}
            work[row["phase"], row["num_tokens"]] = physical.estimate(inputs, inputs).latency_us
    pairs = []
    for pair in source["directed_same_effective_capacity_holdouts_within8x"]:
        phase = pair["phase"]
        anchor = next(row for row in source["uniform_samples"]
                      if row["phase"] == phase and row["num_tokens"] == pair["anchor_rows"])
        prediction = anchor["latency_us"] * work[phase, pair["holdout_rows"]] / work[phase, pair["anchor_rows"]]
        pairs.append({**pair, "fixed192_prediction_us": prediction,
                      "fixed192_error_percent": 100 * (prediction / pair["measured_holdout_us"] - 1)})
    results[sku] = {
        "capacity_fingerprints": fingerprints,
        "matches_by_alignment": {label: sum(row[key] == row["recorded_effective_capacity"] for row in fingerprints)
                                  for label, key in (("fixed192", "fixed192_capacity"),
                                                     ("july384", "july384_capacity"),
                                                     ("pinned1920", "pinned1920_capacity"))},
        "current_failures": source["directed_same_effective_capacity_failures_over25percent"],
        "fixed192_scenario_failures": sum(abs(pair["fixed192_error_percent"]) > 25 for pair in pairs),
        "fixed192_scenario_max_abs_error_percent": max(abs(pair["fixed192_error_percent"]) for pair in pairs),
        "pairs": pairs,
    }

files = (("april-api.hpp", APRIL, "csrc/apis/mega.hpp"),
         ("april-wrapper.py", APRIL, "deep_gemm/mega/__init__.py"),
         ("april-heuristics.hpp", APRIL, "csrc/jit_kernels/heuristics/mega_moe.hpp"),
         ("april-kernel.cuh", APRIL, "deep_gemm/include/deep_gemm/impls/sm100_fp8_fp4_mega_moe.cuh"),
         ("july-api.hpp", JULY, "csrc/apis/mega.hpp"),
         ("july-heuristics.hpp", JULY, "csrc/jit_kernels/heuristics/mega_moe.hpp"),
         (f"{JULY}-layout.cuh", JULY, "deep_gemm/include/deep_gemm/layout/mega_moe.cuh"))
report = {
    "scope": "public historical geometry fingerprints and a fixed192 sensitivity scenario",
    "runtime_change": False, "selected_source_kernel_revision": None,
    "formal_accuracy_evidence": False, "fitted_parameters": False,
    "source_checks": [{"path": str(HISTORY / name), "sha256": hashlib.sha256((HISTORY / name).read_bytes()).hexdigest(),
                       "commit": commit, "url": f"https://raw.githubusercontent.com/vllm-project/DeepGEMM/{commit}/{path}"}
                      for name, commit, path in files],
    "pinned_target_commit": PINNED,
    "hardware": results,
    "limits": [
        "matching capacity fingerprints is not proof of the unrecorded original kernel commit; forks/collector overrides may match too",
        "the fixed192 scenario changes only offline work partitioning and is not a reconstructed complete April kernel",
        "historical ring/shared/schedule/timing differences remain outside this sensitivity; uniform source anchors and raw data untouched",
        "do not adopt the historical scenario or tune a version to serving/reference error without provenance evidence",
    ],
}
destination = STAGE / "kernel-history-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for sku, result in results.items():
    print(sku, result["matches_by_alignment"], "failures", result["current_failures"],
          "-> fixed192 scenario", result["fixed192_scenario_failures"],
          "max", result["fixed192_scenario_max_abs_error_percent"])
