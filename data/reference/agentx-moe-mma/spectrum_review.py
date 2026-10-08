"""Compare the expected local proxy with newly sampled collector histograms.

These are synthetic source-family diagnostics, not measured serving routes or
the unrecorded histograms behind the imported timing table.
"""

import ast
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
source = HELPER.read_text()
names = {"sample_power_law", "_round_robin_adjust_per_rank",
         "_assign_experts_from_counts", "_generate_power_law_distribution"}
functions = [node for node in ast.parse(source).body
             if isinstance(node, ast.FunctionDef) and node.name in names]
assert len(functions) == len(names)
namespace = {"os": os}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(HELPER), "exec"), namespace)
torch.set_num_threads(1)
torch.set_default_device("cpu")
context = HardwareContext.load(ROOT / "data")
device, model = context.device("b200"), context.model("DeepSeek-V4-Pro")
link = LinkCatalog.load(ROOT / "data/topologies/links.yaml").get(device.scale_up_link)
work_model = MegaMoEWorkModel(model, device, 8,
                             nvlink_bytes_per_s=link.bandwidth_per_direction_bytes_per_s)
cases = []
for global_rows in (8, 16, 32, 64, 128, 256, 512, 1024, 8192, 32768):
    local_rows = global_rows // 8
    block = block_m_for_input(local_rows, 8, 384, 6)
    proxy = work_model.estimate({rank: local_rows for rank in range(8)},
                                {rank: local_rows for rank in range(8)})
    samples = []
    for seed in (0, 1, 2):
        torch.manual_seed(seed)
        counts, selected = namespace["_generate_power_law_distribution"](
            global_rows, 384, 6, 8, 1.01)
        assert int(counts.sum()) == global_rows * 6
        assert int(counts.max()) <= global_rows
        assert selected.shape == (global_rows, 6)
        assert bool((torch.sort(selected, dim=1).values[:, 1:]
                     > torch.sort(selected, dim=1).values[:, :-1]).all())
        ranks = counts.reshape(8, 48).tolist()
        tile_counts = [sum(math.ceil(n / block) for n in row if n > 0) for row in ranks]
        mma_rows = [sum(16 * math.ceil(n / 16) for n in row if n > 0) for row in ranks]
        selections = [sum(row) for row in ranks]
        samples.append({"diagnostic_seed": seed, "rank_selections": selections,
                        "rank_m_tiles": tile_counts, "rank_mma_rows": mma_rows,
                        "max_m_tiles": max(tile_counts), "max_mma_rows": max(mma_rows),
                        "max_tiles_over_proxy": max(tile_counts) / proxy.max_expert_tiles,
                        "max_mma_over_proxy": max(mma_rows) / proxy.expert_mma_rows})
    cases.append({"global_valid_rows": global_rows, "local_uniform_rows": local_rows,
                  "block_m": block, "proxy_m_tiles": proxy.max_expert_tiles,
                  "proxy_mma_rows": proxy.expert_mma_rows, "samples": samples})
report = {
    "scope": "new synthetic source-family histogram and critical-rank occupancy diagnostics",
    "runtime_change": False, "fitted_parameters": False,
    "formal_accuracy_evidence": False, "matches_original_table_histograms": False,
    "helper_path": str(HELPER), "helper_sha256": hashlib.sha256(HELPER.read_bytes()).hexdigest(),
    "method": "execute only four original CPU helper functions with explicit new seeds0/1/2; compare integer per-rank tasks to current midpoint local proxy",
    "cases": cases,
    "limits": [
        "the imported timing rows do not record their original expert histograms; these samples are not recovered historical routes",
        "same expected local spectrum on every rank suppresses rank variance; mean local tiles do not determine the max-rank tile distribution",
        "ceil(expected count / BLOCK_M) is not expected ceil(integer count / BLOCK_M); fractional occupancy is a proxy, not actual tasks",
        "this audit does not measure component startup, HBM/L2 traffic, overlap or mixed issue cycles; do not fit serving throughput from it",
    ],
}
destination = STAGE / "spectrum-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for case in cases:
    print(case["local_uniform_rows"], "tiles ratios",
          [round(sample["max_tiles_over_proxy"], 3) for sample in case["samples"]])
