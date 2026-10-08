"""Reproduce pinned native allocation intervals offline, without budget edits."""

import hashlib
import json
from pathlib import Path
import subprocess


STAGE = Path(__file__).resolve().parent
ROOT = STAGE.parents[2]
SOURCE = Path("/tmp/tokensim-deepgemm-buffer-audit")
BLOCKS = (8, 16, 32, 64, 96, 128, 192, 240)
ALIGNMENT = 1920


def ceil_div(a, b):
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def allocation(tokens, sms, *, ep=8, experts=384, topk=6, hidden=7168, inter=3072, shared=1):
    assert 0 < ep <= 72 and experts % ep == 0 and sms >= 2
    assert hidden % 256 == inter % 128 == 0 and tokens > 0 and shared >= 0
    capacity = align(tokens, ALIGNMENT)
    local_experts, clusters = experts // ep, sms // 2
    l1, l2 = 2 * inter // 256, hidden // 256
    largest = 0
    candidates = []
    for block in BLOCKS:
        total = ceil_div(capacity * ep * min(topk, local_experts), block) + local_experts
        first_l2 = ceil_div(ceil_div(clusters, l2) * l1, clusters)
        interleave = ceil_div(l1 + (total - 1) * max(0, l1 - l2), clusters) + 1
        warmup = min(max(first_l2, interleave), ceil_div(total * l1, clusters))
        live = min(total, ceil_div(min(warmup * clusters, total * l1), l1)
                   + ceil_div(total * max(0, l2 - l1), l2)
                   + ceil_div(clusters, min(l1, l2)))
        largest = max(largest, live * block)
        candidates.append({"block_m": block, "pool_blocks": total,
                           "live_pool_blocks": live, "ring_tokens": live * block})
    ring = align(largest, ALIGNMENT)
    sf_ring = max(ring // block * align(block, 128) for block in BLOCKS)
    pool = align(capacity * ep * min(topk, local_experts) + local_experts * 239, ALIGNMENT)
    shared_sf = ceil_div(capacity, 8) * 128
    # sizeof is checked against the previously built host ABI probe below.
    components = {
        "workspace": align(21140864 + experts * capacity * 4 + pool * 12, 16),
        "input_x": capacity * hidden,
        "input_scales": capacity * hidden // 32,
        "topk_ids": capacity * topk * 8,
        "topk_weights": capacity * topk * 4,
        "shared_l1_x_alias": 0,
        "shared_l1_scales": shared_sf * hidden // 32 if shared else 0,
        "shared_l2_x": capacity * inter * shared,
        "shared_l2_scales": shared_sf * inter * shared // 32,
        "ring_l1_x": ring * hidden,
        "ring_l1_scales": sf_ring * hidden // 32,
        "ring_l1_weights": ring * 4,
        "ring_l2_x": ring * inter,
        "ring_l2_scales": sf_ring * inter // 32,
        "combine_output": capacity * (topk + bool(shared)) * hidden * 2,
    }
    offset, intervals = 0, []
    for name, size in components.items():
        if name == "shared_l1_x_alias":
            continue
        intervals.append({"storage": name, "start": offset, "end": offset + size,
                          "bytes": size})
        offset += size
    assert offset == sum(components.values())
    assert all(left["end"] == right["start"] for left, right in zip(intervals, intervals[1:]))
    return {"unsharded_scheduler_tokens": tokens, "aligned_tokens": capacity,
            "expert_parallel_size": ep, "SM_scenario": sms,
            "SM_count_verified_for_reference": False,
            "shared_experts": shared, "ring_tokens": ring, "SF_ring_tokens": sf_ring,
            "components_bytes": components, "storage_intervals": intervals,
            "shared_l1_input_alias": "input_x", "candidate_rings": candidates,
            "total_bytes": offset, "total_GiB": offset / 2**30}


abi = subprocess.check_output([str(SOURCE / "layout-size")], text=True).strip()
assert abi == "21140864 128"
previous = json.loads((ROOT / "data/reference/agentx-target-budget/memory-scope-followup.json").read_text())
for recorded in previous["cases"]:
    calculated = allocation(recorded["configured_tokens"], recorded["SM_count_scenario"])
    for field in ("components_bytes", "ring_tokens", "SF_ring_tokens", "total_bytes"):
        assert calculated[field] == recorded[field], (recorded["hardware_label"], field)

cases = []
for tokens in (8192, 16384):
    for ep in (4, 8):
        for sms in (132, 148):
            for shared in (0, 1):
                cases.append(allocation(tokens, sms, ep=ep, shared=shared))
report = {
    "scope": "offline pinned FP8-dispatch/MXFP4 MegaMoE allocation intervals and ring derivation",
    "runtime_change": False, "formal_accuracy_evidence": False,
    "source_checks": [{"path": str(SOURCE / name), "sha256": hashlib.sha256((SOURCE / name).read_bytes()).hexdigest()}
                      for name in ("__init__.py", "api.hpp", "layout.cuh", "scheduler.cuh", "sym_buffer.cuh", "layout-size.cpp", "layout-size")],
    "host_ABI_probe_output": abi, "all_previous_component_derivations_reproduced": True,
    "cases": cases,
    "limits": [
        "native pinned allocation geometry is independent of unmatched borrowed timing collector/kernel provenance",
        "nominal SM scenarios are assumptions, not reference hardware queries or allocated-byte measurements",
        "same cache key is one persistent storage across target/MTP layers; other keys must be counted separately",
        "EP4/8 and shared0/1 cases check geometry only, not selection of an execution backend for formal curve points",
        "standalone BF16 y, topk, target/drafter hidden, attention workspace and graph/eager peak remain separate identities",
        "existing generic runtime/NCCL reserve is not removed or overlap-credited",
    ],
}
destination = STAGE / "buffer-geometry-probe.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
print("All historical derivations reproduced;", len(cases), "new geometry cases; no budget edits")
