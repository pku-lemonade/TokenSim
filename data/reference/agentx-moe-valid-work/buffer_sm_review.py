"""Bound persistent native-ring sensitivity without changing memory budgets."""

import hashlib
import json
from pathlib import Path


EVIDENCE = Path(__file__).resolve().parent
ROOT = EVIDENCE.parents[2]
BLOCKS = (8, 16, 32, 64, 96, 128, 192, 240)


def ceil_div(a, b):
    return (a + b - 1) // b


def ring_tokens(tokens, sms, *, ep=8, experts=384, topk=6, hidden=7168, inter=3072):
    capacity = ceil_div(tokens, 1920) * 1920
    clusters = sms // 2
    l1, l2 = 2 * inter // 256, hidden // 256
    largest = 0
    for block in BLOCKS:
        total = ceil_div(capacity * ep * min(topk, experts // ep), block) + experts // ep
        first_l2 = ceil_div(ceil_div(clusters, l2) * l1, clusters)
        interleave = ceil_div(l1 + (total - 1) * max(0, l1 - l2), clusters) + 1
        warmup = min(max(first_l2, interleave), ceil_div(total * l1, clusters))
        live = min(total, ceil_div(min(warmup * clusters, total * l1), l1)
                   + ceil_div(total * max(0, l2 - l1), l2)
                   + ceil_div(clusters, min(l1, l2)))
        largest = max(largest, live * block)
    return ceil_div(largest, 1920) * 1920


previous = json.loads((ROOT / "data/reference/agentx-target-budget/memory-scope-followup.json").read_text())
for case in previous["cases"]:
    assert ring_tokens(case["configured_tokens"], case["SM_count_scenario"]) == case["ring_tokens"]

cases = []
for sku, tokens in (("b200", 8192), ("b300", 16384)):
    base = next(case for case in previous["cases"] if case["hardware_label"] == sku)
    buckets = {}
    # Assumption sweep only. It does not identify the physical SM count.
    for sms in range(64, 257):
        ring = ring_tokens(tokens, sms)
        buckets.setdefault(ring, []).append(sms)
    # kMinCandidateBlockM=8 dominates SF ring capacity: 16 scale rows/row.
    ring_bytes_per_token = 7168 + 16 * 7168 // 32 + 4 + 3072 + 16 * 3072 // 32
    for ring, sms in sorted(buckets.items()):
        assert sms == list(range(min(sms), max(sms) + 1))
        cases.append({"hardware_label": sku, "unsharded_scheduler_tokens": tokens,
                      "SM_scenario_inclusive_range": [min(sms), max(sms)],
                      "ring_tokens": ring, "SF_ring_tokens": 16 * ring,
                      "total_native_buffer_bytes": base["total_bytes"]
                          + (ring - base["ring_tokens"]) * ring_bytes_per_token,
                      "SM_count_verified_for_reference": False})

sources = [Path("/tmp/tokensim-deepgemm-buffer-audit") / name
           for name in ("__init__.py", "api.hpp", "layout.cuh", "scheduler.cuh")]
report = {
    "scope": "offline persistent ring SM-count sensitivity; no runtime capacity changes",
    "runtime_change": False, "formal_accuracy_evidence": False,
    "source_checks": [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in sources],
    "previous_recorded_scenarios_reproduced": True,
    "cases": cases,
    "interpretation": [
        "ring allocation is piecewise constant after source 1920-token alignment; identical132/148 scenarios do not establish SM-independent sizing",
        "B200 recorded ring is unchanged for SM scenarios98..217; B300 recorded ring is unchanged for64..169 in the audited64..256 range",
        "persistent capacity uses the full unsharded scheduler budget; active SP shard rows do not divide this storage",
        "native storage identities/lifetimes and generic runtime reserve must still be audited before replacing memory envelopes",
    ],
}
destination = EVIDENCE / "buffer-sm-sensitivity-followup.json"
destination.write_text(json.dumps(report, indent=2) + "\n")
print(destination)
for case in cases:
    print(case["hardware_label"], case["SM_scenario_inclusive_range"], case["ring_tokens"],
          case["total_native_buffer_bytes"])
