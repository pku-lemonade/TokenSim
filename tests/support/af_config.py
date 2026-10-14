"""Af config."""

from __future__ import annotations

from TokenSim.config.cluster_config import ClusterConfig


def _cluster_dict(
    *,
    phases=("prefill", "decode"),
    af_block_size=1,
    attention_workers=1,
    ffn_workers=1,
    duplex="full",
    trace_enabled=True,
):
    link = {
        "name": "af-link",
        "phases": list(phases),
        "attention_group": "a-pool",
        "ffn_group": "f-pool",
        "network": "af-net",
        "resource_group": "fabric",
        "duplex": duplex,
    }
    if duplex == "full":
        link.update({"a_to_f_capacity": 1, "f_to_a_capacity": 1})
    else:
        link["half_duplex_capacity"] = 1
    worker_groups = [
        {
            "name": "a-pool",
            "capabilities": ["af_attention"],
            "hardware": "a-device",
            "num_workers": attention_workers,
            "network": "af-net",
        },
        {
            "name": "f-pool",
            "capabilities": ["af_ffn"],
            "hardware": "f-device",
            "num_workers": ffn_workers,
            "network": "af-net",
        },
    ]
    if "prefill" not in phases:
        worker_groups.insert(
            0,
            {
                "name": "prefill-pool",
                "role": "prefill",
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            },
        )
    if "decode" not in phases:
        worker_groups.append(
            {
                "name": "decode-pool",
                "role": "decode",
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            }
        )
    return {
        "num_workers": sum(group["num_workers"] for group in worker_groups),
        "networks": {"af-net": "profile"},
        "worker_groups": worker_groups,
        "af": {
            "enabled": True,
            "phases": list(phases),
            "af_block_size": af_block_size,
            "trace_enabled": trace_enabled,
            "communication_domains": {
                "node_size": 8,
                "node_count": (sum(g["num_workers"] for g in worker_groups) + 7) // 8,
                "placement_strategy": "manual",
            },
            "links": [link],
        },
    }


def _two_link_cluster(*, shared_resource):
    resource_groups = (
        ("shared-fabric", "shared-fabric")
        if shared_resource
        else (
            "prefill-fabric",
            "decode-fabric",
        )
    )
    return ClusterConfig(
        num_workers=3,
        networks={"af-net": "profile"},
        worker_groups=[
            {
                "name": "prefill-a",
                "capabilities": ["af_attention"],
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            },
            {
                "name": "decode-a",
                "capabilities": ["af_attention"],
                "hardware": "a-device",
                "num_workers": 1,
                "network": "af-net",
            },
            {
                "name": "f-pool",
                "capabilities": ["af_ffn"],
                "hardware": "f-device",
                "num_workers": 1,
                "network": "af-net",
            },
        ],
        af={
            "enabled": True,
            "communication_domains": {
                "node_size": 8,
                "node_count": 1,
                "placement_strategy": "manual",
            },
            "phases": ["prefill", "decode"],
            "af_block_size": 1,
            "trace_enabled": True,
            "links": [
                {
                    "name": "prefill-link",
                    "phases": ["prefill"],
                    "attention_group": "prefill-a",
                    "ffn_group": "f-pool",
                    "network": "af-net",
                    "resource_group": resource_groups[0],
                    "duplex": "full",
                    "a_to_f_capacity": 1,
                    "f_to_a_capacity": 1,
                },
                {
                    "name": "decode-link",
                    "phases": ["decode"],
                    "attention_group": "decode-a",
                    "ffn_group": "f-pool",
                    "network": "af-net",
                    "resource_group": resource_groups[1],
                    "duplex": "full",
                    "a_to_f_capacity": 1,
                    "f_to_a_capacity": 1,
                },
            ],
        },
    )


def _load_cluster(**overrides):
    return ClusterConfig(**_cluster_dict(**overrides))
