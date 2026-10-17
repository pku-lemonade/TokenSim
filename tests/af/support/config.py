"""Small implicit and explicit AF cluster configurations."""

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


def _replica(group, name, dp_rank, tp_size, ep_size, worker_offset, expert_tp=1):
    return {
        "name": name,
        "rank_mapping": [
            {
                "domain_rank": dp_rank * tp_size + tp_rank,
                "tp_rank": tp_rank,
                "pp_rank": 0,
                "dp_rank": dp_rank,
                "ep_rank": (
                    (dp_rank * tp_size + tp_rank) // expert_tp if ep_size > 1 else 0
                ),
                "physical_worker": {
                    "group_name": group,
                    "worker_index": worker_offset + tp_rank,
                },
            }
            for tp_rank in range(tp_size)
        ],
    }


def _cluster(
    a_tp=2,
    a_dp=2,
    f_tp=2,
    f_dp=2,
    f_ep=None,
    expert_placement_strategy="linear",
    placement_strategy="manual",
):
    f_ep = f_tp * f_dp if f_ep is None else f_ep
    a_workers = a_tp * a_dp
    f_workers = f_tp * f_dp
    return ClusterConfig(
        num_workers=a_workers + f_workers,
        networks={"af-net": "test"},
        worker_groups=[
            {
                "name": "a-pool",
                "capabilities": ["af_attention"],
                "hardware": "a-device",
                "num_workers": a_workers,
                "network": "af-net",
            },
            {
                "name": "f-pool",
                "capabilities": ["af_ffn"],
                "hardware": "f-device",
                "num_workers": f_workers,
                "network": "af-net",
            },
        ],
        af={
            "enabled": True,
            "communication_domains": {
                "node_size": 8,
                "node_count": (
                    (a_workers + 7) // 8 + (f_workers + 7) // 8
                    if placement_strategy == "dedicated"
                    else (a_workers + f_workers + 7) // 8
                ),
                "placement_strategy": placement_strategy,
            },
            "phases": ["prefill", "decode"],
            "af_block_size": 1,
            "links": [
                {
                    "name": "af-link",
                    "phases": ["prefill", "decode"],
                    "attention_group": "a-pool",
                    "ffn_group": "f-pool",
                    "attention_domain": "a-domain",
                    "ffn_domain": "f-domain",
                    "network": "af-net",
                    "resource_group": "af-fabric",
                    "duplex": "full",
                    "a_to_f_capacity": 1,
                    "f_to_a_capacity": 1,
                }
            ],
            "execution_domains": [
                {
                    "name": "a-domain",
                    "capability": "af_attention",
                    "phases": ["prefill", "decode"],
                    "worker_group": "a-pool",
                    "parallel": {
                        "tensor_parallel_size": a_tp,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": a_dp,
                    },
                    "logical_instances": [
                        _replica("a-pool", f"a{rank}", rank, a_tp, 1, rank * a_tp)
                        for rank in range(a_dp)
                    ],
                    "communication_plan": (
                        [
                            {
                                "operation": "all_reduce",
                                "after_stage": "a_compute",
                                "participants": list(range(a_tp)),
                                "network": "af-net",
                                "resource_group": "af-fabric",
                                "resource_lane": "a_to_f",
                            },
                            {
                                "operation": "all_gather",
                                "after_stage": "residual",
                                "participants": list(range(a_tp)),
                                "network": "af-net",
                                "resource_group": "af-fabric",
                                "resource_lane": "f_to_a",
                            },
                        ]
                        if a_tp > 1
                        else []
                    ),
                },
                {
                    "name": "f-domain",
                    "capability": "af_ffn",
                    "phases": ["prefill", "decode"],
                    "worker_group": "f-pool",
                    "parallel": {
                        "tensor_parallel_size": f_tp,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": f_dp,
                        "expert_parallel_size": f_ep,
                        "expert_placement_strategy": expert_placement_strategy,
                    },
                    "logical_instances": [
                        _replica(
                            "f-pool",
                            f"f{rank}",
                            rank,
                            f_tp,
                            f_ep,
                            rank * f_tp,
                            f_tp * f_dp // f_ep,
                        )
                        for rank in range(f_dp)
                    ],
                    "communication_plan": (
                        [
                            {
                                "operation": "all_reduce",
                                "after_stage": "expert_compute",
                                "participants": list(range(f_tp)),
                                "network": "af-net",
                                "resource_group": "af-fabric",
                                "resource_lane": "f_to_a",
                            }
                        ]
                        if f_tp > 1
                        else []
                    ),
                },
            ],
        },
    )
