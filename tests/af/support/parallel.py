"""Af parallel."""

from __future__ import annotations

from TokenSim.config.cluster.core import ClusterConfig


def _replica(group, name, dp_rank, tp_size, ep_size, worker_offset):
    return {
        "name": name,
        "rank_mapping": [
            {
                "domain_rank": dp_rank * tp_size + tp_rank,
                "tp_rank": tp_rank,
                "pp_rank": 0,
                "dp_rank": dp_rank,
                "ep_rank": dp_rank * tp_size + tp_rank if ep_size > 1 else 0,
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
                "node_count": (a_workers + f_workers + 7) // 8,
                "placement_strategy": "manual",
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
