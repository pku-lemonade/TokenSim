"""Af config validation."""

from __future__ import annotations

from dataclasses import asdict

from TokenSim.config.af.links import AFLinkConfig
from TokenSim.config.af.settings import AFConfig
from TokenSim.config.cluster_config import ClusterConfig, WorkerGroupConfig


def _full_link(**overrides) -> AFLinkConfig:
    values = {
        "name": "af-link",
        "attention_group": "attention",
        "ffn_group": "ffn",
        "network": "net1",
        "resource_group": "fabric",
        "duplex": "full",
        "a_to_f_capacity": 1,
        "f_to_a_capacity": 1,
    }
    values.update(overrides)
    return AFLinkConfig(**values)


def _raw_link(link: AFLinkConfig) -> dict:
    return {key: value for key, value in asdict(link).items() if value is not None}


def _af_group(name: str, capability: str, **overrides) -> WorkerGroupConfig:
    values = {
        "name": name,
        "capabilities": (capability,),
        "hardware": "device",
        "num_workers": 1,
        "network": "net1",
    }
    values.update(overrides)
    return WorkerGroupConfig(**values)


def _cluster(
    *,
    links=None,
    extra_groups=None,
    networks=None,
    communication_fabrics=None,
) -> ClusterConfig:
    groups = [
        _af_group("attention", "af_attention"),
        _af_group("ffn", "af_ffn"),
    ]
    groups.extend(extra_groups or [])
    return ClusterConfig(
        num_workers=sum(group.num_workers for group in groups),
        networks=networks or {"net1": "test-network"},
        worker_groups=groups,
        af=AFConfig(
            enabled=True,
            links=[_full_link()] if links is None else links,
            communication_fabrics=communication_fabrics or [],
        ),
    )


def _domain_instances(
    group: str, tp: int, dp: int, *, rank_reuse: bool = False, expert_tp=1
):
    return [
        {
            "name": f"{group}-dp{dp_rank}",
            "rank_mapping": [
                {
                    "domain_rank": dp_rank * tp + tp_rank,
                    "tp_rank": tp_rank,
                    "pp_rank": 0,
                    "dp_rank": dp_rank,
                    "ep_rank": (
                        (dp_rank * tp + tp_rank) // expert_tp if rank_reuse else 0
                    ),
                    "physical_worker": {
                        "group_name": group,
                        "worker_index": dp_rank * tp + tp_rank,
                    },
                }
                for tp_rank in range(tp)
            ],
        }
        for dp_rank in range(dp)
    ]


def _explicit_decode_cluster_raw(
    *,
    a_tp: int = 1,
    a_dp: int = 1,
    f_tp: int = 2,
    f_dp: int = 2,
    f_ep: int = 1,
) -> dict:
    a_workers = a_tp * a_dp
    f_workers = f_tp * f_dp
    return {
        "num_workers": 1 + a_workers + f_workers,
        "networks": {"net1": "network"},
        "worker_groups": [
            {
                "name": "prefill",
                "capabilities": ["prefill"],
                "hardware": "prefill-device",
                "num_workers": 1,
                "network": "net1",
            },
            {
                "name": "attention",
                "capabilities": ["af_attention"],
                "hardware": "a-device",
                "num_workers": a_workers,
                "network": "net1",
            },
            {
                "name": "ffn",
                "capabilities": ["af_ffn"],
                "hardware": "f-device",
                "num_workers": f_workers,
                "network": "net1",
            },
        ],
        "af": {
            "enabled": True,
            "communication_fabrics": [
                {
                    "resource_group": "af-fabric",
                    "network": "net1",
                    "duplex": "full",
                    "lanes": [
                        {"name": "a_to_f", "capacity": 1},
                        {"name": "f_to_a", "capacity": 1},
                    ],
                }
            ],
            "links": [
                {
                    "name": "decode-af",
                    "attention_group": "attention",
                    "ffn_group": "ffn",
                    "attention_domain": "a-domain",
                    "ffn_domain": "f-domain",
                    "resource_group": "af-fabric",
                }
            ],
            "execution_domains": [
                {
                    "name": "a-domain",
                    "capability": "af_attention",
                    "worker_group": "attention",
                    "parallel": {
                        "tensor_parallel_size": a_tp,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": a_dp,
                    },
                    "logical_instances": _domain_instances("attention", a_tp, a_dp),
                },
                {
                    "name": "f-domain",
                    "capability": "af_ffn",
                    "worker_group": "ffn",
                    "parallel": {
                        "tensor_parallel_size": f_tp,
                        "pipeline_parallel_size": 1,
                        "data_parallel_size": f_dp,
                        "expert_parallel_size": f_ep,
                    },
                    "logical_instances": _domain_instances(
                        "ffn",
                        f_tp,
                        f_dp,
                        rank_reuse=f_ep > 1,
                        expert_tp=f_tp * f_dp // f_ep,
                    ),
                },
            ],
        },
    }
