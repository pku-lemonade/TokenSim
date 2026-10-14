"""Small physical pipeline configurations for runtime and serving checks."""

from copy import deepcopy
from types import SimpleNamespace

from TokenSim.config.cluster_config import ClusterConfig
from tests.af.support.config import _cluster


def pipeline_cluster(a_pp=2, f_pp=2, layers=2):
    # Keep the existing TP/fabric contract and expand physical stage mappings.
    from dataclasses import asdict

    raw = asdict(
        _cluster(a_tp=2, a_dp=1, f_tp=2, f_dp=1, f_ep=1, placement_strategy="dedicated")
    )

    def clean(value):
        if isinstance(value, dict):
            return {
                key: clean(item)
                for key, item in value.items()
                if item is not None and not key.startswith("_")
            }
        if isinstance(value, (list, tuple)):
            return [clean(item) for item in value]
        return value

    raw = clean(raw)
    for domain, stages in zip(raw["af"]["execution_domains"], (a_pp, f_pp)):
        domain["parallel"]["pipeline_parallel_size"] = stages
        original = domain["logical_instances"][0]["rank_mapping"]
        expanded = []
        for stage in range(stages):
            for rank in original:
                mapping = deepcopy(rank)
                mapping["domain_rank"] = stage * 2 + rank["tp_rank"]
                mapping["pp_rank"] = stage
                mapping["physical_worker"]["worker_index"] = mapping["domain_rank"]
                expanded.append(mapping)
        domain["logical_instances"][0]["rank_mapping"] = expanded
    for group, stages in zip(raw["worker_groups"], (a_pp, f_pp)):
        group["num_workers"] *= stages
    raw["num_workers"] = sum(group["num_workers"] for group in raw["worker_groups"])
    raw["af"]["communication_domains"]["node_count"] = sum(
        (group["num_workers"] + 7) // 8 for group in raw["worker_groups"]
    )
    return ClusterConfig(**raw)


def decode_request(request_id):
    return SimpleNamespace(
        id=request_id, is_prefill=False, prefill_len=8, context_len=8, generation_idx=0
    )
