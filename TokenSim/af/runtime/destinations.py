"""Compile F-to-A return routes only between stages sharing a model layer."""

from TokenSim.af.runtime.records import AFPartialDestination
from TokenSim.config.cache_config import stage_layer_range


def compile_partial_destinations(cluster, ownerships, total_layers):
    attention, ffn = cluster.af_domains
    stages = {
        (domain.name, rank.domain_rank): rank.pp_rank
        for domain in (attention, ffn)
        for instance in domain.logical_instances
        for rank in instance.rank_mapping
    }
    ranges = {
        (domain.name, stage): stage_layer_range(
            total_layers, domain.parallel.pipeline_parallel_size, stage
        )
        for domain in (attention, ffn)
        for stage in range(domain.parallel.pipeline_parallel_size)
    }
    attention_stages = {}
    for owner in ownerships:
        if owner.domain == attention.name:
            stage = stages[(owner.domain, owner.domain_rank)]
            attention_stages.setdefault(stage, []).append(owner)
    destinations = []
    for source in ownerships:
        if source.domain != ffn.name:
            continue
        source_range = ranges[
            (source.domain, stages[(source.domain, source.domain_rank)])
        ]
        for stage, targets in attention_stages.items():
            target_range = ranges[(attention.name, stage)]
            if max(source_range.start, target_range.start) >= min(
                source_range.stop, target_range.stop
            ):
                continue
            for target in targets:
                interval = source.interval.intersection(target.interval)
                if interval is not None:
                    destinations.append(
                        AFPartialDestination(
                            source_domain=source.domain,
                            source_rank=source.domain_rank,
                            source_dp_rank=source.dp_rank,
                            target_domain=target.domain,
                            target_rank=target.domain_rank,
                            target_dp_rank=target.dp_rank,
                            interval=interval,
                        )
                    )
    return tuple(
        sorted(
            destinations,
            key=lambda item: (
                item.source_domain,
                item.source_rank,
                item.target_domain,
                item.target_rank,
                item.interval.start,
            ),
        )
    )
