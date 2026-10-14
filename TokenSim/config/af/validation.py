"""Cluster domains."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TokenSim.config.cluster_config import WorkerGroupConfig

from TokenSim.config.af.domains import DomainRankMapping, ExecutionDomainConfig
from TokenSim.errors import ConfigurationError


class ClusterDomainValidationMixin:
    def _validate_execution_domains(self) -> None:
        groups = {group.name: group for group in self.worker_groups}
        domain_names: set[str] = set()
        for domain in self.af.execution_domains:
            if domain.name in domain_names:
                raise ConfigurationError(
                    f"duplicate execution domain name {domain.name!r}"
                )
            domain_names.add(domain.name)
            group = groups.get(domain.worker_group)
            if group is None:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} references unknown worker_group "
                    f"{domain.worker_group!r}"
                )
            if domain.capability not in group.capabilities:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} worker_group "
                    f"{domain.worker_group!r} must declare {domain.capability!r}"
                )
            if not domain.logical_instances:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} requires explicit logical_instances "
                    f"and rank_mapping for required_rank_count="
                    f"{domain.required_rank_count}"
                )
            if domain.parallel.pipeline_parallel_size != 1:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} has "
                    f"pipeline_parallel_size={domain.parallel.pipeline_parallel_size}; "
                    "AF phase one only supports pipeline_parallel_size=1"
                )
            _ = domain.expert_parallel
            if len(domain.logical_instances) != domain.parallel.data_parallel_size:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} logical_instances count must "
                    f"equal data_parallel_size={domain.parallel.data_parallel_size}; "
                    f"got {len(domain.logical_instances)}"
                )
            expected_domain_ranks = set(range(domain.required_rank_count))
            seen_dp_ranks: set[int] = set()
            instance_names: set[str] = set()
            physical_owners: dict[str, tuple[str, int]] = {}
            for instance in domain.logical_instances:
                if instance.name in instance_names:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} has duplicate logical instance "
                        f"{instance.name!r}"
                    )
                instance_names.add(instance.name)
                actual_ranks = {
                    mapping.domain_rank for mapping in instance.rank_mapping
                }
                if len(actual_ranks) != len(instance.rank_mapping):
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} logical instance "
                        f"{instance.name!r} has duplicate domain ranks"
                    )
                instance_dp_ranks = {
                    mapping.dp_rank for mapping in instance.rank_mapping
                }
                if len(instance_dp_ranks) != 1:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} logical instance "
                        f"{instance.name!r} must contain exactly one dp_rank; got "
                        f"{sorted(instance_dp_ranks)}"
                    )
                instance_dp_rank = next(iter(instance_dp_ranks))
                if instance_dp_rank in seen_dp_ranks:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} has duplicate dp_rank "
                        f"{instance_dp_rank} across logical instances"
                    )
                seen_dp_ranks.add(instance_dp_rank)
                start = instance_dp_rank * domain.ranks_per_replica
                expected_instance_ranks = set(
                    range(start, start + domain.ranks_per_replica)
                )
                if actual_ranks != expected_instance_ranks:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} logical instance "
                        f"{instance.name!r} rank_mapping must cover dp replica "
                        f"{instance_dp_rank} domain ranks "
                        f"{sorted(expected_instance_ranks)}; got {sorted(actual_ranks)}"
                    )
                for mapping in instance.rank_mapping:
                    self._validate_domain_rank_mapping(domain, group, mapping)
                    physical_identity = mapping.physical_worker.identity
                    previous = physical_owners.get(physical_identity)
                    if previous is not None:
                        previous_instance, previous_rank = previous
                        raise ConfigurationError(
                            f"execution domain {domain.name!r} logical instance "
                            f"{instance.name!r} rank {mapping.domain_rank} maps to physical "
                            f"worker {physical_identity!r}, already mapped by logical instance "
                            f"{previous_instance!r} rank {previous_rank}; placement must be "
                            "strictly one-to-one"
                        )
                    physical_owners[physical_identity] = (
                        instance.name,
                        mapping.domain_rank,
                    )
            expected_dp_ranks = set(range(domain.parallel.data_parallel_size))
            if seen_dp_ranks != expected_dp_ranks:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} logical instances must cover "
                    f"dp ranks {sorted(expected_dp_ranks)}; got {sorted(seen_dp_ranks)}"
                )
            if {
                mapping.domain_rank
                for instance in domain.logical_instances
                for mapping in instance.rank_mapping
            } != expected_domain_ranks:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} rank mappings must cover "
                    f"domain ranks {sorted(expected_domain_ranks)}"
                )
            expected_workers = {
                f"{domain.worker_group}:{worker_index}"
                for worker_index in range(group.num_workers)
            }
            missing_workers = expected_workers - set(physical_owners)
            if missing_workers:
                raise ConfigurationError(
                    f"execution domain {domain.name!r} has unmapped physical workers "
                    f"{sorted(missing_workers)}; every worker in group "
                    f"{domain.worker_group!r} must map to exactly one logical instance rank"
                )
            collective_stages: set[str] = set()
            fabrics = {
                fabric.resource_group: fabric
                for fabric in self.af.communication_fabrics
            }
            for collective in domain.communication_plan:
                if collective.after_stage in collective_stages:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} has ambiguous collective "
                        f"plans after_stage={collective.after_stage!r}"
                    )
                collective_stages.add(collective.after_stage)
                expected_tp_ranks = set(range(domain.parallel.tensor_parallel_size))
                invalid = set(collective.participants) - expected_tp_ranks
                if invalid:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} has out-of-range replica-local "
                        f"tp_rank participants {sorted(invalid)} for "
                        f"TP={domain.parallel.tensor_parallel_size}"
                    )
                if collective.resource_group is None:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} requires resource_group"
                    )
                fabric = fabrics.get(collective.resource_group)
                if fabric is None:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} references missing communication "
                        f"fabric resource_group={collective.resource_group!r}"
                    )
                if collective.resource_lane is None:
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} requires explicit resource_lane"
                    )
                fabric.lane(collective.resource_lane)
                if (
                    collective.network is not None
                    and collective.network != fabric.network
                ):
                    raise ConfigurationError(
                        f"execution domain {domain.name!r} collective "
                        f"{collective.operation!r} network={collective.network!r} "
                        f"conflicts with fabric {collective.resource_group!r} "
                        f"network={fabric.network!r}"
                    )
                for dp_rank in range(domain.parallel.data_parallel_size):
                    domain.resolve_collective_participants(collective, dp_rank)

    @staticmethod
    def _validate_domain_rank_mapping(
        domain: ExecutionDomainConfig,
        group: WorkerGroupConfig,
        mapping: DomainRankMapping,
    ) -> None:
        parallel = domain.parallel
        ranks_per_replica = domain.ranks_per_replica
        rank_in_replica = mapping.domain_rank % ranks_per_replica
        expected_dp = mapping.domain_rank // ranks_per_replica
        expected_tp = rank_in_replica % parallel.tensor_parallel_size
        expected_ep = domain.expert_parallel.ep_rank(
            dp_rank=expected_dp, tp_rank=expected_tp
        )
        actual = (
            mapping.tp_rank,
            mapping.pp_rank,
            mapping.dp_rank,
            mapping.ep_rank,
        )
        expected = (expected_tp, 0, expected_dp, expected_ep)
        if actual != expected:
            raise ConfigurationError(
                f"execution domain {domain.name!r} domain_rank={mapping.domain_rank} "
                f"has coordinates {actual}; expected {expected}"
            )
        physical = mapping.physical_worker
        if physical.group_name != domain.worker_group:
            raise ConfigurationError(
                f"execution domain {domain.name!r} rank {mapping.domain_rank} maps to "
                f"worker group {physical.group_name!r}, expected {domain.worker_group!r}"
            )
        if physical.worker_index >= group.num_workers:
            raise ConfigurationError(
                f"execution domain {domain.name!r} rank {mapping.domain_rank} maps to "
                f"out-of-range physical worker {physical.identity}"
            )
