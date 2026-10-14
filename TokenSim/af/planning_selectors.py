"""Planning selectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from TokenSim.config.cluster_config import ExecutionDomainConfig, LogicalInstanceConfig
from TokenSim.errors import ConfigurationError


class GlobalRoundRobinReplicaSelector:
    def __init__(self) -> None:
        self._cursors: dict[tuple[str, str], int] = {}

    def select(
        self, phase: str, domain: ExecutionDomainConfig
    ) -> LogicalInstanceConfig:
        replicas = self.ordered_replicas(phase, domain)
        key = (phase, domain.name)
        cursor = self._cursors.get(key, 0)
        self._cursors[key] = cursor + 1
        return replicas[cursor % len(replicas)]

    @classmethod
    def ordered_replicas(
        cls, phase: str, domain: ExecutionDomainConfig
    ) -> tuple[LogicalInstanceConfig, ...]:
        if phase not in domain.phases:
            raise ConfigurationError(
                f"execution domain {domain.name!r} does not cover phase {phase!r}"
            )
        replicas = tuple(
            sorted(
                domain.logical_instances,
                key=lambda item: (cls._instance_dp_rank(item), item.name),
            )
        )
        if not replicas:
            raise ConfigurationError(
                f"execution domain {domain.name!r} has no DP replicas"
            )
        dp_ranks = [cls._instance_dp_rank(instance) for instance in replicas]
        if len(dp_ranks) != len(set(dp_ranks)):
            raise ConfigurationError(
                f"execution domain {domain.name!r} has duplicate DP replica ranks "
                f"{dp_ranks}"
            )
        return replicas

    @staticmethod
    def _instance_dp_rank(instance: LogicalInstanceConfig) -> int:
        ranks = {mapping.dp_rank for mapping in instance.rank_mapping}
        if len(ranks) != 1:
            raise ConfigurationError(
                f"logical instance {instance.name!r} must represent one DP replica"
            )
        return next(iter(ranks))


@dataclass(frozen=True)
class AFReplicaSelection:
    phase: str
    a_owner: LogicalInstanceConfig
    f_service: LogicalInstanceConfig
    routed_expert_instances: tuple[LogicalInstanceConfig, ...]
    expert_parallel_mode: Literal["expert_tp", "rank_reuse"]

    @property
    def dense_ffn_instance(self) -> LogicalInstanceConfig:
        return self.f_service

    @property
    def shared_expert_instance(self) -> LogicalInstanceConfig:
        return self.f_service


class IndependentRoundRobinAFReplicaSelector:
    """Select A ownership and F service replicas from independent global streams."""

    def __init__(self) -> None:
        self._a_owner_selector = GlobalRoundRobinReplicaSelector()
        self._f_service_selector = GlobalRoundRobinReplicaSelector()

    def select_a_owner(
        self, phase: str, a_domain: ExecutionDomainConfig
    ) -> LogicalInstanceConfig:
        self._require_capability(a_domain, "af_attention", "A owner")
        return self._a_owner_selector.select(phase, a_domain)

    def select_f_service(
        self, phase: str, f_domain: ExecutionDomainConfig
    ) -> LogicalInstanceConfig:
        self._require_capability(f_domain, "af_ffn", "F service")
        return self._f_service_selector.select(phase, f_domain)

    def select_for_block(
        self,
        phase: str,
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
    ) -> AFReplicaSelection:
        a_owner = self.select_a_owner(phase, a_domain)
        return self.select_for_preselected_a(
            phase,
            a_domain,
            f_domain,
            a_owner,
        )

    def select_for_preselected_a(
        self,
        phase: str,
        a_domain: ExecutionDomainConfig,
        f_domain: ExecutionDomainConfig,
        a_owner: LogicalInstanceConfig,
    ) -> AFReplicaSelection:
        """Validate scheduler-owned A placement and independently select F service."""
        self._require_capability(a_domain, "af_attention", "A owner")
        self._require_capability(f_domain, "af_ffn", "F service")
        a_replicas = GlobalRoundRobinReplicaSelector.ordered_replicas(phase, a_domain)
        f_replicas = GlobalRoundRobinReplicaSelector.ordered_replicas(phase, f_domain)
        ep_view = f_domain.expert_parallel

        matching_a_replicas = tuple(
            replica for replica in a_replicas if replica.name == a_owner.name
        )
        if len(matching_a_replicas) != 1 or matching_a_replicas[0] != a_owner:
            raise ConfigurationError(
                f"preselected A owner {a_owner.name!r} is outside execution domain "
                f"{a_domain.name!r} for phase {phase!r}"
            )
        f_service = self._f_service_selector.select(phase, f_domain)
        routed_instances = (f_service,) if ep_view.mode == "expert_tp" else f_replicas
        return AFReplicaSelection(
            phase=phase,
            a_owner=a_owner,
            f_service=f_service,
            routed_expert_instances=routed_instances,
            expert_parallel_mode=ep_view.mode,
        )

    @staticmethod
    def _require_capability(
        domain: ExecutionDomainConfig, expected: str, selection_role: str
    ) -> None:
        if domain.capability != expected:
            raise ConfigurationError(
                f"{selection_role} selector requires capability {expected!r}; "
                f"domain {domain.name!r} has {domain.capability!r}"
            )
