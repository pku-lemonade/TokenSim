"""Cluster normalization."""

from __future__ import annotations

from TokenSim.config.af_fabric import AFCommunicationFabricConfig
from TokenSim.config.constants import AF_PHASE_ORDER
from TokenSim.config.execution_domain import (
    DomainParallelConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
    PhysicalWorkerRef,
)
from TokenSim.config.worker import WorkerGroupConfig
from TokenSim.errors import ConfigurationError


class ClusterNormalizationMixin:
    def _normalize_execution_domains(self) -> None:
        groups = {group.name: group for group in self.worker_groups}
        if self.af.execution_domains:
            for domain in self.af.execution_domains:
                if domain.logical_instances:
                    continue
                group = groups.get(domain.worker_group)
                if (
                    group is None
                    or domain.required_rank_count != 1
                    or group.num_workers != 1
                ):
                    continue
                domain.logical_instances = self._single_rank_instances(group)
            return

        endpoints: dict[tuple[str, str], set[str]] = {}
        for link in self.af.links:
            endpoints.setdefault((link.attention_group, "af_attention"), set()).update(
                link.phases
            )
            endpoints.setdefault((link.ffn_group, "af_ffn"), set()).update(link.phases)
        for group in self.worker_groups:
            for capability, phases in (
                ("prefill", {"prefill"}),
                ("decode_full", {"decode"}),
            ):
                if capability in group.capabilities:
                    endpoints.setdefault((group.name, capability), set()).update(phases)

        domains = []
        for (group_name, capability), phases in sorted(endpoints.items()):
            group = groups.get(group_name)
            if group is None:
                continue
            parallel = DomainParallelConfig(data_parallel_size=group.num_workers)
            domains.append(
                ExecutionDomainConfig(
                    name=f"implicit:{group_name}:{capability}",
                    capability=capability,
                    phases=tuple(phase for phase in AF_PHASE_ORDER if phase in phases),
                    worker_group=group_name,
                    parallel=parallel,
                    logical_instances=self._single_rank_instances(group),
                )
            )
        self.af.execution_domains = domains

    def _normalize_communication_fabrics(self) -> None:
        fabrics: dict[str, AFCommunicationFabricConfig] = {}
        sources: dict[str, str] = {}
        for fabric in self.af.communication_fabrics:
            group = fabric.resource_group
            if group in fabrics:
                raise ConfigurationError(
                    f"duplicate communication fabric resource_group {group!r}"
                )
            fabrics[group] = fabric
            sources[group] = "communication fabric registry"

        for link in self.af.links:
            if not link.has_inline_fabric:
                continue
            inline = link.inline_fabric()
            existing = fabrics.get(link.resource_group)
            if existing is None:
                fabrics[link.resource_group] = inline
                sources[link.resource_group] = f"AF link {link.name!r}"
                continue
            if existing.network != inline.network or existing.duplex != inline.duplex:
                raise ConfigurationError(
                    f"AF resource_group {link.resource_group!r} conflicts between "
                    f"{sources[link.resource_group]} and AF link {link.name!r}"
                )
            for lane in inline.lanes:
                try:
                    existing_lane = existing.lane(lane.name)
                except ConfigurationError as exc:
                    raise ConfigurationError(
                        f"AF resource_group {link.resource_group!r} conflicts between "
                        f"{sources[link.resource_group]} and AF link {link.name!r}: "
                        f"missing lane {lane.name!r}"
                    ) from exc
                if existing_lane.capacity != lane.capacity:
                    raise ConfigurationError(
                        f"AF resource_group {link.resource_group!r} conflicts between "
                        f"{sources[link.resource_group]} and AF link {link.name!r}: "
                        f"lane {lane.name!r} capacity differs"
                    )

        for link in self.af.links:
            fabric = fabrics.get(link.resource_group)
            if fabric is None:
                raise ConfigurationError(
                    f"AF link {link.name!r} references missing communication fabric "
                    f"resource_group={link.resource_group!r}"
                )
            link.bind_fabric(fabric)
        self.af.communication_fabrics = list(fabrics.values())
        self._communication_fabric_sources = dict(sources)

    @staticmethod
    def _single_rank_instances(
        group: WorkerGroupConfig,
    ) -> list[LogicalInstanceConfig]:
        return [
            LogicalInstanceConfig(
                name=f"{group.name}:{worker_index}",
                rank_mapping=[
                    DomainRankMapping(
                        domain_rank=worker_index,
                        tp_rank=0,
                        pp_rank=0,
                        dp_rank=worker_index,
                        ep_rank=0,
                        physical_worker=PhysicalWorkerRef(
                            group_name=group.name,
                            worker_index=worker_index,
                        ),
                    )
                ],
            )
            for worker_index in range(group.num_workers)
        ]
