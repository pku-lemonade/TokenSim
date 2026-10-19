"""Normalize and resolve AF execution domains and communication fabrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from TokenSim.config.cluster_config import WorkerGroupConfig

from typing import Literal

from TokenSim.config.af.communication import (
    AFCommunicationFabricConfig,
    AFCommunicationResourceConfig,
)
from TokenSim.config.af.domains import (
    DomainParallelConfig,
    DomainRankMapping,
    ExecutionDomainConfig,
    LogicalInstanceConfig,
    PhysicalWorkerRef,
)
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError


class ClusterTopologyMixin:
    def _validate_af_topology(self) -> None:
        groups = {group.name: group for group in self.worker_groups}
        if len(self.af.links) != 1:
            raise ConfigurationError("AF decode requires exactly one link")

        for fabric in self.af.communication_fabrics:
            if fabric.network not in self.networks:
                source = getattr(self, "_communication_fabric_sources", {}).get(
                    fabric.resource_group,
                    f"communication fabric {fabric.resource_group!r}",
                )
                raise ConfigurationError(
                    f"{source} references unknown network {fabric.network!r}"
                )

        for link in self.af.links:
            if link.network not in self.networks:
                raise ConfigurationError(
                    f"AF link {link.name!r} references unknown network {link.network!r}"
                )
            if link.attention_group == link.ffn_group and not link.colocated:
                raise ConfigurationError(
                    f"AF link {link.name!r} attention_group and ffn_group must differ"
                )
            if link.colocated and (
                link.attention_group != link.ffn_group
                or not link.attention_domain
                or not link.ffn_domain
                or link.attention_domain == link.ffn_domain
            ):
                raise ConfigurationError(
                    "colocated execution needs one worker group and distinct explicit A/F domains"
                )
            for endpoint_field, required_capability in (
                ("attention_group", "af_attention"),
                ("ffn_group", "af_ffn"),
            ):
                group_name = getattr(link, endpoint_field)
                group = groups.get(group_name)
                if group is None:
                    raise ConfigurationError(
                        f"AF link {link.name!r} {endpoint_field} references unknown group "
                        + f"{group_name!r}"
                    )
                if required_capability not in group.capabilities:
                    raise ConfigurationError(
                        f"AF link {link.name!r} endpoint group {group_name!r} must declare "
                        + f"{required_capability!r}"
                    )
            domains = (
                self._resolve_link_domain(link, "attention"),
                self._resolve_link_domain(link, "ffn"),
            )
            if {domain.name for domain in domains} != set(self.execution_domains):
                raise ConfigurationError(
                    "AF execution domains must contain only the decode A/F endpoints"
                )

    @property
    def af_link(self) -> AFLinkConfig:
        if not self.af.enabled or len(self.af.links) != 1:
            raise ConfigurationError("AF decode requires exactly one link")
        return self.af.links[0]

    @property
    def execution_domains(self) -> dict[str, ExecutionDomainConfig]:
        return {domain.name: domain for domain in self.af.execution_domains}

    @property
    def af_domains(self) -> tuple[ExecutionDomainConfig, ExecutionDomainConfig]:
        return (
            self._resolve_link_domain(self.af_link, "attention"),
            self._resolve_link_domain(self.af_link, "ffn"),
        )

    def af_request_owner_identities(self) -> tuple[str, ...]:
        if not self.af.enabled:
            return ()
        attention_domain, _ = self.af_domains
        owners = []
        for instance in attention_domain.logical_instances:
            first_mapping = min(
                instance.rank_mapping, key=lambda mapping: mapping.domain_rank
            )
            owners.append(first_mapping.physical_worker.identity)
        return tuple(dict.fromkeys(owners))

    def _resolve_link_domain(
        self,
        link: AFLinkConfig,
        side: Literal["attention", "ffn"],
    ) -> ExecutionDomainConfig:
        explicit_name = (
            link.attention_domain if side == "attention" else link.ffn_domain
        )
        capability = "af_attention" if side == "attention" else "af_ffn"
        group_name = link.attention_group if side == "attention" else link.ffn_group
        candidates = [
            domain
            for domain in self.af.execution_domains
            if domain.worker_group == group_name
            and domain.capability == capability
            and (explicit_name is None or domain.name == explicit_name)
        ]
        if len(candidates) != 1:
            qualifier = explicit_name or f"{group_name}/{capability}"
            raise ConfigurationError(
                f"AF link {link.name!r} {side}_domain {qualifier!r} must resolve "
                f"to exactly one execution domain; found {len(candidates)}"
            )
        return candidates[0]

    @property
    def af_communication_resources(
        self,
    ) -> dict[tuple[str, str], AFCommunicationResourceConfig]:
        resources: dict[tuple[str, str], AFCommunicationResourceConfig] = {}
        if not self.af.enabled:
            return resources
        for fabric in self.af.communication_fabrics:
            for resource in fabric.communication_resources():
                if resource.resource_key in resources:
                    raise ConfigurationError(
                        f"duplicate AF communication resource {resource.resource_key!r}"
                    )
                resources[resource.resource_key] = resource
        return resources

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

        endpoints = set()
        for link in self.af.links:
            endpoints.add((link.attention_group, "af_attention"))
            endpoints.add((link.ffn_group, "af_ffn"))
        domains = []
        for group_name, capability in sorted(endpoints):
            group = groups.get(group_name)
            if group is None:
                continue
            domains.append(
                ExecutionDomainConfig(
                    name=f"implicit:{group_name}:{capability}",
                    capability=capability,
                    worker_group=group_name,
                    parallel=DomainParallelConfig(data_parallel_size=group.num_workers),
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
