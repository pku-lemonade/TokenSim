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
from TokenSim.config.constants import AF_PHASE_ORDER
from TokenSim.errors import ConfigurationError


class ClusterTopologyMixin:
    def _validate_af_topology(self) -> None:
        groups = {group.name: group for group in self.worker_groups}
        link_names: set[str] = set()
        phase_links: dict[str, list[AFLinkConfig]] = {
            phase: [] for phase in self.af.phases
        }
        resource_signatures: dict[str, tuple[str, str, tuple[int, ...], str]] = {}

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
            if link.name in link_names:
                raise ConfigurationError(f"duplicate AF link name {link.name!r}")
            link_names.add(link.name)
            unselected = set(link.phases) - set(self.af.phases)
            if unselected:
                raise ConfigurationError(
                    f"AF link {link.name!r} covers unselected phases {sorted(unselected)}"
                )
            for phase in link.phases:
                phase_links[phase].append(link)
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
                        + f"{required_capability!r} for phases {list(link.phases)}"
                    )
            for phase in link.phases:
                self._resolve_link_domain(link, phase, "attention")
                self._resolve_link_domain(link, phase, "ffn")
            signature = (
                link.network,
                link.duplex,
                link.effective_capacities,
                link.name,
            )
            existing = resource_signatures.get(link.resource_group)
            if existing is not None and existing[:3] != signature[:3]:
                raise ConfigurationError(
                    f"AF resource_group {link.resource_group!r} conflicts between links "
                    + f"{existing[3]!r} and {link.name!r}"
                )
            resource_signatures[link.resource_group] = signature

        for phase, links in phase_links.items():
            if len(links) != 1:
                raise ConfigurationError(
                    f"AF phase {phase!r} requires exactly one link; found {len(links)}"
                )
        if "prefill" not in self.af.phases and not any(
            "prefill" in group.capabilities for group in self.worker_groups
        ):
            raise ConfigurationError(
                "Decode-only AF requires a worker group with 'prefill' capability"
            )
        if "decode" not in self.af.phases and not any(
            "decode_full" in group.capabilities for group in self.worker_groups
        ):
            raise ConfigurationError(
                "Prefill-only AF requires a worker group with 'decode_full' capability"
            )

    @property
    def phase_to_af_link(self) -> dict[str, AFLinkConfig]:
        if not self.af.enabled:
            return {}
        return {phase: link for link in self.af.links for phase in link.phases}

    @property
    def execution_domains(self) -> dict[str, ExecutionDomainConfig]:
        return {domain.name: domain for domain in self.af.execution_domains}

    @property
    def phase_to_af_domains(
        self,
    ) -> dict[str, tuple[ExecutionDomainConfig, ExecutionDomainConfig]]:
        if not self.af.enabled:
            return {}
        return {
            phase: (
                self._resolve_link_domain(link, phase, "attention"),
                self._resolve_link_domain(link, phase, "ffn"),
            )
            for phase, link in self.phase_to_af_link.items()
        }

    def af_request_owner_identities(self, phase: str) -> tuple[str, ...]:
        if phase not in self.phase_to_af_domains:
            return ()
        attention_domain, _ = self.phase_to_af_domains[phase]
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
        phase: str,
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
            and phase in domain.phases
            and (explicit_name is None or domain.name == explicit_name)
        ]
        if len(candidates) != 1:
            qualifier = explicit_name or f"{group_name}/{capability}/{phase}"
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
