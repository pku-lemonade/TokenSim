"""Cluster topology."""

from __future__ import annotations

from typing import Literal

from TokenSim.config.af.communication import AFCommunicationResourceConfig
from TokenSim.config.af.domains import ExecutionDomainConfig
from TokenSim.config.af.links import AFLinkConfig
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
            if link.attention_group == link.ffn_group:
                raise ConfigurationError(
                    f"AF link {link.name!r} attention_group and ffn_group must differ"
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
