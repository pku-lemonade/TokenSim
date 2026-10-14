"""Transfer setup."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from TokenSim.af.estimate_cache import initialize_cache
from TokenSim.af.snapshot import HardwareSnapshot, snapshot
from TokenSim.comm.traffic import collective_plan
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype
from TokenSim.hardware.domains import CommunicationDomainConfig, physical_domain_config
from TokenSim.hardware.placement import validate_domain_placement


class TransferSetupMixin:
    def __init__(
        self,
        hardware: Any,
        cluster_config: Any,
        worker_configs: Sequence[Any],
        placement: Any | None = None,
        *,
        dtype: str = "fp16",
        operator_backend: str | None = None,
        backend_by_device: Mapping[str, str | None] | None = None,
        fallback: str = "table_first",
    ) -> None:
        self.hardware = HardwareSnapshot(hardware)
        self.cluster_config = snapshot(cluster_config)
        self.placement = snapshot(placement)
        self.dtype = normalize_dtype(dtype)
        self.operator_backend = operator_backend
        self.backend_by_device = snapshot(dict(backend_by_device or {}))
        self.fallback = fallback
        self._worker_ids: dict[str, int] = {}
        initialize_cache(self, cluster_config.af.service_estimate_cache)
        self._hardware_by_identity: dict[str, str] = {}
        for worker_id, worker in enumerate(worker_configs):
            ref = getattr(worker, "physical_worker_ref", None)
            if ref is None:
                continue
            self._worker_ids[ref.identity] = worker_id
            self._hardware_by_identity[ref.identity] = worker.hardware
        # Fail before the event loop when an AF network has no link class.
        self.link_classes = {
            link.network: self.link_class(link.network)
            for link in cluster_config.af.links
        }
        if (
            placement is None
            and cluster_config.af.transfer_topology_mode != "flat_link"
        ):
            raise ConfigurationError(
                "AF physical topology requires placement; explicitly select flat_link for legacy link pricing"
            )
        physical = physical_domain_config(cluster_config.af)
        self.domain_config = snapshot(
            cluster_config.af.communication_domains or CommunicationDomainConfig()
        )
        if physical is not None:
            validate_domain_placement(self.placement, worker_configs, physical)
        self.node_by_worker = (
            snapshot(
                {
                    worker_id: (
                        self.placement.node_id(worker_id)
                        if physical is not None
                        else self.placement.device_index(worker_id)
                        // self.domain_config.node_size
                    )
                    for worker_id in self._worker_ids.values()
                }
            )
            if placement is not None
            else {}
        )
        records = []
        resolved_backends = {}
        if fallback != "analytical_only":
            for hardware_name in sorted(set(self._hardware_by_identity.values())):
                device = self.hardware.device(hardware_name)
                backend = self.backend_by_device.get(
                    hardware_name, self.operator_backend
                )
                package = self.hardware.operator_package(device.device_id, backend)
                if package is not None:
                    backend = getattr(package.meta, "backend", backend)
                    resolved_backends[hardware_name] = backend
                    for record in package.meta.extra.get(
                        "communication_phase_services", ()
                    ):
                        explicit_scope = record.get("participant_hardware_backends")
                        count = record.get("participants")
                        if type(count) is not int or count < 1:
                            raise ConfigurationError(
                                "phase measurement participants must be a positive integer"
                            )
                        if explicit_scope is None:
                            scope = ((device.device_id, backend),) * count
                        else:
                            if (
                                not isinstance(explicit_scope, (list, tuple))
                                or len(explicit_scope) != count
                                or any(
                                    not isinstance(item, (list, tuple))
                                    or len(item) != 2
                                    for item in explicit_scope
                                )
                            ):
                                raise ConfigurationError(
                                    "phase measurement hardware/backend scope must match participants"
                                )
                            scope = tuple(
                                (self.hardware.device(item[0]).device_id, item[1])
                                for item in explicit_scope
                            )
                        if any(
                            value is not None
                            and (not isinstance(value, str) or not value)
                            for _, value in scope
                        ):
                            raise ConfigurationError(
                                "phase measurement package backend must be a nonempty string or unspecified"
                            )
                        transport = record.get("transport_backend", "unspecified")
                        if not isinstance(transport, str) or not transport:
                            raise ConfigurationError(
                                "phase measurement transport backend must be a nonempty string"
                            )
                        candidate = {
                            **record,
                            "dataset_version": package.meta.dataset_version,
                            "measurement_device": device.device_id,
                            "measurement_backend": backend,
                            "participant_hardware_backends": scope,
                            "transport_backend": transport,
                            "applicability_origin": (
                                "explicit"
                                if explicit_scope is not None
                                else "owning_package_homogeneous"
                            ),
                        }
                        # A/F device aliases may expose the same measurement package.
                        if candidate not in records:
                            records.append(candidate)
        self.phase_measurements = snapshot(tuple(records))
        self._scope_by_worker = snapshot(
            {
                worker_id: (
                    self.hardware.device(
                        self._hardware_by_identity[identity]
                    ).device_id,
                    resolved_backends.get(
                        self._hardware_by_identity[identity],
                        self.backend_by_device.get(
                            self._hardware_by_identity[identity], self.operator_backend
                        ),
                    ),
                )
                for identity, worker_id in self._worker_ids.items()
            }
        )
        self._estimate_namespace += (
            self._fingerprints(self.phase_measurements),
            self._fingerprints(self._scope_by_worker),
        )
        transport_profile = None
        profile = cluster_config.af.transport_service_profile
        if profile is not None:
            if records:
                raise ConfigurationError(
                    "transport profile conflicts with operator-package phase measurements"
                )
            if (
                cluster_config.af.transfer_topology_mode != "physical"
                or physical is None
                or len(set(self.node_by_worker.values())) != 1
            ):
                raise ConfigurationError(
                    "measured direct transport profile requires explicit same-node physical domains"
                )
            from TokenSim.af.transport_profile import TransportServiceProfile

            transport_profile = TransportServiceProfile(
                profile,
                environment=cluster_config.af.calibration_environment,
                domains=self.domain_config,
                device_indices={
                    identity: self.placement.device_index(worker)
                    for identity, worker in self._worker_ids.items()
                },
            )
            self._estimate_namespace += (
                ("transport_service", transport_profile.fingerprint),
            )
        self.transport_profile = transport_profile
        if cluster_config.af.transfer_topology_mode == "physical":
            for domain in self.cluster_config.af.execution_domains:
                for instance in domain.logical_instances:
                    by_tp = {
                        mapping.tp_rank: mapping for mapping in instance.rank_mapping
                    }
                    for collective in domain.communication_plan:
                        ranks = tuple(
                            self._worker_ids[by_tp[tp].physical_worker.identity]
                            for tp in collective.participants
                        )
                        collective_plan(
                            collective.operation,
                            ranks,
                            collective.payload_bytes or 0,
                            self.node_by_worker,
                            self.dtype,
                        )
