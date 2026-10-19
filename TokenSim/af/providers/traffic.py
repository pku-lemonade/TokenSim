"""Transfer provider setup, traffic plans and flat collective services."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from TokenSim.af.caching.estimates import (
    HardwareSnapshot,
    cached_plan,
    initialize_cache,
    snapshot,
)
from TokenSim.af.providers.queries import AFServiceEstimate, AFTransferQuery
from TokenSim.comm.traffic import (
    all_to_all_plan,
    collective_plan,
    point_to_point_plan,
    ring_volume,
    validate_all_to_all_parameters,
    validated_bytes,
)
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import normalize_dtype
from TokenSim.hardware.domains import (
    CommunicationDomainConfig,
    MODEL_VERSION,
    physical_domain_config,
)
from TokenSim.hardware.placement import validate_domain_placement


class TransferPlanMixin:
    def validate_query(self, query):
        if query.operation != "all_to_all":
            return
        if type(query.payload_bytes) is not int:
            raise ConfigurationError(
                "all-to-all payload_bytes must be a non-negative integer"
            )
        validated_bytes(query.payload_bytes, query.dtype)
        endpoints = query.physical_endpoints
        if query.group_size is not None and (
            type(query.group_size) is not int or query.group_size < 1
        ):
            raise ConfigurationError("all-to-all group_size must be a positive integer")
        if (
            not endpoints
            or len(set(endpoints)) != len(endpoints)
            or any(item not in self._worker_ids for item in endpoints)
            or query.group_size not in (None, len(endpoints))
            or (
                query.participant_ranks
                and (
                    len(set(query.participant_ranks)) != len(endpoints)
                    or len(query.participant_ranks) != len(endpoints)
                )
            )
        ):
            raise ConfigurationError(
                "all-to-all needs actual unique mapped participants matching group_size"
            )
        validate_all_to_all_parameters(
            query.local_fraction,
            query.remote_fraction,
            query.load_imbalance,
            query.destination_redistribution,
        )
        if self.cluster_config.af.transfer_topology_mode == "flat_link" and (
            query.local_fraction is not None
            or query.remote_fraction is not None
            or query.load_imbalance != 1
            or query.destination_redistribution
        ):
            raise ConfigurationError(
                "flat all-to-all supports uniform defaults only; use physical for locality, imbalance or redistribution"
            )

    def communication_plan(self, query):
        """Cache wire geometry; invocation provenance and readiness stay outside."""
        self.validate_query(query)
        endpoints = tuple(query.physical_endpoints)
        if (
            not endpoints
            or len(set(endpoints)) != len(endpoints)
            or any(item not in self._worker_ids for item in endpoints)
        ):
            raise ConfigurationError(
                "communication participants must be nonempty, distinct and mapped"
            )
        ranks = tuple(self._worker_ids[item] for item in endpoints)
        operation = (
            "all_reduce" if query.operation == "tp_collective" else query.operation
        )
        collective = operation in {
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
        }
        if (
            collective
            and query.group_size is not None
            and query.group_size != len(ranks)
        ):
            raise ConfigurationError(
                "collective group_size differs from actual participants"
            )
        if (
            collective
            and query.participant_ranks
            and len(query.participant_ranks) != len(set(query.participant_ranks))
        ):
            raise ConfigurationError("duplicate collective participant ranks")
        if (
            collective
            and query.participant_ranks
            and len(query.participant_ranks) != len(ranks)
        ):
            raise ConfigurationError("collective participant/endpoints mismatch")
        if not collective and len(ranks) != 2:
            raise ConfigurationError(
                "endpoint communication requires exactly two ranks"
            )
        payload = validated_bytes(query.payload_bytes)
        # Byte-oriented metadata need not align to dtype; measured service still does.
        key = (
            MODEL_VERSION,
            operation,
            payload,
            query.dtype,
            endpoints,
        )
        # The provider pins node placement. Builders below consume only this
        # geometry and the statistical/backend options, never routing records,
        # tensor item identities, logical byte annotations or query topology.
        transport_backend = getattr(query, "transport_backend", "unspecified")
        if not isinstance(transport_backend, str) or not transport_backend:
            raise ConfigurationError("transport backend must be a nonempty string")
        key += (
            query.local_fraction,
            query.remote_fraction,
            query.load_imbalance,
            query.destination_redistribution,
            transport_backend,
        )

        def build():
            if operation == "all_to_all":
                plan = all_to_all_plan(
                    ranks,
                    payload,
                    self.node_by_worker,
                    local_fraction=query.local_fraction,
                    remote_fraction=query.remote_fraction,
                    load_imbalance=query.load_imbalance,
                    destination_redistribution=query.destination_redistribution,
                    dtype=query.dtype,
                )
            elif collective:
                plan = collective_plan(
                    operation, ranks, payload, self.node_by_worker, query.dtype
                )
            else:
                plan = point_to_point_plan(
                    ranks[0],
                    ranks[1],
                    payload,
                    self.node_by_worker,
                    wire_dtype=query.dtype,
                )
            return replace(
                plan,
                phases=tuple(
                    replace(phase, transport_backend=transport_backend)
                    for phase in plan.phases
                ),
            )

        return cached_plan(self, key, build)

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
            from TokenSim.af.calibration.services import TransportServiceProfile

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

    def _collective(self, query: AFTransferQuery) -> AFServiceEstimate:
        payload = query.payload_bytes or 0
        operation = (
            "all_reduce" if query.operation == "tp_collective" else query.operation
        )
        group_size = (
            query.group_size
            if query.group_size is not None
            else len(query.physical_endpoints)
        )
        if (
            not query.physical_endpoints
            or len(set(query.physical_endpoints)) != len(query.physical_endpoints)
            or group_size != len(query.physical_endpoints)
        ):
            raise ConfigurationError(
                "collective needs actual unique participants matching group_size"
            )
        volume = ring_volume(operation, group_size, payload, query.dtype)
        if payload == 0:
            return self._collective_estimate(
                0.0, "collective:empty", "collective", "exact", query, {}
            )
        # No placement: a ring all-reduce over the AF link class.
        link = self.link_class(query.network)
        ring_bytes = volume.send_bytes_per_rank
        service_us = (
            (
                volume.rounds * link.latency_us
                + volume.send_bytes_per_rank
                / link.bandwidth_per_direction_bytes_per_s
                * 1e6
            )
            if payload
            else 0.0
        )
        return self._collective_estimate(
            service_us,
            f"link:{link.link_id}:{link.source_id}",
            f"links:{link.link_id}",
            "analytical",
            query,
            {
                "algorithm": "ring",
                "operation": operation,
                "logical_bytes": payload,
                "input_bytes_per_rank": volume.input_bytes,
                "output_bytes_per_rank": volume.output_bytes,
                "rounds": volume.rounds,
                "wire_bytes": ring_bytes,
                "wire_bytes_scope": "per_rank",
                "group_size": group_size,
                "link": link.link_id,
            },
        )

    def _collective_estimate(
        self,
        latency_us: float,
        source_id: str,
        dataset_version: str,
        match_type: str,
        query: AFTransferQuery,
        extra: Mapping[str, Any],
    ) -> AFServiceEstimate:
        return AFServiceEstimate(
            service_duration=latency_us / 1e6,
            source_id=source_id,
            payload_bytes=query.payload_bytes,
            dataset_version=dataset_version,
            match_type=match_type,
            details={
                "provider": "collective_model",
                "model_version": MODEL_VERSION,
                "timing_scope": "transport_service",
                "operation": query.operation,
                "direction": query.direction,
                "payload_bytes": query.payload_bytes,
                "logical_participants": query.logical_participants,
                "participant_ranks": query.participant_ranks,
                "physical_endpoints": query.physical_endpoints,
                "resource_identities": query.resource_identities,
                "total_latency_us": latency_us,
                **dict(extra),
            },
        )
