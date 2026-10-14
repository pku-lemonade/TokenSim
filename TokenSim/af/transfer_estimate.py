"""Transfer estimate."""

from __future__ import annotations

from collections import Counter
import math

from TokenSim.af.estimate_cache import cached_estimate, cached_phase_estimate
from TokenSim.af.latency import AFServiceEstimate, AFTransferQuery
from TokenSim.af.operator_composition import (
    _compose_dataset_version,
    _compose_source_id,
    _worst_match,
)
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import MODEL_VERSION


TRANSFER_OPERATIONS = (
    "af_a_to_f",
    "af_f_to_a",
    "af_control_length",
    "af_control_payload",
    "af_router_metadata",
    "ep_dispatch",
    "ep_combine",
    "tp_collective",
    "all_reduce",
    "all_gather",
    "reduce_scatter",
    "all_to_all",
)


class TransferEstimateMixin:
    @cached_phase_estimate
    def phase_estimate(self, phase):
        if self.transport_profile is not None:
            if phase.operation != "send_recv" or len(phase.participants) != 2:
                raise ConfigurationError(
                    "direct transport profile cannot price collective phases"
                )
            source, target = phase.participants
            duration = self.transport_profile.estimate_us(
                source=self.placement.device_index(source),
                target=self.placement.device_index(target),
                backend=phase.transport_backend,
                dtype=phase.dtype,
                size=phase.wire_bytes,
            )
            exact = any(
                row["source_device"] == self.placement.device_index(source)
                and row["target_device"] == self.placement.device_index(target)
                and row["backend"] == phase.transport_backend
                and row["dtype"] == phase.dtype
                and any(point["bytes"] == phase.wire_bytes for point in row["points"])
                for row in self.transport_profile.document["records"]
            )
            return AFServiceEstimate(
                duration * 1e-6,
                self.transport_profile.document["profile_id"],
                phase.logical_input_bytes,
                self.transport_profile.fingerprint,
                "exact" if exact else "interpolated",
                {
                    "timing_scope": "transport_service",
                    "transport_backend": phase.transport_backend,
                    "model_version": MODEL_VERSION,
                    "phase_id": phase.phase_id,
                    "resource_identities": phase.resource_ids,
                    "resource_wire_bytes": phase.resource_wire_bytes,
                    "contains_cpu_submit": False,
                    "contains_external_wait": False,
                    "measurement_boundary": self.transport_profile.document["evidence"][
                        "measurement_boundary"
                    ],
                    "interpolation": "linear_within_training_range",
                },
            )
        algorithm = (
            "direct" if phase.operation in {"send_recv", "all_to_all"} else "ring"
        )
        nodes = {
            node: index
            for index, node in enumerate(
                sorted({self.node_by_worker[rank] for rank in phase.participants})
            )
        }
        signature = {
            "model_version": MODEL_VERSION,
            "algorithm": algorithm,
            "participant_hardware_backends": tuple(
                self._scope_by_worker[rank] for rank in phase.participants
            ),
            "transport_backend": phase.transport_backend,
            "phase_kind": phase.phase_id.split(":")[0],
            "operation": phase.operation,
            "dtype": phase.dtype,
            "participants": len(phase.participants),
            "nodes": len({self.node_by_worker[rank] for rank in phase.participants}),
            "logical_input_bytes": phase.logical_input_bytes,
            "wire_bytes": phase.wire_bytes,
            "wire_schedule": tuple(
                tuple(
                    (
                        nodes[message.source_node],
                        nodes[message.target_node],
                        message.wire_bytes,
                    )
                    for message in messages
                )
                for messages in phase.rounds
            ),
            "domain_parameters": tuple(
                (key, getattr(self.domain_config, key))
                for key in (
                    "local_bandwidth_bytes_per_s",
                    "inter_tx_bandwidth_bytes_per_s",
                    "inter_rx_bandwidth_bytes_per_s",
                    "local_latency_us",
                    "inter_latency_us",
                )
            ),
        }
        matches = [
            record
            for record in self.phase_measurements
            if phase.wire_bytes
            and all(record.get(key) == value for key, value in signature.items())
        ]
        if len(matches) > 1:
            raise ConfigurationError(
                "conflicting matching communication phase measurements"
            )
        if matches:
            record = matches[0]
            if record.get("timing_scope") != "transport_service":
                raise ConfigurationError(
                    "communication phase measurement must declare transport_service timing scope"
                )
            duration = record.get("latency_us")
            if (
                isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(duration)
                or duration < 0
            ):
                raise ConfigurationError("invalid measured communication phase latency")
            if not isinstance(record.get("source_id"), str) or not record["source_id"]:
                raise ConfigurationError(
                    "measured communication phase requires a nonempty source_id"
                )
            return AFServiceEstimate(
                duration * 1e-6,
                record["source_id"],
                phase.logical_input_bytes,
                record["dataset_version"],
                "exact",
                {
                    **signature,
                    "timing_scope": "transport_service",
                    "phase_id": phase.phase_id,
                    "measurement_device": record["measurement_device"],
                    "measurement_backend": record["measurement_backend"],
                    "applicability_origin": record["applicability_origin"],
                    "resource_wire_bytes": phase.resource_wire_bytes,
                    "resource_identities": phase.resource_ids,
                },
            )
        if self.fallback == "table_only" and phase.wire_bytes:
            raise ConfigurationError(
                "measured-only communication needs matching algorithm and phase transport_service records; whole collective timing cannot be split"
            )
        return AFServiceEstimate(
            phase.service(self.domain_config),
            f"analytical:{MODEL_VERSION}:{phase.phase_id}",
            phase.logical_input_bytes,
            MODEL_VERSION,
            "analytical",
            {
                "timing_scope": "transport_service",
                "model_version": MODEL_VERSION,
                "participant_hardware_backends": signature[
                    "participant_hardware_backends"
                ],
                "transport_backend": phase.transport_backend,
                "phase_id": phase.phase_id,
                "algorithm": algorithm,
                "participants": phase.participants,
                "fan": phase.fan,
                "logical_bytes": phase.logical_input_bytes,
                "wire_bytes": phase.wire_bytes,
                "resource_wire_bytes": phase.resource_wire_bytes,
                "resource_identities": phase.resource_ids,
                "rounds": len(phase.rounds),
                "relative_timetable": phase.relative_timetable(self.domain_config),
                "deprecated_fields_ignored": (
                    "half_bandwidth_bytes",
                    "collective_efficiency",
                ),
            },
        )

    @cached_estimate
    def estimate(self, query: AFTransferQuery) -> AFServiceEstimate:
        if query.operation not in TRANSFER_OPERATIONS:
            raise ValueError(
                f"unsupported AF communication operation={query.operation!r}"
            )
        if query.payload_bytes is None:
            raise ValueError(
                f"operation {query.operation!r} requires explicit payload_bytes"
            )
        if type(query.payload_bytes) is not int or query.payload_bytes < 0:
            raise ValueError("transfer payload_bytes must be a non-negative integer")
        if query.operation in {
            "tp_collective",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
            "all_to_all",
        }:
            endpoints = query.physical_endpoints
            if (
                not endpoints
                or len(set(endpoints)) != len(endpoints)
                or any(item not in self._worker_ids for item in endpoints)
            ):
                raise ConfigurationError(
                    "collective participants must be nonempty, distinct and mapped"
                )
            if query.participant_ranks and (
                len(query.participant_ranks) != len(endpoints)
                or len(set(query.participant_ranks)) != len(endpoints)
            ):
                raise ConfigurationError(
                    "collective participant ranks must be unique and match endpoints"
                )
        if self.cluster_config.af.transfer_topology_mode == "physical":
            self.link_class(query.network)
            plan = self.communication_plan(query)
            services = {
                phase.phase_id: self.phase_estimate(phase) for phase in plan.phases
            }
            match_type = _worst_match(item.match_type for item in services.values())
            resource_wire = Counter()
            for phase in plan.phases:
                for resource, size in phase.resource_wire_bytes:
                    resource_wire[resource] += size
            return AFServiceEstimate(
                plan.service(
                    self.domain_config,
                    lambda phase: services[phase.phase_id].service_duration,
                ),
                _compose_source_id(item.source_id for item in services.values())
                or f"analytical:{MODEL_VERSION}",
                query.payload_bytes,
                _compose_dataset_version(
                    item.dataset_version for item in services.values()
                )
                or MODEL_VERSION,
                match_type,
                {
                    "timing_scope": "transport_service",
                    "model_version": MODEL_VERSION,
                    "operation": query.operation,
                    "logical_bytes": query.payload_bytes,
                    "output_bytes_per_rank": plan.output_bytes_per_rank,
                    "expected_output_bytes": plan.expected_output_bytes,
                    "wire_bytes": plan.wire_bytes,
                    "routing_provenance": plan.provenance,
                    "resource_wire_bytes": tuple(sorted(resource_wire.items())),
                    "phase_ids": tuple(phase.phase_id for phase in plan.phases),
                },
            )
        if query.operation in {
            "tp_collective",
            "all_reduce",
            "all_gather",
            "reduce_scatter",
        }:
            return self._collective(query)
        if query.operation == "all_to_all":
            count = len(query.physical_endpoints)
            if (
                count < 1
                or len(set(query.physical_endpoints)) != count
                or query.group_size not in (None, count)
            ):
                raise ConfigurationError(
                    "flat all-to-all needs actual unique participants matching group_size"
                )
            link = self.link_class(query.network)
            wire_bytes = query.payload_bytes * (count - 1) / count
            return self._collective_estimate(
                link.analytical_transfer_us(wire_bytes),
                f"link:{link.link_id}:{link.source_id}",
                f"links:{link.link_id}",
                "analytical",
                query,
                {
                    "algorithm": "direct",
                    "wire_bytes": wire_bytes,
                    "wire_bytes_scope": "per_rank",
                    "group_size": count,
                    "model_version": MODEL_VERSION,
                    "routing_provenance": "uniform_fallback",
                },
            )
        link = self.link_class(query.network)
        payload = query.payload_bytes
        service_us = link.analytical_transfer_us(payload)
        source_id = f"link:{link.link_id}:{link.source_id}"
        dataset_version = f"links:{link.link_id}"
        return AFServiceEstimate(
            service_duration=service_us / 1e6,
            source_id=source_id,
            payload_bytes=payload,
            dataset_version=dataset_version,
            match_type="analytical",
            details={
                "provider": "topology_links",
                "model_version": MODEL_VERSION,
                "timing_scope": "transport_service",
                "transport_backend": query.transport_backend,
                "backend_calibrated": False,
                "link": (link.link_id if link is not None else None),
                "fixed_latency_us": link.latency_us if link is not None else None,
                "bandwidth_bytes_per_s": (
                    link.bandwidth_per_direction_bytes_per_s
                    if link is not None
                    else None
                ),
                "efficiency": 1.0,
                "deprecated_fields_ignored": (
                    "half_bandwidth_bytes",
                    "collective_efficiency",
                ),
                "payload_bytes": payload,
                "operation": query.operation,
                "direction": query.direction,
                "logical_participants": query.logical_participants,
                "participant_ranks": query.participant_ranks,
                "physical_endpoints": query.physical_endpoints,
                "resource_identities": query.resource_identities,
                "topology": dict(query.topology),
                "total_latency_us": service_us,
            },
        )
