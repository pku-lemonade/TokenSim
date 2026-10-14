"""Transfer plan."""

from __future__ import annotations

from dataclasses import replace

from TokenSim.af.caching.estimates import cached_plan, traffic_feature_fingerprint
from TokenSim.comm.traffic import (
    all_to_all_plan,
    collective_plan,
    point_to_point_plan,
    validate_all_to_all_parameters,
    validated_bytes,
)
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import MODEL_VERSION


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
        """Static traffic only. Invocation identities and readiness stay outside."""
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
            query.participant_ranks,
            self.node_by_worker,
            self.placement,
            query.logical_input_bytes_per_rank,
            query.logical_output_bytes_per_rank,
            traffic_feature_fingerprint(query.routing_edges),
            traffic_feature_fingerprint(query.tensor_slices),
            query.topology,
        )
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
