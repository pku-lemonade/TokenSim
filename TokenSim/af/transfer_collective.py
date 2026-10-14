"""Transfer collective."""

from __future__ import annotations

from typing import Any, Mapping

from TokenSim.af.latency import AFServiceEstimate, AFTransferQuery
from TokenSim.comm.traffic import ring_volume
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.domains import MODEL_VERSION


class TransferCollectiveMixin:
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
