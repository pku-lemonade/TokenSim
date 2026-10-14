"""Planner transfer."""

from __future__ import annotations

from TokenSim.af.planning.nodes import ExecutionNode, _TransferCopy
from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError


class TransferStagePlannerMixin:
    def _append_transfer_stage(
        self,
        nodes: list[ExecutionNode],
        *,
        logical_stage_id: str,
        operation: str,
        direction: str,
        copies: tuple[_TransferCopy, ...],
        link: AFLinkConfig,
        layer_type: str,
        layer_id: int,
    ) -> dict[str, tuple[str, ...]]:
        if not copies:
            raise ConfigurationError(
                f"AF logical transfer stage {logical_stage_id!r} has no endpoint copies"
            )
        effective = self.runtime_plan.dtypes_for_link(link.name)
        activation_dtype = effective.activation_dtype
        transfer_dtype = (
            effective.a_to_f_dtype if direction == "a_to_f" else effective.f_to_a_dtype
        )
        needs_cast = activation_dtype != transfer_dtype

        records: dict[tuple[object, ...], dict[str, object]] = {}
        for copy in copies:
            key = (
                copy.source_binding.domain,
                copy.source_binding.domain_rank,
                copy.target_binding.domain,
                copy.target_binding.domain_rank,
                copy.tensor_slice.tensor_item_id,
                copy.tensor_slice.interval,
                copy.tensor_slice.partial_id,
                copy.source_layout,
                copy.target_layout,
            )
            record = records.setdefault(
                key,
                {
                    "copy": copy,
                    "predecessors": [],
                    "consumers": [],
                    "routing_edges": [],
                },
            )
            record["predecessors"].append(copy.predecessor_id)
            record["consumers"].append(copy.consumer_id)
            record["routing_edges"].extend(copy.routing_edges)

        encode_ids: dict[tuple[object, ...], str] = {}
        if needs_cast:
            encode_groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
            for record in records.values():
                copy = record["copy"]
                encode_key = (
                    copy.source_binding.physical_identity,
                    copy.tensor_slice.tensor_item_id,
                    copy.tensor_slice.interval,
                    copy.tensor_slice.partial_id,
                    activation_dtype,
                    transfer_dtype,
                )
                encode_groups.setdefault(encode_key, []).append(record)
            for index, (encode_key, encode_records) in enumerate(encode_groups.items()):
                record = encode_records[0]
                copy = record["copy"]
                node_id = f"{logical_stage_id}:encode:{index}"
                encode_ids[encode_key] = node_id
                encode_edges = tuple(
                    dict.fromkeys(
                        edge
                        for item in encode_records
                        for edge in item["routing_edges"]
                    )
                )
                nodes.append(
                    self._runtime_compute_node(
                        node_id=node_id,
                        operation="encode",
                        stage="encode",
                        predecessors=tuple(
                            dict.fromkeys(
                                predecessor
                                for item in encode_records
                                for predecessor in item["predecessors"]
                            )
                        ),
                        binding=copy.source_binding,
                        layer_type=layer_type,
                        layer_id=layer_id,
                        token_count=1,
                        original_token_count=1,
                        output_token_count=1,
                        tensor_slices=(copy.tensor_slice,),
                        source_layout=copy.source_layout,
                        target_layout=copy.source_layout,
                        routing_edges=encode_edges,
                        source_dtype=activation_dtype,
                        target_dtype=transfer_dtype,
                        shape={
                            "tokens": 1,
                            "elements": copy.tensor_slice.element_count or 0,
                            "hidden_size": copy.tensor_slice.element_count or 0,
                        },
                    )
                )

        grouped: dict[
            tuple[object, ...], list[tuple[tuple[object, ...], dict[str, object]]]
        ] = {}
        record_predecessors: dict[tuple[object, ...], tuple[str, ...]] = {}
        for record_key, record in records.items():
            copy = record["copy"]
            direct_predecessors = tuple(dict.fromkeys(record["predecessors"]))
            if len(direct_predecessors) != 1:
                raise ConfigurationError(
                    f"AF tensor copy in logical stage {logical_stage_id!r} must "
                    "resolve to exactly one data producer before transfer: "
                    f"item={copy.tensor_slice.tensor_item_id!r}, "
                    f"source_rank={copy.source_binding.domain_rank}, "
                    f"target_rank={copy.target_binding.domain_rank}, "
                    f"predecessors={direct_predecessors!r}"
                )
            if needs_cast:
                encode_key = (
                    copy.source_binding.physical_identity,
                    copy.tensor_slice.tensor_item_id,
                    copy.tensor_slice.interval,
                    copy.tensor_slice.partial_id,
                    activation_dtype,
                    transfer_dtype,
                )
                actual_predecessors = (encode_ids[encode_key],)
            else:
                actual_predecessors = direct_predecessors
            record_predecessors[record_key] = actual_predecessors
            endpoint_key = (
                copy.source_binding.domain,
                copy.source_binding.domain_rank,
                copy.target_binding.domain,
                copy.target_binding.domain_rank,
                copy.source_layout,
                copy.target_layout,
                actual_predecessors,
            )
            grouped.setdefault(endpoint_key, []).append((record_key, record))

        leg_for_record: dict[tuple[object, ...], str] = {}
        for leg_index, group in enumerate(grouped.values()):
            first = group[0][1]["copy"]
            leg_id = f"{logical_stage_id}:leg:{leg_index}"
            slices = tuple(record["copy"].tensor_slice for _, record in group)
            routing_edges = tuple(
                dict.fromkeys(
                    edge for _, record in group for edge in record["routing_edges"]
                )
            )
            predecessors = record_predecessors[group[0][0]]
            for record_key, _record in group:
                leg_for_record[record_key] = leg_id
            source_endpoint = self._endpoint(first.source_binding)
            target_endpoint = self._endpoint(first.target_binding)
            nodes.append(
                ExecutionNode(
                    node_id=leg_id,
                    operation=operation,
                    kind="communication",
                    predecessor_ids=predecessors,
                    stage=None,
                    direction=direction,
                    source_domain=first.source_binding.domain,
                    target_domain=first.target_binding.domain,
                    logical_participants=(
                        first.source_binding.instance,
                        first.target_binding.instance,
                    ),
                    domain_ranks=(
                        first.source_binding.domain_rank,
                        first.target_binding.domain_rank,
                    ),
                    physical_workers=(
                        first.source_binding.physical_identity,
                        first.target_binding.physical_identity,
                    ),
                    dp_rank=first.target_binding.dp_rank,
                    tp_rank=first.target_binding.tp_rank,
                    ep_rank=first.target_binding.ep_rank,
                    network=link.network,
                    resource_group=link.resource_group,
                    resource_lane=("shared" if link.duplex == "half" else direction),
                    payload_bytes=sum(item.payload_bytes or 0 for item in slices),
                    layer_type=layer_type,
                    logical_stage_id=logical_stage_id,
                    leg_id=leg_id,
                    source_endpoint=source_endpoint,
                    target_endpoint=target_endpoint,
                    participant_endpoints=(source_endpoint, target_endpoint),
                    tensor_slices=slices,
                    source_layout=first.source_layout,
                    target_layout=first.target_layout,
                    routing_edges=routing_edges,
                    metadata={
                        "original_token_count": (
                            len(
                                {
                                    (edge.request_id, edge.token_id)
                                    for edge in routing_edges
                                }
                            )
                            if routing_edges
                            else None
                        ),
                        "assignment_count": (
                            len(routing_edges) if routing_edges else None
                        ),
                        "payload_breakdown": tuple(
                            {
                                "tensor_item_id": item.tensor_item_id,
                                "interval": item.interval,
                                "partial_id": item.partial_id,
                                "routing_weight": item.routing_weight,
                                "payload_bytes": item.payload_bytes,
                            }
                            for item in slices
                        ),
                    },
                )
            )

        arrivals: dict[str, list[str]] = {}
        decode_ids: dict[tuple[object, ...], str] = {}
        for record_index, (record_key, record) in enumerate(records.items()):
            copy = record["copy"]
            leg_id = leg_for_record[record_key]
            arrival_id = leg_id
            if needs_cast:
                decode_key = (
                    copy.target_binding.physical_identity,
                    copy.tensor_slice.tensor_item_id,
                    copy.tensor_slice.interval,
                    copy.tensor_slice.partial_id,
                    transfer_dtype,
                    activation_dtype,
                )
                arrival_id = decode_ids.get(decode_key)
                if arrival_id is None:
                    arrival_id = f"{logical_stage_id}:decode:{record_index}"
                    decode_ids[decode_key] = arrival_id
                    nodes.append(
                        self._runtime_compute_node(
                            node_id=arrival_id,
                            operation="decode",
                            stage="decode",
                            predecessors=(leg_id,),
                            binding=copy.target_binding,
                            layer_type=layer_type,
                            layer_id=layer_id,
                            token_count=1,
                            original_token_count=1,
                            output_token_count=1,
                            tensor_slices=(copy.tensor_slice,),
                            source_layout=copy.target_layout,
                            target_layout=copy.target_layout,
                            routing_edges=tuple(dict.fromkeys(record["routing_edges"])),
                            source_dtype=transfer_dtype,
                            target_dtype=activation_dtype,
                            shape={
                                "tokens": 1,
                                "elements": copy.tensor_slice.element_count or 0,
                                "hidden_size": copy.tensor_slice.element_count or 0,
                            },
                        )
                    )
            for consumer in dict.fromkeys(record["consumers"]):
                arrivals.setdefault(consumer, []).append(arrival_id)
        return {
            consumer: tuple(dict.fromkeys(node_ids))
            for consumer, node_ids in arrivals.items()
        }
