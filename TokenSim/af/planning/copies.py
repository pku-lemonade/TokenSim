"""Planner copies."""

from __future__ import annotations

from typing import Any, Mapping

from TokenSim.af.planning.batching import AFBlock
from TokenSim.af.planning.nodes import ExecutionNode, _TransferCopy
from TokenSim.af.providers.queries import AFEndpoint, AFRoutingEdge, AFTensorSlice
from TokenSim.dtypes import dtype_size_bytes
from TokenSim.errors import ConfigurationError


class TensorCopyPlannerMixin:
    def _activation_copies(
        self,
        *,
        tokens: tuple[tuple[int, int], ...],
        a_bindings: tuple[Any, ...],
        target_specs: tuple[tuple[Any, str, tuple[tuple[int, int], ...]], ...],
        producer_ids: Mapping[int, str],
        collective: ExecutionNode | None,
        barrier_predecessor: str | None,
        transfer_dtype: str,
        routing_edges_by_consumer: (
            Mapping[str, tuple[AFRoutingEdge, ...]] | None
        ) = None,
    ) -> tuple[_TransferCopy, ...]:
        a_ownerships = self.runtime_plan.ownerships_for_replica(
            a_bindings[0].domain, a_bindings[0].dp_rank, a_bindings[0].pp_rank
        )
        source_by_rank = {item.domain_rank: item for item in a_bindings}
        ownership_by_rank = {item.domain_rank: item for item in a_ownerships}
        source_layout = (
            collective.target_layout
            if collective is not None
            else "replicated" if len(a_bindings) == 1 else "sharded"
        )
        if source_layout not in {"sharded", "replicated"}:
            raise ConfigurationError(
                "AF FFN input requires a sharded or replicated A activation, "
                f"got source_layout={source_layout!r}"
            )
        replicated = source_layout == "replicated"
        canonical_candidates = (
            tuple(
                item
                for item in a_bindings
                if item.domain_rank in collective.domain_ranks
            )
            if replicated and collective is not None
            else a_bindings
        )
        if not canonical_candidates:
            raise ConfigurationError("AF activation has no canonical source candidate")
        canonical = min(canonical_candidates, key=lambda item: item.domain_rank)
        eligible_ranks = (
            set(collective.domain_ranks)
            if collective is not None
            else set(source_by_rank)
        )
        sharded_owners = tuple(
            item for item in a_ownerships if item.domain_rank in eligible_ranks
        )
        byte_width = dtype_size_bytes(transfer_dtype, role="a_to_f_dtype")
        copies: list[_TransferCopy] = []
        edges_by_consumer = dict(routing_edges_by_consumer or {})
        for target, consumer_id, consumer_tokens in target_specs:
            for token in consumer_tokens:
                matching_edges = tuple(
                    edge
                    for edge in edges_by_consumer.get(consumer_id, ())
                    if (edge.request_id, edge.token_id) == token
                )
                source_parts = (
                    (
                        (
                            canonical,
                            ownership_by_rank[canonical.domain_rank],
                            (0, self.hidden_size),
                        ),
                    )
                    if replicated
                    else tuple(
                        (
                            source_by_rank[a_owner.domain_rank],
                            a_owner,
                            (a_owner.interval.start, a_owner.interval.end),
                        )
                        for a_owner in sharded_owners
                    )
                )
                for source, a_owner, interval in source_parts:
                    predecessor = barrier_predecessor or (
                        producer_ids[source.domain_rank]
                    )
                    tensor_slice = AFTensorSlice(
                        tensor_item_id=f"request:{token[0]}:token:{token[1]}:activation",
                        interval=interval,
                        layout=source_layout,
                        owner_domain=a_owner.domain,
                        owner_domain_rank=a_owner.domain_rank,
                        owner_dp_rank=a_owner.dp_rank,
                        owner_tp_rank=a_owner.tp_rank,
                        owner_physical_worker=a_owner.physical_worker,
                        canonical_producer=(
                            canonical.physical_identity if replicated else None
                        ),
                        element_count=interval[1] - interval[0],
                        payload_bytes=(interval[1] - interval[0]) * byte_width,
                    )
                    copies.append(
                        _TransferCopy(
                            source,
                            target,
                            tensor_slice,
                            predecessor,
                            consumer_id,
                            source_layout,
                            "replicated",
                            matching_edges,
                        )
                    )
        self._validate_copy_coverage(
            copies,
            expected_intervals={
                (
                    consumer_id,
                    target.domain_rank,
                    f"request:{token[0]}:token:{token[1]}:activation",
                    None,
                ): (0, self.hidden_size)
                for target, consumer_id, consumer_tokens in target_specs
                for token in consumer_tokens
            },
            preserve_source_partials=False,
            context="A-to-F FFN input",
        )
        return tuple(copies)

    def _output_copies(
        self,
        *,
        items: tuple[
            tuple[tuple[int, int], str, str | None, AFRoutingEdge | None], ...
        ],
        source_bindings: tuple[Any, ...],
        producer_ids: Mapping[int, str],
        collective: ExecutionNode | None,
        target_bindings: tuple[Any, ...],
        consumer_ids: Mapping[int, str],
        tensor_layout: str,
        transfer_dtype: str,
    ) -> tuple[_TransferCopy, ...]:
        source_layout = (
            collective.target_layout
            if collective is not None
            else "replicated" if len(source_bindings) == 1 else tensor_layout
        )
        replicated = source_layout == "replicated"
        eligible_sources = tuple(
            item
            for item in source_bindings
            if collective is None or item.domain_rank in collective.domain_ranks
        )
        canonical_candidates = (
            eligible_sources if collective is not None else source_bindings
        )
        if not canonical_candidates:
            raise ConfigurationError("AF output has no canonical source candidate")
        canonical = min(canonical_candidates, key=lambda item: item.domain_rank)
        source_owners = {
            item.domain_rank: self._ownership(item) for item in source_bindings
        }
        byte_width = dtype_size_bytes(transfer_dtype, role="f_to_a_dtype")
        copies: list[_TransferCopy] = []
        target_layout = "partial" if tensor_layout == "partial" else "sharded"
        expected_intervals: dict[tuple[object, ...], tuple[int, int]] = {}
        for _token, tensor_item_id, partial_id, routing_edge in items:
            for target in target_bindings:
                target_owner = self._ownership(target)
                sources = (canonical,) if replicated else eligible_sources
                for source in sources:
                    interval = target_owner.interval
                    if source_layout == "sharded":
                        interval = source_owners[
                            source.domain_rank
                        ].interval.intersection(target_owner.interval)
                        if interval is None:
                            continue
                    predecessor = (
                        collective.node_id
                        if collective is not None
                        else producer_ids[source.domain_rank]
                    )
                    tensor_slice = AFTensorSlice(
                        tensor_item_id=tensor_item_id,
                        interval=(interval.start, interval.end),
                        layout=source_layout,
                        owner_domain=target.domain,
                        owner_domain_rank=target.domain_rank,
                        owner_dp_rank=target.dp_rank,
                        owner_tp_rank=target.tp_rank,
                        owner_physical_worker=target.physical_identity,
                        partial_id=partial_id,
                        canonical_producer=(
                            canonical.physical_identity if replicated else None
                        ),
                        routing_weight=(
                            routing_edge.routing_weight
                            if routing_edge is not None
                            else None
                        ),
                        element_count=interval.size,
                        payload_bytes=interval.size * byte_width,
                    )
                    copies.append(
                        _TransferCopy(
                            source,
                            target,
                            tensor_slice,
                            predecessor,
                            consumer_ids[target.domain_rank],
                            source_layout,
                            target_layout,
                            (routing_edge,) if routing_edge is not None else (),
                        )
                    )
                coverage_key: tuple[object, ...] = (
                    consumer_ids[target.domain_rank],
                    target.domain_rank,
                    tensor_item_id,
                    partial_id,
                )
                expected_intervals[coverage_key] = (
                    target_owner.interval.start,
                    target_owner.interval.end,
                )
        self._validate_copy_coverage(
            copies,
            expected_intervals=expected_intervals,
            preserve_source_partials=source_layout == "partial",
            context="F-to-A output",
        )
        return tuple(copies)

    @staticmethod
    def _routing_edge(assignment: Any) -> AFRoutingEdge:
        return AFRoutingEdge(
            iteration_id=assignment.iteration_id,
            phase=assignment.phase,
            decode_step=assignment.decode_step,
            layer_id=assignment.layer_id,
            request_id=assignment.request_id,
            token_id=assignment.token_id,
            top_k_index=assignment.top_k_index,
            expert_id=assignment.expert_id,
            routing_weight=assignment.routing_weight,
            a_dp_rank=assignment.a_dp_rank,
            a_tp_rank=assignment.a_tp_rank,
        )

    @staticmethod
    def _validate_copy_coverage(
        copies: list[_TransferCopy],
        *,
        expected_intervals: Mapping[tuple[object, ...], tuple[int, int]],
        preserve_source_partials: bool,
        context: str,
    ) -> None:
        grouped: dict[tuple[object, ...], list[_TransferCopy]] = {}
        for copy in copies:
            key = (
                copy.consumer_id,
                copy.target_binding.domain_rank,
                copy.tensor_slice.tensor_item_id,
                copy.tensor_slice.partial_id,
            )
            grouped.setdefault(key, []).append(copy)

        for key, expected in expected_intervals.items():
            candidates = grouped.get(key, [])
            if not candidates:
                raise ConfigurationError(
                    f"{context} has no tensor slices for consumer={key[0]!r}, "
                    f"target_rank={key[1]}, item={key[2]!r}"
                )
            copy_groups = (
                {
                    copy.source_binding.domain_rank: tuple(
                        item
                        for item in candidates
                        if item.source_binding.domain_rank
                        == copy.source_binding.domain_rank
                    )
                    for copy in candidates
                }.values()
                if preserve_source_partials
                else (tuple(candidates),)
            )
            for copy_group in copy_groups:
                intervals = sorted(item.tensor_slice.interval for item in copy_group)
                cursor = expected[0]
                for start, end in intervals:
                    if start != cursor:
                        raise ConfigurationError(
                            f"{context} has a gap or overlap for consumer={key[0]!r}, "
                            f"target_rank={key[1]}, item={key[2]!r}: "
                            f"expected_start={cursor}, actual=[{start}, {end})"
                        )
                    cursor = end
                if cursor != expected[1]:
                    raise ConfigurationError(
                        f"{context} does not cover the required interval for "
                        f"consumer={key[0]!r}, target_rank={key[1]}, "
                        f"item={key[2]!r}: covered_end={cursor}, "
                        f"required_end={expected[1]}"
                    )

    def _ownership(self, binding: Any) -> Any:
        for ownership in self.runtime_plan.ownerships_for_replica(
            binding.domain, binding.dp_rank, binding.pp_rank
        ):
            if ownership.domain_rank == binding.domain_rank:
                return ownership
        # In a rank-reuse F pool every EP rank keeps a replicated copy of the
        # residual tensor. ``ownerships_for_replica`` intentionally collapses
        # those identical intervals for coverage validation, but execution
        # still needs the rank-specific physical owner when constructing
        # output copies. Resolve that replicated record here.
        domain_plan = self.runtime_plan.domain_runtime(binding.domain)
        if (
            domain_plan.capability == "af_ffn"
            and domain_plan.expert_parallel_mode != "expert_tp"
        ):
            for ownership in self.runtime_plan.tensor_ownerships:
                if (
                    ownership.domain == binding.domain
                    and ownership.domain_rank == binding.domain_rank
                    and ownership.dp_rank == binding.dp_rank
                ):
                    return ownership
        raise ConfigurationError(
            f"AF runtime has no tensor ownership for domain={binding.domain!r}, "
            f"rank={binding.domain_rank}"
        )

    @staticmethod
    def _endpoint(binding: Any) -> AFEndpoint:
        return AFEndpoint(
            domain=binding.domain,
            domain_rank=binding.domain_rank,
            physical_worker=binding.physical_identity,
            device_type=binding.hardware,
            dp_rank=binding.dp_rank,
            tp_rank=binding.tp_rank,
            ep_rank=binding.ep_rank,
        )

    @staticmethod
    def _block_tokens(block: AFBlock) -> tuple[tuple[int, int], ...]:
        if block.query_lengths:
            return tuple(
                (request_id, token)
                for request_id, offset, queries in zip(
                    block.request_ids, block.query_offsets, block.query_lengths
                )
                for token in range(offset, offset + queries)
            )
        return tuple(
            (request_id, max(0, context_len - 1))
            for request_id, context_len in zip(block.request_ids, block.context_lengths)
        )
