from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Callable, Literal

from TokenSim.af.models import AFBlock, AFBlockPlacement, AFIteration, AFPhase
from TokenSim.config.cluster_config import AFLinkConfig
from TokenSim.errors import ConfigurationError
from TokenSim.moe.routing import LayerRoutingSnapshot


AFLayerType = Literal["dense", "moe"]
RoutingSnapshotFactory = Callable[
    [int, AFPhase, int, tuple[Any, ...], AFBlockPlacement | None],
    LayerRoutingSnapshot,
]


def phase_for_requests(requests: Sequence[Any]) -> AFPhase:
    if not requests:
        raise ConfigurationError("AF iteration requires at least one request")
    prefill = [bool(request.is_prefill) for request in requests]
    if all(prefill):
        return "prefill"
    if not any(prefill):
        return "decode"
    raise ConfigurationError(
        "AF iteration requires a homogeneous Prefill or Decode batch"
    )


class AFBlockBuilder:
    def build(
        self,
        iteration_id: int,
        requests: Sequence[Any],
        af_block_size: int,
        source_a_worker_id: int,
        link: AFLinkConfig,
        phase: AFPhase | None = None,
        placement_factory: (
            Callable[[int, tuple[Any, ...]], AFBlockPlacement] | None
        ) = None,
        histogram_factory: Callable[[tuple[Any, ...]], dict[int, int]] | None = None,
        layer_types: Mapping[int, AFLayerType] | None = None,
        routing_snapshot_factory: RoutingSnapshotFactory | None = None,
    ) -> AFIteration:
        resolved_phase = phase or phase_for_requests(requests)
        if type(af_block_size) is not int or af_block_size < 1:
            raise ConfigurationError("af_block_size must be a positive integer")
        if resolved_phase not in link.phases:
            raise ConfigurationError(
                f"AF link {link.name!r} does not cover phase {resolved_phase!r}"
            )
        normalized_layer_types = _normalize_layer_types(layer_types)
        moe_layer_ids = tuple(
            layer_id
            for layer_id, layer_type in normalized_layer_types.items()
            if layer_type == "moe"
        )
        if moe_layer_ids and routing_snapshot_factory is None:
            raise ConfigurationError(
                "AF block builder requires routing_snapshot_factory when layer_types "
                "contains MoE layers"
            )
        if routing_snapshot_factory is not None and layer_types is None:
            raise ConfigurationError(
                "AF block builder requires layer_types when routing_snapshot_factory "
                "is provided"
            )

        blocks: list[AFBlock] = []
        for block_id, start in enumerate(range(0, len(requests), af_block_size)):
            block_requests = tuple(requests[start : start + af_block_size])
            if resolved_phase == "prefill":
                token_count = sum(
                    int(
                        getattr(
                            request,
                            "prefill_compute_len",
                            getattr(request, "prefill_len"),
                        )
                    )
                    for request in block_requests
                )
            else:
                token_count = len(block_requests)
            placement = (
                placement_factory(block_id, block_requests)
                if placement_factory is not None
                else None
            )
            routing_snapshots = (
                tuple(
                    routing_snapshot_factory(
                        iteration_id,
                        resolved_phase,
                        layer_id,
                        block_requests,
                        placement,
                    )
                    for layer_id in moe_layer_ids
                )
                if routing_snapshot_factory is not None
                else ()
            )
            blocks.append(
                AFBlock(
                    iteration_id=iteration_id,
                    block_id=block_id,
                    phase=resolved_phase,
                    request_ids=tuple(int(request.id) for request in block_requests),
                    request_count=len(block_requests),
                    token_count=token_count,
                    context_lengths=tuple(
                        int(request.context_len) for request in block_requests
                    ),
                    source_a_worker_id=source_a_worker_id,
                    link_name=link.name,
                    placement=placement,
                    expert_histogram=tuple(
                        sorted(
                            (
                                histogram_factory(block_requests)
                                if histogram_factory
                                else {}
                            ).items()
                        )
                    ),
                    routing_snapshots=routing_snapshots,
                )
            )
        return AFIteration(
            iteration_id=iteration_id,
            phase=resolved_phase,
            blocks=tuple(blocks),
            request_ids=tuple(int(request.id) for request in requests),
        )


def _normalize_layer_types(
    layer_types: Mapping[int, AFLayerType] | None,
) -> dict[int, AFLayerType]:
    if layer_types is None:
        return {}
    normalized: dict[int, AFLayerType] = {}
    for layer_id, layer_type in layer_types.items():
        if type(layer_id) is not int or layer_id < 0:
            raise ConfigurationError(
                "AF layer_types keys must be non-negative integer layer ids"
            )
        if layer_type not in {"dense", "moe"}:
            raise ConfigurationError(
                f"AF layer {layer_id} type must be 'dense' or 'moe', "
                f"got {layer_type!r}"
            )
        normalized[layer_id] = layer_type
    return dict(sorted(normalized.items()))
