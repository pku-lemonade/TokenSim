"""AF admission geometry, block records and deterministic micro-batch construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, TYPE_CHECKING

from TokenSim.config.af.links import AFLinkConfig
from TokenSim.errors import ConfigurationError
from TokenSim.moe.routing import LayerRoutingSnapshot


if TYPE_CHECKING:
    from TokenSim.moe.routing import LayerRoutingSnapshot


AFPhase = Literal["prefill", "decode"]


@dataclass(frozen=True)
class AFExpertPlacement:
    expert_id: int
    ep_rank: int
    domain_ranks: tuple[int, ...]
    physical_workers: tuple[str, ...]


@dataclass(frozen=True)
class AFBlockPlacement:
    a_domain: str
    a_instance: str
    a_dp_rank: int
    f_domain: str
    f_instance: str
    f_dp_rank: int
    expert_placement: tuple[AFExpertPlacement, ...] = ()


@dataclass(frozen=True)
class AFBlock:
    iteration_id: int
    block_id: int
    phase: AFPhase
    request_ids: tuple[int, ...]
    request_count: int
    token_count: int
    context_lengths: tuple[int, ...]
    source_a_worker_id: int
    link_name: str
    placement: AFBlockPlacement | None = None
    expert_histogram: tuple[tuple[int, int], ...] = ()
    routing_snapshots: tuple["LayerRoutingSnapshot", ...] = ()
    query_lengths: tuple[int, ...] = ()
    query_offsets: tuple[int, ...] = ()
    logits_rows: int | None = None
    forward_phase: AFPhase | None = None

    def __post_init__(self) -> None:
        from TokenSim.moe.routing import LayerRoutingSnapshot

        snapshots = tuple(self.routing_snapshots)
        by_layer: dict[int, LayerRoutingSnapshot] = {}
        decode_steps: set[int] = set()
        block_request_ids = set(self.request_ids)
        for snapshot in snapshots:
            if not isinstance(snapshot, LayerRoutingSnapshot):
                raise ConfigurationError(
                    "AF block routing_snapshots must contain LayerRoutingSnapshot objects"
                )
            if snapshot.layer_id in by_layer:
                raise ConfigurationError(
                    f"AF block {self.block_id} contains duplicate routing snapshot "
                    f"for layer {snapshot.layer_id}"
                )
            if snapshot.iteration_id != self.iteration_id:
                raise ConfigurationError(
                    f"AF block {self.block_id} routing snapshot iteration mismatch: "
                    f"block={self.iteration_id}, snapshot={snapshot.iteration_id}"
                )
            if snapshot.phase != self.phase:
                raise ConfigurationError(
                    f"AF block {self.block_id} routing snapshot phase mismatch: "
                    f"block={self.phase!r}, snapshot={snapshot.phase!r}, "
                    f"layer={snapshot.layer_id}"
                )
            if snapshot.original_token_count != self.token_count:
                raise ConfigurationError(
                    f"AF block {self.block_id} routing snapshot token count mismatch: "
                    f"block={self.token_count}, snapshot={snapshot.original_token_count}, "
                    f"layer={snapshot.layer_id}"
                )
            decode_steps.add(snapshot.decode_step)
            snapshot_request_ids = {
                request_id for request_id, _ in snapshot.original_tokens
            }
            unknown_request_ids = snapshot_request_ids - block_request_ids
            if unknown_request_ids:
                raise ConfigurationError(
                    f"AF block {self.block_id} routing snapshot layer "
                    f"{snapshot.layer_id} references requests outside the block: "
                    f"{sorted(unknown_request_ids)}"
                )
            by_layer[snapshot.layer_id] = snapshot
        if len(decode_steps) > 1:
            raise ConfigurationError(
                f"AF block {self.block_id} routing snapshots span multiple steps: "
                f"{sorted(decode_steps)}"
            )
        object.__setattr__(
            self,
            "routing_snapshots",
            tuple(by_layer[layer_id] for layer_id in sorted(by_layer)),
        )

    @property
    def routing_layer_ids(self) -> tuple[int, ...]:
        return tuple(snapshot.layer_id for snapshot in self.routing_snapshots)

    def routing_snapshot_for_layer(
        self, layer_id: int
    ) -> "LayerRoutingSnapshot | None":
        return next(
            (
                snapshot
                for snapshot in self.routing_snapshots
                if snapshot.layer_id == layer_id
            ),
            None,
        )


@dataclass
class AFIteration:
    iteration_id: int
    phase: AFPhase
    blocks: tuple[AFBlock, ...]
    request_ids: tuple[int, ...]
    start_at: float | None = None
    end_at: float | None = None
    block_completed_at: dict[int, float] = field(default_factory=dict)
    barrier_wait: dict[int, float] = field(default_factory=dict)

    @property
    def makespan(self) -> float | None:
        if self.start_at is None or self.end_at is None:
            return None
        return self.end_at - self.start_at


@dataclass(frozen=True)
class AFRequestWork:
    queries: int
    offset: int
    context: int
    logits_rows: int
    forward_phase: str


def request_work(request, phase):
    context = int(request.context_len)
    building = bool(request.is_prefill or getattr(request, "needs_recompute", False))
    default_queries = (
        int(getattr(request, "prefill_compute_len", request.prefill_len))
        if building
        else 1
    )
    queries = int(
        getattr(
            request, "query_tokens", getattr(request, "step_tokens", default_queries)
        )
    )
    if building:
        offset = int(getattr(request, "num_computed_tokens", context - queries))
        if request.is_prefill and not getattr(request, "needs_recompute", False):
            offset = max(
                offset,
                int(getattr(request, "cached_prefill_tokens", 0)),
                context - default_queries,
            )
    else:
        queries = int(
            getattr(request, "scheduled_query_tokens", 0)
            or getattr(request, "scheduled_tokens", 0)
            or 1
        )
        offset = max(0, context - queries)
    if queries < 1 or offset < 0 or offset + queries > context:
        raise ConfigurationError(
            "AF scheduled queries must lie inside the request context"
        )
    return AFRequestWork(
        queries,
        offset,
        offset + queries,
        int(offset + queries == context) if building else queries,
        "prefill" if building else phase,
    )


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
            work = tuple(
                request_work(request, resolved_phase) for request in block_requests
            )
            token_count = sum(item.queries for item in work)
            forward_phases = {item.forward_phase for item in work}
            if len(forward_phases) != 1:
                raise ConfigurationError("AF blocks require one forward phase")
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
                    context_lengths=tuple(item.context for item in work),
                    query_lengths=tuple(item.queries for item in work),
                    query_offsets=tuple(item.offset for item in work),
                    logits_rows=sum(item.logits_rows for item in work),
                    forward_phase=next(iter(forward_phases)),
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
