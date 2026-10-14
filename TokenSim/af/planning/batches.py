from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TYPE_CHECKING

from TokenSim.errors import ConfigurationError


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
