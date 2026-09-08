from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


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
