from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


AFPhase = Literal["prefill", "decode"]


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
