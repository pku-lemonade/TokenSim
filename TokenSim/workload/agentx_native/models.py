"""Compact request graphs read directly from the original WEKA trace files."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(slots=True)
class RawRequest:
    t: float
    input_length: int
    output_length: int
    hash_ids: Sequence[int]
    model: str
    api_time: float | None = None
    think_time: float | None = None
    type: str = "s"

    @property
    def duration_s(self) -> float:
        return (
            max(0.0, self.api_time)
            if self.api_time is not None and math.isfinite(self.api_time)
            else 0.0
        )

    @property
    def end_s(self) -> float:
        return self.t + self.duration_s


@dataclass(frozen=True, slots=True, order=True)
class TurnKey:
    conversation_id: str
    turn_index: int


@dataclass(frozen=True, slots=True)
class SourceKey:
    trace_id: str
    outer_index: int
    inner_index: int | None = None


@dataclass(slots=True)
class GraphTurn:
    key: TurnKey
    source: SourceKey
    request: RawRequest
    source_kind: str
    # Historical API time only infers this wait and dependencies. Simulation
    # computes service time; it must not add request.duration_s a second time.
    wait_after_previous_s: float
    predecessors: tuple[TurnKey, ...] = ()

    @property
    def requested_output_tokens(self) -> int:
        return max(1, self.request.output_length)


@dataclass(slots=True)
class Conversation:
    conversation_id: str
    parent_conversation_id: str | None
    agent_depth: int
    turns: list[GraphTurn]
    is_auxiliary: bool = False
    branch_ids: list[str] = field(default_factory=list)

    @property
    def replay_scope_id(self) -> str:
        # Captured subagents have independent interval frontiers. The root and
        # its flat workers share the trace scope; overflow workers of a tagged
        # subagent share that subagent's scope. SPAWN/JOIN links the scopes.
        marker = "::sa:"
        if marker not in self.conversation_id:
            return self.parent_conversation_id or self.conversation_id
        trace_id, agent_id = self.conversation_id.split(marker, 1)
        for worker_marker in (":aux:", ":fa:", ":wg:"):
            if worker_marker in agent_id:
                agent_id = agent_id.rsplit(worker_marker, 1)[0]
        return f"{trace_id}{marker}{agent_id}"


@dataclass(frozen=True, slots=True)
class Branch:
    branch_id: str
    child_conversation_ids: tuple[str, ...]
    spawning_turn_index: int
    join_turn_index: int | None
    start_s: float
    recorded_end_s: float


@dataclass(slots=True)
class TraceGraph:
    trace_id: str
    block_size: int
    conversations: list[Conversation]
    branches: list[Branch]
    raw_request_count: int
    time_start_s: float
    time_end_s: float
    parse_stats: dict[str, int] = field(default_factory=dict)

    @property
    def root(self) -> Conversation:
        return self.conversations[0]

    @property
    def request_count(self) -> int:
        return sum(len(c.turns) for c in self.conversations)

    def all_turns(self):
        for conversation in self.conversations:
            yield from conversation.turns

    @property
    def recorded_activity_end_s(self) -> float:
        return max(
            [turn.request.end_s for turn in self.all_turns()]
            + [branch.recorded_end_s for branch in self.branches]
        )
