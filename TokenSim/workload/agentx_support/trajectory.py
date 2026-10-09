"""Deterministic snapshot selection and shared trace-pool sampling.

This implements the metadata-only part of AIPerf 49634d1c trajectory_source.py
and its primer spread. No service time, actual replay or synthetic text runs
here; the replay state machine consumes these plans.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from TokenSim.errors import WorkloadValidationError
from TokenSim.workload.agentx_support.models import TraceGraph, TurnKey


@dataclass(frozen=True, slots=True)
class SnapshotState:
    conversation_id: str
    next_turn_index: int
    dispatch_offset_s: float
    waiting_on_children: bool
    join_memberships: tuple[tuple[str, int], ...] = ()

    @property
    def primer_turn_index(self) -> int | None:
        return self.next_turn_index - 1 if self.next_turn_index else None


@dataclass(frozen=True, slots=True)
class Snapshot:
    graph: TraceGraph
    lane_index: int
    t_star_ms: float
    states: tuple[SnapshotState, ...]
    completed_boundaries: tuple[tuple[str, int], ...]

    @property
    def root_next_turn_index(self) -> int:
        return next(
            (
                s.next_turn_index
                for s in self.states
                if s.conversation_id == self.graph.trace_id
            ),
            0,
        )


class SequentialTracePool:
    def __init__(self, graphs: list[TraceGraph]):
        if not graphs:
            raise WorkloadValidationError("AgentX trace pool cannot be empty")
        self.graphs = tuple(graphs)
        self.draw_count = 0

    def next_graph(self) -> TraceGraph:
        graph = self.graphs[self.draw_count % len(self.graphs)]
        self.draw_count += 1
        return graph


def snapshot_at(
    graph: TraceGraph, lane_index: int, t_star_ms: float
) -> Snapshot | None:
    conversations = {c.conversation_id: c for c in graph.conversations}

    def next_index(cid):
        return next(
            (
                i
                for i, turn in enumerate(conversations[cid].turns)
                if turn.request.t * 1000.0 >= t_star_ms
            ),
            None,
        )

    root_next = next_index(graph.trace_id)
    child_states = []
    pending_joins = set()
    for branch in graph.branches:
        if t_star_ms < branch.start_s * 1000.0:
            spawn_ts = graph.root.turns[branch.spawning_turn_index].request.t * 1000.0
            if t_star_ms < spawn_ts or (
                root_next is not None and root_next <= branch.spawning_turn_index
            ):
                continue
        states = []
        for cid in branch.child_conversation_ids:
            child_next = next_index(cid)
            if child_next is None:
                continue
            child_ts = conversations[cid].turns[child_next].request.t * 1000.0
            memberships = (
                ((branch.branch_id, branch.join_turn_index),)
                if branch.join_turn_index is not None
                else ()
            )
            states.append(
                SnapshotState(
                    cid,
                    child_next,
                    max(0.0, child_ts - t_star_ms) / 1000,
                    False,
                    memberships,
                )
            )
        child_states.extend(states)
        if states and branch.join_turn_index is not None:
            pending_joins.add(branch.join_turn_index)
    states = []
    if root_next is not None:
        timestamp = graph.root.turns[root_next].request.t * 1000.0
        states.append(
            SnapshotState(
                graph.trace_id,
                root_next,
                max(0.0, timestamp - t_star_ms) / 1000,
                root_next in pending_joins,
            )
        )
    states.extend(child_states)
    if not states or not any(not s.waiting_on_children for s in states):
        return None
    boundaries = []
    for cid in sorted(conversations):
        index = next_index(cid)
        completed = index if index is not None else len(conversations[cid].turns)
        if completed:
            boundaries.append((cid, completed))
    return Snapshot(graph, lane_index, t_star_ms, tuple(states), tuple(boundaries))


def sample_snapshot(
    graph: TraceGraph,
    lane_index: int,
    *,
    workload_seed: int = 42,
    min_ratio: float = 0.25,
    max_ratio: float = 0.75,
) -> Snapshot | None:
    if not 0 <= min_ratio <= max_ratio <= 1:
        raise WorkloadValidationError(
            "AgentX snapshot ratios must satisfy 0 <= min <= max <= 1"
        )
    seed = int.from_bytes(
        hashlib.sha256(
            f"{workload_seed}:{graph.trace_id}:{lane_index}".encode()
        ).digest()[:8],
        "big",
    )
    start_ms, end_ms = graph.time_start_s * 1000.0, graph.time_end_s * 1000.0
    duration = end_ms - start_ms
    lo, hi = start_ms + min_ratio * duration, start_ms + max_ratio * duration
    t_star = float(lo if lo == hi else np.random.default_rng(seed).uniform(lo, hi))
    return snapshot_at(graph, lane_index, t_star)


def initial_snapshots(
    pool: SequentialTracePool, concurrency: int, *, workload_seed: int = 42
) -> list[Snapshot]:
    if concurrency <= 0:
        raise WorkloadValidationError("AgentX concurrency must be positive")
    snapshots = []
    for _ in range(concurrency + 2 * len(pool.graphs)):
        snapshot = sample_snapshot(
            pool.next_graph(), len(snapshots), workload_seed=workload_seed
        )
        if snapshot is not None:
            snapshots.append(snapshot)
        if len(snapshots) == concurrency:
            return snapshots
    raise WorkloadValidationError(
        "AgentX pool cannot supply the configured concurrency; refusing reduced load"
    )


def primer_schedule(
    snapshots: list[Snapshot], *, system_idle_cap_s: float | None = 10
) -> list[tuple[float, int, TurnKey]]:
    primers = []
    for snapshot in snapshots:
        conversations = {c.conversation_id: c for c in snapshot.graph.conversations}
        for state in snapshot.states:
            index = state.primer_turn_index
            if index is None:
                continue
            turn = conversations[state.conversation_id].turns[index]
            lead_s = (snapshot.t_star_ms - turn.request.t * 1000.0) / 1000
            if system_idle_cap_s is not None:
                lead_s = min(lead_s, system_idle_cap_s)
            primers.append((lead_s, snapshot.lane_index, turn.key))
    max_lead = max((lead for lead, _, _ in primers), default=0)
    return sorted((max_lead - lead, lane, key) for lead, lane, key in primers)


def cache_scope(
    benchmark_id: str, recycle_pass: int, lane_index: int, trace_id: str
) -> str:
    # The phase is deliberately absent; a continuing tree's warmup and
    # profiling requests retain the same scope. New plays always differ.
    base_trace_id = trace_id.split("::", 1)[0]
    identity = f"{benchmark_id}:{recycle_pass}:{lane_index}:{base_trace_id}"
    return hashlib.sha256(identity.encode()).hexdigest()[:12]
