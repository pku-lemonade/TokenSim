"""Replay configuration, warmup and profiling orchestration."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
import simpy
from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.workload.agentx_support.models import TraceGraph, TurnKey
from TokenSim.workload.agentx_support.trajectory import (
    SequentialTracePool,
    Snapshot,
    cache_scope,
    initial_snapshots,
    primer_schedule,
)
from .records import REPLAY_PROTOCOL, engine_hashes, TraceTurn, StreamCursor, TreePlay
from .clock import ReplayClock
from .lifecycle import RequestLifecycle


class AgentXReplay(ReplayClock, RequestLifecycle):
    """Replay configuration, warmup and profiling orchestration."""

    def __init__(
        self,
        env,
        engine,
        graphs: list[TraceGraph],
        *,
        block_size,
        concurrency,
        profile_duration,
        workload_seed=0,
        # Protocol defaults follow AIPerf's agentic replay.
        grace_period=30.0,
        cancellation_timeout=10.0,
        warmup=True,
        warmup_requests_per_lane=10,
        warmup_grace_period=1800.0,
        tree_idle_cap=300.0,
        system_idle_cap=10.0,
        issue_mode="closed_loop",
        benchmark_id="agentx",
        event_sink=None,
        request_sink=None,
    ):
        for name, value in (
            ("profile_duration", profile_duration),
            ("grace_period", grace_period),
            ("cancellation_timeout", cancellation_timeout),
            ("warmup_grace_period", warmup_grace_period),
        ):
            if not math.isfinite(value) or value < 0:
                raise WorkloadValidationError(f"{name} must be finite and nonnegative")
        if profile_duration <= 0 or concurrency <= 0 or warmup_requests_per_lane < 0:
            raise WorkloadValidationError(
                "positive profile duration/concurrency and nonnegative warmup quota required"
            )
        for cap in (tree_idle_cap, system_idle_cap):
            if cap is not None and (not math.isfinite(cap) or cap <= 0):
                raise WorkloadValidationError("idle caps must be positive or None")
        if not hasattr(engine, "cancel_requests"):
            raise WorkloadValidationError(
                "native replay requires engine cancellation support"
            )
        self.env, self.engine = env, engine
        self.graphs = graphs
        self.pool = SequentialTracePool(graphs)
        self.block_size, self.concurrency = block_size, concurrency
        self.profile_duration, self.grace_period = profile_duration, grace_period
        self.cancellation_timeout = cancellation_timeout
        self.workload_seed = self.random_seed = workload_seed
        self.warmup, self.warmup_quota = warmup, warmup_requests_per_lane
        self.warmup_grace_period = warmup_grace_period
        self.tree_idle_cap, self.system_idle_cap = tree_idle_cap, system_idle_cap
        if issue_mode not in ("closed_loop", "recorded"):
            raise WorkloadValidationError("issue_mode must be closed_loop or recorded")
        self.issue_mode = issue_mode
        self.benchmark_id = benchmark_id
        self.event_sink, self.request_sink = event_sink, request_sink
        self.snapshots = (
            initial_snapshots(self.pool, concurrency, workload_seed=workload_seed)
            if warmup
            else []
        )
        self.initial_graphs = (
            [s.graph for s in self.snapshots]
            if warmup
            else [self.pool.next_graph() for _ in range(concurrency)]
        )
        self.initial_trace_ids = [g.trace_id for g in self.initial_graphs]
        self.recycle_trace_ids = [[] for _ in range(concurrency)]
        self.pass_counts = defaultdict(int)
        self.trees: list[TreePlay] = []
        self.requests, self.warmup_requests = [], []
        self.records = []
        self.phase_counts = {phase: Counter() for phase in ("warmup", "profiling")}
        self.pressure_sent = [0] * concurrency
        self.profile_started_at = self.profile_finished_at = None
        self.issue_window_end_at = None
        self.phase = "initial"
        self.accepting = True
        self.play_count = 0
        self._next_id = 0
        self._in_flight: dict[
            int, tuple[Request, TreePlay | None, str | None, bool]
        ] = {}
        self._changed = env.event()
        self._global_idle_started = env.now
        self._epoch = 0
        self.tree_idle_jumps = self.system_idle_jumps = 0
        self.tree_idle_seconds_skipped = self.system_idle_seconds_skipped = 0.0
        self.trace_ids_used = {"warmup": set(), "profiling": set()}
        self.source_ids_used = {"warmup": set(), "profiling": set()}
        self.in_flight_at_window_end = 0
        self.cancel_unreturned = 0
        self._cached_wake = None
        self._recorded_timer_heap = []
        self._recorded_timer_sequence = 0

    @property
    def profile_elapsed(self):
        if self.profile_started_at is None:
            return 0.0
        end = (
            self.env.now
            if self.profile_finished_at is None
            else self.profile_finished_at
        )
        return max(0.0, end - self.profile_started_at)

    @property
    def metric_observation_duration(self):
        completed = [r for r in self.requests if r.completed_at is not None]
        if not completed:
            return 0.0
        return max(r.completed_at for r in completed) - min(
            r.arrival_at for r in completed
        )

    def runtime_metadata(self):
        return {
            "replay_protocol": REPLAY_PROTOCOL,
            "trace_count": len(self.graphs),
            "dataset_request_count": sum(g.request_count for g in self.graphs),
            "play_count": self.play_count,
            "warmup_request_count": len(self.warmup_requests),
            "warmup_duration_s": self.profile_started_at or 0,
            "workload_seed": self.workload_seed,
            "initial_trace_ids": self.initial_trace_ids,
            "recycle_trace_ids": self.recycle_trace_ids,
            "phase_request_counts": {p: dict(c) for p, c in self.phase_counts.items()},
            "phase_trace_counts": {
                p: len(ids) for p, ids in self.trace_ids_used.items()
            },
            "phase_unique_original_requests": {
                p: len(ids) for p, ids in self.source_ids_used.items()
            },
            "issue_window_s": self.profile_duration,
            "grace_period_s": self.grace_period,
            "cancellation_timeout_s": self.cancellation_timeout,
            "profile_elapsed_s": self.profile_elapsed,
            "metric_observation_duration_s": self.metric_observation_duration,
            "in_flight_at_cutoff": self.in_flight_at_window_end,
            "cancel_unreturned": self.cancel_unreturned,
            "tree_idle_jumps": self.tree_idle_jumps,
            "system_idle_jumps": self.system_idle_jumps,
            "tree_idle_seconds_skipped": self.tree_idle_seconds_skipped,
            "system_idle_seconds_skipped": self.system_idle_seconds_skipped,
            "issue_mode": self.issue_mode,
        }

    def capacity_request(self):
        turns = [turn for graph in self.graphs for turn in graph.all_turns()]
        maximum_input = max(t.request.input_length for t in turns)
        maximum_context = max(
            t.request.input_length + t.requested_output_tokens for t in turns
        )
        return Request(
            -1, maximum_input, max(1, maximum_context - maximum_input), self.block_size
        )

    def _notify(self):
        if not self._changed.triggered:
            self._changed.succeed()
        self._changed = self.env.event()

    def _event(self, kind, **data):
        if self.event_sink:
            self.event_sink(
                {"time_s": self.env.now, "phase": self.phase, "kind": kind, **data}
            )

    def _tree(self, graph, lane, snapshot: Snapshot | None = None):
        pass_index = self.pass_counts[graph.trace_id]
        self.pass_counts[graph.trace_id] += 1
        states = {s.conversation_id: s for s in snapshot.states} if snapshot else {}
        boundaries = dict(snapshot.completed_boundaries) if snapshot else {}
        streams = {}
        completed = set()
        for ordinal, conv in enumerate(graph.conversations):
            index = boundaries.get(conv.conversation_id, 0)
            completed.update(TurnKey(conv.conversation_id, i) for i in range(index))
            state = states.get(conv.conversation_id)
            terminal = index >= len(conv.turns) and bool(snapshot)
            active = state is not None if snapshot else conv is graph.root
            if state:
                index = state.next_turn_index
            if state:
                due = self.env.now + state.dispatch_offset_s
            elif active and index < len(conv.turns):
                due = (
                    self.env.now
                    if self.issue_mode == "closed_loop"
                    else self._recorded_due(
                        snapshot.t_star_ms / 1000 if snapshot else graph.time_start_s,
                        conv.turns[index].request.t,
                    )
                )
            else:
                due = None
            streams[conv.conversation_id] = StreamCursor(
                conv, index, active, due, terminal=terminal, ordinal=ordinal
            )
        opened = {
            b.branch_id
            for b in graph.branches
            if snapshot
            and (
                b.spawning_turn_index < boundaries.get(graph.trace_id, 0)
                or any(cid in states for cid in b.child_conversation_ids)
            )
        }
        play = TreePlay(
            graph,
            lane,
            pass_index,
            snapshot.t_star_ms / 1000 if snapshot else graph.time_start_s,
            streams,
            completed,
            opened,
            cache_scope(self.benchmark_id, pass_index, lane, graph.trace_id),
            clock_origin_s=self.env.now,
            idle_started_at=self.env.now,
        )
        self.play_count += 1
        self._register_play_timers(play)
        return play

    def run(self):
        self.trees = (
            [
                self._tree(snapshot.graph, snapshot.lane_index, snapshot)
                for snapshot in self.snapshots
            ]
            if self.warmup
            else [
                self._tree(graph, lane)
                for lane, graph in enumerate(self.initial_graphs)
            ]
        )
        if self.warmup:
            self.phase = "primers"
            jobs = [
                self.env.process(
                    self._primer(offset, self.trees[lane], key, self._epoch)
                )
                for offset, lane, key in primer_schedule(
                    self.snapshots, system_idle_cap_s=self.system_idle_cap
                )
            ]
            if jobs:
                yield simpy.events.AllOf(self.env, jobs)
            self.phase = "pressure"
            for play in self.trees:
                for cursor in play.streams.values():
                    if cursor.active and not cursor.terminal:
                        cursor.due = self.env.now
            while not all(count >= self.warmup_quota for count in self.pressure_sent):
                self._issue_ready()
                if all(count >= self.warmup_quota for count in self.pressure_sent):
                    break
                yield from self._wait(math.inf)
            self.accepting = False
            self.phase = "warmup_drain"
            yield from self._drain(self.env.now + self.warmup_grace_period)
            self.engine.reset_profile_stats()
        self.accepting = True
        self.phase = "profiling"
        self._epoch += 1
        self.profile_started_at = self.env.now
        self.issue_window_end_at = self.env.now + self.profile_duration
        self._global_idle_started = self.env.now
        self._recycle()
        offsets = []
        for play in self.trees:
            play.clock_origin_s = self.env.now
            for cursor in play.streams.values():
                if (
                    cursor.active
                    and not cursor.terminal
                    and cursor.index < len(cursor.conversation.turns)
                ):
                    turn = cursor.conversation.turns[cursor.index]
                    due = (
                        self._recorded_due(
                            play.origin_s, turn.request.t, play.clock_origin_s
                        )
                        if self.issue_mode == "recorded"
                        else self.env.now + max(0, turn.request.t - play.origin_s)
                    )
                    self._set_due(cursor, due)
                    if self._joins_ready(play, cursor):
                        offsets.append(max(0, turn.request.t - play.origin_s))
        anchor = min(offsets, default=0)
        for play in self.trees:
            play.clock_origin_s -= anchor
            play.idle_started_at = self.env.now
            play.idle_cap_expired = False
            for cursor in play.streams.values():
                if cursor.due is not None:
                    self._set_due(cursor, max(self.env.now, cursor.due - anchor))
        while self.env.now < self.issue_window_end_at:
            self._idle_shifts()
            self._issue_ready()
            yield from self._wait(self.issue_window_end_at)
        self.accepting = False
        self._epoch += 1
        self.phase = "grace"
        self.in_flight_at_window_end = len(self._in_flight)
        yield from self._drain(self.env.now + self.grace_period)
        self.phase = "complete"
        self.profile_finished_at = self.env.now
        # Successful metrics include grace completions. Error/cancelled
        # requests remain in the state accounting, not in latency averages.
        self.requests = [
            r for r in self.requests if r.is_done and not r.cancel_requested
        ]
