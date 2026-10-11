"""Closed-loop AgentX replay, warmup handoff, idle clocks and cancellation."""

from __future__ import annotations

from array import array
import hashlib
import heapq
import math
import sys
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import simpy

from TokenSim.errors import SimulationStateError, WorkloadValidationError
from TokenSim.llm.llm_request import Request
from TokenSim.workload.agentx_support.models import Conversation, TraceGraph, TurnKey
from TokenSim.workload.agentx_support.trajectory import (
    SequentialTracePool,
    Snapshot,
    cache_scope,
    initial_snapshots,
    primer_schedule,
)

REPLAY_PROTOCOL = "aiperf-49634d1c-native"


def engine_hashes(
    hashes, *, trace_block_size: int, engine_block_size: int, input_tokens: int
) -> list[str | int]:
    if engine_block_size % trace_block_size:
        raise WorkloadValidationError(
            "engine block size must be a multiple of trace hash block size"
        )
    width = engine_block_size // trace_block_size
    complete = min(len(hashes) // width, input_tokens // engine_block_size)
    if width == 1:
        return list(hashes[:complete])
    # Each engine block hashes its trace hashes as big-endian u64 values.
    words = array("Q", hashes[: complete * width])
    if sys.byteorder == "little":
        words.byteswap()
    data = words.tobytes()
    stride = 8 * width
    sha256 = hashlib.sha256
    return [
        sha256(data[i : i + stride]).hexdigest()
        for i in range(0, complete * stride, stride)
    ]


@dataclass
class StreamCursor:
    conversation: Conversation
    index: int
    active: bool
    due: float | None
    in_flight: Request | None = None
    terminal: bool = False
    ordinal: int = 0


@dataclass
class TreePlay:
    graph: TraceGraph
    lane: int
    pass_index: int
    origin_s: float
    streams: dict[str, StreamCursor]
    completed: set[TurnKey]
    opened_branches: set[str]
    scope: str
    in_flight: int = 0
    clock_origin_s: float = 0.0
    idle_started_at: float | None = None
    idle_cap_expired: bool = False
    active_streams: dict[str, StreamCursor] = field(default_factory=dict)
    unfinished_streams: int = 0

    def __post_init__(self):
        self.active_streams = {
            cid: c for cid, c in self.streams.items() if c.active and not c.terminal
        }
        self.unfinished_streams = sum(not c.terminal for c in self.streams.values())

    def set_terminal(self, cursor: StreamCursor) -> None:
        if not cursor.terminal:
            cursor.terminal = True
            self.unfinished_streams -= 1
        self.active_streams.pop(cursor.conversation.conversation_id, None)

    def finished(self) -> bool:
        return self.in_flight == 0 and self.unfinished_streams == 0


class AgentXReplay:
    def __init__(
        self,
        env,
        engine,
        graphs: list[TraceGraph],
        *,
        block_size=256,
        concurrency=196,
        profile_duration=3600.0,
        grace_period=30.0,
        cancellation_timeout=10.0,
        workload_seed=42,
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

    def _recorded_due(self, origin_s, request_s, clock_origin_s=None):
        # A recycled tree starts its own source clock at the recycle time.
        anchor = self.env.now if clock_origin_s is None else clock_origin_s
        return anchor + max(0.0, request_s - origin_s)

    def _set_due(self, cursor, due):
        cursor.due = due
        if self.issue_mode == "recorded" and due is not None:
            self._recorded_timer_sequence += 1
            heapq.heappush(
                self._recorded_timer_heap,
                (due, self._recorded_timer_sequence, cursor),
            )

    def _register_play_timers(self, play):
        if self.issue_mode != "recorded":
            return
        for cursor in play.active_streams.values():
            if cursor.due is not None:
                self._set_due(cursor, cursor.due)

    def _next_recorded_timer(self):
        while self._recorded_timer_heap:
            due, _, cursor = self._recorded_timer_heap[0]
            if (
                cursor.terminal
                or not cursor.active
                or cursor.in_flight is not None
                or cursor.due != due
                or due <= self.env.now + 1e-9
            ):
                heapq.heappop(self._recorded_timer_heap)
                continue
            return due
        return None

    def _next_due(self, play, cursor, turn):
        if self.issue_mode == "recorded":
            return self._recorded_due(
                play.origin_s, turn.request.t, play.clock_origin_s
            )
        return self.env.now + turn.wait_after_previous_s

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

    def _joins_ready(self, play, cursor):
        if cursor.conversation is not play.graph.root:
            return True
        return all(
            b.branch_id in play.opened_branches
            and all(play.streams[cid].terminal for cid in b.child_conversation_ids)
            for b in play.graph.branches
            if b.join_turn_index == cursor.index
        )

    def _ready(self, play, cursor):
        if (
            not cursor.active
            or cursor.terminal
            or cursor.in_flight
            or cursor.index >= len(cursor.conversation.turns)
        ):
            return False
        turn = cursor.conversation.turns[cursor.index]
        return (
            cursor.due is not None
            and cursor.due <= self.env.now + 1e-9
            and self._joins_ready(play, cursor)
            and all(p in play.completed for p in turn.predecessors)
        )

    def _make_request(self, play, turn, *, warmup):
        req = Request(
            self._next_id,
            turn.request.input_length,
            1 if warmup else turn.requested_output_tokens,
            self.block_size,
            chat_id=str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"agentx:{self.workload_seed}:{play.graph.trace_id}:{play.pass_index}:{play.lane}:{turn.key.conversation_id}",
                )
            ),
            parent_chat_id=(
                None if turn.source_kind == "weka_main" else play.graph.trace_id
            ),
            turn=turn.key.turn_index,
            hash_ids=engine_hashes(
                turn.request.hash_ids,
                trace_block_size=play.graph.block_size,
                engine_block_size=self.block_size,
                input_tokens=turn.request.input_length,
            ),
            cache_salt=play.scope,
            reuse_group=play.graph.trace_id,
            measurement_phase="warmup" if warmup else "profile",
            record_timing=not warmup,
        )
        self._next_id += 1
        req.agentx_trace_id = play.graph.trace_id
        req.agentx_play_id = play.pass_index
        req.agentx_lane = play.lane
        req.agentx_stream_id = (
            "root" if turn.source_kind == "weka_main" else turn.key.conversation_id
        )
        req.agentx_conversation_id = turn.key.conversation_id
        req.agentx_source_outer_idx = turn.source.outer_index
        req.agentx_source_inner_idx = turn.source.inner_index
        req.agentx_recorded_model = turn.request.model
        req.trace_hash_ids = turn.request.hash_ids
        req.trace_block_size = play.graph.block_size
        return req

    def _submit(self, req, play, cid=None, *, primer=False):
        phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
        req.arrive(self.env)
        self._in_flight[req.id] = (req, play, cid, primer)
        self.phase_counts[phase]["sent"] += 1
        self.trace_ids_used[phase].add(req.agentx_trace_id)
        self.source_ids_used[phase].add(
            (
                req.agentx_trace_id,
                req.agentx_source_outer_idx,
                req.agentx_source_inner_idx,
            )
        )
        if not primer:
            play.in_flight += 1
            play.idle_started_at = None
            play.idle_cap_expired = False
            play.streams[cid].in_flight = req
        self._global_idle_started = None
        req.completion_event.callbacks.append(
            lambda event, rid=req.id: self._completed(rid, event)
        )
        submit = getattr(self.engine, "add_requests_burst", self.engine.add_requests)
        submit([req])
        (self.warmup_requests if phase == "warmup" else self.requests).append(req)
        self._event(
            "issued",
            request_id=req.id,
            lane=play.lane,
            conversation_id=req.agentx_conversation_id,
            primer=primer,
        )

    def _completed(self, request_id, event):
        wire = self._in_flight.pop(request_id, None)
        if wire is None:
            return
        req, play, cid, primer = wire
        phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
        if not event.ok:
            event.defused = True
        outcome = (
            "cancelled"
            if req.cancel_requested
            else "successful" if event.ok and req.is_done else "error"
        )
        self.phase_counts[phase][outcome] += 1
        self._record_terminal(req, play, outcome)
        if not primer:
            cursor = play.streams[cid]
            play.in_flight -= 1
            cursor.in_flight = None
            if outcome != "cancelled" or phase == "warmup":
                old_turn = cursor.conversation.turns[cursor.index]
                play.completed.add(old_turn.key)
                cursor.index += 1
                if cursor.index >= len(cursor.conversation.turns):
                    play.set_terminal(cursor)
                due = (
                    self.env.now
                    if self.phase == "pressure"
                    else (
                        self._next_due(
                            play, cursor, cursor.conversation.turns[cursor.index]
                        )
                        if not cursor.terminal
                        else None
                    )
                )
                self._set_due(cursor, due)
                if cursor.conversation is play.graph.root:
                    self._open_branches(play, old_turn)
            else:
                play.set_terminal(cursor)
                cursor.due = None
            if play.in_flight == 0:
                play.idle_started_at = self.env.now
        if not self._in_flight:
            self._global_idle_started = self.env.now
        req.release_logical_blocks()
        req.hash_ids = None
        self._notify()

    def _record_terminal(self, req, play, outcome):
        phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
        source = req.first_prefill_source
        record = {
            "request_id": req.id,
            "phase": phase,
            "outcome": outcome,
            "start_s": req.arrival_at,
            "end_s": self.env.now,
            "input_tokens": req.prefill_len,
            "gpu_hit_tokens": source.gpu_hit_tokens if source else None,
            "cpu_hit_tokens": source.cpu_hit_tokens if source else None,
            "computed_input_tokens": source.computed_tokens if source else None,
            "prefill_source_observed": source is not None,
            "prefill_admission_count": req.prefill_admission_count,
            "reloaded_gpu_tokens": req.reloaded_gpu_tokens,
            "reloaded_cpu_tokens": req.reloaded_cpu_tokens,
            "reloaded_generated_gpu_tokens": req.reloaded_generated_gpu_tokens,
            "reloaded_generated_cpu_tokens": req.reloaded_generated_cpu_tokens,
            "latest_gpu_context_hit_tokens": req.gpu_cached_context_tokens,
            "latest_cpu_context_hit_tokens": req.external_cached_context_tokens,
            "recomputation_count": req.recomputation_count,
            "recomputed_tokens": req.recomputed_tokens_total,
            "latest_gpu_hit_tokens": req.cached_prefill_tokens
            - req.external_cached_prefill_tokens,
            "latest_cpu_hit_tokens": req.external_cached_prefill_tokens,
            "latest_computed_input_tokens": req.effective_prefill_tokens,
            "requested_output_tokens": req.decode_len,
            "output_tokens": req.generation_idx,
            "trace_id": req.agentx_trace_id,
            "conversation_id": req.agentx_conversation_id,
            "outer_index": req.agentx_source_outer_idx,
            "inner_index": req.agentx_source_inner_idx,
            "turn_index": req.turn,
            "lane": play.lane,
            "dp_rank": getattr(req, "dp_rank", None),
            "worker_id": getattr(req, "routing_worker_id", None),
            "pass_index": play.pass_index,
            "cancellation_acknowledged": (
                req.cancelled_at is not None if req.cancel_requested else None
            ),
            "length_capped": req.max_model_len is not None
            and req.context_len >= req.max_model_len,
            "ttft_s": req.prefill_latency if req.generation_idx else None,
            "latency_s": (
                req.completed_at - req.arrival_at
                if req.completed_at is not None
                else None
            ),
        }
        if self.request_sink:
            self.request_sink(record)
        else:
            self.records.append(record)

    def _open_branches(self, play, completed_turn):
        for branch in play.graph.branches:
            if (
                branch.spawning_turn_index != completed_turn.key.turn_index
                or branch.branch_id in play.opened_branches
            ):
                continue
            play.opened_branches.add(branch.branch_id)
            for cid in branch.child_conversation_ids:
                cursor = play.streams[cid]
                cursor.active = True
                if not cursor.conversation.turns:
                    play.set_terminal(cursor)
                    continue
                if not cursor.terminal:
                    play.active_streams[cid] = cursor
                if self.phase == "pressure":
                    self._set_due(cursor, self.env.now)
                elif self.issue_mode == "recorded":
                    self._set_due(
                        cursor,
                        self._next_due(
                            play, cursor, cursor.conversation.turns[cursor.index]
                        ),
                    )
                else:
                    offset = max(
                        0.0,
                        cursor.conversation.turns[cursor.index].request.t
                        - completed_turn.request.end_s,
                    )
                    self._set_due(cursor, self.env.now + offset)

    def _recycle(self):
        for lane, play in enumerate(self.trees):
            if not play.finished() or not self.accepting:
                continue
            if (
                self.phase == "pressure"
                and self.pressure_sent[lane] >= self.warmup_quota
            ):
                continue
            graph = self.pool.next_graph()
            self.recycle_trace_ids[lane].append(graph.trace_id)
            self.trees[lane] = self._tree(graph, lane)

    def _issue_ready(self):
        if not self.accepting:
            return False
        self._recycle()
        issued = False
        limit = self.env.now + 1e-9
        for play in self.trees:
            # A ready stream has a due timer and no request in flight.
            if not any(
                cursor.in_flight is None
                and cursor.due is not None
                and cursor.due <= limit
                for cursor in play.active_streams.values()
            ):
                continue
            for cid, cursor in sorted(
                play.active_streams.items(), key=lambda item: item[1].ordinal
            ):
                if (
                    self.phase == "pressure"
                    and self.pressure_sent[play.lane] >= self.warmup_quota
                ):
                    break
                if not self._ready(play, cursor):
                    continue
                req = self._make_request(
                    play,
                    cursor.conversation.turns[cursor.index],
                    warmup=self.phase == "pressure",
                )
                if self.phase == "pressure":
                    self.pressure_sent[play.lane] += 1
                self._submit(req, play, cid)
                issued = True
        return issued

    def _future_timers(self, play):
        # ``active_streams`` holds only active, non-terminal cursors.
        limit = self.env.now + 1e-9
        return [
            cursor
            for cursor in play.active_streams.values()
            if cursor.in_flight is None
            and cursor.due is not None
            and cursor.due > limit
        ]

    def _idle_shifts(self):
        if (
            self.phase != "profiling"
            or not self.accepting
            or self.issue_mode == "recorded"
        ):
            return
        all_timers = []
        for play in self.trees:
            timers = self._future_timers(play)
            if (
                timers
                and play.in_flight == 0
                and play.idle_started_at is not None
                and self.tree_idle_cap is not None
                and (
                    play.idle_cap_expired
                    or self.env.now >= play.idle_started_at + self.tree_idle_cap
                )
            ):
                shift = min(c.due for c in timers) - self.env.now
                for cursor in timers:
                    cursor.due -= shift
                play.idle_cap_expired = True
                self.tree_idle_jumps += 1
                self.tree_idle_seconds_skipped += shift
                self._event("tree_idle_shift", lane=play.lane, skipped_s=shift)
                # The shift made some timers due now.
                timers = self._future_timers(play)
            all_timers.extend(timers)
        timers = all_timers
        if (
            timers
            and not self._in_flight
            and self._global_idle_started is not None
            and self.system_idle_cap is not None
            and self.env.now >= self._global_idle_started + self.system_idle_cap
        ):
            shift = min(c.due for c in timers) - self.env.now
            for cursor in timers:
                cursor.due -= shift
            self.system_idle_jumps += 1
            self.system_idle_seconds_skipped += shift
            self._event("system_idle_shift", skipped_s=shift)

    def _wake_at(self, boundary, timers_by_play):
        if not self.accepting:
            return boundary
        if self.issue_mode == "recorded":
            next_due = self._next_recorded_timer()
            return min(boundary, next_due) if next_due is not None else boundary
        candidates = [boundary]
        for play, timers in zip(self.trees, timers_by_play):
            if (
                self.phase == "pressure"
                and self.pressure_sent[play.lane] >= self.warmup_quota
            ):
                continue
            candidates.extend(c.due for c in timers)
            if (
                timers
                and self.phase == "profiling"
                and play.in_flight == 0
                and self.tree_idle_cap is not None
                and play.idle_started_at is not None
            ):
                candidates.append(
                    self.env.now
                    if play.idle_cap_expired
                    else max(self.env.now, play.idle_started_at + self.tree_idle_cap)
                )
        if (
            self.phase == "profiling"
            and not self._in_flight
            and any(timers_by_play)
            and self.system_idle_cap is not None
            and self._global_idle_started is not None
        ):
            candidates.append(
                max(self.env.now, self._global_idle_started + self.system_idle_cap)
            )
        return min(candidates)

    def _wait(self, boundary):
        (
            self.engine.raise_if_failed()
            if hasattr(self.engine, "raise_if_failed")
            else None
        )
        # Timers do not change while this method decides; collect them once.
        timers_by_play = (
            []
            if self.issue_mode == "recorded"
            else [self._future_timers(play) for play in self.trees]
        )
        has_future_timer = (
            self._next_recorded_timer() is not None
            if self.issue_mode == "recorded"
            else any(timers_by_play)
        )
        if self.accepting and not self._in_flight and not has_future_timer:
            raise SimulationStateError(
                "AgentX graph has no ready request, future timer, or in-flight predecessor"
            )
        wake_at = self._wake_at(boundary, timers_by_play)
        changed = self._changed
        events = [changed]
        failure_event = getattr(self.engine, "failure_event", None)
        if failure_event is not None:
            events.append(failure_event)
        if math.isfinite(wake_at):
            if (
                not self._cached_wake
                or self._cached_wake[0] != wake_at
                or self._cached_wake[1].triggered
            ):
                self._cached_wake = (
                    wake_at,
                    self.env.timeout(max(0, wake_at - self.env.now)),
                )
            events.append(self._cached_wake[1])
        elif not self._in_flight:
            raise SimulationStateError(
                "AgentX dependency graph stalled without an in-flight request or future timer"
            )
        yield simpy.events.AnyOf(self.env, events)

    def _primer(self, offset, play, key, epoch):
        yield self.env.timeout(offset)
        if epoch != self._epoch or not self.accepting:
            return
        turn = play.streams[key.conversation_id].conversation.turns[key.turn_index]
        req = self._make_request(play, turn, warmup=True)
        self._submit(req, play, primer=True)
        failure_event = getattr(self.engine, "failure_event", None)
        yield (
            simpy.events.AnyOf(self.env, [req.completion_event, failure_event])
            if failure_event is not None
            else req.completion_event
        )
        if hasattr(self.engine, "raise_if_failed"):
            self.engine.raise_if_failed()

    def _drain(self, deadline):
        while self._in_flight and self.env.now < deadline:
            yield from self._wait(deadline)
        if self._in_flight:
            pending = [entry[0] for entry in self._in_flight.values()]
            self.engine.cancel_requests(pending)
            end = self.env.now + self.cancellation_timeout
            while self._in_flight and self.env.now < end:
                yield from self._wait(end)
            self.cancel_unreturned += len(self._in_flight)
            for req, play, cid, primer in list(self._in_flight.values()):
                phase = "warmup" if req.measurement_phase == "warmup" else "profiling"
                self.phase_counts[phase]["cancelled"] += 1
                self.phase_counts[phase]["cancel_unreturned"] += 1
                self._record_terminal(req, play, "cancelled")
                if phase == "warmup" and not primer:
                    cursor = play.streams[cid]
                    if cursor.index < len(cursor.conversation.turns):
                        old_turn = cursor.conversation.turns[cursor.index]
                        play.completed.add(old_turn.key)
                        cursor.index += 1
                        cursor.in_flight = None
                        if cursor.index >= len(cursor.conversation.turns):
                            play.set_terminal(cursor)
                        play.in_flight -= 1
                        if cursor.conversation is play.graph.root:
                            self._open_branches(play, old_turn)
            self._in_flight.clear()

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
