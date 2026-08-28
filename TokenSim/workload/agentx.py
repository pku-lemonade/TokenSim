from __future__ import annotations

import json
import math
import random
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence, TYPE_CHECKING

import simpy

from TokenSim.errors import WorkloadValidationError
from TokenSim.llm.llm_request import Request

if TYPE_CHECKING:
    from TokenSim.llm.llm_engine import LLMEngine


_REQUEST_TYPES = {"n", "s"}
_EPSILON = 1e-6


@dataclass
class AgentXRequestSpec:
    t: float
    prefill_len: int
    decode_len: int
    hash_ids: Sequence[int]
    model: str
    api_time: float | None = None


@dataclass
class AgentXSubagentPlan:
    agent_id: str
    spawn_t: float
    end_t: float
    spawn_after: int
    join_before: int
    streams: list[list[AgentXRequestSpec]] = field(default_factory=list)


@dataclass
class AgentXTrace:
    trace_id: str
    block_size: int
    roots: list[AgentXRequestSpec]
    subagents: list[AgentXSubagentPlan]

    @property
    def request_count(self) -> int:
        return len(self.roots) + sum(
            len(stream) for plan in self.subagents for stream in plan.streams
        )

    def all_requests(self) -> Iterable[AgentXRequestSpec]:
        yield from self.roots
        for plan in self.subagents:
            for stream in plan.streams:
                yield from stream


def load_agentx_traces(
    dataset_path: str,
    *,
    trace_count: int | None = None,
    trace_skip_count: int = 0,
) -> list[AgentXTrace]:
    """Load SemiAnalysis WEKA JSONL traces without materializing the whole file."""
    path = Path(dataset_path)
    if not path.exists():
        raise WorkloadValidationError(f"AgentX dataset path not found: {dataset_path}")
    if trace_count is not None and trace_count <= 0:
        raise WorkloadValidationError("agentx_trace_count must be positive")
    if trace_skip_count < 0:
        raise WorkloadValidationError("dataset_skip_count cannot be negative")
    traces: list[AgentXTrace] = []
    for record_index, record in enumerate(_iter_trace_records(path)):
        if record_index < trace_skip_count:
            continue
        trace = _parse_trace(record, record_index)
        traces.append(trace)
        if trace_count is not None and len(traces) >= trace_count:
            break
    if not traces:
        raise WorkloadValidationError(f"no AgentX traces loaded from {dataset_path}")
    return traces


def _iter_trace_records(path: Path) -> Iterable[dict[str, Any]]:
    files = sorted(path.glob("*.json*")) if path.is_dir() else [path]
    if not files:
        raise WorkloadValidationError(f"no JSON trace files found in {path}")
    for file_path in files:
        with file_path.open("r") as file:
            first = file.read(1)
            file.seek(0)
            if first in {"[", "{"} and file_path.suffix == ".json":
                try:
                    payload = json.load(file)
                except json.JSONDecodeError as exc:
                    raise WorkloadValidationError(
                        f"invalid AgentX JSON in {file_path}: {exc}"
                    ) from exc
                records = payload if isinstance(payload, list) else [payload]
                for record in records:
                    if not isinstance(record, dict):
                        raise WorkloadValidationError(
                            f"AgentX records in {file_path} must be objects"
                        )
                    yield record
                continue
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise WorkloadValidationError(
                        f"invalid AgentX JSONL at {file_path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(record, dict):
                    raise WorkloadValidationError(
                        f"AgentX record at {file_path}:{line_number} must be an object"
                    )
                yield record


def _parse_trace(record: dict[str, Any], record_index: int) -> AgentXTrace:
    trace_id = str(record.get("id", f"trace-{record_index}"))
    block_size = _positive_int(record.get("block_size"), "block_size", trace_id)
    if record.get("hash_id_scope", "local") != "local":
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} uses unsupported non-local hash IDs"
        )
    entries = record.get("requests")
    if not isinstance(entries, list):
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} requests must be a list"
        )

    roots: list[AgentXRequestSpec] = []
    pending_subagents: list[tuple[dict[str, Any], int]] = []
    for entry_index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise WorkloadValidationError(
                f"AgentX trace {trace_id} request {entry_index} must be an object"
            )
        request_type = entry.get("type")
        if request_type in _REQUEST_TYPES:
            roots.append(_parse_request(entry, trace_id, entry_index))
        elif request_type == "subagent":
            if not roots:
                raise WorkloadValidationError(
                    f"AgentX trace {trace_id} has a subagent before its first root"
                )
            pending_subagents.append((entry, len(roots) - 1))
        else:
            raise WorkloadValidationError(
                f"AgentX trace {trace_id} request {entry_index} has unknown type {request_type!r}"
            )
    if not roots:
        raise WorkloadValidationError(f"AgentX trace {trace_id} has no root requests")

    subagents: list[AgentXSubagentPlan] = []
    for subagent_index, (entry, spawn_after) in enumerate(pending_subagents):
        spawn_t = _finite_nonnegative(entry.get("t", 0), "subagent t", trace_id)
        inner_rows = entry.get("requests", [])
        if not isinstance(inner_rows, list):
            raise WorkloadValidationError(
                f"AgentX trace {trace_id} subagent requests must be a list"
            )
        inner: list[AgentXRequestSpec] = []
        for inner_index, row in enumerate(inner_rows):
            spec = _parse_request(
                row, trace_id, f"subagent-{subagent_index}-{inner_index}"
            )
            if spec.t + _EPSILON < spawn_t:
                spec.t += spawn_t
            inner.append(spec)
        inner.sort(key=lambda spec: spec.t)
        duration_ms = entry.get("duration_ms")
        try:
            duration_value = float(duration_ms) if duration_ms is not None else None
        except (TypeError, ValueError):
            duration_value = None
        if duration_value is not None and math.isfinite(duration_value):
            end_t = spawn_t + max(0.0, duration_value / 1000.0)
        elif inner:
            end_t = max(spec.t + _api_duration(spec.api_time) for spec in inner)
        else:
            end_t = spawn_t
        join_before = next(
            (
                idx
                for idx, root in enumerate(roots)
                if idx > spawn_after and root.t >= end_t
            ),
            len(roots),
        )
        subagents.append(
            AgentXSubagentPlan(
                agent_id=str(entry.get("agent_id", f"subagent-{subagent_index}")),
                spawn_t=spawn_t,
                end_t=end_t,
                spawn_after=spawn_after,
                join_before=join_before,
                streams=_partition_streams(inner),
            )
        )
    return AgentXTrace(trace_id, block_size, roots, subagents)


def _parse_request(
    row: Any,
    trace_id: str,
    request_index: int | str,
) -> AgentXRequestSpec:
    if not isinstance(row, dict) or row.get("type") not in _REQUEST_TYPES:
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} request {request_index} must have type 'n' or 's'"
        )
    hashes = row.get("hash_ids", [])
    if not isinstance(hashes, list):
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} request {request_index} hash_ids must be a list"
        )
    try:
        hash_type = "q" if any(int(value) < 0 for value in hashes) else "Q"
        compact_hashes: Sequence[int] = array(
            hash_type, (int(value) for value in hashes)
        )
    except (OverflowError, TypeError, ValueError) as exc:
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} request {request_index} has invalid hash_ids"
        ) from exc
    api_time = row.get("api_time")
    try:
        api_time = float(api_time) if api_time is not None else None
    except (TypeError, ValueError):
        api_time = None
    if api_time is not None and (not math.isfinite(api_time) or api_time < 0):
        api_time = None
    try:
        decode_len = max(1, int(row.get("out", 1)))
    except (TypeError, ValueError) as exc:
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} request {request_index} has invalid out"
        ) from exc
    return AgentXRequestSpec(
        t=_finite_nonnegative(row.get("t"), "request t", trace_id),
        prefill_len=_positive_int(row.get("in"), "in", trace_id),
        decode_len=decode_len,
        hash_ids=compact_hashes,
        model=str(row.get("model", "unknown")),
        api_time=api_time,
    )


def _partition_streams(
    requests: list[AgentXRequestSpec],
) -> list[list[AgentXRequestSpec]]:
    """Recover overlapping subagent streams with hash-LCP affinity."""
    streams: list[list[AgentXRequestSpec]] = []
    for request in requests:
        best_stream: list[AgentXRequestSpec] | None = None
        best_lcp = -1
        for stream in streams:
            previous = stream[-1]
            previous_end = previous.t + _api_duration(previous.api_time)
            if previous_end > request.t + _EPSILON:
                continue
            lcp = _hash_lcp(previous.hash_ids, request.hash_ids)
            if lcp > best_lcp:
                best_lcp = lcp
                best_stream = stream
        if best_stream is None:
            streams.append([request])
        else:
            best_stream.append(request)
    return streams


class _SystemIdleCoordinator:
    """Shift pending replay timers together only while the whole engine is idle."""

    def __init__(self, env: simpy.Environment, idle_gap_cap: float | None) -> None:
        self.env = env
        self.idle_gap_cap = idle_gap_cap
        self.in_flight = 0
        self.idle_started_at: float | None = env.now
        self.pending: dict[int, tuple[float, simpy.Event]] = {}
        self._next_id = 0
        self._changed = env.event()
        env.process(self._run())

    def delay(self, seconds: float) -> simpy.Event:
        event = self.env.event()
        timer_id = self._next_id
        self._next_id += 1
        self.pending[timer_id] = (self.env.now + max(0.0, seconds), event)
        self._notify()
        return event

    def request_started(self) -> None:
        self.in_flight += 1
        self.idle_started_at = None
        self._notify()

    def request_finished(self) -> None:
        self.in_flight = max(0, self.in_flight - 1)
        if self.in_flight == 0:
            self.idle_started_at = self.env.now
        self._notify()

    def reset_idle_window(self) -> None:
        if self.in_flight == 0:
            self.idle_started_at = self.env.now
        self._notify()

    def _notify(self) -> None:
        if not self._changed.triggered:
            self._changed.succeed()
        self._changed = self.env.event()

    def _run(self):
        while True:
            if not self.pending:
                changed = self._changed
                yield changed
                continue

            earliest = min(target for target, _ in self.pending.values())
            wake_at = earliest
            if (
                self.idle_gap_cap is not None
                and self.in_flight == 0
                and self.idle_started_at is not None
            ):
                wake_at = min(wake_at, self.idle_started_at + self.idle_gap_cap)

            changed = self._changed
            timeout = self.env.timeout(max(0.0, wake_at - self.env.now))
            outcome = yield simpy.events.AnyOf(self.env, [timeout, changed])
            if changed in outcome:
                continue

            if (
                self.idle_gap_cap is not None
                and self.in_flight == 0
                and self.idle_started_at is not None
                and earliest > self.idle_started_at + self.idle_gap_cap + _EPSILON
            ):
                shift = earliest - (self.idle_started_at + self.idle_gap_cap)
                self.pending = {
                    timer_id: (target - shift, event)
                    for timer_id, (target, event) in self.pending.items()
                }

            ready = [
                timer_id
                for timer_id, (target, _) in self.pending.items()
                if target <= self.env.now + _EPSILON
            ]
            for timer_id in ready:
                _, event = self.pending.pop(timer_id)
                if not event.triggered:
                    event.succeed()


class AgentXReplay:
    """Closed-loop AgentX session-tree replay for a TokenSim engine."""

    def __init__(
        self,
        env: simpy.Environment,
        engine: LLMEngine,
        traces: list[AgentXTrace],
        *,
        block_size: int,
        concurrency: int,
        profile_duration: float | None = None,
        max_requests: int | None = None,
        system_idle_gap_cap: float | None = 10.0,
        warmup: bool = True,
        warmup_min_ratio: float = 0.0,
        warmup_max_ratio: float = 1.0,
        random_seed: int = 0,
        tqdm_submit_func=None,
    ) -> None:
        if concurrency <= 0:
            raise WorkloadValidationError("agentx_concurrency must be positive")
        if profile_duration is not None and profile_duration <= 0:
            raise WorkloadValidationError("agentx_profile_duration must be positive")
        if max_requests is not None and max_requests <= 0:
            raise WorkloadValidationError("agentx_max_requests must be positive")
        if system_idle_gap_cap is not None and (
            not math.isfinite(system_idle_gap_cap) or system_idle_gap_cap <= 0
        ):
            raise WorkloadValidationError("agentx_idle_gap_cap must be positive")
        if not 0 <= warmup_min_ratio <= warmup_max_ratio <= 1:
            raise WorkloadValidationError(
                "AgentX warmup ratios must satisfy 0 <= min <= max <= 1"
            )
        mismatched = [
            trace.trace_id for trace in traces if trace.block_size != block_size
        ]
        if mismatched:
            raise WorkloadValidationError(
                f"AgentX trace block_size must match --block_size={block_size}; "
                f"mismatched traces: {mismatched[:3]}"
            )
        self.env = env
        self.engine = engine
        self.traces = traces
        self.block_size = block_size
        self.concurrency = concurrency
        self.profile_duration = profile_duration
        self.max_requests = max_requests
        self.system_idle_gap_cap = system_idle_gap_cap
        self.warmup = warmup
        self.warmup_min_ratio = warmup_min_ratio
        self.warmup_max_ratio = warmup_max_ratio
        self.random = random.Random(random_seed)
        self.tqdm_submit_func = tqdm_submit_func
        self.requests: list[Request] = []
        self.warmup_requests: list[Request] = []
        self.profile_started_at: float | None = None
        self.profile_finished_at: float | None = None
        self.play_count = 0
        self._next_request_id = 0
        self._idle_coordinator = _SystemIdleCoordinator(env, system_idle_gap_cap)

    @property
    def profile_elapsed(self) -> float:
        if self.profile_started_at is None or self.profile_finished_at is None:
            return 0.0
        return max(0.0, self.profile_finished_at - self.profile_started_at)

    def capacity_request(self) -> Request:
        specs = [spec for trace in self.traces for spec in trace.all_requests()]
        max_prefill = max(spec.prefill_len for spec in specs)
        max_context = max(spec.prefill_len + spec.decode_len for spec in specs)
        return Request(
            id=-1,
            prefill_len=max_prefill,
            decode_len=max(1, max_context - max_prefill),
            block_size=self.block_size,
        )

    def run(self):
        lane_cutoffs: list[float | None] = []
        if self.warmup:
            warmup_processes = []
            for lane in range(self.concurrency):
                trace = self.traces[lane % len(self.traces)]
                cutoff = self._warmup_cutoff(trace)
                lane_cutoffs.append(cutoff)
                warmup_processes.append(
                    self.env.process(self._warm_lane(lane, trace, cutoff))
                )
            if warmup_processes:
                yield simpy.events.AllOf(self.env, warmup_processes)
            while self._pending_async_jobs() > 0:
                yield self.env.timeout(0.001)
            self.engine.reset_profile_stats()
        else:
            lane_cutoffs = [None] * self.concurrency

        self.profile_started_at = self.env.now
        self._idle_coordinator.reset_idle_window()
        lanes = [
            self.env.process(self._run_lane(lane, lane_cutoffs[lane]))
            for lane in range(self.concurrency)
        ]
        if lanes:
            lanes_done = simpy.events.AllOf(self.env, lanes)
            if self.profile_duration is not None:
                deadline = self.profile_started_at + self.profile_duration
                deadline_event = self.env.timeout(self.profile_duration)
                outcome = yield simpy.events.AnyOf(
                    self.env,
                    [lanes_done, deadline_event],
                )
                if deadline_event in outcome:
                    # Stop at the exact measurement deadline. Benchmark.main
                    # runs until this replay process, so post-deadline engine
                    # events cannot leak into metrics.
                    self.profile_finished_at = deadline
                    self.requests = [
                        request
                        for request in self.requests
                        if request.completed_at is not None
                        and request.completed_at <= deadline + _EPSILON
                    ]
                    return
                else:
                    self.profile_finished_at = self.env.now
            else:
                yield lanes_done
                self.profile_finished_at = self.env.now
        else:
            self.profile_finished_at = self.env.now

    def _run_lane(self, lane: int, initial_cutoff: float | None):
        play_id = 0
        while True:
            if play_id > 0 and self.profile_duration is None:
                break
            if self._profile_limit_reached():
                break
            trace = self.traces[(lane + play_id * self.concurrency) % len(self.traces)]
            self.play_count += 1
            yield self.env.process(
                self._run_trace(
                    lane,
                    play_id,
                    trace,
                    cutoff=initial_cutoff if play_id == 0 else None,
                )
            )
            play_id += 1

    def _run_trace(
        self,
        lane: int,
        play_id: int,
        trace: AgentXTrace,
        *,
        cutoff: float | None,
    ):
        start_root = 0
        if cutoff is not None:
            start_root = next(
                (idx for idx, root in enumerate(trace.roots) if root.t > cutoff),
                len(trace.roots),
            )
        branch_events: dict[int, list[simpy.Event]] = {}

        if cutoff is not None:
            for plan in trace.subagents:
                if plan.end_t <= cutoff or plan.spawn_after >= start_root:
                    continue
                plan_events: list[simpy.Event] = [
                    self._delay(max(0.0, plan.end_t - cutoff))
                ]
                for stream_index, stream in enumerate(plan.streams):
                    first = next(
                        (idx for idx, spec in enumerate(stream) if spec.t > cutoff),
                        len(stream),
                    )
                    if first >= len(stream):
                        continue
                    event = self.env.process(
                        self._run_child_stream(
                            lane,
                            play_id,
                            trace,
                            plan,
                            stream_index,
                            stream[first:],
                            ready_from=cutoff,
                        )
                    )
                    plan_events.append(event)
                branch_events.setdefault(plan.join_before, []).extend(plan_events)

        ready_event = self._delay(
            max(0.0, trace.roots[start_root].t - cutoff)
            if cutoff is not None and start_root < len(trace.roots)
            else 0.0
        )
        for root_index in range(start_root, len(trace.roots)):
            dependencies: list[simpy.Event] = [ready_event]
            dependencies.extend(branch_events.get(root_index, []))
            yield simpy.events.AllOf(self.env, dependencies)
            root = trace.roots[root_index]
            completed = yield self.env.process(
                self._submit(lane, play_id, trace, "root", root_index, root)
            )
            if not completed:
                break
            for plan in trace.subagents:
                if plan.spawn_after != root_index:
                    continue
                if cutoff is not None and plan.spawn_t <= cutoff:
                    continue
                ready_from = root.t + _api_duration(root.api_time)
                plan_events: list[simpy.Event] = [
                    self._delay(max(0.0, plan.end_t - ready_from))
                ]
                for stream_index, stream in enumerate(plan.streams):
                    event = self.env.process(
                        self._run_child_stream(
                            lane,
                            play_id,
                            trace,
                            plan,
                            stream_index,
                            stream,
                            ready_from=ready_from,
                        )
                    )
                    plan_events.append(event)
                branch_events.setdefault(plan.join_before, []).extend(plan_events)
            if root_index + 1 < len(trace.roots):
                next_root = trace.roots[root_index + 1]
                ready_event = self._delay(_end_to_start_delay(root, next_root))

        remaining = [event for events in branch_events.values() for event in events]
        if remaining:
            yield simpy.events.AllOf(self.env, remaining)

    def _run_child_stream(
        self,
        lane: int,
        play_id: int,
        trace: AgentXTrace,
        plan: AgentXSubagentPlan,
        stream_index: int,
        stream: list[AgentXRequestSpec],
        *,
        ready_from: float,
    ):
        if not stream:
            yield self._delay(max(0.0, plan.end_t - ready_from))
            return
        yield self._delay(max(0.0, stream[0].t - ready_from))
        previous: AgentXRequestSpec | None = None
        for turn_index, spec in enumerate(stream):
            if previous is not None:
                yield self._delay(_end_to_start_delay(previous, spec))
            completed = yield self.env.process(
                self._submit(
                    lane,
                    play_id,
                    trace,
                    f"{plan.agent_id}:{stream_index}",
                    turn_index,
                    spec,
                )
            )
            if not completed:
                return
            previous = spec
        last = stream[-1]
        yield self._delay(
            max(0.0, plan.end_t - (last.t + _api_duration(last.api_time)))
        )

    def _submit(
        self,
        lane: int,
        play_id: int,
        trace: AgentXTrace,
        stream_id: str,
        turn_index: int,
        spec: AgentXRequestSpec,
    ):
        if self._profile_limit_reached():
            return False
        request = self._make_request(
            lane, play_id, trace, stream_id, turn_index, spec, warmup=False
        )
        self.requests.append(request)
        request.arrive(self.env)
        self._idle_coordinator.request_started()
        self.engine.add_requests([request])
        yield request.completion_event
        self._idle_coordinator.request_finished()
        yield self.env.timeout(0)
        request.release_logical_blocks()
        return True

    def _warm_lane(self, lane: int, trace: AgentXTrace, cutoff: float):
        specs: list[tuple[str, int, AgentXRequestSpec]] = []
        root_candidates = [
            (idx, request)
            for idx, request in enumerate(trace.roots)
            if request.t <= cutoff
        ]
        if root_candidates:
            idx, request = root_candidates[-1]
            specs.append(("root", idx, request))
        for plan in trace.subagents:
            if not (plan.spawn_t <= cutoff < plan.end_t):
                continue
            for stream_index, stream in enumerate(plan.streams):
                candidates = [
                    (idx, request)
                    for idx, request in enumerate(stream)
                    if request.t <= cutoff
                ]
                if candidates:
                    idx, request = candidates[-1]
                    specs.append((f"{plan.agent_id}:{stream_index}", idx, request))
        events = []
        for stream_id, turn_index, spec in specs:
            request = self._make_request(
                lane, 0, trace, stream_id, turn_index, spec, warmup=True
            )
            self.warmup_requests.append(request)
            request.arrive(self.env)
            self.engine.add_requests([request])
            events.append(request.completion_event)
        if events:
            yield simpy.events.AllOf(self.env, events)
            yield self.env.timeout(0)
            for request in self.warmup_requests:
                if request.chat_id == f"agentx-lane-{lane}":
                    request.release_logical_blocks()

    def _make_request(
        self,
        lane: int,
        play_id: int,
        trace: AgentXTrace,
        stream_id: str,
        turn_index: int,
        spec: AgentXRequestSpec,
        *,
        warmup: bool,
    ) -> Request:
        request_id = self._next_request_id
        self._next_request_id += 1
        request = Request(
            id=request_id,
            prefill_len=spec.prefill_len,
            decode_len=1 if warmup else spec.decode_len,
            block_size=self.block_size,
            chat_id=f"agentx-lane-{lane}",
            parent_chat_id=None if stream_id == "root" else trace.trace_id,
            turn=turn_index,
            hash_ids=spec.hash_ids,
            cache_salt=f"agentx-play-{lane}-{play_id}",
            reuse_group=trace.trace_id,
            measurement_phase="warmup" if warmup else "profile",
            record_timing=not warmup,
        )
        request.tqdm_submit_func = None if warmup else self.tqdm_submit_func
        request.agentx_trace_id = trace.trace_id
        request.agentx_play_id = play_id
        request.agentx_stream_id = stream_id
        request.agentx_recorded_model = spec.model
        return request

    def _warmup_cutoff(self, trace: AgentXTrace) -> float:
        if len(trace.roots) == 1:
            return trace.roots[0].t - _EPSILON
        start = trace.roots[0].t
        end = trace.roots[-2].t
        ratio = self.random.uniform(self.warmup_min_ratio, self.warmup_max_ratio)
        return start + (end - start) * ratio

    def _profile_limit_reached(self) -> bool:
        if self.max_requests is not None and len(self.requests) >= self.max_requests:
            return True
        return bool(
            self.profile_duration is not None
            and self.profile_started_at is not None
            and self.env.now - self.profile_started_at >= self.profile_duration
        )

    def _pending_async_jobs(self) -> int:
        seen: set[int] = set()
        pending = 0
        for worker in getattr(self.engine, "workers", []):
            connectors = [getattr(worker, "connector", None)]
            while connectors:
                connector = connectors.pop()
                if connector is None:
                    continue
                connectors.extend(getattr(connector, "children", []))
                stats = getattr(connector, "mooncake_stats", None)
                if stats is None or id(stats) in seen:
                    continue
                seen.add(id(stats))
                pending += int(getattr(stats, "pending_async_jobs", 0))
        return pending

    def _delay(self, seconds: float) -> simpy.Event:
        return self._idle_coordinator.delay(seconds)


def _end_to_start_delay(
    previous: AgentXRequestSpec,
    current: AgentXRequestSpec,
) -> float:
    return max(0.0, current.t - (previous.t + _api_duration(previous.api_time)))


def _api_duration(value: float | None) -> float:
    return value if value is not None and math.isfinite(value) and value > 0 else 0.0


def _hash_lcp(left: Sequence[int], right: Sequence[int]) -> int:
    common = 0
    for left_value, right_value in zip(left, right):
        if left_value != right_value:
            break
        common += 1
    return common


def _positive_int(value: Any, field: str, trace_id: str) -> int:
    try:
        numeric = int(value)
    except (TypeError, ValueError) as exc:
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} field {field} must be an integer"
        ) from exc
    if numeric <= 0:
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} field {field} must be positive"
        )
    return numeric


def _finite_nonnegative(value: Any, field: str, trace_id: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} field {field} must be numeric"
        ) from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise WorkloadValidationError(
            f"AgentX trace {trace_id} field {field} must be finite and non-negative"
        )
    return numeric
