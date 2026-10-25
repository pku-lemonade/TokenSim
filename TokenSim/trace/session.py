"""Engine-owned recording sessions installed only when tracing is enabled."""

from dataclasses import replace
from itertools import chain
import math
from pathlib import Path

from TokenSim.errors import SimulationStateError
from TokenSim.config.cache_config import stage_layer_range
from TokenSim.trace.archive import ArchivedTraceRecorder
from TokenSim.trace.composition import Segment, expand_composition
from TokenSim.trace.events import TraceEvent
from TokenSim.trace.execution import observe_service
from TokenSim.trace.formats import write_trace
from TokenSim.trace.transient import ValidationTraceRecorder
from TokenSim.trace.transfers import TransferObserver


class TraceSession:
    def __init__(self, engine):
        self.engine = engine
        self.options = engine.config.trace
        self.recorder = (
            ValidationTraceRecorder()
            if self.options.mode == "validate"
            else ArchivedTraceRecorder()
        )
        self.pending = {}
        self.request_tails = {}
        self.iteration = 0
        self.run_phase = "profiling"
        self.completed = 0
        self.active_budgets = {}
        self.transfers = TransferObserver(self)
        for worker in engine.request_workers:
            self._attach_worker(worker)
        engine.attach_trace(self)
        original = engine.run_complete_compute

        def compute(worker_id, latency):
            info = self.pending.get(worker_id)
            if info is not None:
                info["submitted"] = True
            try:
                yield from observe_service(
                    original(worker_id, latency),
                    lambda: self.mark_started(worker_id, engine.env.now, latency),
                )
            except BaseException:
                if info is not None:
                    info["stopped_at"] = engine.env.now
                raise
            self.finish(worker_id, engine.env.now - latency, latency)

        engine.run_complete_compute = compute

    def _attach_worker(self, worker):
        backend = worker.latency_backend
        captured = []
        communications = []
        active_communications = []
        original_record = worker.parallel_communicator._record

        def record(kind, target, bytes_, latency, group, estimate):
            active_communications.append(
                dict(
                    kind=kind,
                    bytes=bytes_,
                    seconds=latency,
                    group=tuple(group),
                    estimate=estimate,
                )
            )
            return original_record(kind, target, bytes_, latency, group, estimate)

        worker.parallel_communicator._record = record

        def attach(current):
            original_comm = current._communication

            def communication(*args, **kwargs):
                active_communications.clear()
                result = original_comm(*args, **kwargs)
                communications[:] = active_communications
                return result

            current._communication = communication
            current.trace_composition = lambda estimator, values: captured.extend(
                expand_composition(estimator, values, communications)
            )
            if current.native_attention:
                current.native_attention.trace_layer_costs = {}
            for child in current._layer_backends:
                owned = stage_layer_range(
                    current.model.num_layers,
                    current.parallel_config.pipeline_parallel_size,
                    current.rank_info.pp_rank,
                    current.pipeline_layer_counts,
                )
                child.trace_layer_indices = tuple(
                    layer
                    for layer in owned
                    if current.model.attention_window(layer)
                    == child.model.sliding_window
                    and current.model.attention_geometry(layer)
                    == child.model.attention_geometry(layer)
                )
                attach(child)

        attach(backend)
        original_batch = worker.dynamic_batch

        def batch(requests):
            captured.clear()
            if backend.native_attention:
                backend.native_attention.trace_layer_costs.clear()
            phases = {
                "prefill" if req.is_context_build else "decode" for req in requests
            }
            info = dict(
                ready=worker.env.now,
                requests=tuple(req.id for req in requests),
                tokens=sum(req.query_tokens for req in requests),
                phase=next(iter(phases)) if len(phases) == 1 else "mixed",
                run_phase=self.run_phase,
            )
            info["iteration"] = self.iteration
            self.iteration += 1
            duration = original_batch(requests)
            # Cost grouping must not reorder mixed geometries on the timeline.
            leading = {"step_overhead", "embedding_lookup", "embedding_scale"}
            info["segments"] = sorted(
                captured,
                key=lambda s: (
                    (
                        0
                        if s.layer < 0
                        and (
                            s.operation in leading
                            or s.details.get("collective_scope") == "tp_embedding"
                        )
                        else 2 if s.layer < 0 else 1
                    ),
                    s.layer,
                ),
            )
            info["duration"] = duration
            self.pending[worker.id] = info
            return duration

        worker.dynamic_batch = batch
        original_complete = worker.complete_step

        def complete(requests, latency):
            # Lockstep execution bypasses run_complete_compute.
            self.finish(worker.id, worker.env.now - latency, latency)
            result = original_complete(requests, latency)
            if result > 0:
                plans = tuple(worker.connector._metadata.saves)
                own = getattr(worker.connector, "_own_plans", None)
                if own is not None:
                    plans = tuple(own(plans))
                if plans:
                    self._record_load(worker, plans, result, kind="save")
            for request in requests:
                if request.is_done:
                    self.request_tails.pop(request.id, None)
            return result

        worker.complete_step = complete
        self.transfers.attach(worker)

    def record_connector_load(self, worker, duration):
        plans = tuple(worker.connector._metadata.loads)
        own = getattr(worker.connector, "_own_plans", None)
        if own is not None:
            plans = tuple(own(plans))
        if plans:
            self._record_load(worker, plans, duration)

    def _record_load(self, worker, plans, duration, *, kind="load"):
        now = worker.env.now
        ids = tuple(request_id for plan in plans for request_id in plan.request_ids)
        dependencies = tuple(
            dict.fromkeys(
                self.request_tails[r][0] for r in ids if r in self.request_tails
            )
        )
        node = f"transfer:{self.iteration}"
        self.active_budgets = {
            key: end for key, end in self.active_budgets.items() if end > now
        }
        self.active_budgets[node] = now + duration
        self.recorder.record(
            TraceEvent(
                iteration_id=self.iteration,
                block_id=0,
                layer_id=-1,
                phase="prefill",
                event_type="transfer",
                resource_id=f"kv:{kind}:{worker.id}",
                ready_at=now,
                start_at=now,
                end_at=now + duration,
                queue_duration=0,
                service_duration=duration,
                request_count=len(ids),
                token_count=0,
                source_id=worker.connector.name,
                node_id=node,
                predecessor_ids=dependencies,
                worker_id=worker.id,
                operation="kv_transfer" if kind == "load" else "kv_save",
                run_phase=self.run_phase,
                payload_bytes=sum(plan.bytes for plan in plans),
                details={
                    "request_ids": ids,
                    "direction": kind,
                    "granularity": "module",
                    "cost_scope": "blocking_connector_invocation",
                    "run_phase": self.run_phase,
                    "source_workers": tuple(plan.source_worker_id for plan in plans),
                },
            )
        )
        self._complete(self.iteration)
        self.iteration += 1
        for request_id in ids:
            self.request_tails[request_id] = node, now + duration

    def mark_started(self, worker_id, start, elapsed):
        info = self.pending.get(worker_id)
        if info is not None:
            info.update(start=start, elapsed=elapsed, submitted=True)

    def _forward_events(self, worker_id, info, start, elapsed):
        worker = self.engine.workers[worker_id]
        requests = info["requests"]
        predecessors = tuple(
            dict.fromkeys(
                self.request_tails[r][0] for r in requests if r in self.request_tails
            )
        )
        segments = list(info["segments"])
        if elapsed > info["duration"] + 1e-12:
            segments.append(
                Segment(
                    "rank_synchronize", elapsed - info["duration"], -1, "synchronize"
                )
            )
        participants = worker.parallel_communicator.tp_group() or (worker_id,)
        for rank_index, physical_worker in enumerate(participants):
            time, offset, previous = start, 0.0, predecessors
            for index, segment in enumerate(segments):
                node = f"forward:{worker_id}:{info['iteration']}:rank:{physical_worker}:{index}"
                details = dict(
                    segment.details,
                    request_ids=requests,
                    run_phase=info["run_phase"],
                    placement_assumption=(
                        "symmetric_tp" if len(participants) > 1 else "local"
                    ),
                )
                yield TraceEvent(
                    iteration_id=info["iteration"],
                    block_id=worker_id,
                    layer_id=segment.layer,
                    phase=info["phase"],
                    event_type=segment.category,
                    resource_id=f"worker:{physical_worker}",
                    ready_at=info["ready"] if index == 0 else time,
                    start_at=time,
                    end_at=start + (offset + segment.seconds),
                    queue_duration=time - info["ready"] if index == 0 else 0,
                    service_duration=segment.seconds,
                    request_count=len(requests),
                    token_count=info["tokens"],
                    source_id=segment.source,
                    match_type=segment.match,
                    operation=segment.operation,
                    node_id=node,
                    predecessor_ids=previous,
                    worker_id=physical_worker,
                    tp_rank=rank_index,
                    dp_rank=worker.dp_rank,
                    pp_rank=worker.pp_rank,
                    run_phase=info["run_phase"],
                    details=details,
                )
                offset += segment.seconds
                time = start + offset
                previous = (node,)
            if abs(time - (start + elapsed)) > max(1e-10, 4 * math.ulp(time)):
                raise SimulationStateError(
                    "expanded trace must end at the executed forward boundary"
                )

    def finish(self, worker_id, start, elapsed):
        info = self.pending.pop(worker_id, None)
        if info is None:
            return
        external = dict(self.request_tails.values())
        terminal = None
        for event in self._forward_events(worker_id, info, start, elapsed):
            self.recorder.record(event)
            if event.worker_id == worker_id:
                terminal = event.node_id, event.end_at
        if terminal is not None:
            for request_id in info["requests"]:
                self.request_tails[request_id] = terminal
        self._complete(info["iteration"], external)

    def _pending_events(self):
        for worker_id, info in self.pending.items():
            if "start" not in info:
                continue
            cutoff = min(
                self.engine.env.now, info.get("stopped_at", self.engine.env.now)
            )
            for event in self._forward_events(
                worker_id, info, info["start"], info["elapsed"]
            ):
                if event.start_at < cutoff:
                    end = min(event.end_at, cutoff)
                    yield replace(
                        event,
                        end_at=end,
                        service_duration=end - event.start_at,
                        incomplete=True,
                        details=dict(event.details, incomplete=True),
                    )

    def _complete(self, iteration, external=None):
        self.recorder.completed_predecessors = (
            dict(self.request_tails.values()) if external is None else external
        )
        if self.options.validate:
            self.recorder.validate_invariants(iteration)
        self.recorder.complete_iteration(iteration)
        self.completed += 1

    def events(self):
        from TokenSim.trace.views import expand_af_events

        af = self.engine.af_executor
        histories = [
            self.recorder.events,
            self._pending_events(),
            self.transfers.partial_events(),
        ]
        if af is not None:
            histories.append(expand_af_events(af.trace.events))
        cutoff = self.engine.env.now
        tolerance = max(1e-12, 4 * math.ulp(cutoff))
        for event in chain.from_iterable(histories):
            if event.start_at >= cutoff and event.end_at > cutoff:
                continue
            if event.end_at > cutoff + tolerance:
                yield replace(
                    event,
                    end_at=cutoff,
                    service_duration=cutoff - event.start_at,
                    incomplete=True,
                    details=dict(event.details, incomplete=True),
                )
            elif event.end_at > cutoff:
                yield replace(
                    event,
                    end_at=cutoff,
                    service_duration=max(0, cutoff - event.start_at),
                )
            else:
                yield event

    def export(self, directory, *, termination="completed", result_name=None):
        active_budgets = sum(
            end > self.engine.env.now for end in self.active_budgets.values()
        )
        descriptor = dict(
            mode=self.options.mode,
            schema_version=2,
            completed_scopes=self.completed - active_budgets,
            incomplete=any(info.get("submitted") for info in self.pending.values())
            or bool(self.transfers.pending)
            or bool(active_budgets)
            or termination != "completed",
        )
        if self.options.mode == "full":
            path = (
                Path(self.options.output)
                if self.options.output
                else Path(directory)
                / (
                    f"{Path(result_name).stem}.trace.json"
                    if result_name
                    else "operator_trace.json"
                )
            )
            path = write_trace(
                path,
                self.events(),
                metadata=dict(
                    schema_version=2,
                    termination_reason=termination,
                    incomplete=descriptor["incomplete"],
                    clock="simulation",
                    internal_time_unit="seconds",
                    model_id=self.engine.model_spec.model_id,
                    dtype=self.engine.model_spec.activation_dtype,
                    hidden_size=self.engine.model_spec.hidden_size,
                    num_layers=self.engine.model_spec.num_layers,
                    num_attention_heads=self.engine.model_spec.num_attention_heads,
                    tensor_parallel_size=self.engine.parallel_config.tensor_parallel_size,
                ),
                trace_config=self.options,
                trace_format=self.options.format,
            )
            descriptor["path"] = str(path)
        return descriptor
