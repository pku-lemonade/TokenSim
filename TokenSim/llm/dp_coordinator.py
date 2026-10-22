"""DP lockstep and busy-wave state derived from vLLM 30f5c01d4ebb.

GPU passes synchronize every iteration. The CPU finished-state all-reduce is
checked on step 1 and every dp_sync_interval (16 in the current source).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import simpy

from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.llm.llm_request import Request
from TokenSim.llm.forward import ForwardPass, RankShape


class StepBoundary(simpy.Event):
    """Settle all normal callbacks at this timestamp before scheduling a pass."""

    def __init__(self, env):
        super().__init__(env)
        self._ok, self._value = True, None
        env.schedule(self, priority=simpy.events.NORMAL + 1)


@dataclass
class DPStepStats:
    steps: int = 0
    waves_started: int = 0
    waves_finished: int = 0
    finish_consensus_checks: int = 0
    elapsed_step_seconds: float = 0.0
    real_batches: dict = field(default_factory=lambda: defaultdict(int))
    dummy_batches: dict = field(default_factory=lambda: defaultdict(int))
    local_compute_seconds: dict = field(default_factory=lambda: defaultdict(float))
    synchronization_wait_seconds: dict = field(
        default_factory=lambda: defaultdict(float)
    )


class DPStepCoordinator:
    def __init__(self, env, engine, workers, *, sync_interval=16):
        if sync_interval < 1 or any(w.role != "hybrid" for w in workers):
            raise ConfigurationError(
                "DP lockstep needs hybrid leaders and a positive finished-state interval"
            )
        self.env, self.engine, self.workers = env, engine, workers
        self.sync_interval = sync_interval
        self.worker_ids = {w.id for w in workers}
        self.busy = False
        self.step_counter = self.wave = 0
        self.wave_running = False
        self.stats = DPStepStats()
        self.pending_cancels = defaultdict(dict)
        self.changed = env.event()
        self.action = env.process(self.run())
        self.dummies = {}
        for worker in workers:
            dummy = Request(
                -1000000 - worker.id, 1, 1, worker.block_size, record_timing=False
            )
            dummy.generation_idx = 1
            dummy.num_computed_tokens = 1
            # GPUWorker.execute_dummy_batch uses uniform_decode_query_len=1+K.
            dummy.scheduled_tokens = 1 + (
                worker.cache_config.hybrid_layout.speculative_tokens
                if worker.cache_config.hybrid_layout
                else 0
            )
            dummy.scheduled_query_tokens = dummy.scheduled_tokens
            self.dummies[worker.id] = dummy

    def notify(self):
        if not self.changed.triggered:
            self.changed.succeed()
        self.changed = self.env.event()

    def has_work(self):
        return any(
            w.scheduler.running
            or w.scheduler.waiting
            or getattr(w.scheduler, "loading", {})
            or getattr(w.scheduler, "loaded_waiting", [])
            for w in self.workers
        )

    def cancel_at_boundary(self, worker, request):
        if worker.scheduler.cancel_request(request):
            request.acknowledge_cancel(self.env, worker.id)
        else:
            self.env.process(worker._wait_for_cancel_release(request))

    def consume_messages(self, worker):
        # Local import avoids the engine/coordinator type-definition cycle.
        from TokenSim.llm.llm_engine import (
            Task,
            WorkerStatus,
            _merge_connector_metadata,
        )

        while True:
            msg = yield worker.msg_queue.get()
            try:
                if msg.task == Task.ADD:
                    if msg.metadata is not None:
                        worker.pending_connector_metadata = _merge_connector_metadata(
                            msg.metadata, worker.pending_connector_metadata
                        )
                    worker.scheduler.add_requests(msg.requests or [])
                    worker.status = (
                        WorkerStatus.RUNNING
                        if (worker.scheduler.running or worker.scheduler.waiting)
                        else WorkerStatus.WAITING
                    )
                    self.notify()
                elif msg.task == Task.CANCEL:
                    for request in msg.requests or []:
                        if self.busy:
                            self.pending_cancels[worker.id][request.id] = request
                        else:
                            self.cancel_at_boundary(worker, request)
                    self.notify()
                elif msg.task == Task.STOP:
                    worker.status = WorkerStatus.STOPPED
                    self.notify()
                    return
                elif msg.task != Task.STEP:
                    raise SimulationStateError("unknown task in DP input consumer")
            except Exception as error:
                self.engine.record_worker_failure(worker.id, error)
                self.notify()
                return

    def run(self):
        from TokenSim.llm.llm_engine import WorkerStatus

        while True:
            yield StepBoundary(self.env)
            if self.engine.failure is not None:
                return
            if not self.wave_running:
                if not self.has_work():
                    yield self.changed
                    continue
                self.wave_running = True
                self.step_counter = 0
                self.stats.waves_started += 1
            self.busy = True
            batches, estimates = {}, {}
            try:
                save_latencies = []
                load_latencies = []
                for worker in self.workers:
                    worker.dp_step_counter = self.step_counter
                    running, load_latency = worker.prepare_step()
                    load_latencies.append(load_latency)
                    batches[worker.id] = running
                load_duration = max(load_latencies, default=0.0)
                if load_duration:
                    if self.engine.trace_session is not None:
                        for worker, latency in zip(self.workers, load_latencies):
                            if latency:
                                self.engine.trace_session.record_connector_load(
                                    worker, latency
                                )
                    yield self.env.timeout(load_duration)
                # Kept visible for native MegaMoE estimators added later:
                # every rank's routes belong to this same global iteration.
                token_counts = {
                    w.dp_rank: sum(r.query_tokens for r in batches[w.id])
                    for w in self.workers
                }
                shapes = {}
                for worker in self.workers:
                    batch = batches[worker.id]
                    widths = {r.query_tokens for r in batch}
                    context = any(not r.decode_graph_eligible for r in batch)
                    dummy_width = self.dummies[worker.id].query_tokens
                    shapes[worker.dp_rank] = RankShape(
                        token_counts[worker.dp_rank] if batch else dummy_width,
                        (
                            (
                                next(iter(widths))
                                if len(widths) == 1 and not context
                                else None
                            )
                            if batch
                            else dummy_width
                        ),
                        context,
                        len(batch) if batch else 1,
                        token_counts[worker.dp_rank],
                        len(batch),
                    )
                moe_step = ForwardPass(shapes)
                for worker in self.workers:
                    worker.dp_step_token_counts = token_counts
                    if hasattr(worker.latency_backend, "set_moe_step"):
                        worker.latency_backend.set_moe_step(moe_step)
                    batch = batches[worker.id]
                    estimates[worker.id] = worker.dynamic_batch(
                        batch or [self.dummies[worker.id]]
                    )
                    if batch:
                        self.stats.real_batches[worker.dp_rank] += 1
                    else:
                        self.stats.dummy_batches[worker.dp_rank] += 1
                duration = max(estimates.values())
                if duration <= 0:
                    raise SimulationStateError(
                        "DP passes need positive modeled duration to advance the busy wave"
                    )
                if self.engine.trace_session is not None:
                    for worker in self.workers:
                        self.engine.trace_session.mark_started(
                            worker.id, self.env.now, duration
                        )
                yield self.env.timeout(duration)
                if self.engine.trace_session is not None:
                    for worker in self.workers:
                        self.engine.trace_session.finish(
                            worker.id, self.env.now - duration, duration
                        )
                for worker in self.workers:
                    self.stats.local_compute_seconds[worker.dp_rank] += estimates[
                        worker.id
                    ]
                    self.stats.synchronization_wait_seconds[worker.dp_rank] += (
                        duration - estimates[worker.id]
                    )
                    batch = batches[worker.id]
                    if batch:
                        save_latency = worker.complete_step(batch, duration)
                        save_latencies.append(save_latency)
                        worker.finalize_connector_step()
                    for request in self.pending_cancels.pop(worker.id, {}).values():
                        self.cancel_at_boundary(worker, request)
                    worker.status = (
                        WorkerStatus.RUNNING
                        if (worker.scheduler.running or worker.scheduler.waiting)
                        else WorkerStatus.WAITING
                    )
                # Synchronous external stores (for example the Mooncake
                # comparison backend) report a blocking save after compute.
                # Keep the DP wave in lockstep by charging the slowest save
                # once, while native asynchronous connectors return zero.
                save_duration = max(save_latencies, default=0.0)
                if save_duration:
                    yield self.env.timeout(save_duration)
                self.stats.steps += 1
                self.stats.elapsed_step_seconds += duration
                self.step_counter += 1
                self.busy = False
                yield StepBoundary(self.env)
                if (
                    self.step_counter == 1
                    or self.step_counter % self.sync_interval == 0
                ):
                    self.stats.finish_consensus_checks += 1
                    if not self.has_work():
                        self.wave_running = False
                        self.wave += 1
                        self.step_counter = 0
                        self.stats.waves_finished += 1
                for worker in self.workers:
                    worker.dp_step_counter = self.step_counter
            except Exception as error:
                self.busy = False
                self.engine.record_worker_failure(worker.id, error)
                return

    def reset_stats(self):
        self.stats = DPStepStats()

    def describe(self):
        return {
            "mode": "lockstep",
            "finished_state_sync_interval": self.sync_interval,
            "step_counter": self.step_counter,
            "wave": self.wave,
            "steps": self.stats.steps,
            "waves_started": self.stats.waves_started,
            "waves_finished": self.stats.waves_finished,
            "finish_consensus_checks": self.stats.finish_consensus_checks,
            "elapsed_step_seconds": self.stats.elapsed_step_seconds,
            "real_batches_per_rank": dict(self.stats.real_batches),
            "dummy_batches_per_rank": dict(self.stats.dummy_batches),
            "local_compute_seconds_per_rank": dict(self.stats.local_compute_seconds),
            "synchronization_wait_seconds_per_rank": dict(
                self.stats.synchronization_wait_seconds
            ),
            "control_allreduce": "two CPU int32 values; timing belongs to future framework overhead calibration",
        }
