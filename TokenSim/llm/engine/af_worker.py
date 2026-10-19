"""AF physical-worker bindings and per-iteration compute execution."""

from __future__ import annotations

import simpy
from TokenSim.af.runtime.records import AFRankBinding
from TokenSim.config.config import WorkerConfig, ParallelRankInfo
from .messages import WorkerStatus


class AFComputeWorker:
    """Physical AF-only worker that never owns requests or KV cache."""

    def __init__(
        self,
        env: simpy.Environment,
        id: int,
        config: WorkerConfig,
        *,
        runtime_bindings: tuple[AFRankBinding, ...] = (),
        runtime_state=None,
    ) -> None:
        self.env = env
        self.id = id
        self.role = None
        self.name = config.name
        self.capabilities = config.capabilities
        self.hardware = config.hardware
        self.network = config.network
        self.nettype = config.nettype
        self.rank_info = config.rank_info or ParallelRankInfo()
        self.physical_worker_ref = config.physical_worker_ref
        self.global_rank = self.rank_info.global_rank
        self.tp_rank = self.rank_info.tp_rank
        self.pp_rank = self.rank_info.pp_rank
        self.dp_rank = self.rank_info.dp_rank
        self.rank_in_dp_group = self.rank_info.rank_in_dp_group
        self.kv_cache_group_id = self.rank_info.kv_cache_group_id
        self.status = WorkerStatus.WAITING
        self.preempted_cnt = 0
        self.af_runtime_bindings = runtime_bindings
        self.af_runtime_state = runtime_state
        self.owned_expert_ids = sorted(
            {
                expert_id
                for binding in runtime_bindings
                for expert_id in binding.expert_ids
            }
        )

    def workload(self) -> float:
        return 0.0

    def cpu_workload(self) -> float:
        return 0.0


def bind_af_worker(worker, worker_config, runtime_bindings, runtime_state):
    worker.name = worker_config.name
    worker.capabilities = worker_config.capabilities
    worker.af_runtime_bindings = runtime_bindings
    worker.af_runtime_state = runtime_state
    leader_binding = next(
        (binding for binding in runtime_bindings if binding.is_scheduler_leader),
        None,
    )
    worker.rank_info = (
        ParallelRankInfo(
            global_rank=leader_binding.domain_rank,
            rank_in_dp_group=leader_binding.tp_rank,
            tp_rank=leader_binding.tp_rank,
            pp_rank=leader_binding.pp_rank,
            dp_rank=leader_binding.dp_rank,
            kv_cache_group_id=(f"{leader_binding.domain}:dp{leader_binding.dp_rank}"),
        )
        if leader_binding is not None
        else worker_config.rank_info or ParallelRankInfo()
    )
    worker.physical_worker_ref = worker_config.physical_worker_ref


def _complete_compute(worker, running):
    latency = worker.dynamic_batch(running)
    if latency:
        yield from worker.engine.run_complete_compute(worker.id, latency)
    return latency


def _af_compute(worker, requests):
    engine = worker.engine
    iteration = engine.af_executor.submit(worker.id, requests)
    yield from engine.af_executor.run_iteration(iteration)
    engine.record_af_iteration(iteration)
    return iteration.makespan or 0.0


def run_worker_compute(worker, running):
    engine = worker.engine
    if engine.af_executor is None:
        return (yield from _complete_compute(worker, running))
    if all(req.is_prefill for req in running):
        return (yield from _complete_compute(worker, running))
    groups = {}
    for req in running:
        groups.setdefault((req.is_prefill, req.needs_recompute), []).append(req)
    start = worker.env.now
    jobs = [
        (
            _af_compute(worker, requests)
            if not prefill
            else _complete_compute(worker, requests)
        )
        for (prefill, _), requests in groups.items()
    ]
    if (
        len(jobs) == 1
        or engine.cluster_config.af.communication_timing.native_control_enabled
    ):
        for job in jobs:
            yield from job
    else:
        processes = [worker.env.process(job) for job in jobs]
        for process in processes:
            process.defused = True
        joined = worker.env.all_of(processes)
        joined.defused = True
        try:
            yield joined
        except BaseException:
            for process in processes:
                if process.is_alive:
                    process.interrupt("AF serving batch failed or was cancelled")
            for process in processes:
                if not process.triggered:
                    try:
                        yield process
                    except BaseException:
                        pass
            raise
    return worker.env.now - start
