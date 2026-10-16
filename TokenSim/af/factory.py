"""Construct a whole-model AF runtime from catalog models and cluster data."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.runtime.plan import AFRuntimePlan
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.runtime.host import run_environment
from TokenSim.af.trace.archive import ArchivedTraceRecorder
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.cluster_config import WorkerConfig
from TokenSim.config.model_config import ModelSpec
from TokenSim.errors import ConfigurationError
from TokenSim.hardware import HardwareContext
from TokenSim.hardware.placement import build_topology_placement
from TokenSim.hardware.topology import TopologyPlacement


@dataclass
class AFModelRuntime:
    doc: dict[str, Any]
    hardware: HardwareContext
    model: ModelSpec
    cluster: ClusterConfig
    workers: list[WorkerConfig]
    placement: TopologyPlacement
    plan: AFRuntimePlan


def build_model_runtime(doc, *, hardware, model, block_size=16):
    cluster = ClusterConfig(**doc)
    workers = cluster.workers()
    placement = build_topology_placement(cluster, workers, hardware)
    plan = compile_af_runtime_plan(
        cluster_config=cluster,
        worker_configs=workers,
        model=model,
        hardware=hardware,
        block_size=block_size,
        dtype=model.activation_dtype,
    )
    return AFModelRuntime(doc, hardware, model, cluster, workers, placement, plan)


def simulate_decode(
    config, batch, context, *, steps=1, trace=True, fallback="table_first", seed=42
):
    """Exact fixed-batch decode; every token advances its own SimPy iteration."""
    if any(type(value) is not int or value < 1 for value in (batch, context, steps)):
        raise ValueError("batch, context and steps must be positive")
    peak_context = context + steps
    if peak_context > config.model.max_position_embeddings:
        raise ConfigurationError("fixed-batch decode exceeds the model context limit")
    # This helper submits every request to one decode A replica. It has no
    # scheduler admission loop, so check the resident KV budget explicitly.
    owner = next(
        owner for owner in config.plan.request_owners if owner.phase == "decode"
    )
    leader = config.plan.binding(owner.domain, owner.leader_rank)
    source_worker = next(
        index
        for index, worker in enumerate(config.workers)
        if worker.physical_worker_ref.identity == leader.physical_identity
    )
    replica = config.plan.replica_for_instance(leader.domain, leader.instance)
    required_blocks = batch * (-(-peak_context // config.plan.block_size))
    if required_blocks > replica.logical_gpu_blocks:
        raise ConfigurationError(
            f"fixed-batch decode KV does not fit: required_blocks={required_blocks}, "
            f"available_blocks={replica.logical_gpu_blocks}"
        )
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(
        hardware=config.hardware,
        model=config.model,
        cluster_config=config.cluster,
        worker_configs=config.workers,
        placement=config.placement,
        dtype=config.model.activation_dtype,
        fallback=fallback,
    )
    executor = ModelAFExecutor(
        env,
        config.cluster,
        config.workers,
        compute,
        transfer,
        config.model.model_id,
        config.model.dtype,
        config.plan.total_layers,
        config.model.hidden_size,
        moe_config=config.model.moe,
        random_seed=seed,
        runtime_plan=config.plan,
        trace_recorder=ArchivedTraceRecorder(trace),
        model=config.model,
    )
    rows = []
    for step in range(steps):
        requests = [
            SimpleNamespace(
                id=i,
                is_prefill=False,
                prefill_len=context,
                context_len=context + step,
                generation_idx=step,
            )
            for i in range(batch)
        ]
        iteration = executor.submit(source_worker, requests, "decode")
        env.process(executor.run_iteration(iteration))
        try:
            run_environment(env)
        except BaseException:
            executor.trace.close()
            raise
        rows.append(
            dict(
                iteration_id=iteration.iteration_id,
                batch_size=batch,
                context_len=context + step,
                makespan_us=iteration.makespan * 1e6,
            )
        )
    return executor, rows, compute, transfer
