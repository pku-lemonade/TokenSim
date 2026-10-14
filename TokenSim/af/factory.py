"""Construct a whole-model AF runtime from catalog models and cluster data."""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.runtime.plan import AFRuntimePlan
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.trace.recorder import AFTraceRecorder
from TokenSim.config.cluster_config import ClusterConfig
from TokenSim.config.cluster_config import WorkerConfig
from TokenSim.config.model_config import ModelSpec
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
        dtype=model.dtype,
    )
    return AFModelRuntime(doc, hardware, model, cluster, workers, placement, plan)


def simulate_decode(
    config, batch, context, *, steps=1, trace=True, fallback="table_first", seed=42
):
    """Exact fixed-batch decode; every token advances its own SimPy iteration."""
    if batch < 1 or context < 1 or steps < 1:
        raise ValueError("batch, context and steps must be positive")
    env = simpy.Environment()
    compute, transfer = build_operator_table_af_providers(
        hardware=config.hardware,
        model=config.model,
        cluster_config=config.cluster,
        worker_configs=config.workers,
        placement=config.placement,
        dtype=config.model.dtype,
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
        trace_recorder=AFTraceRecorder(trace),
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
        iteration = executor.submit(0, requests, "decode")
        env.process(executor.run_iteration(iteration))
        env.run()
        rows.append(
            dict(
                iteration_id=iteration.iteration_id,
                batch_size=batch,
                context_len=context + step,
                makespan_us=iteration.makespan * 1e6,
            )
        )
    return executor, rows, compute, transfer
