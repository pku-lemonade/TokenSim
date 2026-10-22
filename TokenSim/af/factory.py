"""Construct a whole-model AF runtime from catalog models and cluster data."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import simpy

from TokenSim.af.execution.model import ModelAFExecutor
from TokenSim.af.providers.operator_table import build_operator_table_af_providers
from TokenSim.af.runtime.plan import AFRuntimePlan
from TokenSim.af.runtime.compiler import compile_af_runtime_plan
from TokenSim.af.runtime.host import host_collection, run_environment
from TokenSim.af.trace.archive import ArchivedTraceRecorder
from TokenSim.af.trace.config import AFTraceConfig
from TokenSim.trace.config import TraceConfig
from TokenSim.af.trace.formats import make_trace_recorder, validate_trace_format
from TokenSim.af.trace.pipeline import PipelinedTraceRecorder
from TokenSim.af.trace.transient import ValidationTraceRecorder
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


@host_collection()
def simulate_decode(
    config,
    batch,
    context,
    *,
    steps=1,
    trace=True,
    fallback="table_first",
    seed=42,
    trace_config: TraceConfig | None = None,
    trace_output: str | Path | None = None,
    trace_format=None,
):
    """Exact fixed-batch decode; every token advances its own SimPy iteration.

    An optional trace_output prepares the selected format in its directory.
    Call write_trace after simulation to publish the complete file.
    """
    if any(type(value) is not int or value < 1 for value in (batch, context, steps)):
        raise ValueError("batch, context and steps must be positive")
    if trace_config is not None and not isinstance(trace_config, TraceConfig):
        raise ConfigurationError("trace_config must be a TraceConfig")
    if trace_output is not None and not isinstance(trace_output, (str, Path)):
        raise ConfigurationError("trace_output must be a path")
    options = trace_config or AFTraceConfig(mode="full" if trace else "off")
    trace_format = trace_format or options.format
    trace_output = trace_output or options.output
    validate_trace_format(trace_format, require_dependency=options.mode == "full")
    if trace is False and options.enabled:
        raise ConfigurationError("trace=False conflicts with an enabled trace_config")
    if (
        config.cluster.af.execution_protocol == "vllm_afd"
        and config.cluster.af_domains[0].parallel.data_parallel_size > 1
    ):
        raise ConfigurationError(
            "fixed-batch native decode requires one A replica; "
            "multi-A vllm_afd must submit every peer through AFService or AFExecutor"
        )
    peak_context = context + steps
    if peak_context > config.model.max_position_embeddings:
        raise ConfigurationError("fixed-batch decode exceeds the model context limit")
    from TokenSim.af.runtime.lpx_memory import memory_policies, validate_queries

    validate_queries(
        memory_policies(config.hardware, config.workers),
        min(batch, config.cluster.af.af_block_size),
        peak_context,
    )
    # This helper submits every request to one decode A replica. It has no
    # scheduler admission loop, so check the resident KV budget explicitly.
    owner = next(iter(config.plan.request_owners))
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
    if options.mode == "validate":
        recorder = ValidationTraceRecorder()
    elif options.mode == "full" and trace_output is not None:
        recorder = (
            PipelinedTraceRecorder(Path(trace_output).parent, trace_config=options)
            if trace_format == "json"
            else make_trace_recorder(Path(trace_output).parent, options, trace_format)
        )
    else:
        recorder = ArchivedTraceRecorder(options.enabled)
    recorder.validate_on_completion = options.validate
    recorder.trace_config = trace_config
    recorder.trace_format = trace_format
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
        trace_recorder=recorder,
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
        iteration = executor.submit(source_worker, requests)
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
