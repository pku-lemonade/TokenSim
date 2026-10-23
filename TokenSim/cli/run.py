"""Build and execute a simulation and export its results."""

from __future__ import annotations

from dataclasses import replace
from itertools import islice
from pathlib import Path
import time
import simpy
from TokenSim.config.config import ClusterConfig, KVTransferConfig
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.config.validation import number
from TokenSim.config.router_config import load_router_mapping
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware import HardwareContext
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.workload.agentx import AgentXReplay
from TokenSim.workload.agentx_support.parser import iter_trace_graphs
from TokenSim.workload.agentx_support.logs import ReplayLogs
from TokenSim.workload.source import get_requests, LLMSource
from TokenSim.reporting import RunReport, export_result, print_all_stats
from TokenSim.progress import TqdmManager
from .configuration import (
    simulation_config,
    build_parallel_config,
    validate_moe_parallel_config,
)
from TokenSim.reporting import write_failure

ROOT = Path(__file__).resolve().parents[2]


def main(args, preloaded_agentx_traces=None):
    if args.sim_time is not None:
        number(args.sim_time, "sim_time", positive=True)
    config = simulation_config(args)
    cluster = ClusterConfig.from_file(args.cluster)
    from TokenSim.af.providers.queries import resolve_af_latency_providers

    af_compute_provider, af_transfer_provider = (
        resolve_af_latency_providers(
            cluster.af,
            getattr(args, "af_compute_provider", None),
            getattr(args, "af_transfer_provider", None),
        )
        if cluster.af.enabled
        else (None, None)
    )
    hardware = HardwareContext.load(args.data_root)
    workload_config = (
        WorkloadConfig.from_file(args.workload_config)
        if args.workload_config
        else WorkloadConfig()
    ).from_args(args)
    model = hardware.models.resolve(args.model)
    overrides = {}
    for field, argument in (
        ("distribution", "moe_routing_distribution"),
        ("hot_expert_fraction", "moe_hot_expert_fraction"),
    ):
        value = getattr(args, argument)
        if value is not None:
            overrides[field] = value
    if args.moe_hot_experts is not None:
        try:
            overrides["hot_experts"] = [
                int(item) for item in args.moe_hot_experts.split(",") if item.strip()
            ]
        except ValueError as error:
            raise ConfigurationError(
                "moe_hot_experts must contain integer expert IDs"
            ) from error
    if overrides:
        if not model.is_moe:
            raise ConfigurationError("MoE routing overrides require a MoE model")
        routing = replace(model.moe.routing, **overrides)
        model = replace(model, moe=replace(model.moe, routing=routing))
    run = RunReport(
        results_path=(
            Path(args.results_path)
            if args.results_path
            else (ROOT if (ROOT / "data").is_dir() else Path.cwd())
            / "results"
            / model.model_id
            / Path(args.cluster).parent.name
            / Path(args.cluster).stem
        ),
        qps=(
            float("inf")
            if workload_config.distribution == "burst"
            else workload_config.qps
        ),
        batching=args.batching,
        agentx_concurrency=(
            args.agentx_concurrency if args.workload_type == "agentx_weka" else None
        ),
    )
    parallel = build_parallel_config(args, cluster)
    validate_moe_parallel_config(model, parallel)
    kv_override = (
        KVTransferConfig.from_file(args.kv_transfer_config)
        if args.kv_transfer_config
        else None
    )
    kv_config = cluster.effective_kv_transfer(kv_override)
    if args.router_mapping_path:
        config = replace(
            config, router_mapping=load_router_mapping(args.router_mapping_path)
        )
    env = simpy.Environment()
    engine = LLMEngine(
        env=env,
        hardware=hardware,
        cluster_config=cluster,
        model_spec=model,
        workload_config=workload_config,
        parallel_config=parallel,
        kv_transfer_config=kv_config,
        config=config,
        af_compute_provider=af_compute_provider,
        af_transfer_provider=af_transfer_provider,
        af_dtype=getattr(args, "af_dtype", None),
    )
    progress = TqdmManager(verbose=args.verbose, program_id=args.program_id)
    replay = logs = None
    requests = []
    if args.workload_type == "agentx_weka":
        if not args.dataset_path:
            raise ConfigurationError("agentx_weka requires --dataset_path")
        if (
            args.dataset_skip_count < 0
            or args.agentx_trace_count is not None
            and args.agentx_trace_count <= 0
        ):
            raise ConfigurationError(
                "trace skip must be nonnegative and trace count positive"
            )
        graphs = preloaded_agentx_traces
        if graphs is None:
            stop = (
                None
                if args.agentx_trace_count is None
                else args.dataset_skip_count + args.agentx_trace_count
            )
            graphs = list(
                islice(
                    iter_trace_graphs(args.dataset_path), args.dataset_skip_count, stop
                )
            )
        if not graphs:
            raise ConfigurationError("AgentX selection contains no trace graphs")
        if engine.trace_session is not None:
            engine.trace_session.run_phase = (
                "warmup" if args.agentx_warmup else "profiling"
            )
        logs = ReplayLogs(run.results_path / "replay", engine)
        replay = AgentXReplay(
            env,
            engine,
            graphs,
            block_size=config.block_size,
            concurrency=args.agentx_concurrency,
            profile_duration=args.agentx_profile_duration,
            workload_seed=(
                args.random_seed
                if args.agentx_workload_seed is None
                else args.agentx_workload_seed
            ),
            warmup=args.agentx_warmup,
            issue_mode=args.agentx_issue_mode,
            system_idle_cap=(
                args.agentx_idle_gap_cap if args.agentx_idle_gap_cap > 0 else None
            ),
            grace_period=args.agentx_grace_period,
            cancellation_timeout=args.agentx_cancellation_timeout,
            warmup_requests_per_lane=args.agentx_warmup_requests_per_lane,
            warmup_grace_period=args.agentx_warmup_grace_period,
            tree_idle_cap=(
                args.agentx_tree_idle_cap if args.agentx_tree_idle_cap > 0 else None
            ),
            event_sink=lambda record: logs.write("events", record),
            request_sink=lambda record: logs.write("requests", record),
        )
        progress.set_total(
            sum(g.request_count for g in graphs) * args.agentx_concurrency
        )
        probe = replay.capacity_request()
        engine.validate_request_capacity([probe])
        probe.release_logical_blocks()
        source = env.process(replay.run())
        env.process(logs.monitor(env, replay))
    else:
        requests, _, _ = get_requests(
            args,
            workload_config,
            config.block_size,
            lambda count: progress.update(count),
            num_experts=model.moe.num_experts,
        )
        progress.set_total(len(requests))
        engine.validate_request_capacity(requests)
        source = env.process(
            LLMSource(
                env,
                engine,
                requests,
                run.qps,
                workload_config.distribution,
                random_seed=args.random_seed,
            )
        )
    wall_start = time.perf_counter()
    engine.start_debug_clock(wall_start)
    error = None
    try:
        if args.sim_time is not None:
            if replay is not None:
                horizon = env.timeout(args.sim_time)
                env.run(until=env.any_of([source, horizon]))
                if not source.triggered:
                    run.termination_reason = "sim_time_limit"
            else:
                # Stop at the cutoff, but keep the clock at the natural end
                # when the workload finishes first: run(until=...) would
                # advance an idle clock to the cutoff and dilute throughput.
                while env.peek() < args.sim_time:
                    env.step()
                if env.peek() != float("inf"):
                    env.run(until=args.sim_time)
                    if any(not r.is_done for r in requests):
                        run.termination_reason = "sim_time_limit"
        elif replay is not None:
            env.run(until=source)
        else:
            env.run()
        engine.raise_if_failed()
        if run.termination_reason == "completed":
            connectors = [
                w.connector for w in engine.request_workers if w.connector.async_offload
            ]
            deadline = env.now + (replay.cancellation_timeout if replay else 10.0)
            if replay and replay.issue_window_end_at is not None:
                deadline = (
                    replay.issue_window_end_at
                    + replay.grace_period
                    + replay.cancellation_timeout
                )
            while any(c.loads or c.write_jobs or c.pending_writes for c in connectors):
                if env.now >= deadline:
                    raise SimulationStateError(
                        "KV transfers exceeded the cancellation drain deadline"
                    )
                env.run(
                    until=env.any_of(
                        [c.changed for c in connectors]
                        + [env.timeout(deadline - env.now)]
                    )
                )
            for w in engine.request_workers:
                manager = getattr(w.scheduler, "block_manager", None)
                if manager is not None and hasattr(manager, "validate_conservation"):
                    manager.validate_conservation()
                if w.connector.async_offload:
                    w.connector.store.validate()
    except Exception as exc:
        error = exc
        run.termination_reason = "failed"
    finally:
        if logs:
            try:
                logs.snapshot(env, replay)
            except Exception as exc:
                if error is None:
                    error = exc
                    run.termination_reason = "failed"
            logs.close()
    wall_time = time.perf_counter() - wall_start
    if replay:
        requests = replay.requests
        run.agentx_metadata = replay.runtime_metadata()
        run.agentx_metadata.update(
            random_seed=args.random_seed,
            phase=replay.phase,
            partial=run.termination_reason != "completed",
        )
        duration = replay.metric_observation_duration or replay.profile_elapsed
        successful = [r for r in requests if r.is_done and not r.cancel_requested]
    else:
        duration = env.now
        successful = [r for r in requests if r.is_done and not r.cancel_requested]
    unfinished = (
        replay.unfinished_request_ids
        if replay
        else [r.id for r in requests if not r.is_done]
    )
    if unfinished and run.termination_reason == "completed":
        error = SimulationStateError("event queue exhausted with unfinished requests")
        run.termination_reason = "event_queue_exhausted"
    run.failure = str(error) if error else None
    if error or run.termination_reason != "completed":
        write_failure(run, env, engine, unfinished, error)
    result = export_result(
        run=run,
        g_time=engine.timing_recorder.timing,
        engine=engine,
        cluster=cluster,
        duration=duration,
        request_count=len(requests),
        requests=successful if replay else requests,
        prefill_lens=[r.prefill_len for r in requests],
        decode_lens=[r.decode_len for r in requests],
        notdone=unfinished,
        simulator_wall_time=wall_time,
    )
    if successful and duration > 0:
        print_all_stats(
            engine.timing_recorder.timing,
            successful,
            engine,
            duration,
            len(successful),
            [r.prefill_len for r in successful],
            slo=result.get("slo"),
        )
    if error:
        raise error
    return engine
