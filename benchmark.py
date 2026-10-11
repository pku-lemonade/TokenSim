#!/usr/bin/env python3
"""Run a workload against the configured model, hardware and serving policy."""

import argparse
from dataclasses import fields, replace
from itertools import islice
import json
import os
from pathlib import Path
import sys
import time

import simpy

from TokenSim.config.config import ClusterConfig, KVTransferConfig, ParallelConfig
from TokenSim.config.parallel_config import EXPERT_PARALLEL_SCOPES
from TokenSim.config.workload_config import WorkloadConfig
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.simulation import SimulationConfig
from TokenSim.config.paths import default_data_root
from TokenSim.config.validation import number
from TokenSim.errors import ConfigurationError, SimulationStateError, TokenSimError
from TokenSim.hardware import HardwareContext
from TokenSim.latency import FALLBACK_POLICIES
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import Request
from TokenSim.llm.llm_scheduler import DEFAULT_MAX_NUM_BATCHED_TOKENS
from TokenSim.workload.agentx import AgentXReplay
from TokenSim.workload.agentx_support.parser import iter_trace_graphs
from TokenSim.workload.agentx_support.logs import ReplayLogs
from util.request import get_requests, LLMSource
from util.results import export_result, print_all_stats
from util.progress import TqdmManager

ROOT = Path(__file__).resolve().parent
DATA_ROOT = default_data_root()


def simulation_config(args):
    defaults = SimulationConfig()
    names = {"latency_backend_type": "latency_backend"}
    values = {
        field.name: getattr(
            args, names.get(field.name, field.name), getattr(defaults, field.name)
        )
        for field in fields(defaults)
    }
    values["max_num_batched_tokens"] = max_num_batched_tokens(args)
    values["moe_execution"] = {
        "backend": args.moe_backend,
        "shared_expert_fusion": not args.disable_moe_shared_fusion,
    }
    return SimulationConfig(**values)


def write_failure(args, env, engine, unfinished, error):
    path = Path(args.results_path)
    path.mkdir(parents=True, exist_ok=True)
    diagnostic = {
        "termination_reason": args.termination_reason,
        "simulation_time": env.now,
        "failure": str(error) if error else None,
        "unfinished_request_ids": unfinished,
        "workers": [
            {
                "worker_id": w.id,
                "running": [r.id for r in w.scheduler.running],
                "waiting": [r.id for r in w.scheduler.waiting],
            }
            for w in engine.workers
        ],
    }
    (path / "failure.json").write_text(json.dumps(diagnostic, indent=2) + "\n")
    if error:
        with (path / "failed.txt").open("a") as stream:
            stream.write(str(error) + "\n")


def main(args, preloaded_agentx_traces=None):
    if args.sim_time is not None:
        number(args.sim_time, "sim_time", positive=True)
    config = simulation_config(args)
    hardware = HardwareContext.load(args.data_root)
    cluster = ClusterConfig.from_file(args.cluster)
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
    args.qps = (
        float("inf") if workload_config.distribution == "burst" else workload_config.qps
    )
    args.results_path = str(
        Path(args.results_path)
        if args.results_path
        else (ROOT if (ROOT / "data").is_dir() else Path.cwd())
        / "results"
        / model.model_id
        / Path(args.cluster).parent.name
        / Path(args.cluster).stem
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
        mapping = json.loads(Path(args.router_mapping_path).read_text())
        mapping = mapping.get("mapping", mapping)
        if not isinstance(mapping, dict):
            raise ConfigurationError("router mapping must contain a JSON object")
        config = replace(config, router_mapping=mapping)
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
        logs = ReplayLogs(Path(args.results_path) / "replay", engine)
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
                args.qps,
                workload_config.distribution,
                random_seed=args.random_seed,
            )
        )
    wall_start = time.perf_counter()
    engine.start_debug_clock(wall_start)
    error = None
    args.termination_reason = "completed"
    try:
        if args.sim_time is not None:
            if replay is not None:
                horizon = env.timeout(args.sim_time)
                env.run(until=env.any_of([source, horizon]))
                if not source.triggered:
                    args.termination_reason = "sim_time_limit"
            else:
                # Stop at the cutoff, but keep the clock at the natural end
                # when the workload finishes first: run(until=...) would
                # advance an idle clock to the cutoff and dilute throughput.
                while env.peek() < args.sim_time:
                    env.step()
                if env.peek() != float("inf"):
                    env.run(until=args.sim_time)
                    if any(not r.is_done for r in requests):
                        args.termination_reason = "sim_time_limit"
        elif replay is not None:
            env.run(until=source)
        else:
            env.run()
        engine.raise_if_failed()
        if args.termination_reason == "completed":
            connectors = [
                w.connector
                for w in engine.workers
                if getattr(w.connector, "native_async", False)
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
            for w in engine.workers:
                manager = getattr(w.scheduler, "block_manager", None)
                if manager is not None and hasattr(manager, "validate_conservation"):
                    manager.validate_conservation()
                if getattr(w.connector, "native_async", False):
                    w.connector.store.validate()
    except Exception as exc:
        error = exc
        args.termination_reason = "failed"
    finally:
        if logs:
            try:
                logs.snapshot(env, replay)
            except Exception as exc:
                if error is None:
                    error = exc
                    args.termination_reason = "failed"
            logs.close()
    wall_time = time.perf_counter() - wall_start
    if replay:
        requests = replay.requests
        args.agentx_runtime_metadata = replay.runtime_metadata()
        args.agentx_runtime_metadata.update(
            random_seed=args.random_seed,
            phase=replay.phase,
            partial=args.termination_reason != "completed",
        )
        duration = replay.metric_observation_duration or replay.profile_elapsed
        successful = [r for r in requests if r.is_done and not r.cancel_requested]
    else:
        duration = env.now
        successful = [r for r in requests if r.is_done and not r.cancel_requested]
    unfinished = [r.id for r in requests if not r.is_done]
    if unfinished and args.termination_reason == "completed":
        error = SimulationStateError("event queue exhausted with unfinished requests")
        args.termination_reason = "event_queue_exhausted"
    args.simulation_failure = str(error) if error else None
    if error or args.termination_reason != "completed":
        write_failure(args, env, engine, unfinished, error)
    export_result(
        args=args,
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
        )
    if error:
        raise error
    return engine


LATENCY_BACKEND_KEYWORDS = ("operator_table", "analytical")


def max_num_batched_tokens(args: argparse.Namespace) -> int | None:
    """Per-step token budget of the paged-attention scheduler; ``None`` disables chunking."""
    if not getattr(args, "chunked_prefill", True):
        return None
    return int(getattr(args, "max_num_batched_tokens", DEFAULT_MAX_NUM_BATCHED_TOKENS))


def get_latency_backend_type(latency_backend: str) -> str:
    """Map the CLI value to a backend type."""
    if latency_backend in LATENCY_BACKEND_KEYWORDS:
        return latency_backend
    raise ConfigurationError(
        f"unsupported latency backend {latency_backend!r}; "
        "use 'operator_table' (tables with analytical fallback) or 'analytical'"
    )


def build_parallel_config(
    args: argparse.Namespace,
    cluster: ClusterConfig,
) -> ParallelConfig:
    base = cluster.effective_parallel_config()
    return base.override(
        tensor_parallel_size=getattr(args, "tensor_parallel_size", None),
        pipeline_parallel_size=getattr(args, "pipeline_parallel_size", None),
        data_parallel_size=getattr(args, "data_parallel_size", None),
        data_parallel_rank=getattr(args, "data_parallel_rank", None),
        data_parallel_size_local=getattr(args, "data_parallel_size_local", None),
        enable_expert_parallel=getattr(args, "enable_expert_parallel", None),
        expert_parallel_scope=getattr(args, "expert_parallel_scope", None),
        expert_parallel_size=getattr(args, "expert_parallel_size", None),
        expert_placement_strategy=getattr(args, "expert_placement_strategy", None),
        all2all_backend=getattr(args, "all2all_backend", None),
    )


def validate_moe_parallel_config(
    model: ModelSpec,
    parallel_config: ParallelConfig,
) -> None:
    if parallel_config.enable_expert_parallel and not model.is_moe:
        raise ConfigurationError("enable_expert_parallel requires a MoE model config")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim_time", type=float, default=None, help="Sim time.")
    parser.add_argument("--qps", type=float, default=None, help="Qps.")
    parser.add_argument(
        "--batching",
        choices=["static", "dynamic", "paged-attn"],
        default="dynamic",
        help="Batching.",
    )
    parser.add_argument(
        "--distribution",
        choices=["burst", "uniform", "poisson"],
        default=None,
        help="Distribution.",
    )
    parser.add_argument("--request_count", type=int, default=100, help="Request count.")
    parser.add_argument("--prefill_mean_len", type=int, help="Prefill mean len.")
    parser.add_argument("--prefill_range_len", type=int, help="Prefill range len.")
    parser.add_argument("--decode_mean_len", type=int, help="Decode mean len.")
    parser.add_argument("--decode_range_len", type=int, help="Decode range len.")
    parser.add_argument(
        "--decode_len_distribution",
        choices=["uniform", "exponential", "capped_exponential", "burst"],
        default=None,
        help="Decode len distribution.",
    )
    parser.add_argument("--block_size", type=int, default=16, help="Block size.")
    parser.add_argument(
        "--model",
        type=str,
        default="LLaMa-7B",
        help="Model catalog name or model YAML path.",
    )
    parser.add_argument(
        "--workload_config",
        default=None,
        help="Optional JSON request-generation and SLO configuration.",
    )
    parser.add_argument(
        "--cluster",
        type=str,
        default=str(DATA_ROOT / "clusters/1_a100/h1.json"),
        help="Cluster.",
    )
    parser.add_argument(
        "--kv_transfer_config", type=str, default=None, help="Kv transfer config."
    )
    parser.add_argument(
        "--prefill_worker_pool_type",
        choices=["round_robin", "least_gpu_memory", "balanced_load"],
        default="round_robin",
        help="Prefill worker pool type.",
    )
    parser.add_argument(
        "--decode_worker_pool_type",
        choices=["round_robin", "least_gpu_memory", "balanced_load"],
        default="least_gpu_memory",
        help="Decode worker pool type.",
    )
    parser.add_argument(
        "--max_num_seqs",
        "--max_parallem_sum",
        type=int,
        default=64,
        help="Max parallem sum.",
    )
    parser.add_argument(
        "--max_occupy_ratio", type=float, default=1.0, help="Max occupy ratio."
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=1.0,
        help="Gpu memory utilization.",
    )
    parser.add_argument(
        "--router_policy",
        choices=["none", "consistent_hash", "fixed_mapping"],
        default="none",
        help="Router policy.",
    )
    parser.add_argument(
        "--router_backend_url",
        default="http://localhost:8889",
        help="Router backend url.",
    )
    parser.add_argument(
        "--router_mapping_path", default=None, help="Router mapping path."
    )
    parser.add_argument(
        "--dp_lockstep",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dp lockstep.",
    )
    parser.add_argument(
        "--dp_sync_interval", type=int, default=16, help="Dp sync interval."
    )
    parser.add_argument(
        "--long_prefill_token_threshold",
        type=int,
        default=0,
        help="Long prefill token threshold.",
    )
    parser.add_argument(
        "--prefill_schedule_interval",
        type=int,
        default=1,
        help="Prefill schedule interval.",
    )
    parser.add_argument(
        "--adaptive_long_prefill_threshold",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Adaptive long prefill threshold.",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help="Token budget of one paged-attn step (vLLM V1 chunked prefill): decodes are served first, prompts longer than the remaining budget are prefilled in chunks.",
    )
    parser.add_argument(
        "--chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable to prefill every prompt in a single step regardless of length.",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=None, help="Tensor parallel size."
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=None,
        help="Pipeline parallel size.",
    )
    parser.add_argument(
        "--data_parallel_size", type=int, default=None, help="Data parallel size."
    )
    parser.add_argument(
        "--data_parallel_rank", type=int, default=None, help="Data parallel rank."
    )
    parser.add_argument(
        "--data_parallel_size_local",
        type=int,
        default=None,
        help="Data parallel size local.",
    )
    parser.add_argument(
        "--enable_expert_parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable expert parallel.",
    )
    parser.add_argument(
        "--expert_parallel_scope",
        choices=list(EXPERT_PARALLEL_SCOPES),
        default=None,
        help="Which ranks share one copy of the experts: every TP x DP rank (global) or one DP replica (per_dp).",
    )
    parser.add_argument(
        "--expert_parallel_size",
        type=int,
        default=None,
        help="Ranks per expert-parallel group; validated against the TP/DP layout and the scope.",
    )
    parser.add_argument(
        "--expert_placement_strategy",
        choices=["linear", "round_robin"],
        default=None,
        help="Expert placement strategy.",
    )
    parser.add_argument(
        "--all2all_backend",
        choices=[
            "allgather_reducescatter",
            "naive",
            "deepep_high_throughput",
            "deepep_low_latency",
        ],
        default=None,
        help="All2all backend.",
    )
    parser.add_argument(
        "--moe_routing_distribution",
        choices=["uniform", "skew", "hot", "burst"],
        default=None,
        help="Moe routing distribution.",
    )
    parser.add_argument(
        "--moe_hot_experts", type=str, default=None, help="Moe hot experts."
    )
    parser.add_argument(
        "--moe_hot_expert_fraction",
        type=float,
        default=None,
        help="Moe hot expert fraction.",
    )
    parser.add_argument(
        "--verbose",
        type=str,
        choices=["none", "simple", "tqdm"],
        default="tqdm",
        help="Verbose.",
    )
    parser.add_argument(
        "--debug_print",
        "--debug-print",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Debug-print.",
    )
    parser.add_argument("--program_id", type=int, default=0, help="Program id.")
    parser.add_argument("--results_path", type=str, default="", help="Results path.")
    parser.add_argument("--dataset_path", type=str, default=None, help="Dataset path.")
    parser.add_argument(
        "--dataset_skip_count", type=int, default=0, help="Dataset skip count."
    )
    parser.add_argument(
        "--trace_timestamp_scale",
        type=float,
        default=None,
        help="Scale loaded timestamped trace arrivals as new_ts = old_ts * scale.",
    )
    parser.add_argument(
        "--trace_target_qps",
        type=float,
        default=None,
        help="Scale timestamped trace arrivals to this QPS using scale = original_trace_qps / trace_target_qps.",
    )
    parser.add_argument(
        "--workload_type",
        choices=["synthetic", "json_pairs", "qwen_jsonl", "agentx_weka"],
        default="synthetic",
        help="Workload type.",
    )
    parser.add_argument(
        "--agentx_concurrency", type=int, default=1, help="Agentx concurrency."
    )
    parser.add_argument(
        "--agentx_trace_count", type=int, default=None, help="Agentx trace count."
    )
    parser.add_argument(
        "--agentx_profile_duration",
        type=float,
        default=3600.0,
        help="Agentx profile duration.",
    )
    parser.add_argument(
        "--agentx_issue_mode",
        choices=("closed_loop", "recorded"),
        default="closed_loop",
        help="Native diagnostic issue timing: closed-loop dependencies or recorded trace timestamps.",
    )
    parser.add_argument(
        "--agentx_idle_gap_cap", type=float, default=10.0, help="Agentx idle gap cap."
    )
    parser.add_argument(
        "--agentx_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Agentx warmup.",
    )
    parser.add_argument("--random_seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--latency_backend",
        type=str,
        default="operator_table",
        help="'operator_table' (per-operator tables with analytical fallback) or 'analytical' (formulas only).",
        choices=list(LATENCY_BACKEND_KEYWORDS),
    )
    parser.add_argument(
        "--latency_fallback",
        choices=list(FALLBACK_POLICIES),
        default="table_first",
        help="How operator_table handles shapes missing from the tables.",
    )
    parser.add_argument(
        "--operator_backend",
        type=str,
        default=None,
        help="Operator-data backend to load for every worker (e.g. trtllm, vllm, analytical).",
        choices=["vllm", "sglang", "trtllm", "analytical"],
    )
    parser.add_argument(
        "--moe_backend",
        choices=("generic", "deep_gemm_amxf4_mega_moe"),
        default="generic",
        help="Moe backend.",
    )
    parser.add_argument(
        "--disable_moe_shared_fusion",
        action="store_true",
        help="Disable moe shared fusion.",
    )
    parser.add_argument(
        "--attention_backend",
        choices=("generic", "module"),
        default="generic",
        help="Attention cost source: generic operators or the model's measured module binding.",
    )
    parser.add_argument(
        "--mtp_acceptance_length",
        type=float,
        default=1.0,
        help="Average accepted output tokens per speculative verification step.",
    )
    parser.add_argument(
        "--mtp_num_speculative_tokens",
        type=int,
        default=None,
        help="MTP draft token count K;0 disables drafting, while positive K is independent of accepted length.",
    )
    parser.add_argument(
        "--mhc_backend",
        choices=("generic", "module"),
        default="generic",
        help="Residual-mixing cost source: generic operators or the model's measured module binding.",
    )
    parser.add_argument(
        "--decode_context_bucket",
        type=int,
        default=128,
        help="Decode attention lookups round context length up to this multiple.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=str(DATA_ROOT),
        help="Directory holding devices/, topologies/, models/ and operator_data/.",
    )
    parser.add_argument(
        "--agentx_workload_seed",
        type=int,
        default=None,
        help="Trace warmup sampling seed; defaults to random_seed.",
    )
    parser.add_argument(
        "--agentx_grace_period",
        type=float,
        default=30.0,
        help="Seconds to allow issued requests to finish.",
    )
    parser.add_argument(
        "--agentx_cancellation_timeout",
        type=float,
        default=10.0,
        help="Seconds to drain cancellation and transfer ownership.",
    )
    parser.add_argument(
        "--sticky_sessions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Retain bounded chat affinity with the ordinary DP placement policy.",
    )
    return parser


def cli(argv=None):
    args = build_parser().parse_args(argv)
    try:
        main(args)
    except TokenSimError as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
