#!/usr/bin/env python3

import argparse
import json
import simpy
from pathlib import Path
import os
import time

from util.results import export_result, print_all_stats
from util.request import get_requests, LLMSource
from util.tqdm import TqdmManager
from TokenSim.llm.llm_engine import LLMEngine
from TokenSim.llm.llm_request import g_time, reset_g_time, Request
from TokenSim.llm.llm_scheduler import DEFAULT_MAX_NUM_BATCHED_TOKENS
from TokenSim.config.config import ClusterConfig, KVTransferConfig, ParallelConfig
from TokenSim.config.parallel_config import EXPERT_PARALLEL_SCOPES
from TokenSim.config.psla_config import PSLAConfig
from TokenSim.errors import ConfigurationError, SimulationStateError
from TokenSim.hardware import HardwareContext
from TokenSim.latency import FALLBACK_POLICIES
from TokenSim.workload.agentx import AgentXReplay, load_agentx_traces


def check_results(
    args: argparse.Namespace,
    requests: list[Request],
    engine: LLMEngine,
    model_config: PSLAConfig,
    cluster: ClusterConfig,
    duration: float,
    request_count: int,
    prefill_lens: list[int],
    decode_lens: list[int],
    simulator_wall_time: float,
):
    notdone = [r.id for r in requests if not r.is_done]
    if notdone:
        failed_path = Path.cwd() / "results"
        failed_path.mkdir(parents=True, exist_ok=True)

        failed = "Failed " + args.cluster + "_" + str(args.qps) + ": " + str(notdone)
        with open(failed_path / "failed.txt", "a", newline="\n") as file:
            file.write(failed + "\n")

        if args.results_path == "":
            diagnostic_path = failed_path
        else:
            diagnostic_path = Path(args.results_path)
        diagnostic_path.mkdir(parents=True, exist_ok=True)
        snapshot = {
            "failure": "simpy_event_queue_exhausted_with_unfinished_requests",
            "cluster": args.cluster,
            "qps": args.qps,
            "simulation_time": duration,
            "simulator_wall_time": simulator_wall_time,
            "unfinished_requests": [
                {
                    "id": req.id,
                    "status": req.status.name,
                    "generation_idx": req.generation_idx,
                    "decode_len": req.decode_len,
                    "needs_recompute": req.needs_recompute,
                    "recompute_tokens": req.recompute_tokens,
                }
                for req in requests
                if not req.is_done
            ],
            "workers": [
                {
                    "id": worker.id,
                    "role": worker.role,
                    "status": worker.status.name,
                    "running": [req.id for req in worker.scheduler.running],
                    "waiting": [req.id for req in worker.scheduler.waiting],
                    "gpu_blocks": (
                        worker.scheduler.block_manager.get_gpu_status()
                        if hasattr(worker.scheduler, "block_manager")
                        else None
                    ),
                }
                for worker in engine.workers
            ],
        }
        failure_file = diagnostic_path / f"failure_{args.qps}.json"
        temporary_file = failure_file.with_suffix(".json.tmp")
        temporary_file.write_text(json.dumps(snapshot, indent=4) + "\n")
        os.replace(temporary_file, failure_file)
        raise SimulationStateError(
            f"simulation ended with {len(notdone)} unfinished requests; "
            f"diagnostics written to {failure_file}"
        )

    print_all_stats(g_time, requests, engine, duration, request_count, prefill_lens)
    export_result(
        args=args,
        g_time=g_time,
        engine=engine,
        model_config=model_config,
        cluster=cluster,
        request_count=request_count,
        prefill_lens=prefill_lens,
        decode_lens=decode_lens,
        requests=requests,
        notdone=notdone,
        duration=duration,
        simulator_wall_time=simulator_wall_time,
    )


def main(
    args: argparse.Namespace,
    preloaded_agentx_traces=None,
):
    reset_g_time()
    latency_backend_type = get_latency_backend_type(args.latency_backend)
    hardware = HardwareContext.load(args.data_root)

    cluster = ClusterConfig.from_file(args.cluster)
    kv_transfer_override = (
        KVTransferConfig.from_file(args.kv_transfer_config)
        if args.kv_transfer_config
        else None
    )
    kv_transfer_config = cluster.effective_kv_transfer(kv_transfer_override)
    model_config = PSLAConfig.from_file(args.model_config_path).from_args(args)
    parallel_config = build_parallel_config(args, cluster, model_config)
    validate_moe_parallel_config(model_config, parallel_config)

    tqdm_manager = TqdmManager(verbose=args.verbose, program_id=args.program_id)
    env = simpy.Environment()
    is_agentx = args.workload_type == "agentx_weka"
    agentx_traces = None
    if is_agentx:
        if not args.dataset_path:
            raise ConfigurationError(
                "--dataset_path is required for agentx_weka workloads"
            )
        agentx_traces = preloaded_agentx_traces
        if agentx_traces is None:
            agentx_traces = load_agentx_traces(
                args.dataset_path,
                trace_count=args.agentx_trace_count,
                trace_skip_count=args.dataset_skip_count,
            )
        requests: list[Request] = []
        prefill_lens: list[int] = []
        decode_lens: list[int] = []
        estimated_total = (
            args.agentx_max_requests
            or sum(trace.request_count for trace in agentx_traces)
            * args.agentx_concurrency
        )
        tqdm_manager.set_total(estimated_total)
    else:
        requests, prefill_lens, decode_lens = get_requests(
            args=args,
            model_config=model_config,
            block_size=args.block_size,
            tqdm_submit_func=lambda req_num: tqdm_manager.update(req_num),
        )
        tqdm_manager.set_total(len(requests))

    engine = LLMEngine(
        env=env,
        block_size=args.block_size,
        batching=args.batching,
        kv_transfer_config=kv_transfer_config,
        psla_config=model_config,
        cluster_config=cluster,
        parallel_config=parallel_config,
        hardware=hardware,
        prefill_worker_pool_type=args.prefill_worker_pool_type,
        decode_worker_pool_type=args.decode_worker_pool_type,
        max_parallem_sum=args.max_parallem_sum,
        max_occupy_ratio=args.max_occupy_ratio,
        latency_backend_type=latency_backend_type,
        latency_fallback=args.latency_fallback,
        operator_backend=args.operator_backend,
        decode_context_bucket=args.decode_context_bucket,
        random_seed=args.random_seed,
        debug_print=getattr(args, "debug_print", False),
        max_num_batched_tokens=max_num_batched_tokens(args),
    )
    replay = None
    if is_agentx:
        replay = AgentXReplay(
            env,
            engine,
            agentx_traces,
            block_size=args.block_size,
            concurrency=args.agentx_concurrency,
            profile_duration=args.agentx_profile_duration,
            max_requests=args.agentx_max_requests,
            system_idle_gap_cap=(
                args.agentx_idle_gap_cap if args.agentx_idle_gap_cap > 0 else None
            ),
            warmup=args.agentx_warmup,
            warmup_min_ratio=args.agentx_warmup_min_ratio,
            warmup_max_ratio=args.agentx_warmup_max_ratio,
            random_seed=args.random_seed,
            tqdm_submit_func=lambda req_num: tqdm_manager.update(req_num),
        )
        capacity_request = replay.capacity_request()
        engine.validate_request_capacity([capacity_request])
        capacity_request.release_logical_blocks()
        source = replay.run()
    else:
        engine.validate_request_capacity(requests)
        source = LLMSource(
            env=env,
            engine=engine,
            requests=requests,
            qps=args.qps,
            distribution=model_config.distribution,
        )

    source_process = env.process(source)

    wall_start = time.perf_counter()
    engine.start_debug_clock(wall_start)
    if args.sim_time is not None:
        env.run(args.sim_time)
    elif replay is not None and replay.profile_duration is not None:
        env.run(until=source_process)
    else:
        env.run()
    engine.raise_if_failed()
    simulator_wall_time = time.perf_counter() - wall_start

    if replay is not None:
        requests = replay.requests
        prefill_lens = [request.prefill_len for request in requests]
        decode_lens = [request.decode_len for request in requests]
        duration = replay.profile_elapsed
        if not requests or duration <= 0:
            raise SimulationStateError(
                "AgentX replay completed without profile requests; increase "
                "--agentx_profile_duration or use --no-agentx_warmup for a short smoke run"
            )
        args.agentx_runtime_metadata = {
            "trace_count": len(agentx_traces),
            "play_count": replay.play_count,
            "warmup_request_count": len(replay.warmup_requests),
            "warmup_duration_s": replay.profile_started_at or 0.0,
            "random_seed": replay.random_seed,
            "initial_trace_ids": replay.initial_trace_ids,
            "recycle_trace_ids": replay.recycle_trace_ids,
        }
    else:
        duration = env.now
    request_count = len(requests)

    check_results(
        args,
        requests,
        engine,
        model_config,
        cluster,
        duration,
        request_count,
        prefill_lens,
        decode_lens,
        simulator_wall_time,
    )


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
    model_config: PSLAConfig,
) -> ParallelConfig:
    base = cluster.effective_parallel_config(
        model_parallel_config=model_config.parallel_config
    )
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
    model_config: PSLAConfig,
    parallel_config: ParallelConfig,
) -> None:
    if parallel_config.enable_expert_parallel and not model_config.moe_config.enabled:
        raise ConfigurationError("enable_expert_parallel requires a MoE model config")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--sim_time", type=float, default=None)

    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument(
        "--batching", choices=["static", "dynamic", "paged-attn"], default="dynamic"
    )
    parser.add_argument(
        "--distribution", choices=["burst", "uniform", "poisson"], default="uniform"
    )
    parser.add_argument("--request_count", type=int, default=100)
    parser.add_argument("--prefill_mean_len", type=int)
    parser.add_argument("--prefill_range_len", type=int)
    parser.add_argument("--decode_mean_len", type=int)
    parser.add_argument("--decode_range_len", type=int)
    parser.add_argument(
        "--decode_len_distribution",
        choices=["uniform", "exponential", "capped_exponential", "burst"],
        default="uniform",
    )

    parser.add_argument("--block_size", type=int, default=16)
    parser.add_argument(
        "--model",
        dest="model_config_path",
        type=str,
        default="./data/psla/llama-7b.json",
    )
    parser.add_argument("--cluster", type=str, default="./data/clusters/1_a100/h1.json")
    parser.add_argument("--kv_transfer_config", type=str, default=None)
    parser.add_argument(
        "--prefill_worker_pool_type",
        choices=["round_robin", "least_gpu_memory", "balanced_load"],
        default="round_robin",
    )
    parser.add_argument(
        "--decode_worker_pool_type",
        choices=["round_robin", "least_gpu_memory", "balanced_load"],
        default="least_gpu_memory",
    )
    parser.add_argument("--max_parallem_sum", type=int, default=99999)
    parser.add_argument("--max_occupy_ratio", type=float, default=1.0)
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=DEFAULT_MAX_NUM_BATCHED_TOKENS,
        help=(
            "Token budget of one paged-attn step (vLLM V1 chunked prefill): decodes are "
            "served first, prompts longer than the remaining budget are prefilled in chunks."
        ),
    )
    parser.add_argument(
        "--chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable to prefill every prompt in a single step regardless of length.",
    )
    parser.add_argument("--tensor_parallel_size", type=int, default=None)
    parser.add_argument("--pipeline_parallel_size", type=int, default=None)
    parser.add_argument("--data_parallel_size", type=int, default=None)
    parser.add_argument("--data_parallel_rank", type=int, default=None)
    parser.add_argument("--data_parallel_size_local", type=int, default=None)
    parser.add_argument(
        "--enable_expert_parallel",
        action=argparse.BooleanOptionalAction,
        default=None,
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
    )
    parser.add_argument(
        "--moe_routing_distribution",
        choices=["uniform", "skew", "hot", "burst"],
        default=None,
    )
    parser.add_argument("--moe_hot_experts", type=str, default=None)
    parser.add_argument("--moe_hot_expert_fraction", type=float, default=None)

    parser.add_argument(
        "--verbose", type=str, choices=["none", "simple", "tqdm"], default="tqdm"
    )
    parser.add_argument(
        "--debug-print",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--program_id", type=int, default=0)
    parser.add_argument("--results_path", type=str, default="")
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--dataset_skip_count", type=int, default=0)
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
        help=(
            "Scale timestamped trace arrivals to this QPS using "
            "scale = original_trace_qps / trace_target_qps."
        ),
    )
    parser.add_argument(
        "--workload_type",
        choices=["synthetic", "json_pairs", "qwen_jsonl", "agentx_weka"],
        default="synthetic",
    )
    parser.add_argument("--agentx_concurrency", type=int, default=1)
    parser.add_argument("--agentx_trace_count", type=int, default=8)
    parser.add_argument("--agentx_profile_duration", type=float, default=None)
    parser.add_argument("--agentx_max_requests", type=int, default=None)
    parser.add_argument("--agentx_idle_gap_cap", type=float, default=10.0)
    parser.add_argument(
        "--agentx_warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--agentx_warmup_min_ratio", type=float, default=0.0)
    parser.add_argument("--agentx_warmup_max_ratio", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=0)

    parser.add_argument(
        "--latency_backend",
        type=str,
        default="operator_table",
        help=(
            "'operator_table' (per-operator tables with analytical fallback) "
            "or 'analytical' (formulas only)."
        ),
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
        default="./data",
        help="Directory holding devices/, topologies/, models/ and operator_data/.",
    )

    args = parser.parse_args()

    if args.distribution == "burst":
        args.qps = float("inf")
    main(args)
