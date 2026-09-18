#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing
import sys
import time
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark import main as run_benchmark
from TokenSim.workload.agentx import load_agentx_traces


@dataclass(frozen=True)
class Case:
    name: str
    cluster: str
    concurrency: int
    kv_transfer_config: str | None


CASES = {
    case.name: case
    for case in (
        Case("b200-hbm", "./data/clusters/64_b200/tp8_dp8_ep.json", 196, None),
        Case(
            "b200-dram",
            "./data/clusters/64_b200/tp8_dp8_ep.json",
            196,
            "./data/kv_transfer/mooncake_store_dram_agentx.json",
        ),
        Case("b300-hbm", "./data/clusters/64_b300/tp8_dp8_ep.json", 384, None),
        Case(
            "b300-dram",
            "./data/clusters/64_b300/tp8_dp8_ep.json",
            384,
            "./data/kv_transfer/mooncake_store_dram_agentx.json",
        ),
    )
}

_SHARED_TRACES = None


def benchmark_args(
    *,
    case: Case,
    dataset_path: str,
    result_root: Path,
    trace_count: int,
    profile_duration: float,
) -> Namespace:
    hardware, cache_mode = case.name.split("-", 1)
    return Namespace(
        sim_time=None,
        qps=1.0,
        batching="paged-attn",
        distribution="uniform",
        request_count=100,
        prefill_mean_len=None,
        prefill_range_len=None,
        decode_mean_len=None,
        decode_range_len=None,
        decode_len_distribution="uniform",
        block_size=64,
        model_config_path="./data/psla/deepseek-v4-proxy.json",
        cluster=case.cluster,
        kv_transfer_config=case.kv_transfer_config,
        prefill_worker_pool_type="round_robin",
        decode_worker_pool_type="least_gpu_memory",
        max_parallem_sum=2_000_000,
        max_occupy_ratio=1.0,
        tensor_parallel_size=None,
        pipeline_parallel_size=None,
        data_parallel_size=None,
        data_parallel_rank=None,
        data_parallel_size_local=None,
        enable_expert_parallel=None,
        expert_placement_strategy=None,
        all2all_backend=None,
        moe_routing_distribution=None,
        moe_hot_experts=None,
        moe_hot_expert_fraction=None,
        verbose="none",
        debug_print=False,
        program_id=0,
        results_path=str(result_root / hardware / cache_mode),
        dataset_path=dataset_path,
        dataset_skip_count=0,
        trace_timestamp_scale=None,
        trace_target_qps=None,
        workload_type="agentx_weka",
        agentx_concurrency=case.concurrency,
        agentx_trace_count=trace_count,
        agentx_profile_duration=profile_duration,
        agentx_max_requests=None,
        agentx_idle_gap_cap=10.0,
        agentx_warmup=True,
        agentx_warmup_min_ratio=0.0,
        agentx_warmup_max_ratio=1.0,
        random_seed=0,
        latency_backend="operator_table",
        latency_fallback="table_first",
        operator_backend="vllm",
        decode_context_bucket=128,
        data_root="./data",
    )


def run_case(
    case_name: str,
    dataset_path: str,
    result_root: str,
    profile_duration: float,
) -> dict[str, float | str]:
    if _SHARED_TRACES is None:
        raise RuntimeError("AgentX traces were not initialized")
    case = CASES[case_name]
    print(f"Running {case.name}: concurrency={case.concurrency}", flush=True)
    case_start = time.perf_counter()
    run_benchmark(
        benchmark_args(
            case=case,
            dataset_path=dataset_path,
            result_root=Path(result_root),
            trace_count=len(_SHARED_TRACES),
            profile_duration=profile_duration,
        ),
        preloaded_agentx_traces=_SHARED_TRACES,
    )
    wall_time = time.perf_counter() - case_start
    print(f"Completed {case.name} in {wall_time:.2f}s", flush=True)
    gc.collect()
    return {"name": case.name, "wall_time_s": wall_time}


def run_case_child(
    case_name: str,
    dataset_path: str,
    result_root: str,
    profile_duration: float,
    queue,
) -> None:
    try:
        queue.put(
            (
                True,
                run_case(
                    case_name,
                    dataset_path,
                    result_root,
                    profile_duration,
                ),
            )
        )
    except BaseException as exc:
        queue.put((False, {"name": case_name, "error": repr(exc)}))
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path")
    parser.add_argument("result_root")
    parser.add_argument("--trace-count", type=int, default=393)
    parser.add_argument("--profile-duration", type=float, default=1800.0)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=sorted(CASES),
        default=list(CASES),
    )
    parser.add_argument("--parallel", action="store_true")
    args = parser.parse_args()

    result_root = Path(args.result_root)
    result_root.mkdir(parents=True, exist_ok=True)
    load_start = time.perf_counter()
    global _SHARED_TRACES
    _SHARED_TRACES = load_agentx_traces(
        args.dataset_path,
        trace_count=args.trace_count,
    )
    traces = _SHARED_TRACES
    load_elapsed = time.perf_counter() - load_start
    print(
        f"Loaded {len(traces)} AgentX traces once in {load_elapsed:.2f}s; "
        f"requests={sum(trace.request_count for trace in traces)}"
    )

    manifest = {
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "trace_count": len(traces),
        "dataset_request_count": sum(trace.request_count for trace in traces),
        "profile_duration_s": args.profile_duration,
        "load_wall_time_s": load_elapsed,
        "cases": [],
    }
    if args.parallel and len(args.cases) > 1:
        context = multiprocessing.get_context("fork")
        queue = context.Queue()
        processes = [
            context.Process(
                target=run_case_child,
                args=(
                    case_name,
                    args.dataset_path,
                    str(result_root),
                    args.profile_duration,
                    queue,
                ),
                name=f"agentx-{case_name}",
            )
            for case_name in args.cases
        ]
        for process in processes:
            process.start()
        outcomes = [queue.get() for _ in processes]
        for process in processes:
            process.join()
        failures = [payload for ok, payload in outcomes if not ok]
        if failures:
            raise RuntimeError(f"AgentX comparison cases failed: {failures}")
        manifest["cases"].extend(payload for _, payload in outcomes)
    else:
        for case_name in args.cases:
            manifest["cases"].append(
                run_case(
                    case_name,
                    args.dataset_path,
                    str(result_root),
                    args.profile_duration,
                )
            )

    (result_root / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
