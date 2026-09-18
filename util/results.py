from __future__ import annotations

import json
from pathlib import Path
from dataclasses import asdict
from typing import TYPE_CHECKING, List, Any

import numpy as np

from TokenSim.config.psla_config import PSLAConfig, LLMResult, MetricData
from TokenSim.config.config import ClusterConfig
from TokenSim.llm.llm_request import LLMTime, Request
from TokenSim.moe.stats import MoEStats
from TokenSim.mooncake.metrics import MooncakeStats

if TYPE_CHECKING:
    from TokenSim.llm.llm_engine import LLMEngine


def get_prefix_reuse_stats(requests: list[Request]) -> dict[str, float | int]:
    reuse_hit_blocks = sum(req.reuse_hit_blocks for req in requests)
    reuse_miss_blocks = sum(req.reuse_miss_blocks for req in requests)
    reuse_hit_tokens = sum(req.cached_prefill_tokens for req in requests)
    effective_prefill_tokens = sum(
        (
            req.prefill_compute_len
            if req.effective_prefill_tokens is not None
            else req.prefill_len
        )
        for req in requests
    )
    total_blocks = reuse_hit_blocks + reuse_miss_blocks
    prefix_cache_hit_rate = reuse_hit_blocks / total_blocks if total_blocks else 0
    return {
        "reuse_hit_blocks": reuse_hit_blocks,
        "reuse_miss_blocks": reuse_miss_blocks,
        "reuse_hit_tokens": reuse_hit_tokens,
        "effective_prefill_tokens": effective_prefill_tokens,
        "prefix_cache_hit_rate": prefix_cache_hit_rate,
    }


def get_connector_stats(engine: LLMEngine) -> dict[str, float | int]:
    stats = getattr(engine, "connector_stats", None)
    if stats is None:
        return {}
    for worker in getattr(engine, "workers", []):
        for connector in _flatten_connectors(getattr(worker, "connector", None)):
            stats = stats.aggregate(connector.stats)
    for pool_name in ("prefill_workers", "decode_workers"):
        pool = getattr(engine, pool_name, None)
        if pool is not None:
            stats.placement_decision_count += getattr(
                pool,
                "placement_decision_count",
                0,
            )
    return stats.as_dict()


def get_parallel_stats(engine: LLMEngine) -> dict[str, Any]:
    parallel_config = getattr(engine, "parallel_config", None)
    workers = getattr(engine, "workers", [])
    aggregate = None
    for worker in workers:
        communicator = getattr(worker, "parallel_communicator", None)
        stats = getattr(communicator, "stats", None)
        if stats is None:
            continue
        aggregate = stats if aggregate is None else aggregate.aggregate(stats)
    stats_dict = aggregate.as_dict() if aggregate is not None else {}
    dp_counts: dict[int, int] = {}
    for pool_name in ("prefill_workers", "decode_workers"):
        pool = getattr(engine, pool_name, None)
        for dp_rank, count in getattr(pool, "dp_placement_counts", {}).items():
            dp_counts[dp_rank] = dp_counts.get(dp_rank, 0) + count
    per_rank = []
    for worker in workers:
        per_rank.append(
            {
                "worker_id": worker.id,
                "role": worker.role,
                "global_rank": getattr(worker, "global_rank", worker.id),
                "tp_rank": getattr(worker, "tp_rank", 0),
                "pp_rank": getattr(worker, "pp_rank", 0),
                "dp_rank": getattr(worker, "dp_rank", 0),
                "kv_cache_group_id": getattr(worker, "kv_cache_group_id", "dp0-pp0"),
                "owned_expert_ids": getattr(worker, "owned_expert_ids", []),
                "gpu_memory_percent": worker.workload(),
                "cpu_transfer_gb": worker.cpu_workload(),
            }
        )
    if parallel_config is None:
        config_dict = {}
        expected_ranks = len(workers)
    else:
        config_dict = parallel_config.to_dict()
        expected_ranks = parallel_config.world_size
    return {
        "parallel_config": config_dict,
        "parallel_expected_rank_count": expected_ranks,
        "parallel_actual_rank_count": len(workers),
        "parallel_per_rank_utilization": per_rank,
        "parallel_dp_placement_counts": dp_counts,
        **stats_dict,
    }


def get_cache_capacity_stats(engine: LLMEngine) -> dict[str, int | float]:
    """Report KV capacity once per data-parallel replica."""
    replica_workers: dict[int, Any] = {}
    for worker in getattr(engine, "workers", []):
        if getattr(worker, "tp_rank", 0) != 0 or getattr(worker, "pp_rank", 0) != 0:
            continue
        replica_workers.setdefault(getattr(worker, "dp_rank", 0), worker)
    if not replica_workers:
        return {}

    capacities: list[int] = []
    first_cache = None
    for worker in replica_workers.values():
        cache = getattr(worker, "cache_config", None)
        block_manager = getattr(getattr(worker, "scheduler", None), "block_manager", None)
        if cache is None or block_manager is None:
            continue
        first_cache = first_cache or cache
        capacities.append(int(block_manager.num_total_gpu_blocks) * int(cache.block_size))
    if not capacities or first_cache is None:
        return {}
    return {
        "kv_cache_block_size": int(first_cache.block_size),
        "kv_cache_bytes_per_token_per_rank": int(first_cache.size_per_token),
        "kv_cache_capacity_tokens_per_dp_rank": min(capacities),
        "kv_cache_capacity_tokens_total": sum(capacities),
        "model_param_size_bytes_per_rank": float(getattr(first_cache, "model_param_size", 0)),
    }


def get_latency_stats(engine: LLMEngine) -> dict[str, Any]:
    """Provenance of the latency estimates: backend, datasets, match types, missing shapes."""
    from TokenSim.latency.operator_table import OperatorStats

    aggregate = OperatorStats()
    descriptions: dict[str, dict[str, Any]] = {}
    for worker in getattr(engine, "workers", []):
        backend = getattr(worker, "latency_backend", None)
        if backend is None:
            continue
        stats = getattr(backend, "stats", None)
        if isinstance(stats, OperatorStats):
            aggregate = aggregate.aggregate(stats)
        else:
            fallback = getattr(backend, "fallback_backend", None)
            fallback_stats = getattr(fallback, "stats", None)
            if isinstance(fallback_stats, OperatorStats):
                aggregate = aggregate.aggregate(fallback_stats)
        try:
            description = backend.describe()
        except Exception:  # pragma: no cover - defensive
            description = {"backend": type(backend).__name__}
        device_id = getattr(getattr(worker, "device", None), "device_id", getattr(worker, "hardware", "?"))
        descriptions.setdefault(str(device_id), description)
    return {
        "latency_backends": descriptions,
        **aggregate.as_dict(),
        "operator_missing_shapes": aggregate.missing_shape_records()[:200],
    }


def get_moe_stats(engine: LLMEngine) -> dict[str, Any]:
    aggregate: MoEStats | None = None
    placement = getattr(engine, "expert_placement", None)
    moe_config = getattr(engine, "moe_config", None)
    for worker in getattr(engine, "workers", []):
        latency_backend = getattr(worker, "latency_backend", None)
        stats = getattr(latency_backend, "moe_stats", None)
        if stats is None:
            continue
        aggregate = stats if aggregate is None else aggregate.aggregate(stats)
    if aggregate is None:
        aggregate = MoEStats()
    if placement is not None:
        aggregate.expert_placement = placement.to_dict()
    if moe_config is not None:
        aggregate.effective_moe_config = moe_config.to_dict()
    return aggregate.as_dict()


def get_mooncake_stats(engine: LLMEngine) -> dict[str, Any]:
    aggregate = MooncakeStats()
    seen_stats: set[int] = set()
    for worker in getattr(engine, "workers", []):
        for connector in _flatten_connectors(getattr(worker, "connector", None)):
            stats = getattr(connector, "mooncake_stats", None)
            if stats is None:
                continue
            stats_id = id(stats)
            if stats_id in seen_stats:
                continue
            seen_stats.add(stats_id)
            aggregate = aggregate.aggregate(stats)
    return aggregate.as_dict()


def _flatten_connectors(connector: Any) -> list[Any]:
    if connector is None:
        return []
    result = [connector]
    for child in getattr(connector, "children", []):
        result.extend(_flatten_connectors(child))
    return result


def get_agentx_metrics(
    args: Any,
    requests: list[Request],
    duration: float,
    num_gpus: int = 1,
) -> dict[str, Any] | None:
    if getattr(args, "workload_type", None) != "agentx_weka":
        return None
    ttft = [request.prefill_latency for request in requests]
    e2e = [request.total_time for request in requests]
    tpot = [
        request.decode_time_sum / (request.decode_len - 1)
        for request in requests
        if request.decode_len > 1
    ]
    interactivity = [1.0 / value for value in tpot if value > 0]
    normalized_interactivity = [
        request.decode_len / request.total_time
        for request in requests
        if request.total_time > 0
    ]
    output_tokens = sum(request.decode_len for request in requests)
    metadata = dict(getattr(args, "agentx_runtime_metadata", {}) or {})
    return {
        "methodology": "agentx-closed-loop-simulation",
        "official_submission_compatible": False,
        "concurrency": getattr(args, "agentx_concurrency", 1),
        "trace_count": metadata.get("trace_count", 0),
        "play_count": metadata.get("play_count", 0),
        "warmup_request_count": metadata.get("warmup_request_count", 0),
        "warmup_duration_s": metadata.get("warmup_duration_s", 0.0),
        "profile_duration_s": duration,
        "request_count": len(requests),
        "root_request_count": sum(
            getattr(request, "agentx_stream_id", None) == "root" for request in requests
        ),
        "subagent_request_count": sum(
            getattr(request, "agentx_stream_id", None) != "root" for request in requests
        ),
        "input_tokens": sum(request.prefill_len for request in requests),
        "output_tokens": output_tokens,
        "request_throughput_rps": len(requests) / duration if duration else 0.0,
        "output_token_throughput_tps": output_tokens / duration if duration else 0.0,
        "output_token_throughput_per_gpu_tps": (
            output_tokens / duration / max(1, num_gpus) if duration else 0.0
        ),
        "ttft_s": _percentile_summary(ttft),
        "tpot_s": _percentile_summary(tpot),
        "e2e_s": _percentile_summary(e2e),
        "interactivity_tps": _percentile_summary(interactivity),
        "e2e_normalized_interactivity_tps": _percentile_summary(
            normalized_interactivity
        ),
    }


def _percentile_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
        }
    return {
        "count": len(values),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": float(max(values)),
    }


def _flatten_connectors(connector: Any) -> list[Any]:
    if connector is None:
        return []
    result = [connector]
    for child in getattr(connector, "children", []):
        result.extend(_flatten_connectors(child))
    return result


def print_all_stats(
    g_time: LLMTime,
    requests: list[Request],
    engine: LLMEngine,
    duration: float,
    request_count: int,
    prefill_lens: list[int],
):
    request_time = MetricData.from_list(g_time.request_time)
    prefill_time = MetricData.from_list(g_time.prefill_time)
    decode_time = MetricData.from_list(g_time.decode_time)
    decode_max_time = MetricData.from_list(g_time.decode_max_time)
    prefill_idle_time = MetricData.from_list(g_time.prefill_idle)
    decode_idle_time = MetricData.from_list(g_time.decode_idle)

    # Print timing metrics.
    print_metrics(
        request_time,
        prefill_time,
        decode_time,
        decode_max_time,
        prefill_idle_time,
        decode_idle_time,
    )
    prefill_latency = [req.prefill_latency for req in requests]
    decode_latency = [
        req.decode_time_sum / max(1, req.generation_idx - 1) for req in requests
    ]
    # Print latency statistics.
    print_latency_stats(prefill_latency, decode_latency)
    # Print throughput statistics.
    print_throughput_stats(duration, request_count, prefill_lens)
    print(f"Total preemptions: {sum([wkr.preempted_cnt for wkr in engine.workers])}")
    print_prefix_reuse_stats(requests)
    print_connector_stats(engine)
    print_mooncake_stats(engine)
    print_parallel_stats(engine)
    print_moe_stats(engine)
    print_operator_latency_stats(engine)
    # Print SLO statistics.
    print_slo_stats(duration, g_time)


def print_metrics(
    request_time: MetricData,
    prefill_time: MetricData,
    decode_time: MetricData,
    decode_max_time: MetricData,
    prefill_idle_time: MetricData,
    decode_idle_time: MetricData,
):
    """Print aggregate timing metrics.

    Args:
        request_time: End-to-end request timing metric.
        prefill_time: Prefill processing timing metric.
        decode_time: Decode processing timing metric.
        decode_max_time: Maximum decode timing metric.
        prefill_idle_time: Prefill worker idle-time metric.
        decode_idle_time: Decode worker idle-time metric.
    """
    print(f"{request_time=}")
    print(f"{prefill_time=}")
    print(f"{decode_time=}")
    print(f"{decode_max_time=}")
    print(f"{prefill_idle_time=}")
    print(f"{decode_idle_time=}")


def print_latency_stats(prefill_latency: List[float], decode_latency: List[float]):
    print(f"Average prefill latency: {sum(prefill_latency) / len(prefill_latency)}")
    print(f"Max prefill latency: {max(prefill_latency)}")
    print(f"Min prefill latency: {min(prefill_latency)}")
    print(f"Average decode latency: {sum(decode_latency) / len(decode_latency)}")
    print(f"Max decode latency: {max(decode_latency)}")
    print(f"Min decode latency: {min(decode_latency)}")


def print_throughput_stats(dur: float, request_count: int, prefill_lens: List[int]):
    print(f"total time: {dur}")
    print(f"Thoughput: {request_count / dur} r/s, {sum(prefill_lens) / dur} token/s")


def print_prefix_reuse_stats(requests: list[Request]):
    stats = get_prefix_reuse_stats(requests)
    print(
        "Prefix cache: "
        + f"hit_blocks={stats['reuse_hit_blocks']}, "
        + f"miss_blocks={stats['reuse_miss_blocks']}, "
        + f"hit_tokens={stats['reuse_hit_tokens']}, "
        + f"effective_prefill_tokens={stats['effective_prefill_tokens']}, "
        + f"hit_rate={stats['prefix_cache_hit_rate']}"
    )


def print_connector_stats(engine: LLMEngine):
    stats = get_connector_stats(engine)
    if not stats:
        return
    print(
        "KV connector: "
        + f"transfers={stats['connector_transfer_count']}, "
        + f"blocks={stats['connector_transfer_blocks']}, "
        + f"bytes={stats['connector_transfer_bytes']}, "
        + f"latency={stats['connector_transfer_latency']}, "
        + f"load_wait={stats['connector_load_wait_time']}, "
        + f"save_wait={stats['connector_save_wait_time']}"
    )


def print_mooncake_stats(engine: LLMEngine):
    stats = get_mooncake_stats(engine)
    if not (
        stats.get("mooncake_get_count")
        or stats.get("mooncake_put_count")
        or stats.get("mooncake_transferred_bytes")
    ):
        return
    print(
        "Mooncake: "
        + f"gets={stats['mooncake_get_count']}, "
        + f"puts={stats['mooncake_put_count']}, "
        + f"hits={stats['mooncake_store_hit_count']}, "
        + f"misses={stats['mooncake_store_miss_count']}, "
        + f"mem_hits={stats['mooncake_memory_tier_hit_count']}, "
        + f"disk_hits={stats['mooncake_disk_tier_hit_count']}, "
        + f"ssd_read_blocks={stats['mooncake_ssd_read_blocks']}, "
        + f"ssd_write_blocks={stats['mooncake_ssd_write_blocks']}, "
        + f"bytes={stats['mooncake_transferred_bytes']}, "
        + f"latency={stats['mooncake_transfer_latency']}"
    )


def print_parallel_stats(engine: LLMEngine):
    stats = get_parallel_stats(engine)
    config = stats.get("parallel_config", {})
    print(
        "Parallel: "
        + f"config={config}, "
        + f"ranks={stats.get('parallel_actual_rank_count')}/"
        + f"{stats.get('parallel_expected_rank_count')}, "
        + f"dp_placements={stats.get('parallel_dp_placement_counts')}, "
        + f"sync_events={stats.get('parallel_sync_event_count', 0)}, "
        + f"sync_latency={stats.get('parallel_latency_total', 0)}"
    )


def print_moe_stats(engine: LLMEngine):
    stats = get_moe_stats(engine)
    config = stats.get("effective_moe_config", {})
    if not config.get("is_moe_model", False):
        print("MoE: disabled")
        return
    print(
        "MoE: "
        + f"experts={config.get('num_experts')}, "
        + f"top_k={config.get('num_experts_per_tok')}, "
        + f"ep_ranks={stats.get('moe_ep_rank_count')}, "
        + f"imbalance={stats.get('moe_expert_load_imbalance_ratio')}, "
        + f"compute_latency={stats.get('moe_compute_latency')}, "
        + f"all2all_latency={stats.get('moe_all2all_latency')}, "
        + f"straggler_latency={stats.get('moe_straggler_latency')}"
    )


def print_operator_latency_stats(engine: LLMEngine):
    stats = get_latency_stats(engine)
    backends = stats.get("latency_backends", {})
    print(
        "Latency: "
        + f"backends={ {k: v.get('backend') + ':' + str(v.get('operator_backend')) for k, v in backends.items()} }, "
        + f"queries={stats.get('operator_query_count', 0)}, "
        + f"match_types={stats.get('operator_match_type_counts', {})}, "
        + f"missing_shapes={stats.get('operator_missing_shape_count', 0)}"
    )
    components = stats.get("operator_component_seconds", {})
    if components:
        print(
            "Latency breakdown (s): "
            + ", ".join(f"{name}={value:.4f}" for name, value in components.items())
        )


def print_slo_stats(
    dur: float,
    timing: LLMTime,
    prefill_slo: float = 15,
    decode_slo: float = 0.15,
):
    prefill_slo_good_cnt = sum(
        [1 if t <= prefill_slo else 0 for t in timing.prefill_time]
    )
    decode_slo_good_cnt = sum([1 if t <= decode_slo else 0 for t in timing.decode_time])
    req_slo_good_cnt = sum(
        [
            1 if (p <= prefill_slo and d <= decode_slo) else 0
            for p, d in zip(timing.prefill_time, timing.decode_time)
        ]
    )

    decode_max_slo_good_cnt = sum(
        [1 if t <= decode_slo else 0 for t in timing.decode_max_time]
    )
    req_max_slo_good_cnt = sum(
        [
            1 if (p <= prefill_slo and d <= decode_slo) else 0
            for p, d in zip(timing.prefill_time, timing.decode_max_time)
        ]
    )

    print(
        f"Prefill SLO Good Throughput: {prefill_slo_good_cnt} r, {prefill_slo_good_cnt / dur} r/s"
    )
    print(
        f"Max-Decode SLO Good Throughput: {decode_max_slo_good_cnt} r, {decode_max_slo_good_cnt / dur} r/s"
    )
    print(
        f"Prefill and Max-Decode SLO Good Throughput: {req_max_slo_good_cnt} r, {req_max_slo_good_cnt / dur} r/s"
    )
    print(
        f"Decode SLO Good Throughput: {decode_slo_good_cnt} r, {decode_slo_good_cnt / dur} r/s"
    )
    print(
        f"Prefill and Decode SLO Good Throughput: {req_slo_good_cnt} r, {req_slo_good_cnt / dur} r/s"
    )


def export_result(
    args: Any,
    g_time: LLMTime,
    engine: LLMEngine,
    model_config: PSLAConfig,
    cluster: ClusterConfig,
    request_count: int,
    prefill_lens: list[int],
    decode_lens: list[int],
    requests: list[Request],
    notdone: list[int],
    duration: float,
    simulator_wall_time: float = 0,
):
    if getattr(args, "workload_type", None) == "agentx_weka":
        request_time = MetricData.from_list([req.total_time for req in requests])
        prefill_time = MetricData.from_list([req.prefill_latency for req in requests])
        decode_time = MetricData.from_list(
            [
                req.decode_time_sum / max(1, req.generation_idx - 1)
                for req in requests
            ]
        )
    else:
        request_time = MetricData.from_list(g_time.request_time)
        prefill_time = MetricData.from_list(g_time.prefill_time)
        decode_time = MetricData.from_list(g_time.decode_time)
    prefix_reuse_stats = get_prefix_reuse_stats(requests)
    connector_stats = get_connector_stats(engine)
    parallel_stats = get_parallel_stats(engine)
    cache_capacity_stats = get_cache_capacity_stats(engine)
    moe_stats = get_moe_stats(engine)
    mooncake_stats = get_mooncake_stats(engine)
    latency_stats = get_latency_stats(engine)

    result = LLMResult(
        qps=args.qps,
        cluster=cluster,
        request_time=request_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        request_count=request_count,
        prefill_lens=prefill_lens,
        decode_lens=decode_lens,
        batching=args.batching,
        duration=duration,
        simulator_wall_time=simulator_wall_time,
        simulated_time=duration,
        output_qps=request_count / duration,
        output_token_ps=(sum(prefill_lens) + sum(decode_lens)) / duration,
        notdone=notdone,
        preemption_count=sum(worker.preempted_cnt for worker in engine.workers),
        recomputation_count=sum(req.recomputation_count for req in requests),
        recomputed_tokens=sum(req.recomputed_tokens_total for req in requests),
        recompute_service_time=sum(req.recompute_service_time for req in requests),
        **prefix_reuse_stats,
        **connector_stats,
        **parallel_stats,
        **cache_capacity_stats,
        **moe_stats,
        **mooncake_stats,
        **latency_stats,
    )

    if args.results_path == "":
        results_path = (
            Path.cwd()
            / "results"
            / model_config.name
            / Path(args.cluster).parent.name
            / Path(args.cluster).stem
        )
    else:
        results_path = Path(args.results_path)

    results_path.mkdir(parents=True, exist_ok=True)

    result_dict = asdict(result)
    result_dict["cluster"] = asdict(result.cluster)
    result_dict["request_time"] = asdict(result.request_time)
    result_dict["prefill_time"] = asdict(result.prefill_time)
    result_dict["decode_time"] = asdict(result.decode_time)

    if getattr(args, "workload_type", None) == "agentx_weka":
        result_name = f"agentx_c{args.agentx_concurrency}.json"
    else:
        result_name = f"result_{args.qps}.json"
    result_file = results_path / result_name
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)
    missing = latency_stats.get("operator_missing_shapes") or []
    if missing:
        # Shapes the tables could not answer: the collection checklist for the
        # next profiling run on the target device.
        missing_file = results_path / f"missing_shapes_{args.qps}.json"
        with open(missing_file, "w") as f:
            json.dump(missing, f, indent=2)
