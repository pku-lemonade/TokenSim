from __future__ import annotations

from TokenSim.kv_transfer.connectors.base import flatten_connectors

import json
from pathlib import Path
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, List, Any

import numpy as np

from TokenSim.results import LLMResult, MetricData
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
        for connector in flatten_connectors(getattr(worker, "connector", None)):
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
    # TP/EP group membership and the topology level each spans, taken from
    # rank 0; homogeneous clusters lay every group out the same way.
    groups: dict[str, Any] = {}
    for worker in workers:
        communicator = getattr(worker, "parallel_communicator", None)
        if communicator is not None:
            groups = communicator.describe()
            break
    return {
        "parallel_config": config_dict,
        "parallel_expected_rank_count": expected_ranks,
        "parallel_actual_rank_count": len(workers),
        "parallel_per_rank_utilization": per_rank,
        "parallel_dp_placement_counts": dp_counts,
        "parallel_groups": groups,
        **stats_dict,
    }


def get_cache_capacity_stats(engine: LLMEngine) -> dict[str, Any]:
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
        block_manager = getattr(
            getattr(worker, "scheduler", None), "block_manager", None
        )
        if cache is None or block_manager is None:
            continue
        first_cache = first_cache or cache
        layout = getattr(cache, "hybrid_layout", None)
        capacities.append(
            layout.equivalent_tokens(block_manager.num_total_gpu_blocks)
            if layout is not None
            else int(block_manager.num_total_gpu_blocks) * int(cache.block_size)
        )
    if not capacities or first_cache is None:
        return {}
    result = {
        "kv_cache_block_size": int(first_cache.block_size),
        "kv_cache_bytes_per_token_per_rank": int(first_cache.size_per_token),
        "kv_cache_capacity_tokens_per_dp_rank": min(capacities),
        "kv_cache_capacity_tokens_total": sum(capacities),
        "model_param_size_bytes_per_rank": float(
            getattr(first_cache, "model_param_size", 0)
        ),
    }
    if getattr(first_cache, "hybrid_layout", None) is not None:
        result["kv_cache_capacity_token_kind"] = (
            "group-aware-log-equivalent; allocation uses physical slots"
        )
        result["kv_cache_bytes_per_token_kind"] = (
            "physical stride / logical block; not uniform token storage"
        )
        result["worker_cache_budgets"] = [
            {
                "worker_id": worker.id,
                "dp_rank": worker.dp_rank,
                "cache": worker.cache_config.to_dict(),
                "pool_slots": worker.scheduler.block_manager.num_total_gpu_blocks,
                "peak_used_slots": worker.scheduler.block_manager.peak_used_slots,
                "final_used_slots": worker.scheduler.block_manager.get_num_allocated_blocks(),
                "evictions": worker.scheduler.block_manager.eviction_count,
            }
            for worker in replica_workers.values()
        ]
    return result


def get_latency_stats(engine: LLMEngine) -> dict[str, Any]:
    """Provenance of the latency estimates: backend, datasets, match types, missing shapes."""
    from TokenSim.latency.operator_table import OperatorStats

    aggregate = OperatorStats()
    descriptions: dict[str, dict[str, Any]] = {}
    modules: dict[str, dict[str, Any]] = {}
    for worker in getattr(engine, "workers", []):
        backend = getattr(worker, "latency_backend", None)
        if backend is None:
            continue
        _add_module_stats(modules, backend)
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
        device_id = getattr(
            getattr(worker, "device", None),
            "device_id",
            getattr(worker, "hardware", "?"),
        )
        descriptions.setdefault(str(device_id), description)
    return {
        "latency_backends": descriptions,
        **aggregate.as_dict(),
        **({"module_stats": modules} if _finish_module_stats(modules) else {}),
        # Grouped by table and discrete key with the requested axis ranges;
        # the full report goes to missing_shapes_<qps>.json.
        "operator_missing_shapes": aggregate.missing_shape_records()[:200],
    }


def _merge_sum(total: dict[str, Any], part: dict[str, Any]) -> None:
    """Add ``part`` into ``total``, recursing into nested dictionaries."""
    for key, value in part.items():
        if isinstance(value, dict):
            _merge_sum(total.setdefault(key, {}), value)
        else:
            total[key] = total.get(key, 0) + value


def _add_module_stats(modules: dict[str, dict[str, Any]], backend) -> None:
    """Accumulate one worker's operator-module statistics, keyed by module."""
    draft = getattr(backend, "native_mtp", None)
    for name, counts in (
        ("target_logits", getattr(backend, "target_logits_counts", None)),
        ("main_forward", getattr(backend, "main_forward_counts", None)),
    ):
        if counts is not None:
            _merge_sum(modules.setdefault(name, {}), dict(counts))
    attention = getattr(backend, "native_attention", None)
    if attention is not None:
        _merge_sum(modules.setdefault("attention", {}), attention.stats_dict())
    estimators = [getattr(backend, "native_moe", None)]
    if draft:
        estimators += [draft.initial_moe, draft.decode_moe]
    for estimator in filter(None, estimators):
        stats = estimator.stats_dict()
        rows = stats.pop("max_rank_rows")
        moe = modules.setdefault("moe", {})
        _merge_sum(moe, stats)
        moe["max_rank_rows"] = max(moe.get("max_rank_rows", 0), rows)
    # Main estimates resolve one layer shape and are paid per local layer;
    # each draft estimate is one actual pass.
    for role, estimator, layers in (
        (
            "main",
            getattr(backend, "native_mhc", None),
            getattr(backend, "layers_local", 0),
        ),
        ("draft", getattr(draft, "mhc", None), 1),
    ):
        if estimator is not None:
            mixing = modules.setdefault(
                "residual_mixing",
                {
                    "main_seconds": 0.0,
                    "draft_seconds": 0.0,
                    "counts": {"main": {}, "draft": {}},
                },
            )
            mixing[f"{role}_seconds"] += estimator.seconds * layers
            _merge_sum(mixing["counts"][role], dict(estimator.counts))
    for role, head in (
        ("target", getattr(backend, "residual_output", None)),
        ("draft", getattr(draft, "hc_head", None)),
    ):
        if head is not None:
            _merge_sum(
                modules.setdefault("residual_output", {}),
                {
                    "counts": {f"{role}_calls": head.calls, f"{role}_rows": head.rows},
                    "seconds": {role: head.seconds},
                },
            )
    if draft:
        _merge_sum(
            modules.setdefault("speculative", {}),
            {
                "counts": dict(draft.counts),
                "component_seconds": dict(draft.component_seconds),
            },
        )


def _finish_module_stats(modules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Drop modules that never ran and add derived totals."""
    for name in ("target_logits", "main_forward"):
        if not modules.get(name):
            modules.pop(name, None)
    for name in ("attention", "moe", "speculative"):
        if name in modules and not modules[name].get("counts"):
            del modules[name]
    mixing = modules.get("residual_mixing")
    if mixing is not None:
        if not any(mixing["counts"].values()):
            del modules["residual_mixing"]
        else:
            mixing["seconds"] = mixing["main_seconds"] + mixing["draft_seconds"]
    return modules


def analytical_share_by_table(latency_stats: dict[str, Any]) -> dict[str, float]:
    """Fraction of each table's queries that fell back to the analytical model."""
    shares: dict[str, float] = {}
    for table, counts in (
        latency_stats.get("operator_table_match_counts") or {}
    ).items():
        total = sum(counts.values())
        if total:
            shares[table] = counts.get("analytical", 0) / total
    comm = latency_stats.get("parallel_comm_match_type_counts") or {}
    if sum(comm.values()):
        shares["comm"] = comm.get("analytical", 0) / sum(comm.values())
    return shares


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
        for connector in flatten_connectors(getattr(worker, "connector", None)):
            stats = getattr(connector, "mooncake_stats", None)
            if stats is None:
                continue
            stats_id = id(stats)
            if stats_id in seen_stats:
                continue
            seen_stats.add(stats_id)
            aggregate = aggregate.aggregate(stats)
    return aggregate.as_dict()


def _is_root_stream(stream_id: str | None) -> bool:
    """True if stream_id belongs to a root stream (single or multi-stream)."""
    return stream_id is not None and (
        stream_id == "root" or stream_id.startswith("root:")
    )


@dataclass
class RunReport:
    """How a run ended and where its results go, as decided by the runner."""

    results_path: Path
    qps: float
    batching: str
    termination_reason: str = "completed"
    failure: str | None = None
    # AgentX replay: concurrency and the replay's runtime metadata.
    agentx_concurrency: int | None = None
    agentx_metadata: dict[str, Any] | None = None

    @property
    def result_name(self) -> str:
        if self.agentx_concurrency is not None:
            return f"agentx_c{self.agentx_concurrency}.json"
        return f"result_{self.qps}.json"


def get_agentx_metrics(
    run: RunReport,
    requests: list[Request],
    duration: float,
    num_gpus: int = 1,
) -> dict[str, Any] | None:
    if run.agentx_concurrency is None:
        return None

    requests = [r for r in requests if r.is_done and not r.cancel_requested]
    root_requests = [
        r for r in requests if _is_root_stream(getattr(r.source, "stream_id", None))
    ]
    subagent_requests = [
        r for r in requests if not _is_root_stream(getattr(r.source, "stream_id", None))
    ]

    ttft = [request.prefill_latency for request in requests]
    e2e = [request.total_time for request in requests]
    # TPOT/ITL: mean inter-token latency excluding first token (matches vLLM).
    tpot = [
        request.decode_time_sum / (request.generation_idx - 1)
        for request in requests
        if request.generation_idx > 1
    ]
    interactivity = [1.0 / value for value in tpot if value > 0]
    normalized_interactivity = [
        request.generation_idx / request.total_time
        for request in requests
        if request.total_time > 0
    ]

    input_tokens = sum(request.prefill_len for request in requests)
    output_tokens = sum(request.generation_idx for request in requests)
    cached_input_tokens = sum(request.cached_prefill_tokens for request in requests)
    external_input_tokens = sum(
        request.external_cached_prefill_tokens for request in requests
    )
    computed_input_tokens = sum(
        (request.effective_prefill_tokens or request.prefill_len)
        for request in requests
    )
    metadata = dict(run.agentx_metadata or {})
    if metadata.get("replay_protocol"):
        sources = [r.first_prefill_source for r in requests]
        if any(source is None for source in sources):
            raise ValueError(
                "successful native requests need first-admission source observations"
            )
        gpu_input_tokens = sum(s.gpu_hit_tokens for s in sources)
        external_input_tokens = sum(s.cpu_hit_tokens for s in sources)
        cached_input_tokens = gpu_input_tokens + external_input_tokens
        computed_input_tokens = sum(s.computed_tokens for s in sources)

    def summary(values):
        if not values and metadata.get("replay_protocol"):
            return {
                key: 0 if key == "count" else None
                for key in ("count", "mean", "p50", "p90", "p95", "p99", "max")
            }
        return _percentile_summary(values)

    # In-flight request count at profile end (set by benchmark if available).
    in_flight_at_cutoff = metadata.get("in_flight_at_cutoff", 0)

    result: dict[str, Any] = {
        "methodology": "agentx-closed-loop-simulation",
        "official_submission_compatible": False,
        "concurrency": run.agentx_concurrency,
        "trace_count": metadata.get("trace_count", 0),
        "play_count": metadata.get("play_count", 0),
        "warmup_request_count": metadata.get("warmup_request_count", 0),
        "warmup_duration_s": metadata.get("warmup_duration_s", 0.0),
        "random_seed": metadata.get("random_seed"),
        "workload_seed": metadata.get("workload_seed"),
        "initial_trace_ids": metadata.get("initial_trace_ids", []),
        "trace_sampling_strategy": (
            "shared-sequential-pool"
            if metadata.get("replay_protocol")
            else "seed-shuffled-pool"
        ),
        "profile_duration_s": duration,
        "metric_observation_defined": bool(requests),
        "throughput_denominator_kind": (
            "successful-request-boundaries"
            if metadata.get("replay_protocol") and requests
            else (
                "phase-elapsed-no-success"
                if metadata.get("replay_protocol")
                else "legacy-profile-window"
            )
        ),
        "replay_protocol": metadata.get("replay_protocol"),
        "phase_request_counts": metadata.get("phase_request_counts"),
        "phase_trace_counts": metadata.get("phase_trace_counts"),
        "phase_unique_original_requests": metadata.get(
            "phase_unique_original_requests"
        ),
        "issue_window_s": metadata.get("issue_window_s"),
        "grace_period_s": metadata.get("grace_period_s"),
        "cancellation_timeout_s": metadata.get("cancellation_timeout_s"),
        "cancel_unreturned": metadata.get("cancel_unreturned"),
        "tree_idle_jumps": metadata.get("tree_idle_jumps"),
        "system_idle_jumps": metadata.get("system_idle_jumps"),
        "in_flight_at_cutoff": in_flight_at_cutoff,
        # --- Raw counts (always present for reconciliation) ---
        "request_count": len(requests),
        "root_request_count": len(root_requests),
        "subagent_request_count": len(subagent_requests),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "computed_input_tokens": computed_input_tokens,
        # --- Derived throughput ---
        "request_throughput_rps": len(requests) / duration if duration else 0.0,
        "input_token_throughput_tps": input_tokens / duration if duration else 0.0,
        "input_token_throughput_per_gpu_tps": (
            input_tokens / duration / max(1, num_gpus) if duration else 0.0
        ),
        "output_token_throughput_tps": output_tokens / duration if duration else 0.0,
        "output_token_throughput_per_gpu_tps": (
            output_tokens / duration / max(1, num_gpus) if duration else 0.0
        ),
        # --- Latency distributions (all requests) ---
        "ttft_s": summary(ttft),
        "tpot_s": summary(tpot),
        "e2e_s": summary(e2e),
        "interactivity_tps": summary(interactivity),
        "e2e_normalized_interactivity_tps": summary(normalized_interactivity),
    }
    # Native replay records the input-token decomposition per request. Older
    # Mooncake paths expose their own counters; absent native observations
    # must not silently become a zero CPU hit rate.
    if metadata.get("replay_protocol"):
        result.update(
            cache_source_observation="first-admission",
            gpu_cached_input_tokens=cached_input_tokens - external_input_tokens,
            cpu_cached_input_tokens=external_input_tokens,
            gpu_cache_hit_rate=(
                (cached_input_tokens - external_input_tokens) / input_tokens
                if input_tokens
                else None
            ),
            cpu_cache_hit_rate=(
                external_input_tokens / input_tokens if input_tokens else None
            ),
            readmission_count=sum(
                max(0, r.prefill_admission_count - 1) for r in requests
            ),
            reloaded_gpu_tokens=sum(r.reloaded_gpu_tokens for r in requests),
            reloaded_cpu_tokens=sum(r.reloaded_cpu_tokens for r in requests),
            reloaded_generated_gpu_tokens=sum(
                r.reloaded_generated_gpu_tokens for r in requests
            ),
            reloaded_generated_cpu_tokens=sum(
                r.reloaded_generated_cpu_tokens for r in requests
            ),
            recomputed_tokens=sum(r.recomputed_tokens_total for r in requests),
            latest_gpu_hit_tokens=sum(
                r.cached_prefill_tokens - r.external_cached_prefill_tokens
                for r in requests
            ),
            latest_cpu_hit_tokens=sum(
                r.external_cached_prefill_tokens for r in requests
            ),
        )

    # --- Breakdown by root / subagent ---
    for label, subset in [("root", root_requests), ("subagent", subagent_requests)]:
        sub_ttft = [r.prefill_latency for r in subset]
        sub_e2e = [r.total_time for r in subset]
        sub_tpot = [
            r.decode_time_sum / (r.decode_len - 1) for r in subset if r.decode_len > 1
        ]
        result[f"{label}_ttft_s"] = _percentile_summary(sub_ttft)
        result[f"{label}_tpot_s"] = _percentile_summary(sub_tpot)
        result[f"{label}_e2e_s"] = _percentile_summary(sub_e2e)
        result[f"{label}_output_tokens"] = sum(r.decode_len for r in subset)
        result[f"{label}_input_tokens"] = sum(r.prefill_len for r in subset)

    return result


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


def print_all_stats(
    g_time: LLMTime,
    requests: list[Request],
    engine: LLMEngine,
    duration: float,
    request_count: int,
    prefill_lens: list[int],
    slo: dict[str, Any] | None = None,
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
    print_throughput_stats(
        duration, request_count, [r.generation_idx for r in requests]
    )
    print(f"Total preemptions: {sum([wkr.preempted_cnt for wkr in engine.workers])}")
    print_prefix_reuse_stats(requests)
    print_connector_stats(engine)
    print_mooncake_stats(engine)
    print_parallel_stats(engine)
    print_moe_stats(engine)
    print_operator_latency_stats(engine)
    print_slo_stats(slo)


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


def print_throughput_stats(dur: float, request_count: int, output_tokens: List[int]):
    print(f"total time: {dur}")
    print(
        f"Throughput: {request_count / dur} r/s, {sum(output_tokens) / dur} output token/s"
    )


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
    groups = stats.get("parallel_groups", {})
    print(
        "Parallel: "
        + f"config={config}, "
        + f"ranks={stats.get('parallel_actual_rank_count')}/"
        + f"{stats.get('parallel_expected_rank_count')}, "
        + f"dp_placements={stats.get('parallel_dp_placement_counts')}, "
        + f"tp_group={groups.get('tp_group_fan')}@{groups.get('tp_group_link')}, "
        + f"ep_group={groups.get('ep_group_size')}x{groups.get('ep_group_count')}@{groups.get('ep_group_link')}, "
        + f"sync_events={stats.get('parallel_sync_event_count', 0)}, "
        + f"sync_latency={stats.get('parallel_latency_total', 0)}"
    )


def print_moe_stats(engine: LLMEngine):
    stats = get_moe_stats(engine)
    config = stats.get("effective_moe_config", {})
    if not config.get("is_moe_model", False):
        print("MoE: disabled")
        return
    all2all = get_parallel_stats(engine).get("parallel_ep_all2all_latency", 0.0)
    print(
        "MoE: "
        + f"experts={config.get('num_experts')}, "
        + f"top_k={config.get('num_experts_per_tok')}, "
        + f"ep_ranks={stats.get('moe_ep_rank_count')}, "
        + f"imbalance={stats.get('moe_expert_load_imbalance_ratio')}, "
        + f"compute_latency={stats.get('moe_compute_latency')}, "
        + f"all2all_latency={all2all}, "
        + f"straggler_latency={stats.get('moe_straggler_latency')}"
    )


def print_operator_latency_stats(engine: LLMEngine):
    stats = get_latency_stats(engine)
    stats.update(get_parallel_stats(engine))
    backends = stats.get("latency_backends", {})
    print(
        "Latency: "
        + f"backends={ {k: v.get('backend') + ':' + str(v.get('operator_backend')) for k, v in backends.items()} }, "
        + f"queries={stats.get('operator_query_count', 0)}, "
        + f"match_types={stats.get('operator_match_type_counts', {})}, "
        + f"missing_shapes={stats.get('operator_missing_shape_count', 0)} "
        + f"in {stats.get('operator_missing_shape_groups', 0)} groups"
    )
    formula_share = {
        table: f"{share:.0%}"
        for table, share in analytical_share_by_table(stats).items()
        if share > 0
    }
    if formula_share:
        # Loud on purpose: any share here is simulated time resting on formulas
        # rather than measurements, see missing_shapes_<qps>.json.
        print(f"Latency WARNING analytical share per table: {formula_share}")
    for group in stats.get("operator_missing_shapes", [])[:5]:
        print(
            f"  missing {group['table']} {group['key']} axes={group['axes']} "
            + f"queries={group['query_count']} kinds={group['kinds']}"
        )
    components = stats.get("operator_component_seconds", {})
    if components:
        print(
            "Latency breakdown (s): "
            + ", ".join(f"{name}={value:.4f}" for name, value in components.items())
        )


def slo_report(
    workload_config,
    first_token: list[float],
    decode_token: list[float],
    duration: float,
) -> dict[str, Any] | None:
    """Attainment of the workload's latency targets; ``None`` without targets.

    ``first_token`` and ``decode_token`` hold each request's time to first
    token and mean inter-token latency. A ``max`` target bounds every request
    and goodput counts the requests within all of them; ``p50`` and ``p99``
    targets bound the run's percentiles.
    """
    report: dict[str, Any] = {}
    within = [True] * len(first_token)
    bounded = False
    for name, values in (
        ("first_token_latency", first_token),
        ("decode_token_latency", decode_token),
    ):
        target = getattr(workload_config, name, None)
        bounds = {
            key: value
            for key, value in asdict(target or MetricData(None, None, None)).items()
            if value is not None
        }
        if not bounds:
            continue
        observed = asdict(MetricData.from_list(values))
        report[name] = {
            "target": bounds,
            "observed": {key: observed[key] for key in bounds},
            "met": {
                key: observed[key] is not None and bool(observed[key] <= bound)
                for key, bound in bounds.items()
            },
        }
        if "max" in bounds:
            bounded = True
            within = [
                ok and value <= bounds["max"] for ok, value in zip(within, values)
            ]
    if not report:
        return None
    if bounded:
        report["goodput_requests"] = sum(within)
        report["goodput_requests_per_s"] = sum(within) / duration if duration else 0.0
    return report


def print_slo_stats(report: dict[str, Any] | None) -> None:
    if not report:
        return
    for name in ("first_token_latency", "decode_token_latency"):
        if name in report:
            entry = report[name]
            print(
                f"SLO {name}: "
                + ", ".join(
                    f"{key} {entry['observed'][key]} <= {bound}: {entry['met'][key]}"
                    for key, bound in entry["target"].items()
                )
            )
    if "goodput_requests" in report:
        print(
            f"SLO goodput: {report['goodput_requests']} r, "
            f"{report['goodput_requests_per_s']} r/s"
        )


def export_result(
    run: RunReport,
    g_time: LLMTime,
    engine: LLMEngine,
    cluster: ClusterConfig,
    request_count: int,
    prefill_lens: list[int],
    decode_lens: list[int],
    requests: list[Request],
    notdone: list[int],
    duration: float,
    simulator_wall_time: float = 0,
):
    if run.agentx_concurrency is not None:
        request_time = MetricData.from_list([req.total_time for req in requests])
        first_token = [req.prefill_latency for req in requests]
        decode_token = [
            req.decode_time_sum / max(1, req.generation_idx - 1) for req in requests
        ]
    else:
        request_time = MetricData.from_list(g_time.request_time)
        first_token, decode_token = g_time.prefill_time, g_time.decode_time
    prefill_time = MetricData.from_list(first_token)
    decode_time = MetricData.from_list(decode_token)
    prefix_reuse_stats = get_prefix_reuse_stats(requests)
    connector_stats = get_connector_stats(engine)
    parallel_stats = get_parallel_stats(engine)
    cache_capacity_stats = get_cache_capacity_stats(engine)
    moe_stats = get_moe_stats(engine)
    mooncake_stats = get_mooncake_stats(engine)
    latency_stats = get_latency_stats(engine)

    result = LLMResult(
        qps=run.qps,
        cluster=cluster,
        request_time=request_time,
        prefill_time=prefill_time,
        decode_time=decode_time,
        request_count=request_count,
        prefill_lens=prefill_lens,
        decode_lens=decode_lens,
        batching=run.batching,
        duration=duration,
        simulator_wall_time=simulator_wall_time,
        simulated_time=duration,
        output_qps=sum(r.is_done for r in requests) / duration if duration else 0,
        output_token_ps=(
            sum(r.generation_idx for r in requests) / duration if duration else 0
        ),
        notdone=notdone,
        termination_reason=run.termination_reason,
        failure=run.failure,
        preemption_count=sum(worker.preempted_cnt for worker in engine.workers),
        recomputation_count=sum(req.recomputation_count for req in requests),
        recomputed_tokens=sum(req.recomputed_tokens_total for req in requests),
        recompute_service_time=sum(req.recompute_service_time for req in requests),
        prefill_chunk_count=sum(req.prefill_chunks for req in requests),
        max_num_batched_tokens=getattr(engine, "max_num_batched_tokens", None),
        router_stats=(
            engine.router.describe() if getattr(engine, "router", None) else None
        ),
        dp_step_stats=(
            engine.dp_coordinator.describe()
            if getattr(engine, "dp_coordinator", None)
            else None
        ),
        scheduler_stats=[
            {
                "worker_id": w.id,
                "long_prefill_token_threshold": getattr(
                    w.scheduler, "long_prefill_token_threshold", None
                ),
                "prefill_schedule_interval": getattr(
                    w, "prefill_schedule_interval", None
                ),
                "counts": dict(getattr(w.scheduler, "stats", {})),
            }
            for w in engine.workers
        ],
        **prefix_reuse_stats,
        **connector_stats,
        **parallel_stats,
        **cache_capacity_stats,
        **moe_stats,
        **mooncake_stats,
        **latency_stats,
        agentx_metrics=get_agentx_metrics(
            run,
            requests,
            duration,
            num_gpus=len(engine.workers),
        ),
        slo=slo_report(engine.workload_config, first_token, decode_token, duration),
    )

    results_path = Path(run.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    result_dict = {
        key: value for key, value in asdict(result).items() if value is not None
    }
    result_dict["cluster"] = asdict(result.cluster)
    result_dict["request_time"] = asdict(result.request_time)
    result_dict["prefill_time"] = asdict(result.prefill_time)
    result_dict["decode_time"] = asdict(result.decode_time)

    result_file = results_path / run.result_name
    with open(result_file, "w") as f:
        json.dump(result_dict, f, indent=4)
    if latency_stats.get("operator_missing_shape_groups"):
        # Shapes the tables could not answer, grouped by table and discrete key
        # with the requested axis ranges: the collection checklist for the next
        # profiling run on the target device (compute and communication alike).
        missing_file = results_path / f"missing_shapes_{run.qps}.json"
        with open(missing_file, "w") as f:
            json.dump(_missing_shape_report(engine), f, indent=2)
    return result_dict


def _missing_shape_report(engine: LLMEngine) -> dict[str, Any]:
    from TokenSim.operator_data.coverage import MissingShapeReport

    report = MissingShapeReport()
    for worker in getattr(engine, "workers", []):
        worker_report = getattr(worker, "missing_report", None)
        if worker_report is not None:
            report = report.merge(worker_report)
    return report.to_dict()
