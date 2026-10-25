"""Cache, communication, storage and operator statistics for a simulation run."""

from __future__ import annotations

from typing import TYPE_CHECKING

from TokenSim.kv_transfer.connectors.base import flatten_connectors
from typing import Any
from TokenSim.llm.llm_request import Request
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
        scheduler = getattr(worker, "scheduler", None)
        if getattr(scheduler, "mtp_enabled", False):
            acceptance = modules.setdefault("speculative", {}).setdefault(
                "acceptance", {}
            )
            acceptance[str(worker.id)] = {
                **scheduler.mtp_acceptance.describe(
                    scheduler.speculative_tokens, scheduler.mtp_acceptance_length
                ),
                "seed": scheduler.mtp_seed,
                **{
                    name: value
                    for name, value in scheduler.stats.items()
                    if name.startswith("mtp_")
                },
            }
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
        if name in modules and not (
            modules[name].get("counts") or modules[name].get("acceptance")
        ):
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
