"""Simulation result schema and latency summaries."""

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import numpy as np

from TokenSim.config.config import ClusterConfig


@dataclass
class MetricData:
    p50: float | None
    p99: float | None
    max: float | None

    @classmethod
    def from_list(cls, latencies):
        if not len(latencies):
            return cls(None, None, None)
        return cls(np.median(latencies), np.percentile(latencies, 99), max(latencies))

    @classmethod
    def inf(cls):
        return cls(999999, 999999, 999999)


@dataclass(config=ConfigDict(extra="forbid"))
class LLMResult:
    cluster: ClusterConfig
    qps: float
    request_count: int
    prefill_lens: list[int]
    decode_lens: list[int]
    batching: str
    request_time: MetricData
    prefill_time: MetricData
    decode_time: MetricData
    duration: float
    output_qps: float
    output_token_ps: float
    notdone: list[int]
    preemption_count: int = 0
    recomputation_count: int = 0
    recomputed_tokens: int = 0
    recompute_service_time: float = 0
    # Chunked prefill: context-building steps taken by all prompts and the
    # per-step token budget in force (None: whole prompts in one step).
    prefill_chunk_count: int = 0
    max_num_batched_tokens: int | None = None
    simulator_wall_time: float = 0
    simulated_time: float = 0
    reuse_hit_blocks: int = 0
    reuse_miss_blocks: int = 0
    reuse_hit_tokens: int = 0
    effective_prefill_tokens: int = 0
    prefix_cache_hit_rate: float = 0
    kv_cache_block_size: int = 0
    kv_cache_bytes_per_token_per_rank: int = 0
    kv_cache_capacity_tokens_per_dp_rank: int = 0
    kv_cache_capacity_tokens_total: int = 0
    kv_cache_capacity_token_kind: str | None = None
    kv_cache_bytes_per_token_kind: str | None = None
    worker_cache_budgets: list[dict[str, Any]] | None = None
    router_stats: dict[str, Any] | None = None
    dp_step_stats: dict[str, Any] | None = None
    scheduler_stats: list[dict[str, Any]] | None = None
    model_param_size_bytes_per_rank: float = 0
    connector_transfer_count: int = 0
    connector_transfer_blocks: int = 0
    connector_transfer_bytes: int = 0
    connector_transfer_latency: float = 0
    connector_load_wait_time: float = 0
    connector_save_wait_time: float = 0
    connector_producer_count: int = 0
    connector_consumer_count: int = 0
    connector_both_count: int = 0
    connector_placement_decision_count: int = 0
    parallel_config: dict | None = None
    parallel_expected_rank_count: int = 1
    parallel_actual_rank_count: int = 1
    parallel_per_rank_utilization: list[dict] | None = None
    parallel_dp_placement_counts: dict[int, int] | None = None
    parallel_groups: dict[str, Any] | None = None
    parallel_latency_total: float = 0
    parallel_tp_collective_latency: float = 0
    parallel_pp_transfer_latency: float = 0
    parallel_ep_all2all_latency: float = 0
    parallel_sync_event_count: int = 0
    parallel_tp_shard_event_count: int = 0
    parallel_link_type_counts: dict[str, int] | None = None
    parallel_comm_match_type_counts: dict[str, int] | None = None
    latency_backends: dict[str, Any] | None = None
    operator_query_count: int = 0
    operator_match_type_counts: dict[str, int] | None = None
    operator_table_match_counts: dict[str, Any] | None = None
    operator_component_seconds: dict[str, float] | None = None
    attention_module_seconds: float | None = None
    attention_module_counts: dict[str, int] | None = None
    attention_module_decode_batches: dict[str, int] | None = None
    attention_module_mtp_kv: dict[str, float] | None = None
    attention_module_composition: dict[str, float] | None = None
    target_logits: dict[str, int] | None = None
    main_forward_counts: dict[str, int] | None = None
    residual_output_counts: dict[str, int] | None = None
    residual_output_seconds: dict[str, float] | None = None
    native_mhc_seconds: float | None = None
    native_mhc_main_seconds: float | None = None
    native_mhc_draft_seconds: float | None = None
    native_mhc_counts: dict[str, dict[str, int]] | None = None
    speculative_counts: dict[str, int] | None = None
    speculative_component_seconds: dict[str, float] | None = None
    native_megamoe_routed_seconds: float | None = None
    native_megamoe_shared_seconds: float | None = None
    native_megamoe_max_rank_rows: int | None = None
    native_megamoe_counts: dict[str, int] | None = None
    native_megamoe_composition: dict[str, float] | None = None
    operator_missing_shape_count: int = 0
    operator_missing_shape_groups: int = 0
    operator_missing_per_table: dict[str, Any] | None = None
    operator_step_count: int = 0
    operator_missing_shapes: list[dict[str, Any]] | None = None
    effective_moe_config: dict | None = None
    moe_ep_rank_count: int = 1
    moe_expert_placement: dict | None = None
    moe_per_expert_load: dict[str, int] | None = None
    moe_per_rank_expert_load: dict[str, int] | None = None
    moe_max_expert_load: int = 0
    moe_mean_expert_load: float = 0
    moe_expert_load_imbalance_ratio: float = 0
    moe_compute_latency: float = 0
    moe_all2all_latency: float = 0
    moe_straggler_latency: float = 0
    moe_all2all_event_count: int = 0
    moe_routed_token_count: int = 0
    mooncake_get_count: int = 0
    mooncake_put_count: int = 0
    mooncake_store_hit_count: int = 0
    mooncake_store_miss_count: int = 0
    mooncake_memory_tier_hit_count: int = 0
    mooncake_disk_tier_hit_count: int = 0
    mooncake_ssd_read_blocks: int = 0
    mooncake_ssd_read_bytes: int = 0
    mooncake_ssd_write_blocks: int = 0
    mooncake_ssd_write_bytes: int = 0
    mooncake_transferred_bytes: int = 0
    mooncake_effective_transfer_bandwidth_gbps: float = 0
    mooncake_transfer_latency: float = 0
    mooncake_p2p_latency: float = 0
    mooncake_store_latency: float = 0
    mooncake_admission_count: int = 0
    mooncake_admission_rejection_count: int = 0
    mooncake_eviction_count: int = 0
    mooncake_pending_async_jobs: int = 0
    mooncake_cache_query_tokens: int = 0
    mooncake_local_gpu_hit_tokens: int = 0
    mooncake_memory_hit_tokens: int = 0
    mooncake_disk_hit_tokens: int = 0
    mooncake_load_wait_time: float = 0
    mooncake_save_wait_time: float = 0
    mooncake_pool_keys: list[str] | None = None
    mooncake_offload_tiers: list[str] | None = None
    mooncake_offload_profiles: list[dict[str, Any]] | None = None
    agentx_metrics: dict[str, Any] | None = None
    termination_reason: str = "completed"
    failure: str | None = None

    @classmethod
    def from_file(cls, filename):
        return cls(**json.loads(Path(filename).read_text()))
