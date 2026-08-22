from pydantic.dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from typing import Any

from TokenSim.config.config import ClusterConfig, ParallelConfig
from TokenSim.moe.config import MoEModelConfig, RoutingConfig


@dataclass
class MetricData:
    p50: float
    p99: float
    max: float

    @classmethod
    def from_list(cls, latencies):
        return cls(np.median(latencies), np.percentile(latencies, 99), max(latencies))

    @classmethod
    def inf(cls):
        return cls(999999, 999999, 999999)


@dataclass
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
    simulator_wall_time: float = 0
    simulated_time: float = 0
    reuse_hit_blocks: int = 0
    reuse_miss_blocks: int = 0
    reuse_hit_tokens: int = 0
    effective_prefill_tokens: int = 0
    prefix_cache_hit_rate: float = 0
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
    parallel_latency_total: float = 0
    parallel_tp_collective_latency: float = 0
    parallel_pp_transfer_latency: float = 0
    parallel_ep_all2all_latency: float = 0
    parallel_sync_event_count: int = 0
    parallel_roofline_conversion_count: int = 0
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
    mooncake_local_gpu_hit_tokens: int = 0
    mooncake_memory_hit_tokens: int = 0
    mooncake_disk_hit_tokens: int = 0
    mooncake_load_wait_time: float = 0
    mooncake_save_wait_time: float = 0
    mooncake_pool_keys: list[str] | None = None
    mooncake_offload_tiers: list[str] | None = None
    mooncake_offload_profiles: list[dict[str, Any]] | None = None

    @classmethod
    def from_file(cls, filename):
        return cls(**json.loads(Path(filename).read_text()))


@dataclass
class PSLAConfig:
    name: str
    model: str
    distribution: str
    prefill_mean_len: int
    prefill_range_len: int
    decode_mean_len: int
    decode_range_len: int
    decode_len_distribution: str
    first_token_latency: MetricData
    decode_token_latency: MetricData
    qps: float
    parallel_config: ParallelConfig | None = None
    is_moe_model: bool = False
    num_experts: int = 0
    num_experts_per_tok: int = 1
    moe_intermediate_size: int | None = None
    num_shared_experts: int = 0
    num_moe_layers: int = 0
    first_k_dense_replace: int = 0
    moe_layer_freq: int = 1
    interleave_moe_layer_step: int = 1
    moe_routing: RoutingConfig | None = None
    hidden_size: int | None = None
    intermediate_size: int | None = None
    num_attention_heads: int | None = None
    num_key_value_heads: int | None = None

    def __post_init__(self) -> None:
        self.moe_routing = RoutingConfig.from_value(self.moe_routing)
        self._moe_config = MoEModelConfig.from_model_config(self)

    @classmethod
    def from_file(cls, filename):
        return cls(**json.loads(Path(filename).read_text()))

    def from_args(self, args):
        self.distribution = args.distribution or self.distribution
        self.prefill_mean_len = args.prefill_mean_len or self.prefill_mean_len
        self.prefill_range_len = args.prefill_range_len or self.prefill_range_len
        self.decode_mean_len = args.decode_mean_len or self.decode_mean_len
        self.decode_range_len = args.decode_range_len or self.decode_range_len
        self.decode_len_distribution = (
            args.decode_len_distribution or self.decode_len_distribution
        )
        self._apply_moe_args(args)
        self._moe_config = MoEModelConfig.from_model_config(self)
        return self

    @property
    def moe_config(self) -> MoEModelConfig:
        return self._moe_config

    def _apply_moe_args(self, args) -> None:
        distribution = getattr(args, "moe_routing_distribution", None)
        hot_experts = getattr(args, "moe_hot_experts", None)
        hot_fraction = getattr(args, "moe_hot_expert_fraction", None)
        if distribution is None and hot_experts is None and hot_fraction is None:
            return
        if hot_experts is None:
            parsed_hot_experts = (
                self.moe_routing.hot_experts if self.moe_routing else None
            )
        elif hot_experts == "":
            parsed_hot_experts = []
        else:
            parsed_hot_experts = [int(item) for item in hot_experts.split(",")]
        self.moe_routing = RoutingConfig(
            distribution=(
                distribution
                if distribution is not None
                else (self.moe_routing.distribution if self.moe_routing else "uniform")
            ),
            hot_experts=parsed_hot_experts,
            hot_expert_fraction=(
                hot_fraction
                if hot_fraction is not None
                else (self.moe_routing.hot_expert_fraction if self.moe_routing else 0.5)
            ),
        )
