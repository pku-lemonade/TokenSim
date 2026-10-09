"""Validated serving configuration shared by the engine and its workers."""

from dataclasses import dataclass, field
import math

from TokenSim.config.moe_execution import MoEExecutionConfig
from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class SimulationConfig:
    block_size: int = 16
    batching: str = "dynamic"
    max_num_seqs: int = 64
    max_num_batched_tokens: int | None = 8192
    max_occupy_ratio: float = 1.0
    gpu_memory_utilization: float = 1.0
    prefill_worker_pool_type: str = "round_robin"
    decode_worker_pool_type: str = "least_gpu_memory"
    latency_backend_type: str = "operator_table"
    latency_fallback: str = "table_first"
    operator_backend: str | None = None
    decode_context_bucket: int = 128
    random_seed: int = 0
    debug_print: bool = False
    router_policy: str = "none"
    router_backend_url: str = "http://localhost:8889"
    router_mapping: dict | None = None
    sticky_sessions: bool = False
    dp_lockstep: bool = False
    dp_sync_interval: int = 16
    moe_execution: MoEExecutionConfig = field(default_factory=MoEExecutionConfig)
    attention_backend: str = "generic"
    mhc_backend: str = "generic"
    mtp_acceptance_length: float = 1.0
    mtp_num_speculative_tokens: int | None = None
    mtp_seed: int | None = None
    long_prefill_token_threshold: int = 0
    prefill_schedule_interval: int = 1
    adaptive_long_prefill_threshold: bool = False

    def __post_init__(self):
        for name in (
            "block_size",
            "max_num_seqs",
            "decode_context_bucket",
            "dp_sync_interval",
            "prefill_schedule_interval",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ConfigurationError(f"{name} must be a positive integer")
        if self.max_num_batched_tokens is not None and (
            isinstance(self.max_num_batched_tokens, bool)
            or not isinstance(self.max_num_batched_tokens, int)
            or self.max_num_batched_tokens <= 0
        ):
            raise ConfigurationError("max_num_batched_tokens must be positive or None")
        if self.long_prefill_token_threshold < 0:
            raise ConfigurationError("long_prefill_token_threshold must be nonnegative")
        for name in ("max_occupy_ratio", "gpu_memory_utilization"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 < value <= 1:
                raise ConfigurationError(f"{name} must be in (0, 1]")
        choices = {
            "batching": ("dynamic", "static", "paged-attn"),
            "latency_backend_type": ("operator_table", "analytical"),
            "latency_fallback": ("table_first", "table_only", "analytical_only"),
            "attention_backend": ("generic", "module"),
            "mhc_backend": ("generic", "module"),
            "router_policy": ("none", "consistent_hash", "fixed_mapping"),
            "prefill_worker_pool_type": (
                "round_robin",
                "least_gpu_memory",
                "balanced_load",
            ),
            "decode_worker_pool_type": (
                "round_robin",
                "least_gpu_memory",
                "balanced_load",
            ),
        }
        for name, values in choices.items():
            if getattr(self, name) not in values:
                raise ConfigurationError(f"{name} must be one of {values}")
        if (
            self.latency_backend_type == "analytical"
            or self.latency_fallback == "analytical_only"
        ) and (
            self.attention_backend == "module"
            or self.mhc_backend == "module"
            or MoEExecutionConfig.from_value(self.moe_execution).native
        ):
            raise ConfigurationError(
                "measured operator modules require table-based latency"
            )
        object.__setattr__(
            self, "moe_execution", MoEExecutionConfig.from_value(self.moe_execution)
        )
        if self.mtp_seed is None:
            object.__setattr__(self, "mtp_seed", self.random_seed)
