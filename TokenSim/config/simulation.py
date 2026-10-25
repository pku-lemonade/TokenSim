"""Validated serving configuration shared by the engine and its workers."""

from dataclasses import dataclass, field

from TokenSim.config.moe_execution import MoEExecutionConfig
from TokenSim.trace.config import TraceConfig, DEFAULT_TRACE_CONFIG
from TokenSim.config.router_config import validate_router_mapping
from TokenSim.config.speculative_execution import (
    MTPAcceptanceConfig,
    validate_mtp_token_budget,
)
from TokenSim.errors import ConfigurationError
from TokenSim.config.validation import integer, number, boolean


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
    mtp_acceptance: MTPAcceptanceConfig = field(default_factory=MTPAcceptanceConfig)
    long_prefill_token_threshold: int = 0
    prefill_schedule_interval: int = 1
    adaptive_long_prefill_threshold: bool = False
    trace: TraceConfig = DEFAULT_TRACE_CONFIG

    def __post_init__(self):
        if not isinstance(self.trace, TraceConfig):
            object.__setattr__(self, "trace", TraceConfig.from_mapping(self.trace))
        if self.router_mapping is not None:
            object.__setattr__(
                self, "router_mapping", validate_router_mapping(self.router_mapping)
            )
        for name in (
            "block_size",
            "max_num_seqs",
            "decode_context_bucket",
            "dp_sync_interval",
            "prefill_schedule_interval",
        ):
            integer(getattr(self, name), name, minimum=1)
        if self.max_num_batched_tokens is not None:
            integer(self.max_num_batched_tokens, "max_num_batched_tokens", minimum=1)
        for name in ("long_prefill_token_threshold", "random_seed"):
            integer(getattr(self, name), name)
        for name in ("mtp_seed", "mtp_num_speculative_tokens"):
            if getattr(self, name) is not None:
                integer(getattr(self, name), name)
        acceptance = MTPAcceptanceConfig.from_value(self.mtp_acceptance)
        acceptance.validate(
            self.mtp_num_speculative_tokens or 0, self.mtp_acceptance_length
        )
        object.__setattr__(self, "mtp_acceptance", acceptance)
        validate_mtp_token_budget(
            self.max_num_batched_tokens, self.mtp_num_speculative_tokens or 0
        )
        for name in (
            "debug_print",
            "sticky_sessions",
            "dp_lockstep",
            "adaptive_long_prefill_threshold",
        ):
            boolean(getattr(self, name), name)
        for name in ("max_occupy_ratio", "gpu_memory_utilization"):
            number(getattr(self, name), name, positive=True, maximum=1)
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
