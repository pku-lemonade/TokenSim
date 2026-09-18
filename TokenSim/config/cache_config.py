from __future__ import annotations

from typing import Any

from TokenSim.config.constants import _GB
from TokenSim.config.model_config import ModelSpec
from TokenSim.config.parallel_config import ParallelConfig, ParallelRankInfo
from TokenSim.errors import ConfigurationError
from TokenSim.hardware.device import DeviceSpec, dtype_bytes
from TokenSim.moe.placement import ExpertPlacement

# Host memory is modelled as a bounded swap space; see the note in __init__.
_HOST_SWAP_BYTES = 32768 * _GB


class CacheConfig:
    """Per-rank memory accounting: KV bytes per token and available KV blocks."""

    def __init__(
        self,
        block_size: int,
        device: DeviceSpec,
        model: ModelSpec,
        parallel_config: ParallelConfig | None = None,
        rank_info: ParallelRankInfo | None = None,
        expert_placement: ExpertPlacement | None = None,
        usable_memory_fraction: float = 1.0,
        kv_cache_capacity_tokens_per_dp_rank: int | None = None,
    ):
        self.block_size: int = block_size
        self.device = device
        self.model_spec = model
        self.model: str = model.model_id
        self.parallel_config = parallel_config or ParallelConfig.default()
        self.rank_info = rank_info or ParallelRankInfo()
        self.expert_placement = expert_placement
        self.moe_config = model.moe

        self.total_num_layers = model.num_layers
        self.num_layers_per_rank = stage_layer_count(
            model.num_layers,
            self.parallel_config.pipeline_parallel_size,
            self.rank_info.pp_rank,
        )
        self.num_kv_heads = model.num_key_value_heads
        self.local_kv_heads = local_kv_heads(
            self.num_kv_heads,
            self.parallel_config.tensor_parallel_size,
        )
        self.head_dim = model.head_dim
        kv_bytes = dtype_bytes(model.kv_cache_dtype)
        self.size_per_token_unsharded = int(
            2 * model.kv_dim * kv_bytes * model.num_layers
        )
        if model.kv_cache_dim is not None:
            if model.kv_dim % self.parallel_config.tensor_parallel_size != 0:
                raise ConfigurationError(
                    f"model {model.model_id!r}: kv_cache_dim must be divisible by "
                    f"tensor_parallel_size {self.parallel_config.tensor_parallel_size}"
                )
            local_kv_dim = model.kv_dim // self.parallel_config.tensor_parallel_size
            self.size_per_token = int(
                local_kv_dim * 2 * kv_bytes * self.num_layers_per_rank
            )
        else:
            self.size_per_token = int(
                self.local_kv_heads
                * self.head_dim
                * 2
                * kv_bytes
                * self.num_layers_per_rank
            )

        weight_bytes = dtype_bytes(model.dtype)
        self.model_param_size_unsharded = model.total_params() * weight_bytes
        self.model_param_size = self._rank_params() * weight_bytes

        # Runtime reserves (NCCL buffers, CUDA context, framework workspace)
        # come off the top before the optional usable fraction is applied.
        self.reserved_bytes = device.memory_reserved_bytes.value
        capacity = max(0.0, device.memory_capacity_bytes.value - self.reserved_bytes) * usable_memory_fraction
        self.usable_memory_bytes = capacity
        if capacity <= self.model_param_size:
            raise ConfigurationError(
                f"model {model.model_id!r} does not fit on device {device.device_id!r}: "
                + f"model_param_size={self.model_param_size:.3e}, usable_memory={capacity:.3e} "
                + f"(capacity={device.memory_capacity_bytes.value:.3e}, reserved={self.reserved_bytes:.3e})"
            )
        # FIXME: actual host memory can reach terabytes; it only serves as swap
        # space here, and a huge swap hides preemption effects, so it is bounded.
        self.num_cpu_blocks: int = int(
            min(32768, _HOST_SWAP_BYTES / self.size_per_token // self.block_size)
        )
        self.num_gpu_blocks: int = int(
            (capacity - self.model_param_size) / self.size_per_token // self.block_size
        )
        if kv_cache_capacity_tokens_per_dp_rank is not None:
            if kv_cache_capacity_tokens_per_dp_rank <= 0:
                raise ConfigurationError("KV cache capacity override must be positive")
            self.num_gpu_blocks = (
                kv_cache_capacity_tokens_per_dp_rank // self.block_size
            )

    # -- parameter sharding ----------------------------------------------------

    def _rank_params(self) -> float:
        model = self.model_spec
        tp = self.parallel_config.tensor_parallel_size
        pp = self.parallel_config.pipeline_parallel_size
        if not model.is_moe or self.expert_placement is None:
            return model.total_params() / (tp * pp)
        owned_moe_layers = len(
            self.expert_placement.moe_layers_for_pp_rank(self.rank_info.pp_rank)
        )
        dense_layers = max(0, self.num_layers_per_rank - owned_moe_layers)
        params = self.num_layers_per_rank * model.attention_params_per_layer()
        params += dense_layers * model.dense_ffn_params_per_layer()
        params += owned_moe_layers * model.moe.num_shared_experts * model.expert_params()
        params += owned_moe_layers * model.hidden_size * model.moe.num_experts  # router
        owned_experts = len(self.expert_placement.experts_for_rank(self.rank_info))
        params += owned_moe_layers * owned_experts * model.expert_params()
        params += model.embedding_params() / pp
        return params / tp

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_size": self.block_size,
            "size_per_token": self.size_per_token,
            "num_gpu_blocks": self.num_gpu_blocks,
            "num_cpu_blocks": self.num_cpu_blocks,
            "model_param_size": self.model_param_size,
            "reserved_bytes": self.reserved_bytes,
            "usable_memory_bytes": self.usable_memory_bytes,
            "local_kv_heads": self.local_kv_heads,
            "num_layers_per_rank": self.num_layers_per_rank,
        }


def stage_layer_count(total_layers: int, pp_size: int, pp_rank: int) -> int:
    base = total_layers // pp_size
    remainder = total_layers % pp_size
    return base + (1 if pp_rank < remainder else 0)


def num_kv_heads(model: ModelSpec) -> int:
    return int(model.num_key_value_heads)


def local_kv_heads(kv_heads: int, tensor_parallel_size: int) -> int:
    """KV heads held by one TP rank; heads are replicated when kv_heads < tp."""
    if tensor_parallel_size <= kv_heads:
        if kv_heads % tensor_parallel_size != 0:
            raise ConfigurationError(
                f"KV heads {kv_heads} are not divisible by tensor_parallel_size "
                + f"{tensor_parallel_size}"
            )
        return kv_heads // tensor_parallel_size
    if tensor_parallel_size % kv_heads != 0:
        raise ConfigurationError(
            f"tensor_parallel_size {tensor_parallel_size} must be a multiple of KV heads "
            + f"{kv_heads} when replicating KV heads"
        )
    return 1
