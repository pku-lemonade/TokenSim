"""DeepSeek V4 architecture metadata, independent of serving frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError


@dataclass(frozen=True)
class ModelLayerSpec:
    layer_id: int
    attention_kind: str
    compress_ratio: int = 1
    moe_routing: str = "none"
    is_mtp: bool = False


@dataclass(frozen=True)
class DeepSeekV4Spec:
    # Main layers followed by the distinct MTP modules. K draft iterations
    # reuse these modules, rather than allocating K additional weight copies.
    compress_ratios: tuple[int, ...]
    num_hash_layers: int
    num_nextn_predict_layers: int
    q_lora_rank: int
    o_lora_rank: int
    o_groups: int
    qk_rope_head_dim: int
    index_n_heads: int
    index_head_dim: int
    index_topk: int
    hc_mult: int
    hc_sinkhorn_iters: int = 20
    routed_scaling_factor: float = 2.5
    scoring_func: str = "sqrtsoftplus"
    topk_method: str = "noaux_tc"
    norm_topk_prob: bool = True
    weight_block_size: tuple[int, int] = (128, 128)
    expert_scale_group_size: int = 32
    indexer_cache_dtype: str = "mxfp4"

    def __post_init__(self) -> None:
        for name in (
            "q_lora_rank",
            "o_lora_rank",
            "o_groups",
            "qk_rope_head_dim",
            "index_n_heads",
            "index_head_dim",
            "index_topk",
            "hc_mult",
            "hc_sinkhorn_iters",
            "expert_scale_group_size",
        ):
            if getattr(self, name) <= 0:
                raise ConfigurationError(f"DeepSeek V4 {name} must be positive")
        if self.num_hash_layers < 0 or self.num_nextn_predict_layers < 0:
            raise ConfigurationError("DeepSeek V4 layer counts must be nonnegative")
        if not self.compress_ratios or any(
            r not in (1, 4, 128) for r in self.compress_ratios
        ):
            raise ConfigurationError(
                "DeepSeek V4 compression ratios must be 1, 4 or 128"
            )
        if len(self.weight_block_size) != 2 or min(self.weight_block_size) <= 0:
            raise ConfigurationError(
                "DeepSeek V4 weight_block_size must contain two positive dimensions"
            )
        if self.indexer_cache_dtype not in ("mxfp4", "fp8"):
            raise ConfigurationError("DeepSeek V4 indexer cache must be mxfp4 or fp8")

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "DeepSeekV4Spec":
        values = dict(raw)
        values["compress_ratios"] = tuple(int(r) for r in values["compress_ratios"])
        if "weight_block_size" in values:
            values["weight_block_size"] = tuple(
                int(r) for r in values["weight_block_size"]
            )
        return cls(**values)

    def validate_model(
        self,
        *,
        num_layers: int,
        num_attention_heads: int,
        head_dim: int,
        sliding_window: int,
        num_experts: int,
    ) -> None:
        if len(self.compress_ratios) != num_layers + self.num_nextn_predict_layers:
            raise ConfigurationError(
                "DeepSeek V4 compression ratios must cover all main and MTP layers"
            )
        if self.num_hash_layers > num_layers:
            raise ConfigurationError("DeepSeek V4 hash layers exceed main layer count")
        if any(r != 1 for r in self.compress_ratios[num_layers:]):
            raise ConfigurationError(
                "DeepSeek V4 MTP layers must be SWA-only (ratio 1)"
            )
        if num_attention_heads % self.o_groups or self.qk_rope_head_dim > head_dim:
            raise ConfigurationError(
                "invalid DeepSeek V4 grouped output or RoPE geometry"
            )
        if sliding_window <= 0 or num_experts <= 0:
            raise ConfigurationError(
                "DeepSeek V4 requires a sliding window and routed experts"
            )

    def layers(self, num_layers: int) -> tuple[ModelLayerSpec, ...]:
        return tuple(
            ModelLayerSpec(
                layer_id=i,
                attention_kind={1: "swa", 4: "csa", 128: "hca"}[ratio],
                compress_ratio=ratio,
                moe_routing="hash" if i < self.num_hash_layers else "learned",
                is_mtp=i >= num_layers,
            )
            for i, ratio in enumerate(self.compress_ratios)
        )

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["compress_ratios"] = list(self.compress_ratios)
        result["weight_block_size"] = list(self.weight_block_size)
        return result
