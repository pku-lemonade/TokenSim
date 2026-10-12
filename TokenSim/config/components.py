"""Architecture metadata of compressed-attention transformers (for example
DeepSeek-V4), independent of serving frameworks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.config.validation import integer, number, boolean


@dataclass(frozen=True)
class ModelLayerSpec:
    layer_id: int
    attention_kind: str
    compress_ratio: int = 1
    moe_routing: str = "none"
    is_mtp: bool = False


@dataclass(frozen=True)
class TransformerComponents:
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
    # Layers with this compression ratio (CSA) select their compressed KV
    # entries with the lightning indexer; other compressed layers (HCA) attend
    # to every compressed entry.
    indexed_compress_ratio: int = 4
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
            integer(getattr(self, name), name, minimum=1)
        integer(self.num_hash_layers, "num_hash_layers")
        integer(self.num_nextn_predict_layers, "num_nextn_predict_layers")
        integer(self.indexed_compress_ratio, "indexed_compress_ratio", minimum=2)
        number(self.routed_scaling_factor, "routed_scaling_factor", positive=True)
        boolean(self.norm_topk_prob, "norm_topk_prob")
        if not self.compress_ratios:
            raise ConfigurationError(
                "compressed transformer compression ratios must be positive integers"
            )
        for ratio in self.compress_ratios:
            integer(ratio, "compress_ratio", minimum=1)
        if len(self.weight_block_size) != 2:
            raise ConfigurationError(
                "compressed transformer weight_block_size must contain two positive dimensions"
            )
        for size in self.weight_block_size:
            integer(size, "weight_block_size", minimum=1)
        if self.indexer_cache_dtype not in ("mxfp4", "fp8"):
            raise ConfigurationError(
                "compressed transformer indexer cache must be mxfp4 or fp8"
            )

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> "TransformerComponents":
        values = dict(raw)
        values["compress_ratios"] = tuple(values["compress_ratios"])
        if "weight_block_size" in values:
            values["weight_block_size"] = tuple(values["weight_block_size"])
        try:
            return cls(**values)
        except TypeError as error:
            raise ConfigurationError(
                f"invalid transformer components: {error}"
            ) from error

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
                "compressed transformer compression ratios must cover all main and MTP layers"
            )
        if self.num_hash_layers > num_layers:
            raise ConfigurationError(
                "compressed transformer hash layers exceed main layer count"
            )
        if any(r != 1 for r in self.compress_ratios[num_layers:]):
            raise ConfigurationError(
                "compressed transformer MTP layers must be SWA-only (ratio 1)"
            )
        if num_attention_heads % self.o_groups or self.qk_rope_head_dim > head_dim:
            raise ConfigurationError(
                "invalid compressed transformer grouped output or RoPE geometry"
            )
        if sliding_window <= 0:
            raise ConfigurationError(
                "compressed transformer requires a positive sliding window"
            )

    def layers(self, num_layers: int) -> tuple[ModelLayerSpec, ...]:
        return tuple(
            ModelLayerSpec(
                layer_id=i,
                attention_kind=(
                    "swa"
                    if ratio == 1
                    else "csa" if ratio == self.indexed_compress_ratio else "hca"
                ),
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
