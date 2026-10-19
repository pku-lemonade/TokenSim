from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.config.components import TransformerComponents, ModelLayerSpec
from TokenSim.hardware._yaml import load_yaml_mapping, require
from TokenSim.hardware.device import normalize_dtype
from TokenSim.moe.config import MoEModelConfig
from TokenSim.mla import MLADimensions
from TokenSim.config.attention_model import AttentionModelMixin
from TokenSim.config.validation import integer, boolean

GATED_ACTIVATIONS = {"swiglu", "geglu", "reglu"}
ACTIVATIONS = GATED_ACTIVATIONS | {"gelu", "relu", "silu"}


@dataclass(frozen=True)
class ModelSpec(AttentionModelMixin):
    """Architecture parameters of a transformer needed for latency estimation.

    Field names follow Hugging Face ``config.json`` where one exists so that
    configs can be transcribed directly.
    """

    model_id: str
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    # Optional compressed KV width for latent-attention models.
    kv_cache_dim: int | None = None
    vocab_size: int = 32000
    activation: str = "swiglu"
    max_position_embeddings: int = 4096
    dtype: str = "fp16"
    activation_dtype: str | None = None
    kv_cache_dtype: str = "fp16"
    sliding_window: int = 0
    attention_windows: tuple[int, ...] = ()
    tie_word_embeddings: bool = False
    aliases: tuple[str, ...] = ()
    moe: MoEModelConfig = field(default_factory=MoEModelConfig)
    sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    display_name: str = ""
    attention_type: str = "mha"
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    rope_head_dim: int = 0
    nope_head_dim: int = 0
    v_head_dim: int = 0
    architecture: str = "transformer"
    components: TransformerComponents | None = None
    operator_dtypes: Mapping[str, str] = field(default_factory=dict)
    operator_modules: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    qk_norm: bool = False

    def __post_init__(self) -> None:
        if self.attention_type not in {"mha", "gqa", "mqa", "mla"}:
            raise ConfigurationError(
                f"model {self.model_id!r}: unknown attention_type={self.attention_type!r}"
            )
        if self.attention_type == "mla":
            try:
                dims = MLADimensions.from_model(self)
            except ValueError as exc:
                raise ConfigurationError(f"model {self.model_id!r}: {exc}") from exc
            if self.head_dim != dims.head_dim:
                raise ConfigurationError(
                    f"model {self.model_id!r}: MLA head_dim must equal nope_head_dim + rope_head_dim"
                )
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "max_position_embeddings",
        ):
            integer(getattr(self, name), f"model {self.model_id!r}: {name}", minimum=1)
        integer(self.sliding_window, "sliding_window")
        if not isinstance(self.attention_windows, (tuple, list)):
            raise ConfigurationError(
                "attention_windows must be an array of layer windows"
            )
        object.__setattr__(self, "attention_windows", tuple(self.attention_windows))
        if self.attention_windows and len(self.attention_windows) != self.num_layers:
            raise ConfigurationError("attention_windows must specify every model layer")
        for window in self.attention_windows:
            integer(window, "attention window")
        boolean(self.tie_word_embeddings, "tie_word_embeddings")
        boolean(self.qk_norm, "qk_norm")
        if self.qk_norm and (
            self.attention_type == "mla" or self.components is not None
        ):
            raise ConfigurationError(
                "qk_norm requires ordinary attention without a compressed module"
            )
        if not isinstance(self.model_id, str) or not self.model_id:
            raise ConfigurationError("model_id must be a nonempty string")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ConfigurationError(
                f"model {self.model_id!r}: num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.activation not in ACTIVATIONS:
            raise ConfigurationError(
                f"model {self.model_id!r}: activation must be one of {sorted(ACTIVATIONS)}"
            )
        if self.moe.enabled and (
            self.moe.first_k_dense_replace
            + (self.moe.num_moe_layers - 1) * self.moe.moe_layer_freq
            >= self.num_layers
        ):
            raise ConfigurationError("MoE layers extend beyond the model's layer count")
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        object.__setattr__(
            self,
            "activation_dtype",
            normalize_dtype(self.activation_dtype or self.dtype),
        )
        object.__setattr__(self, "kv_cache_dtype", normalize_dtype(self.kv_cache_dtype))
        object.__setattr__(
            self,
            "operator_dtypes",
            {str(k): normalize_dtype(v) for k, v in self.operator_dtypes.items()},
        )
        if self.components is not None:
            if self.activation != "swiglu" or self.num_key_value_heads != 1:
                raise ConfigurationError(
                    "compressed transformer requires SwiGLU and one latent KV head"
                )
            if self.kv_cache_dim not in (None, self.head_dim):
                raise ConfigurationError(
                    "compressed transformer latent KV width must equal head_dim"
                )
            self.components.validate_model(
                num_layers=self.num_layers,
                num_attention_heads=self.num_attention_heads,
                head_dim=self.head_dim,
                sliding_window=self.sliding_window,
                num_experts=self.moe.num_experts,
            )
        if self.kv_cache_dim is not None:
            integer(self.kv_cache_dim, "kv_cache_dim", minimum=1)
        if not self.display_name:
            object.__setattr__(self, "display_name", self.model_id)

    # -- derived dimensions ----------------------------------------------

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def gated(self) -> bool:
        return self.activation in GATED_ACTIVATIONS

    @property
    def ffn_projection_count(self) -> int:
        """Input projections of an FFN: gate and up (2) when gated, else 1."""
        return 2 if self.gated else 1

    @property
    def is_moe(self) -> bool:
        return self.moe.enabled

    @property
    def moe_intermediate_size(self) -> int:
        return int(self.moe.moe_intermediate_size or self.intermediate_size)

    def moe_layer_indices(self) -> list[int]:
        return self.moe.moe_layer_indices(self.num_layers) if self.is_moe else []

    def dense_layer_count(self) -> int:
        return self.num_layers - len(self.moe_layer_indices())

    def layer_specs(self, *, include_mtp: bool = True) -> tuple[ModelLayerSpec, ...]:
        if self.components is not None:
            layers = self.components.layers(self.num_layers)
            return layers if include_mtp else layers[: self.num_layers]
        moe_layers = set(self.moe_layer_indices())
        return tuple(
            ModelLayerSpec(
                i,
                (
                    "sliding"
                    if self.attention_window(i)
                    else "latent" if self.kv_cache_dim else "gqa"
                ),
                moe_routing="learned" if i in moe_layers else "none",
            )
            for i in range(self.num_layers)
        )

    def operator_dtype(self, name: str, fallback: str | None = None) -> str:
        return self.operator_dtypes.get(name, fallback or self.dtype)

    def attention_window(self, layer_id: int) -> int:
        if type(layer_id) is not int or not 0 <= layer_id < self.num_layers:
            raise ConfigurationError("attention window requires a valid layer index")
        return (
            self.attention_windows[layer_id]
            if self.attention_windows
            else self.sliding_window
        )

    def cache_signature(self) -> str:
        """Architecture/precision identity, independent of timing data bindings."""
        import json
        import hashlib

        shape = self.to_dict()
        for name in ("operator_modules", "total_params"):
            shape.pop(name)
        shape["moe"] = self.moe.to_dict()
        shape["moe"].pop("routing")
        payload = json.dumps(shape, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @property
    def names(self) -> tuple[str, ...]:
        return (self.model_id, self.display_name, *self.aliases)

    # -- parameter accounting --------------------------------------------

    def attention_params_per_layer(self) -> int:
        if self.attention_type == "mla":
            return self.mla_dimensions.weight_elements(
                self.num_attention_heads, self.hidden_size
            )
        if self.components is not None:
            spec = self.components
            return (
                self.hidden_size * (spec.q_lora_rank + self.head_dim)
                + spec.q_lora_rank * self.q_dim
                + self.q_dim * spec.o_lora_rank
                + spec.o_groups * spec.o_lora_rank * self.hidden_size
            )
        return (
            self.hidden_size * (self.q_dim + 2 * self.kv_dim)
            + self.q_dim * self.hidden_size
            + (2 * self.head_dim if self.qk_norm else 0)
        )

    def dense_ffn_params_per_layer(self) -> int:
        return (
            (self.ffn_projection_count + 1) * self.hidden_size * self.intermediate_size
        )

    def expert_params(self) -> int:
        """Parameters of one routed expert."""
        return (
            (self.ffn_projection_count + 1)
            * self.hidden_size
            * self.moe_intermediate_size
        )

    def embedding_params(self) -> int:
        return (
            self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)
        )

    def total_params(self) -> int:
        if self.components is not None:
            from TokenSim.config.weight_memory import packed_weight_memory

            return packed_weight_memory(self)["parameter_elements"]
        moe_layers = len(self.moe_layer_indices())
        dense_layers = self.num_layers - moe_layers
        params = self.num_layers * self.attention_params_per_layer()
        params += dense_layers * self.dense_ffn_params_per_layer()
        if moe_layers:
            per_layer = (
                self.moe.num_experts + self.moe.num_shared_experts
            ) * self.expert_params()
            per_layer += self.hidden_size * self.moe.num_experts  # router
            params += moe_layers * per_layer
        params += self.embedding_params()
        return int(params)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any], context: str = "model"
    ) -> "ModelSpec":
        if not isinstance(raw, Mapping):
            raise ConfigurationError(
                f"{context}: model configuration must be an object"
            )
        unknown = set(raw) - {f.name for f in fields(cls)} - {"schema_version"}
        if unknown:
            raise ConfigurationError(
                f"{context}: unknown model fields: {sorted(unknown)}"
            )
        raw = dict(raw)
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "kv_cache_dim",
            "vocab_size",
            "max_position_embeddings",
        ):
            if name in raw and not (name == "kv_cache_dim" and raw[name] is None):
                integer(raw[name], f"{context}: {name}", minimum=1)
        if "sliding_window" in raw:
            integer(raw["sliding_window"], "sliding_window")
        if "tie_word_embeddings" in raw:
            boolean(raw["tie_word_embeddings"], "tie_word_embeddings")
        model_id = require(raw, "model_id", context)
        context = f"model {model_id!r}"
        if raw.get("schema_version", 1) != 1:
            raise ConfigurationError(f"{context}: unsupported schema_version")
        heads = int(require(raw, "num_attention_heads", context))
        hidden = int(require(raw, "hidden_size", context))
        head_dim = int(raw.get("head_dim") or hidden // heads)
        attention_type = str(raw.get("attention_type", "mha")).lower()
        if attention_type == "mla" and "head_dim" not in raw:
            head_dim = int(raw.get("nope_head_dim", 0)) + int(
                raw.get("rope_head_dim", 0)
            )
        moe_raw = raw.get("moe") or None
        moe = MoEModelConfig()
        if moe_raw:
            if not isinstance(moe_raw, Mapping):
                raise ConfigurationError(f"{context}: moe must be a mapping")
            known = {
                "num_experts",
                "num_experts_per_tok",
                "moe_intermediate_size",
                "num_shared_experts",
                "num_moe_layers",
                "first_k_dense_replace",
                "moe_layer_freq",
                "interleave_moe_layer_step",
                "routing",
            }
            if set(moe_raw) - known:
                raise ConfigurationError(
                    f"{context}: unknown MoE model fields: {sorted(set(moe_raw) - known)}"
                )
            for name, value in moe_raw.items():
                if name != "routing" and value is not None:
                    integer(
                        value,
                        f"{context}: moe.{name}",
                        minimum=(
                            0
                            if name in {"num_shared_experts", "first_k_dense_replace"}
                            else 1
                        ),
                    )
            moe = MoEModelConfig(
                is_moe_model=True,
                num_experts=int(require(moe_raw, "num_experts", context)),
                num_experts_per_tok=int(moe_raw.get("num_experts_per_tok", 1)),
                moe_intermediate_size=moe_raw.get("moe_intermediate_size"),
                num_shared_experts=int(moe_raw.get("num_shared_experts", 0)),
                num_moe_layers=int(
                    moe_raw.get("num_moe_layers") or require(raw, "num_layers", context)
                ),
                first_k_dense_replace=int(moe_raw.get("first_k_dense_replace", 0)),
                moe_layer_freq=int(moe_raw.get("moe_layer_freq", 1)),
                interleave_moe_layer_step=int(
                    moe_raw.get("interleave_moe_layer_step", 1)
                ),
                routing=moe_raw.get("routing"),
                hidden_size=hidden,
                intermediate_size=int(require(raw, "intermediate_size", context)),
                num_attention_heads=heads,
                num_key_value_heads=int(raw.get("num_key_value_heads", heads)),
            )
        return cls(
            model_id=model_id,
            hidden_size=hidden,
            intermediate_size=int(require(raw, "intermediate_size", context)),
            num_layers=int(require(raw, "num_layers", context)),
            num_attention_heads=heads,
            num_key_value_heads=int(raw.get("num_key_value_heads", heads)),
            head_dim=head_dim,
            kv_cache_dim=(
                int(raw["kv_cache_dim"])
                if raw.get("kv_cache_dim") is not None
                else None
            ),
            vocab_size=int(raw.get("vocab_size", 32000)),
            activation=str(raw.get("activation", "swiglu")).lower(),
            max_position_embeddings=int(raw.get("max_position_embeddings", 4096)),
            dtype=str(raw.get("dtype", "fp16")),
            activation_dtype=(
                str(raw["activation_dtype"])
                if raw.get("activation_dtype") is not None
                else None
            ),
            kv_cache_dtype=str(raw.get("kv_cache_dtype", raw.get("dtype", "fp16"))),
            sliding_window=int(raw.get("sliding_window", 0) or 0),
            attention_windows=raw.get("attention_windows", ()),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            aliases=tuple(str(alias) for alias in raw.get("aliases", [])),
            moe=moe,
            sources={str(k): dict(v) for k, v in (raw.get("sources") or {}).items()},
            display_name=str(raw.get("display_name", model_id)),
            architecture=str(raw.get("architecture", "transformer")),
            components=(
                TransformerComponents.from_mapping(raw["components"])
                if raw.get("components")
                else None
            ),
            operator_dtypes=dict(raw.get("operator_dtypes") or {}),
            operator_modules={
                str(k): dict(v) for k, v in (raw.get("operator_modules") or {}).items()
            },
            qk_norm=raw.get("qk_norm", False),
            attention_type=attention_type,
            **{name: raw.get(name, 0) for name in MLADimensions.__dataclass_fields__},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "aliases": list(self.aliases),
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "kv_cache_dim": self.kv_cache_dim,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings,
            "sliding_window": self.sliding_window,
            **(
                {"attention_windows": list(self.attention_windows)}
                if self.attention_windows
                else {}
            ),
            "tie_word_embeddings": self.tie_word_embeddings,
            **({"qk_norm": True} if self.qk_norm else {}),
            "activation": self.activation,
            "dtype": self.dtype,
            "activation_dtype": self.activation_dtype,
            "kv_cache_dtype": self.kv_cache_dtype,
            "is_moe": self.is_moe,
            "moe": self.moe.to_dict(),
            "sources": {key: dict(value) for key, value in self.sources.items()},
            "total_params": self.total_params(),
            "attention_type": self.attention_type,
            "mla_dimensions": (
                {
                    name: getattr(self, name)
                    for name in MLADimensions.__dataclass_fields__
                }
                if self.attention_type == "mla"
                else None
            ),
            "architecture": self.architecture,
            "operator_dtypes": dict(self.operator_dtypes),
            "components": self.components.to_dict() if self.components else None,
            "operator_modules": dict(self.operator_modules),
            "layers": [vars(layer) for layer in self.layer_specs()],
        }


from TokenSim.config.model_catalog import ModelCatalog
