from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from TokenSim.errors import ConfigurationError
from TokenSim.hardware._yaml import load_yaml_mapping, require
from TokenSim.hardware.device import normalize_dtype
from TokenSim.moe.config import MoEModelConfig

GATED_ACTIVATIONS = {"swiglu", "geglu", "reglu"}
ACTIVATIONS = GATED_ACTIVATIONS | {"gelu", "relu", "silu"}


@dataclass(frozen=True)
class ModelSpec:
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
    vocab_size: int = 32000
    activation: str = "swiglu"
    max_position_embeddings: int = 4096
    dtype: str = "fp16"
    kv_cache_dtype: str = "fp16"
    sliding_window: int = 0
    tie_word_embeddings: bool = False
    aliases: tuple[str, ...] = ()
    moe: MoEModelConfig = field(default_factory=MoEModelConfig)
    sources: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    display_name: str = ""

    def __post_init__(self) -> None:
        for name in (
            "hidden_size",
            "intermediate_size",
            "num_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        ):
            if int(getattr(self, name)) <= 0:
                raise ConfigurationError(f"model {self.model_id!r}: {name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ConfigurationError(
                f"model {self.model_id!r}: num_attention_heads must be divisible by num_key_value_heads"
            )
        if self.activation not in ACTIVATIONS:
            raise ConfigurationError(
                f"model {self.model_id!r}: activation must be one of {sorted(ACTIVATIONS)}"
            )
        object.__setattr__(self, "dtype", normalize_dtype(self.dtype))
        object.__setattr__(self, "kv_cache_dtype", normalize_dtype(self.kv_cache_dtype))
        if not self.display_name:
            object.__setattr__(self, "display_name", self.model_id)

    # -- derived dimensions ----------------------------------------------

    @property
    def q_dim(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def kv_dim(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def gated(self) -> bool:
        return self.activation in GATED_ACTIVATIONS

    @property
    def ffn_projection_count(self) -> int:
        """``g`` in the task document: gate+up (2) for SwiGLU, 1 otherwise."""
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

    @property
    def names(self) -> tuple[str, ...]:
        return (self.model_id, self.display_name, *self.aliases)

    # -- parameter accounting --------------------------------------------

    def attention_params_per_layer(self) -> int:
        return self.hidden_size * (self.q_dim + 2 * self.kv_dim) + self.q_dim * self.hidden_size

    def dense_ffn_params_per_layer(self) -> int:
        return (self.ffn_projection_count + 1) * self.hidden_size * self.intermediate_size

    def expert_params(self) -> int:
        """Parameters of one routed expert."""
        return (self.ffn_projection_count + 1) * self.hidden_size * self.moe_intermediate_size

    def embedding_params(self) -> int:
        return self.vocab_size * self.hidden_size * (1 if self.tie_word_embeddings else 2)

    def total_params(self) -> int:
        moe_layers = len(self.moe_layer_indices())
        dense_layers = self.num_layers - moe_layers
        params = self.num_layers * self.attention_params_per_layer()
        params += dense_layers * self.dense_ffn_params_per_layer()
        if moe_layers:
            per_layer = (self.moe.num_experts + self.moe.num_shared_experts) * self.expert_params()
            per_layer += self.hidden_size * self.moe.num_experts  # router
            params += moe_layers * per_layer
        params += self.embedding_params()
        return int(params)

    # -- construction -----------------------------------------------------

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], context: str = "model") -> "ModelSpec":
        model_id = str(require(raw, "model_id", context))
        context = f"model {model_id!r}"
        if raw.get("schema_version", 1) != 1:
            raise ConfigurationError(f"{context}: unsupported schema_version")
        heads = int(require(raw, "num_attention_heads", context))
        hidden = int(require(raw, "hidden_size", context))
        head_dim = int(raw.get("head_dim") or hidden // heads)
        moe_raw = raw.get("moe") or None
        moe = MoEModelConfig()
        if moe_raw:
            if not isinstance(moe_raw, Mapping):
                raise ConfigurationError(f"{context}: moe must be a mapping")
            moe = MoEModelConfig(
                is_moe_model=True,
                num_experts=int(require(moe_raw, "num_experts", context)),
                num_experts_per_tok=int(moe_raw.get("num_experts_per_tok", 1)),
                moe_intermediate_size=moe_raw.get("moe_intermediate_size"),
                num_shared_experts=int(moe_raw.get("num_shared_experts", 0)),
                num_moe_layers=int(moe_raw.get("num_moe_layers") or require(raw, "num_layers", context)),
                first_k_dense_replace=int(moe_raw.get("first_k_dense_replace", 0)),
                moe_layer_freq=int(moe_raw.get("moe_layer_freq", 1)),
                interleave_moe_layer_step=int(moe_raw.get("interleave_moe_layer_step", 1)),
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
            vocab_size=int(raw.get("vocab_size", 32000)),
            activation=str(raw.get("activation", "swiglu")).lower(),
            max_position_embeddings=int(raw.get("max_position_embeddings", 4096)),
            dtype=str(raw.get("dtype", "fp16")),
            kv_cache_dtype=str(raw.get("kv_cache_dtype", raw.get("dtype", "fp16"))),
            sliding_window=int(raw.get("sliding_window", 0) or 0),
            tie_word_embeddings=bool(raw.get("tie_word_embeddings", False)),
            aliases=tuple(str(alias) for alias in raw.get("aliases", [])),
            moe=moe,
            sources={str(k): dict(v) for k, v in (raw.get("sources") or {}).items()},
            display_name=str(raw.get("display_name", model_id)),
        )

    def with_moe_override(self, moe_config: MoEModelConfig | None) -> "ModelSpec":
        """Apply MoE metadata supplied by a PSLA file on top of the catalog entry."""
        if moe_config is None or not moe_config.enabled:
            return self
        updates: dict[str, Any] = {"moe": moe_config}
        if moe_config.hidden_size:
            updates["hidden_size"] = moe_config.hidden_size
        if moe_config.intermediate_size:
            updates["intermediate_size"] = moe_config.intermediate_size
        if moe_config.num_attention_heads:
            updates["num_attention_heads"] = moe_config.num_attention_heads
            heads = moe_config.num_attention_heads
            hidden = updates.get("hidden_size", self.hidden_size)
            if hidden % heads == 0 and self.head_dim * self.num_attention_heads != hidden:
                updates["head_dim"] = hidden // heads
        if moe_config.num_key_value_heads:
            updates["num_key_value_heads"] = moe_config.num_key_value_heads
        return replace(self, **updates)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "activation": self.activation,
            "dtype": self.dtype,
            "kv_cache_dtype": self.kv_cache_dtype,
            "is_moe": self.is_moe,
            "total_params": self.total_params(),
        }


class ModelCatalog:
    def __init__(self, models: Iterable[ModelSpec] = ()) -> None:
        self._models: dict[str, ModelSpec] = {}
        self._index: dict[str, str] = {}
        for model in models:
            self.add(model)

    def add(self, model: ModelSpec) -> None:
        if model.model_id in self._models:
            raise ConfigurationError(f"duplicate model_id {model.model_id!r}")
        self._models[model.model_id] = model
        for name in model.names:
            key = _alias_key(name)
            existing = self._index.get(key)
            if existing is not None and existing != model.model_id:
                raise ConfigurationError(
                    f"model alias {name!r} is claimed by both {existing!r} and {model.model_id!r}"
                )
            self._index[key] = model.model_id

    @classmethod
    def load(cls, directory: str | Path) -> "ModelCatalog":
        directory = Path(directory)
        if not directory.is_dir():
            raise ConfigurationError(f"model catalog directory not found: {directory}")
        catalog = cls()
        for path in sorted(directory.glob("*.yaml")):
            catalog.add(ModelSpec.from_mapping(load_yaml_mapping(path), context=str(path)))
        if not catalog:
            raise ConfigurationError(f"model catalog {directory} has no *.yaml entries")
        return catalog

    def get(self, name: str) -> ModelSpec:
        model_id = self._index.get(_alias_key(name))
        if model_id is None:
            raise ConfigurationError(
                f"unknown model {name!r}; known models: {sorted(self._models)}"
            )
        return self._models[model_id]

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and _alias_key(name) in self._index

    def __len__(self) -> int:
        return len(self._models)

    def __iter__(self):
        return iter(self._models.values())


def _alias_key(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())
