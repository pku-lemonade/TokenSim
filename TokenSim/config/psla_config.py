from pydantic.dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from typing import Any

from TokenSim.config.config import ClusterConfig, ParallelConfig
from TokenSim.moe.config import MoEModelConfig, RoutingConfig
from TokenSim.results import MetricData


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
        # ``is not None`` rather than ``or``: an explicit 0 (e.g. --decode_range_len 0)
        # must override the file value instead of being treated as "unset".
        def pick(name):
            value = getattr(args, name, None)
            return getattr(self, name) if value is None else value

        self.qps = pick("qps")
        self.distribution = pick("distribution")
        self.prefill_mean_len = pick("prefill_mean_len")
        self.prefill_range_len = pick("prefill_range_len")
        self.decode_mean_len = pick("decode_mean_len")
        self.decode_range_len = pick("decode_range_len")
        self.decode_len_distribution = pick("decode_len_distribution")
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
