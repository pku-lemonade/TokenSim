from dataclasses import asdict
from typing import Any

from pydantic.dataclasses import dataclass

from TokenSim.errors import ConfigurationError


@dataclass
class ParallelConfig:
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    data_parallel_rank: int = 0
    data_parallel_size_local: int = 1
    tensor_parallel_collective_latency_scale: float = 1.0
    pipeline_parallel_activation_latency_scale: float = 1.0
    enable_expert_parallel: bool = False
    expert_placement_strategy: str = "linear"
    all2all_backend: str = "allgather_reducescatter"

    def __post_init__(self) -> None:
        for field_name in (
            "tensor_parallel_size",
            "pipeline_parallel_size",
            "data_parallel_size",
            "data_parallel_size_local",
        ):
            value = getattr(self, field_name)
            if value < 1:
                raise ConfigurationError(f"{field_name} must be at least 1")
        if self.data_parallel_rank < 0:
            raise ConfigurationError("data_parallel_rank must be non-negative")
        if self.data_parallel_rank >= self.data_parallel_size:
            raise ConfigurationError(
                "data_parallel_rank must be smaller than data_parallel_size"
            )
        if self.data_parallel_size_local > self.data_parallel_size:
            raise ConfigurationError(
                "data_parallel_size_local cannot exceed data_parallel_size"
            )
        if self.expert_placement_strategy not in {"linear", "round_robin"}:
            raise ConfigurationError(
                "expert_placement_strategy must be 'linear' or 'round_robin'"
            )
        if self.all2all_backend not in {
            "allgather_reducescatter",
            "naive",
            "deepep_high_throughput",
            "deepep_low_latency",
        }:
            raise ConfigurationError(
                "all2all_backend must be one of "
                + "['allgather_reducescatter', 'naive', "
                + "'deepep_high_throughput', 'deepep_low_latency']"
            )

    @property
    def ranks_per_dp_group(self) -> int:
        return self.tensor_parallel_size * self.pipeline_parallel_size

    @property
    def world_size(self) -> int:
        return self.ranks_per_dp_group * self.data_parallel_size

    @classmethod
    def default(cls) -> "ParallelConfig":
        return cls()

    def override(
        self,
        *,
        tensor_parallel_size: int | None = None,
        pipeline_parallel_size: int | None = None,
        data_parallel_size: int | None = None,
        data_parallel_rank: int | None = None,
        data_parallel_size_local: int | None = None,
        enable_expert_parallel: bool | None = None,
        expert_placement_strategy: str | None = None,
        all2all_backend: str | None = None,
    ) -> "ParallelConfig":
        return ParallelConfig(
            tensor_parallel_size=(
                self.tensor_parallel_size
                if tensor_parallel_size is None
                else tensor_parallel_size
            ),
            pipeline_parallel_size=(
                self.pipeline_parallel_size
                if pipeline_parallel_size is None
                else pipeline_parallel_size
            ),
            data_parallel_size=(
                self.data_parallel_size if data_parallel_size is None else data_parallel_size
            ),
            data_parallel_rank=(
                self.data_parallel_rank if data_parallel_rank is None else data_parallel_rank
            ),
            data_parallel_size_local=(
                self.data_parallel_size_local
                if data_parallel_size_local is None
                else data_parallel_size_local
            ),
            tensor_parallel_collective_latency_scale=(
                self.tensor_parallel_collective_latency_scale
            ),
            pipeline_parallel_activation_latency_scale=(
                self.pipeline_parallel_activation_latency_scale
            ),
            enable_expert_parallel=(
                self.enable_expert_parallel
                if enable_expert_parallel is None
                else enable_expert_parallel
            ),
            expert_placement_strategy=(
                self.expert_placement_strategy
                if expert_placement_strategy is None
                else expert_placement_strategy
            ),
            all2all_backend=(
                self.all2all_backend if all2all_backend is None else all2all_backend
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParallelRankInfo:
    global_rank: int = 0
    rank_in_dp_group: int = 0
    tp_rank: int = 0
    pp_rank: int = 0
    dp_rank: int = 0
    kv_cache_group_id: str = "dp0-pp0"

    @classmethod
    def from_global_rank(
        cls,
        global_rank: int,
        parallel_config: ParallelConfig,
    ) -> "ParallelRankInfo":
        if global_rank < 0:
            raise ConfigurationError("global rank must be non-negative")
        if global_rank >= parallel_config.world_size:
            return cls(
                global_rank=global_rank,
                rank_in_dp_group=global_rank,
                tp_rank=0,
                pp_rank=0,
                dp_rank=0,
                kv_cache_group_id="dp0-pp0",
            )
        rank_in_dp_group = global_rank % parallel_config.ranks_per_dp_group
        dp_rank = global_rank // parallel_config.ranks_per_dp_group
        pp_rank = rank_in_dp_group // parallel_config.tensor_parallel_size
        tp_rank = rank_in_dp_group % parallel_config.tensor_parallel_size
        return cls(
            global_rank=global_rank,
            rank_in_dp_group=rank_in_dp_group,
            tp_rank=tp_rank,
            pp_rank=pp_rank,
            dp_rank=dp_rank,
            kv_cache_group_id=f"dp{dp_rank}-pp{pp_rank}",
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
